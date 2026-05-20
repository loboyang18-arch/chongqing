"""
V18 — Conv2D 逐小时预测：重庆 24h 电价（默认 da_clearing_price）

环境变量 V18_TARGET_COL 可改为列名（如 rt_clearing_price），或特殊值
spread_rt_minus_da / rt_da_spread（小时级：当日 15min 实时与日前各自 4 点均值之差）。
输出文件名与可视化标签随之切换；默认仍为日前出清电价。

借鉴内蒙 V6 的 2D 网格思想，将多通道 15min 序列组织为
(C_TOTAL, H_SLOTS, LOOKBACK_DAYS) 的图像张量，用 Conv2d 捕捉
日内模式 × 跨日趋势的联合特征。

输入: (C, H_SLOTS, 7)
  - C 个特征通道（Lag0 可用特征 + Lag1 历史 + Lag2 历史 + 时间编码）
  - H_SLOTS 个 15min 时间槽（默认 h-5..h+1 共 7h = 28 槽，可通过环境变量调整）
  - 7 天 lookback（按 Lag0/Lag1/Lag2 映射对齐）
  - 每个样本预测 1 小时目标电价（由 V18_TARGET_COL 指定）
  - 一天产生 24 个样本

训练配置：
  - 3 层 Conv2d + BN + GELU，2 层 FC
  - Warmup 10ep + CosineAnnealing
  - 固定 train/val/test 切分见 src/experiment/splits.py（TRAIN_END / VAL_END / TEST_*）

训练过采样默认关 V18_TRAIN_OVERSAMPLE=1；残差 MC 默认关（V18_RESIDUAL_MC=1 / V18_TRAIN_OVERSAMPLE>1 可开）
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import OUTPUT_DIR
from .model_v16_nhits import (
    build_feature_matrix,
    EFFECTIVE_START, EFFECTIVE_END, TRAIN_END, VAL_END, TEST_END, TEST_START,
    HIST_COLS, FUTR_COLS, N_LAG1, N_HIST, N_FUTR,
)
from price_forecast_eval import quick_shape_report

logger = logging.getLogger(__name__)

# ── Channel definitions (重庆) ────────────────────────────────────
# Lag0: D 日可用的预测类特征（日前即知）
LAG0_COLS = [c for c in FUTR_COLS if c not in (
    "minute_of_day_sin", "minute_of_day_cos", "dow_sin", "dow_cos",
)]
# Lag1: D-7..D-1 出清价 + 可靠性价 + 出清量（见 model_v16_nhits.LAG1_HIST）
LAG1_COLS = list(HIST_COLS[:N_LAG1])
# Lag2: D-8..D-2 负荷/出力/断面/申报等（无出清价）
LAG2_COLS = list(HIST_COLS[N_LAG1:])

C_LAG0 = len(LAG0_COLS) + 4   # + 4 time encodings
C_LAG1 = len(LAG1_COLS)
C_LAG2 = len(LAG2_COLS)
C_TOTAL = C_LAG0 + C_LAG1 + C_LAG2

LOOKBACK_DAYS = 7
SLOTS_PER_HOUR = 4
CONTEXT_BEFORE = int(os.environ.get("V18_CTX_BEFORE", "5"))  # hours before h
CONTEXT_AFTER = int(os.environ.get("V18_CTX_AFTER", "1"))    # hours after h
CONTEXT_HOURS = CONTEXT_BEFORE + 1 + CONTEXT_AFTER           # total hours in window
H_SLOTS = CONTEXT_HOURS * SLOTS_PER_HOUR                     # 28 (default)
SPD = 96                       # slots per day (15min)

TARGET_COL = os.environ.get("V18_TARGET_COL", "da_clearing_price").strip() or "da_clearing_price"

# 特殊目标：小时级 (RT−DA)，由 _build_daily_arrays 从两列合成
V18_SPREAD_TARGET_NAMES = frozenset({"spread_rt_minus_da", "rt_da_spread"})

V18_DELTA_TARGET = os.environ.get("V18_DELTA_TARGET", "0").strip().lower() in ("1", "true", "yes")


def _v18_is_spread_target() -> bool:
    return TARGET_COL in V18_SPREAD_TARGET_NAMES


def _v18_is_rt_target() -> bool:
    return TARGET_COL in ("rt_clearing_price", "realtime_clearing_price")


def _v18_result_csv_path(out_dir: Path) -> Path:
    if _v18_is_rt_target():
        name = "rt_result.csv"
    elif _v18_is_spread_target():
        name = "spread_result.csv"
    else:
        name = "da_result.csv"
    return out_dir / name


def _v18_viz_label() -> str:
    """图例/日志显示名；可由环境变量 V18_VIZ_LABEL 覆盖（V25 等复用 run_v18 时设置）。"""
    override = os.environ.get("V18_VIZ_LABEL", "").strip()
    if override:
        return override
    if _v18_is_rt_target():
        return "V18-Conv2D-RT"
    if _v18_is_spread_target():
        return "V18-Conv2D-Spread"
    return "V18-Conv2D"

# ── Device ─────────────────────────────────────────────────────────
DEVICE = torch.device(
    "mps" if getattr(torch.backends, "mps", None)
             and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ── Constants ──────────────────────────────────────────────────────
MAX_EPOCHS = int(os.environ.get("V18_EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("V18_BS", "64"))
LR = float(os.environ.get("V18_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("V18_WD", "1e-4"))
DROPOUT = float(os.environ.get("V18_DROPOUT", "0.0"))
WARMUP_EPOCHS = 10

# 训练过采样：逻辑上把每个基础样本重复 K 倍（不复制存储），每 epoch 更多 step；
# 第 2..K 份「模拟样本」对归一化标签做一次较轻的残差池抽样，与首份区分。
V18_TRAIN_OVERSAMPLE = max(1, int(os.environ.get("V18_TRAIN_OVERSAMPLE", "1")))
V18_OVERSAMPLE_RESID_SCALE = float(os.environ.get("V18_OVERSAMPLE_RESID_SCALE", "0.28"))

# 残差池 MC 目标增强（仅训练集首份 rep=0；验证/预测不启用）
# 思路：按小时去均值得到归一化残差；rep=0 上以概率 p、最多 NPASS 次叠加（温和）。
V18_RESIDUAL_MC = int(os.environ.get("V18_RESIDUAL_MC", "0"))
V18_RESIDUAL_MC_P = float(os.environ.get("V18_RESIDUAL_MC_P", "0.28"))
V18_RESIDUAL_MC_SCALE = float(os.environ.get("V18_RESIDUAL_MC_SCALE", "0.35"))
V18_RESIDUAL_MC_NPASS = int(os.environ.get("V18_RESIDUAL_MC_NPASS", "1"))

V18_DIR = OUTPUT_DIR / "v18_conv2d"

# 滚动评估（末段连续日按周切，每周之前全部为训练集）
ROLLING_TOTAL_DAYS = int(os.environ.get("V18_ROLLING_TOTAL_DAYS", "28"))
ROLLING_WEEK_DAYS = int(os.environ.get("V18_ROLLING_WEEK_DAYS", "7"))


# ── Helpers ────────────────────────────────────────────────────────
def _log_device():
    logger.info("PyTorch %s | device=%s", torch.__version__, DEVICE)


def _seed(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    if DEVICE.type == "mps":
        torch.mps.manual_seed(s)
    elif DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(s)


def _build_daily_arrays(df: pd.DataFrame):
    """
    按天切分为 96 点 (15min) 日矩阵。
    day_lag0[d] = (96, C_LAG0), day_lag1[d] = (96, C_LAG1), day_lag2[d] = (96, C_LAG2)
    day_targets[d] = (24,) 小时级目标：默认日前出清 4 槽均值；或实时出清；或 RT−DA 价差。
    """
    start_date = df.index.min().normalize().date()
    end_date = df.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")

    day_lag0: Dict = {}
    day_lag1: Dict = {}
    day_lag2: Dict = {}
    day_targets: Dict = {}
    valid: List = []

    feat_cols = LAG0_COLS + LAG1_COLS + LAG2_COLS
    df[feat_cols] = df[feat_cols].ffill()

    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        raw = df.reindex(grid)

        if raw[feat_cols].isna().all().any():
            continue

        l0 = raw[LAG0_COLS].values.astype(np.float32)  # (96, len(LAG0_COLS))
        steps = np.arange(96, dtype=np.float32)
        dow = float(pd.Timestamp(d).dayofweek)
        te = np.column_stack([
            np.sin(2 * np.pi * steps / 96),
            np.cos(2 * np.pi * steps / 96),
            np.full(96, np.sin(2 * np.pi * dow / 7), dtype=np.float32),
            np.full(96, np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        ])  # (96, 4)
        day_lag0[d] = np.concatenate([l0, te], axis=1).astype(np.float32)

        day_lag1[d] = raw[LAG1_COLS].values.astype(np.float32)
        day_lag2[d] = raw[LAG2_COLS].values.astype(np.float32)

        # 目标：单序列 4 槽→小时均值；或 RT−DA 小时价差（同口径）
        if _v18_is_spread_target():
            if "rt_clearing_price" not in raw.columns or "da_clearing_price" not in raw.columns:
                continue
            v_rt = raw["rt_clearing_price"].to_numpy(dtype=np.float64, copy=False)
            v_da = raw["da_clearing_price"].to_numpy(dtype=np.float64, copy=False)
            if v_rt.size != 96 or v_da.size != 96:
                continue
            h_rt = np.nanmean(v_rt.reshape(24, 4), axis=1)
            h_da = np.nanmean(v_da.reshape(24, 4), axis=1)
            spread_h = h_rt - h_da
            if spread_h.size == 24 and np.isfinite(spread_h).all():
                day_targets[d] = spread_h.astype(np.float32)
                valid.append(d)
        elif TARGET_COL in raw.columns:
            v = raw[TARGET_COL].to_numpy(dtype=np.float64, copy=False)
            if v.size == 96:
                hourly_vals = np.nanmean(v.reshape(24, 4), axis=1)
                if hourly_vals.size == 24 and np.isfinite(hourly_vals).all():
                    day_targets[d] = hourly_vals.astype(np.float32)
                    valid.append(d)

    # delta targets (hour-to-hour) — always computed for dual-task support
    day_delta_targets: Dict = {}
    day_anchors: Dict = {}
    sorted_days = sorted(day_targets.keys())
    for i, d in enumerate(sorted_days):
        abs_vals = day_targets[d]
        prev_d = sorted_days[i - 1] if i > 0 else None
        if prev_d is not None and prev_d in day_targets:
            anchor = float(day_targets[prev_d][-1])
        else:
            anchor = float(abs_vals[0])
        day_anchors[d] = anchor
        delta = np.empty(24, dtype=np.float32)
        delta[0] = abs_vals[0] - anchor
        delta[1:] = abs_vals[1:] - abs_vals[:-1]
        day_delta_targets[d] = delta

    if V18_DELTA_TARGET:
        day_targets = day_delta_targets
        logger.info("Delta target mode: predicting hour-to-hour price changes")

    valid = sorted(valid)
    logger.info(
        "Daily arrays: %d days total, %d with valid target",
        len(day_lag0), len(valid),
    )
    return valid, day_lag0, day_lag1, day_lag2, day_targets, day_anchors, day_delta_targets


def _get_hour_slots(day_arrays: Dict, d, h: int,
                    ctx_before: int = None, ctx_after: int = None):
    """
    获取日 d 第 h 小时 ±context 窗口的特征切片。
    窗口: [h - ctx_before, h + ctx_after]（含）共 (ctx_before+1+ctx_after) 小时。
    返回 (n_slots, C)。超出当日 0-23 边界时向相邻日借数据。

    ctx_before/ctx_after 不指定时使用模块级 CONTEXT_BEFORE/CONTEXT_AFTER。
    """
    if ctx_before is None:
        ctx_before = CONTEXT_BEFORE
    if ctx_after is None:
        ctx_after = CONTEXT_AFTER

    n_hours = ctx_before + 1 + ctx_after
    n_slots = n_hours * 4

    arr = day_arrays[d]  # (96, C)
    C = arr.shape[1]

    start_slot = (h - ctx_before) * 4
    end_slot = (h + ctx_after + 1) * 4

    if 0 <= start_slot and end_slot <= 96:
        return arr[start_slot:end_slot]

    result = np.zeros((n_slots, C), dtype=np.float32)
    out_idx = 0

    for hh in range(h - ctx_before, h + ctx_after + 1):
        s = hh * 4
        e = s + 4
        if 0 <= s and e <= 96:
            result[out_idx:out_idx + 4] = arr[s:e]
        elif s < 0:
            prev_d = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            ps, pe = s + 96, e + 96
            if prev_d in day_arrays:
                result[out_idx:out_idx + 4] = day_arrays[prev_d][ps:pe]
            else:
                result[out_idx:out_idx + 4] = arr[0:4]
        else:
            next_d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()
            ns, ne = s - 96, e - 96
            if next_d in day_arrays:
                result[out_idx:out_idx + 4] = day_arrays[next_d][ns:ne]
            else:
                result[out_idx:out_idx + 4] = arr[92:96]
        out_idx += 4

    return result


def _build_residual_mc_pool(
    train_days: List, day_targets: Dict, y_mean: float, y_std: float,
) -> np.ndarray:
    """训练集 (日×24) 上按小时去均值后的归一化目标残差，展平为一维池供 MC 抽样。"""
    rows = []
    for d in train_days:
        if d not in day_targets:
            continue
        y = day_targets[d].astype(np.float64)
        rows.append((y - y_mean) / y_std)
    if not rows:
        return np.array([], dtype=np.float32)
    mat = np.stack(rows, axis=0)
    hour_mean = mat.mean(axis=0, keepdims=True)
    resid = (mat - hour_mean).reshape(-1).astype(np.float32)
    return resid


# ── Dataset ────────────────────────────────────────────────────────
class HourlyConv2dDataset(Dataset):
    """每小时一个样本 → (C_TOTAL, H_SLOTS, 7) 输入 + 1 标量目标。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict,
        day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        ctx_before: int = None, ctx_after: int = None,
        residual_mc_pool: Optional[np.ndarray] = None,
        residual_mc_prob: float = 0.0,
        residual_mc_scale: float = 0.35,
        residual_mc_npass: int = 1,
        train_oversample: int = 1,
        oversample_resid_scale: float = 0.28,
        day_delta_targets: Optional[Dict] = None,
        delta_y_mean: float = 0.0, delta_y_std: float = 1.0,
        hour_start: int = 0,
        hour_end: int = 24,
    ):
        cb = ctx_before if ctx_before is not None else CONTEXT_BEFORE
        ca = ctx_after if ctx_after is not None else CONTEXT_AFTER
        self._h_slots = (cb + 1 + ca) * SLOTS_PER_HOUR
        self.hour_start = int(hour_start)
        self.hour_end = int(hour_end)

        a0 = set(day_lag0.keys())
        a1 = set(day_lag1.keys())
        a2 = set(day_lag2.keys())

        self.items = []
        self.meta = []

        for d in sample_dates:
            if d not in day_targets:
                continue

            # Lag 对齐：Lag0 = [D-6..D], Lag1 = [D-7..D-1], Lag2 = [D-8..D-2]
            dates0 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates1 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS, 0, -1)]
            dates2 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS + 1, 1, -1)]

            ok = (all(dd in a0 for dd in dates0)
                  and all(dd in a1 for dd in dates1)
                  and all(dd in a2 for dd in dates2))
            if not ok:
                continue

            for h in range(int(hour_start), int(hour_end)):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    d0, d1, d2 = dates0[k], dates1[k], dates2[k]
                    s0 = _get_hour_slots(day_lag0, d0, h, cb, ca)
                    s1 = _get_hour_slots(day_lag1, d1, h, cb, ca)
                    s2 = _get_hour_slots(day_lag2, d2, h, cb, ca)
                    layer = np.concatenate([s0, s1, s2], axis=1)
                    layers.append(layer)

                grid = np.stack(layers, axis=-1)        # (12, C_TOTAL, 7)
                grid = grid.transpose(1, 0, 2)          # (C_TOTAL, 12, 7)

                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
                        / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)

                tgt = np.float32((day_targets[d][h] - y_mean) / y_std)
                dtgt = np.float32(0.0)
                if day_delta_targets is not None and d in day_delta_targets:
                    dtgt = np.float32((day_delta_targets[d][h] - delta_y_mean) / delta_y_std)
                self.items.append((grid, tgt, dtgt))
                self.meta.append((d, h))

        self._n_orig = len(self.items)
        self._train_oversample = max(1, int(train_oversample))
        self._oversample_resid_scale = float(oversample_resid_scale)

        self._residual_pool = residual_mc_pool
        self._residual_mc_prob = float(residual_mc_prob)
        self._residual_mc_scale = float(residual_mc_scale)
        self._residual_mc_npass = max(1, int(residual_mc_npass))

    def __len__(self):
        return self._n_orig * self._train_oversample

    def __getitem__(self, idx):
        base = idx % self._n_orig
        rep = idx // self._n_orig
        grid, tgt, dtgt = self.items[base]
        grid = np.copy(grid)
        tgt = np.float32(tgt)
        pool = self._residual_pool

        if rep > 0 and pool is not None and len(pool) > 0:
            tgt = np.float32(
                tgt + self._oversample_resid_scale * float(np.random.choice(pool))
            )
        elif rep == 0 and pool is not None and len(pool) > 0 and self._residual_mc_prob > 0.0:
            for _ in range(self._residual_mc_npass):
                if np.random.random() < self._residual_mc_prob:
                    r = float(np.random.choice(pool))
                    tgt = np.float32(tgt + self._residual_mc_scale * r)
        return torch.from_numpy(grid), torch.tensor(tgt), torch.tensor(dtgt)


# ── Model ──────────────────────────────────────────────────────────
class Conv2dPriceNet(nn.Module):
    """(B, C, H_SLOTS, 7) → Conv2d×3 → FC×2 → (B, 1)

    自动适配不同的 H_SLOTS (12 / 28 / ...)。
    """

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        h_out = h_slots // 2 // 2 - 2   # after 2× pool(2,1) + conv(3, pad=0)
        w_out = 7 - 2                    # 7 → 5 after conv(3, pad=0)
        fc_in = 64 * h_out * w_out
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._fc_in = fc_in

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────
def _eval_mae_hourly(model, loader, y_mean, y_std):
    model.eval()
    _dual = getattr(model, "_dual_head", False)
    ps, ts = [], []
    with torch.no_grad():
        for grid, tgt, _dtgt in loader:
            out = model(grid.to(DEVICE))
            pred = out[0] if _dual else out
            ps.append(pred.cpu().numpy())
            ts.append(tgt.numpy())
    p = np.concatenate(ps) * y_std + y_mean
    t = np.concatenate(ts) * y_std + y_mean
    return float(np.mean(np.abs(p - t)))


def train_model(
    train_days, val_days,
    day_lag0, day_lag1, day_lag2, day_targets,
    norm_mean, norm_std, y_mean, y_std,
    epochs=None, out_dir=None,
    lr=None, weight_decay=None, batch_size=None,
    dropout=None, ctx_before=None, ctx_after=None,
    epoch_callback=None,
    model_cls=None,
    day_delta_targets=None,
):
    epochs = epochs or MAX_EPOCHS
    out_dir = out_dir or V18_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)

    _lr = lr if lr is not None else LR
    _wd = weight_decay if weight_decay is not None else WEIGHT_DECAY
    _bs = batch_size if batch_size is not None else BATCH_SIZE
    _dropout = dropout if dropout is not None else DROPOUT
    _cb = ctx_before if ctx_before is not None else CONTEXT_BEFORE
    _ca = ctx_after if ctx_after is not None else CONTEXT_AFTER
    _h_slots = (_cb + 1 + _ca) * SLOTS_PER_HOUR

    use_resid_mc = V18_RESIDUAL_MC == 1
    need_resid_pool = use_resid_mc or (V18_TRAIN_OVERSAMPLE > 1)
    resid_pool = None
    if need_resid_pool:
        resid_pool = _build_residual_mc_pool(train_days, day_targets, y_mean, y_std)
        logger.info(
            "Residual pool: size=%d (MC=%s, oversample=%d, virt_scale=%.3f)",
            len(resid_pool), "on" if use_resid_mc else "off",
            V18_TRAIN_OVERSAMPLE, V18_OVERSAMPLE_RESID_SCALE,
        )
    if use_resid_mc:
        logger.info(
            "Residual MC on rep=0: p=%.3f scale=%.3f npass=%d",
            V18_RESIDUAL_MC_P, V18_RESIDUAL_MC_SCALE, V18_RESIDUAL_MC_NPASS,
        )

    delta_y_mean, delta_y_std = 0.0, 1.0
    if day_delta_targets:
        dvals = [day_delta_targets[d] for d in train_days if d in day_delta_targets]
        if dvals:
            dall = np.concatenate(dvals)
            delta_y_mean = float(np.mean(dall))
            delta_y_std = max(float(np.std(dall)), 1e-6)

    ds_kw = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        ctx_before=_cb, ctx_after=_ca,
        residual_mc_pool=resid_pool if need_resid_pool else None,
        residual_mc_prob=V18_RESIDUAL_MC_P if use_resid_mc else 0.0,
        residual_mc_scale=V18_RESIDUAL_MC_SCALE,
        residual_mc_npass=V18_RESIDUAL_MC_NPASS if use_resid_mc else 1,
        train_oversample=V18_TRAIN_OVERSAMPLE,
        oversample_resid_scale=V18_OVERSAMPLE_RESID_SCALE,
        day_delta_targets=day_delta_targets,
        delta_y_mean=delta_y_mean, delta_y_std=delta_y_std,
    )
    train_ds = HourlyConv2dDataset(sample_dates=train_days, **ds_kw)
    val_ds = HourlyConv2dDataset(
        sample_dates=val_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        ctx_before=_cb, ctx_after=_ca,
        residual_mc_pool=None,
        residual_mc_prob=0.0,
        residual_mc_scale=V18_RESIDUAL_MC_SCALE,
        residual_mc_npass=1,
        train_oversample=1,
        oversample_resid_scale=V18_OVERSAMPLE_RESID_SCALE,
        day_delta_targets=day_delta_targets,
        delta_y_mean=delta_y_mean, delta_y_std=delta_y_std,
    )

    logger.info(
        "Train samples: %d logical (= %d base × oversample %d) | Val: %d",
        len(train_ds), train_ds._n_orig, V18_TRAIN_OVERSAMPLE, len(val_ds),
    )

    tl = DataLoader(train_ds, _bs, shuffle=True, drop_last=True)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    _model_cls = model_cls or Conv2dPriceNet
    model = _model_cls(c_in=C_TOTAL, h_slots=_h_slots, dropout=_dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s params: %d  (C_in=%d, h_slots=%d, dropout=%.2f)",
                _model_cls.__name__, n_params, C_TOTAL, _h_slots, _dropout)

    opt = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=_wd)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - WARMUP_EPOCHS, 1), eta_min=1e-6,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS],
    )

    _is_dual = getattr(model, "_dual_head", False)
    _delta_lambda = float(os.environ.get("V18_DELTA_LAMBDA", "0.5"))

    final_val_mae = float("inf")
    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for grid, tgt, dtgt in tl:
            grid, tgt, dtgt = grid.to(DEVICE), tgt.to(DEVICE), dtgt.to(DEVICE)
            opt.zero_grad()
            out = model(grid)
            if _is_dual:
                price_pred, delta_pred = out
                loss = F.l1_loss(price_pred, tgt) + _delta_lambda * F.l1_loss(delta_pred, dtgt)
            else:
                loss = F.l1_loss(out, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        do_log = ep % 10 == 0 or ep == epochs - 1
        if do_log:
            train_mae = _eval_mae_hourly(
                model, DataLoader(train_ds, min(512, len(train_ds)), shuffle=False),
                y_mean, y_std,
            )
            if len(val_ds) > 0:
                final_val_mae = _eval_mae_hourly(model, val_l, y_mean, y_std)
                logger.info(
                    "  ep%3d  loss=%.4f  train_mae=%.1f  val_mae=%.1f  lr=%.1e",
                    ep, ep_loss / max(nb, 1), train_mae, final_val_mae,
                    opt.param_groups[0]["lr"],
                )
            else:
                logger.info(
                    "  ep%3d  loss=%.4f  train_mae=%.1f  lr=%.1e",
                    ep, ep_loss / max(nb, 1), train_mae,
                    opt.param_groups[0]["lr"],
                )

        if epoch_callback is not None and len(val_ds) > 0:
            final_val_mae = _eval_mae_hourly(model, val_l, y_mean, y_std)
            epoch_callback(ep, final_val_mae)

    ckpt = out_dir / "seed0.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Saved last-epoch checkpoint → %s", ckpt)
    return model, final_val_mae


def predict_days(model, dates, day_lag0, day_lag1, day_lag2, day_targets,
                 norm_mean, norm_std, y_mean, y_std,
                 hour_start: int = 0, hour_end: int = 24):
    """Predict on given dates, return (p24, a24, valid_dates)."""
    ds = HourlyConv2dDataset(
        sample_dates=dates,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        hour_start=hour_start,
        hour_end=hour_end,
    )
    if len(ds) == 0:
        return np.zeros((0, 24)), np.zeros((0, 24)), []

    _dual = getattr(model, "_dual_head", False)
    loader = DataLoader(ds, min(512, len(ds)), shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for grid, _, _dtgt in loader:
            out = model(grid.to(DEVICE))
            pred = out[0] if _dual else out
            all_preds.append(pred.cpu().numpy())
    preds_flat = np.concatenate(all_preds) * y_std + y_mean

    day_preds: Dict = {}
    for i, (d, h) in enumerate(ds.meta):
        if d not in day_preds:
            day_preds[d] = np.full(24, np.nan)
        day_preds[d][h] = preds_flat[i]

    # 时段专模只填充 [hour_start, hour_end)，有效性仅检查该区间
    valid_dates = sorted(
        d
        for d in day_preds
        if d in day_targets
        and not np.isnan(day_preds[d][hour_start:hour_end]).any()
        and not np.isnan(day_targets[d][hour_start:hour_end]).any()
    )
    if not valid_dates:
        return np.zeros((0, 24)), np.zeros((0, 24)), []
    p24 = np.array([day_preds[d] for d in valid_dates])
    a24 = np.array([day_targets[d] for d in valid_dates])
    return p24, a24, valid_dates


def compute_norm(day_lag0, day_lag1, day_lag2, train_days):
    rows = []
    for d in train_days:
        if d in day_lag0 and d in day_lag1 and d in day_lag2:
            row = np.concatenate([day_lag0[d], day_lag1[d], day_lag2[d]], axis=1)
            rows.append(row)
    stack = np.concatenate(rows, axis=0)
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    std = np.nanstd(stack, axis=0).astype(np.float32) + 1e-8
    return mean, std


# ── Plotting ──────────────────────────────────────────────────────
def _plot_train_last_week(p24, a24, dates, plots_dir):
    """绘制训练集最后一周的逐小时预测 vs 实际。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for p in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/PingFang.ttc"):
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = len(dates)

    # 连续曲线图
    fig, ax = plt.subplots(figsize=(22, 5))
    a_all = np.concatenate([a24[i] for i in range(n)])
    p_all = np.concatenate([p24[i] for i in range(n)])
    x = np.arange(len(a_all))
    ax.plot(x, a_all, "k-", lw=1.5, label="实际(1h)", zorder=3)
    ax.plot(x, p_all, "#E91E63", lw=1.0, alpha=0.85, label="V18训练集(1h)")

    ticks, labels = [], []
    pos = 0
    for i in range(n):
        if pos > 0:
            ax.axvline(pos, color="gray", ls="--", alpha=0.3, lw=0.8)
        ticks.append(pos + 12)
        if np.std(a24[i]) > 1e-6 and np.std(p24[i]) > 1e-6:
            r = np.corrcoef(a24[i], p24[i])[0, 1]
            mae_d = np.mean(np.abs(a24[i] - p24[i]))
            labels.append(f"{dates[i]}\nr={r:.2f} MAE={mae_d:.1f}")
        else:
            labels.append(str(dates[i]))
        pos += 24
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"{_v18_viz_label()} 训练集最后一周拟合 ({dates[0]} ~ {dates[-1]})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "da_train_last_week.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_train_last_week.png")

    # 逐日子图
    cols = min(n, 4)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.5 * cols, 4 * rows_n))
    axes = np.atleast_2d(axes) if rows_n > 1 else np.atleast_1d(axes).reshape(1, -1)
    hours_24 = np.arange(24)
    for j in range(n):
        ax = axes[j // cols][j % cols]
        ax.plot(hours_24, a24[j], "k-", lw=2, label="实际")
        ax.plot(hours_24, p24[j], "#E91E63", lw=1.5, label=f"{_v18_viz_label()} 拟合")
        r = (np.corrcoef(a24[j], p24[j])[0, 1]
             if np.std(a24[j]) > 1e-6 and np.std(p24[j]) > 1e-6 else 0)
        mae_d = np.mean(np.abs(a24[j] - p24[j]))
        ax.set_title(f"{dates[j]}  r={r:.2f}  MAE={mae_d:.1f}", fontsize=9)
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(f"{_v18_viz_label()} 训练集最后一周 — 逐日拟合", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "da_train_last_week_daily.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_train_last_week_daily.png")


# ── Main ───────────────────────────────────────────────────────────
def run_v18(out_dir=None, feature_df: Optional[pd.DataFrame] = None, model_cls=None):
    out_dir = Path(out_dir) if out_dir else V18_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("%s — hourly %s prediction", _v18_viz_label(), TARGET_COL)
    logger.info("  input=(%d ch, %d slots [-%dh..+%dh], %d days)",
                C_TOTAL, H_SLOTS, CONTEXT_BEFORE, CONTEXT_AFTER, LOOKBACK_DAYS)
    logger.info("  Lag0(%d ch): %s + time_enc(4)", len(LAG0_COLS), LAG0_COLS[:3])
    logger.info("  Lag1(%d ch): %s", C_LAG1, LAG1_COLS)
    logger.info("  Lag2(%d ch): %s", C_LAG2, LAG2_COLS[:3])
    logger.info("  epochs=%d, bs=%d, lr=%.1e, wd=%.1e, dropout=%.2f, warmup=%d",
                MAX_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, DROPOUT, WARMUP_EPOCHS)
    logger.info(
        "  train_oversample=%d (logical sample count K×), residual_mc=%d, virt_resid_scale=%.3f",
        V18_TRAIN_OVERSAMPLE, V18_RESIDUAL_MC, V18_OVERSAMPLE_RESID_SCALE,
    )
    _log_device()

    df = feature_df if feature_df is not None else build_feature_matrix()
    if feature_df is not None:
        logger.info("Using pre-built feature matrix: %d rows × %d cols", len(df), len(df.columns))
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets, day_anchors, day_delta_targets = _build_daily_arrays(df)

    merge_val = os.environ.get("V18_MERGE_VAL", "0").strip().lower() in ("1", "true", "yes")
    tr_last = TRAIN_END.date()
    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
    if merge_val:
        train_days = [d for d in valid_dates if d <= val_last]
        val_days = []
        logger.info("V18_MERGE_VAL=1 → val merged into train")
    else:
        train_days = [d for d in valid_dates if d <= tr_last]
        val_days = [d for d in valid_dates if tr_last < d <= val_last]
    test_days = [d for d in valid_dates if ts_first <= d <= ts_last]

    logger.info("Train days: %d (%s ~ %s)", len(train_days),
                train_days[0] if train_days else "?",
                train_days[-1] if train_days else "?")
    logger.info("Val days:   %d (%s ~ %s)", len(val_days),
                val_days[0] if val_days else "?",
                val_days[-1] if val_days else "?")
    logger.info("Test days:  %d (%s ~ %s)", len(test_days),
                test_days[0] if test_days else "?",
                test_days[-1] if test_days else "?")

    norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)

    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    model, _ = train_model(
        train_days=train_days,
        val_days=val_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        out_dir=out_dir,
        model_cls=model_cls,
        day_delta_targets=day_delta_targets if not V18_DELTA_TARGET else None,
    )
    logger.info("Using last-epoch weights (ep %d)", MAX_EPOCHS - 1)

    # ── 训练集最后一周拟合图 ──
    train_last7 = train_days[-7:] if len(train_days) >= 7 else train_days
    p24_tr, a24_tr, dates_tr = predict_days(
        model, train_last7,
        day_lag0, day_lag1, day_lag2, day_targets,
        norm_mean, norm_std, y_mean, y_std,
    )
    if len(dates_tr) > 0:
        _plot_train_last_week(p24_tr, a24_tr, dates_tr, out_dir / "plots")

    save_val_pred = os.environ.get("V18_SAVE_VAL_PRED", "0").strip().lower() in ("1", "true", "yes")
    skip_test_pred = os.environ.get("V18_SKIP_TEST_PRED", "0").strip().lower() in ("1", "true", "yes")

    def _days_to_result(p24_arr, a24_arr, day_list):
        if V18_DELTA_TARGET and day_anchors:
            p24_arr = p24_arr.copy()
            a24_arr = a24_arr.copy()
            for i, d in enumerate(day_list):
                anchor = day_anchors.get(d, 0.0)
                p24_arr[i] = anchor + np.cumsum(p24_arr[i])
                a24_arr[i] = anchor + np.cumsum(a24_arr[i])
        rows = []
        for i, d in enumerate(day_list):
            for h in range(24):
                rows.append({
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": a24_arr[i, h],
                    "predicted": p24_arr[i, h],
                })
        return pd.DataFrame(rows).set_index("ts").sort_index()

    val_result = None
    if save_val_pred and val_days:
        p24_v, a24_v, dates_v = predict_days(
            model, val_days,
            day_lag0, day_lag1, day_lag2, day_targets,
            norm_mean, norm_std, y_mean, y_std,
        )
        if len(dates_v) > 0:
            val_result = _days_to_result(p24_v, a24_v, dates_v)
            val_path = out_dir / "val_result.csv"
            val_result.to_csv(val_path)
            logger.info("Saved: %s (%d rows, %d days)", val_path.name, len(val_result), len(dates_v))

    if skip_test_pred:
        logger.info("V18_SKIP_TEST_PRED=1 → skip test prediction")
        if val_result is not None and len(val_result) > 0:
            af = val_result["actual"].values
            pf = val_result["predicted"].values
            mae = float(np.mean(np.abs(af - pf)))
            rmse = float(np.sqrt(np.mean((af - pf) ** 2)))
            shape = quick_shape_report(af, pf, val_result.index)
            logger.info("=" * 60)
            logger.info("%s VAL RESULTS (selection split)", _v18_viz_label())
            logger.info("  MAE:  %.2f", mae)
            logger.info("  RMSE: %.2f", rmse)
            for k, v in shape.items():
                logger.info("  %-18s %.4f", k, v)
            logger.info("=" * 60)
            return {"mae": mae, "rmse": rmse, **shape}
        return {}

    # ── 测试集预测 ──
    p24, a24, dates = predict_days(
        model, test_days,
        day_lag0, day_lag1, day_lag2, day_targets,
        norm_mean, norm_std, y_mean, y_std,
    )

    if V18_DELTA_TARGET and day_anchors:
        for i, d in enumerate(dates):
            anchor = day_anchors.get(d, 0.0)
            p24[i] = anchor + np.cumsum(p24[i])
            a24[i] = anchor + np.cumsum(a24[i])
        logger.info("Delta→absolute reconstruction done (anchor from previous day h=23)")

    result = _days_to_result(p24, a24, dates)
    result_path = _v18_result_csv_path(out_dir)
    result.to_csv(result_path)
    logger.info("Saved: %s (%d rows, %d days)", result_path.name, len(result), len(dates))

    from price_forecast_eval.viz import run_standard_visualization
    run_standard_visualization(
        result_path,
        out_dir=out_dir / "plots",
        label=_v18_viz_label(),
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    af = result["actual"].values
    pf = result["predicted"].values
    mae = float(np.mean(np.abs(af - pf)))
    rmse = float(np.sqrt(np.mean((af - pf) ** 2)))
    shape = quick_shape_report(af, pf, result.index)

    logger.info("=" * 60)
    logger.info("%s RESULTS", _v18_viz_label())
    logger.info("  MAE:  %.2f", mae)
    logger.info("  RMSE: %.2f", rmse)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    return {"mae": mae, "rmse": rmse, **shape}


def run_v18_rolling_last28(out_dir: Optional[Path] = None) -> Dict:
    """
    滚动测试：取 `valid_dates` 末尾连续 ROLLING_TOTAL_DAYS 个日历日（在数据中存在），
    按 ROLLING_WEEK_DAYS 天切成若干周。第 k 周为测试集时，训练集为「该周第一日之前的所有日」，
    每周重新训练（末 epoch 权重）、预测该周 24h，最后拼接整段末 28 天并汇总指标与可视化。

    环境变量：
      V18_ROLLING_TOTAL_DAYS  默认 28
      V18_ROLLING_WEEK_DAYS   默认 7
      V18_ROLLING_EPOCHS      若设置则覆盖本流程每折训练轮数；否则同 V18_EPOCHS
    """
    out_dir = Path(out_dir) if out_dir else (OUTPUT_DIR / "v18_conv2d_rolling28")
    out_dir.mkdir(parents=True, exist_ok=True)

    re = os.environ.get("V18_ROLLING_EPOCHS", "").strip()
    rolling_epochs = int(re) if re else MAX_EPOCHS

    logger.info("=" * 60)
    logger.info(
        "V18 Rolling — last %d d in data, test window %d d, epochs/fold=%d",
        ROLLING_TOTAL_DAYS, ROLLING_WEEK_DAYS, rolling_epochs,
    )
    _log_device()

    df = build_feature_matrix()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets, *_ = _build_daily_arrays(df)
    valid_dates = sorted(valid_dates)

    n_tail = min(ROLLING_TOTAL_DAYS, len(valid_dates))
    if n_tail < ROLLING_WEEK_DAYS:
        raise ValueError(
            f"Need at least {ROLLING_WEEK_DAYS} tail days, got n_tail={n_tail} (valid_dates={len(valid_dates)})"
        )
    tail = valid_dates[-n_tail:]
    chunks: List = []
    for i in range(0, len(tail), ROLLING_WEEK_DAYS):
        ch = tail[i : i + ROLLING_WEEK_DAYS]
        if ch:
            chunks.append(ch)

    logger.info(
        "Tail block: %s .. %s (%d d) → %d folds",
        tail[0], tail[-1], len(tail), len(chunks),
    )

    all_rows: List[Dict] = []
    fold_records: List[Dict] = []

    for fold, test_days in enumerate(chunks):
        train_days = [d for d in valid_dates if d < test_days[0]]
        if len(train_days) < LOOKBACK_DAYS + 3:
            logger.warning("Fold %d: skip (train_days=%d < need)", fold, len(train_days))
            continue

        fold_dir = out_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 60)
        logger.info(
            "Fold %d | train %d d (%s .. %s) | test %d d (%s .. %s)",
            fold, len(train_days), train_days[0], train_days[-1],
            len(test_days), test_days[0], test_days[-1],
        )

        norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
        tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
        y_mean = float(tgt_stack.mean())
        y_std = float(tgt_stack.std()) + 1e-8

        model, _ = train_model(
            train_days=train_days,
            val_days=test_days,
            day_lag0=day_lag0,
            day_lag1=day_lag1,
            day_lag2=day_lag2,
            day_targets=day_targets,
            norm_mean=norm_mean,
            norm_std=norm_std,
            y_mean=y_mean,
            y_std=y_std,
            epochs=rolling_epochs,
            out_dir=fold_dir,
        )
        logger.info("Fold %d: using last-epoch weights", fold)

        p24, a24, dates = predict_days(
            model, test_days,
            day_lag0, day_lag1, day_lag2, day_targets,
            norm_mean, norm_std, y_mean, y_std,
        )
        if len(dates) == 0:
            logger.warning("Fold %d: no valid prediction days", fold)
            continue

        flat_pred, flat_act = [], []
        for i, d in enumerate(dates):
            for h in range(24):
                ap = float(a24[i, h])
                pp = float(p24[i, h])
                flat_act.append(ap)
                flat_pred.append(pp)
                all_rows.append({
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": ap,
                    "predicted": pp,
                    "fold": fold,
                    "test_week_start": str(test_days[0]),
                })

        fa, fp = np.array(flat_act), np.array(flat_pred)
        mae_f = float(np.mean(np.abs(fa - fp)))
        rmse_f = float(np.sqrt(np.mean((fa - fp) ** 2)))
        fold_records.append({
            "fold": fold,
            "test_week_start": str(test_days[0]),
            "test_week_end": str(test_days[-1]),
            "n_train_days": len(train_days),
            "mae": mae_f,
            "rmse": rmse_f,
        })
        logger.info("Fold %d MAE=%.3f RMSE=%.3f", fold, mae_f, rmse_f)

    if not all_rows:
        raise RuntimeError("Rolling eval produced no rows.")

    result = pd.DataFrame(all_rows).set_index("ts").sort_index()
    result_path = _v18_result_csv_path(out_dir)
    result.to_csv(result_path)
    logger.info("Saved pooled: %s (%d rows)", result_path, len(result))

    summary_path = out_dir / "rolling_summary.json"
    af = result["actual"].values
    pf = result["predicted"].values
    mae_all = float(np.mean(np.abs(af - pf)))
    rmse_all = float(np.sqrt(np.mean((af - pf) ** 2)))
    shape_all = quick_shape_report(af, pf, result.index)
    summary = {
        "rolling_total_days": ROLLING_TOTAL_DAYS,
        "rolling_week_days": ROLLING_WEEK_DAYS,
        "epochs_per_fold": rolling_epochs,
        "pooled_mae": mae_all,
        "pooled_rmse": rmse_all,
        "pooled_shape": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in shape_all.items()},
        "folds": fold_records,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", summary_path)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    from price_forecast_eval.viz import run_standard_visualization

    run_standard_visualization(
        result_path,
        out_dir=plots_dir,
        label="V18-Rolling28d",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    logger.info("=" * 60)
    logger.info("V18 ROLLING POOLED (tail %d d)", n_tail)
    logger.info("  MAE:  %.2f", mae_all)
    logger.info("  RMSE: %.2f", rmse_all)
    for k, v in shape_all.items():
        logger.info("  %-18s %.4f", k, float(v) if isinstance(v, (float, np.floating)) else v)
    logger.info("=" * 60)

    return {"mae": mae_all, "rmse": rmse_all, "summary_path": str(summary_path), **shape_all}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v18()
