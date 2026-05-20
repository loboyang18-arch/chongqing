"""
V22 — 96 点 15 分钟日前出清电价（da_clearing_price）

- 每样本预测一个 15 分钟槽；一天 96 样本，不裁边界，窗口 6+1+5 槽跨日取自 day_lag*。
- 输入张量 (C_TOTAL, 12, 7)：12 = 3h 窗口，7 = Lag0/Lag1/Lag2 对齐的 lookback 日（同 V18）。
- 骨干复用 V18 的 Conv2dPriceNet(h_slots=12)。
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import OUTPUT_DIR
from .model_v16_nhits import (
    build_feature_matrix,
    TRAIN_END, VAL_END, TEST_END, TEST_START,
)
from .model_v18_conv2d import (
    LAG0_COLS, LAG1_COLS, LAG2_COLS,
    C_TOTAL,
    LOOKBACK_DAYS, TARGET_COL, DEVICE, WARMUP_EPOCHS,
    _log_device, _seed,
    compute_norm,
    Conv2dPriceNet,
    V18_TRAIN_OVERSAMPLE,
    V18_OVERSAMPLE_RESID_SCALE,
    V18_RESIDUAL_MC,
    V18_RESIDUAL_MC_P,
    V18_RESIDUAL_MC_SCALE,
    V18_RESIDUAL_MC_NPASS,
)

from price_forecast_eval import quick_shape_report

logger = logging.getLogger(__name__)


def configure_realtime_console_logging() -> None:
    """行缓冲 + 每条日志后 flush，便于终端实时看到训练输出。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError, AttributeError):
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(line_buffering=True)
        except (OSError, ValueError, AttributeError):
            pass

    class _FlushStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    h = _FlushStreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(h)

SPD = 96
SLOT_CTX_BEFORE = 6
SLOT_CTX_AFTER = 5
WINDOW_SLOTS = SLOT_CTX_BEFORE + 1 + SLOT_CTX_AFTER  # 12

MAX_EPOCHS = int(os.environ.get("V22_EPOCHS", "100"))
# 训练日志间隔（epoch）；默认 1 即每轮打印，便于终端实时观察
LOG_INTERVAL = max(1, int(os.environ.get("V22_LOG_INTERVAL", "1")))
BATCH_SIZE = int(os.environ.get("V22_BS", os.environ.get("V18_BS", "64")))
LR = float(os.environ.get("V22_LR", os.environ.get("V18_LR", "1e-3")))
WEIGHT_DECAY = float(os.environ.get("V22_WD", os.environ.get("V18_WD", "1e-4")))

V22_DIR = OUTPUT_DIR / "v22_96_da"


def _build_daily_arrays_96(df: pd.DataFrame):
    """day_targets[d] = (96,) 15min 日前出清价；lag 矩阵同 V18。"""
    start_date = df.index.min().normalize().date()
    end_date = df.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")

    day_lag0: Dict = {}
    day_lag1: Dict = {}
    day_lag2: Dict = {}
    day_targets: Dict = {}
    valid: List = []

    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        raw = df.reindex(grid)

        if raw[LAG0_COLS + LAG1_COLS + LAG2_COLS].isna().all().any():
            continue

        l0 = raw[LAG0_COLS].values.astype(np.float32)
        steps = np.arange(96, dtype=np.float32)
        dow = float(pd.Timestamp(d).dayofweek)
        te = np.column_stack([
            np.sin(2 * np.pi * steps / 96),
            np.cos(2 * np.pi * steps / 96),
            np.full(96, np.sin(2 * np.pi * dow / 7), dtype=np.float32),
            np.full(96, np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        ])
        day_lag0[d] = np.concatenate([l0, te], axis=1).astype(np.float32)
        day_lag1[d] = raw[LAG1_COLS].values.astype(np.float32)
        day_lag2[d] = raw[LAG2_COLS].values.astype(np.float32)

        if TARGET_COL in raw.columns:
            v = raw[TARGET_COL].to_numpy(dtype=np.float64, copy=False)
            if v.size == 96 and np.isfinite(v).all():
                day_targets[d] = v.astype(np.float32)
                valid.append(d)

    valid = sorted(valid)
    logger.info(
        "V22 daily arrays: %d days total, %d with valid 96-slot target",
        len(day_lag0), len(valid),
    )
    return valid, day_lag0, day_lag1, day_lag2, day_targets


def _get_slot_window(
    day_arrays: Dict,
    d,
    slot_s: int,
    before: int = SLOT_CTX_BEFORE,
    after: int = SLOT_CTX_AFTER,
) -> np.ndarray:
    """以日 d 的第 slot_s 个 15min 为中心，取 [slot_s-before, slot_s+after] 共 12 点；跨日用前后日。"""
    n = before + 1 + after
    arr = day_arrays[d]
    c = arr.shape[1]
    out = np.zeros((n, c), dtype=np.float32)
    for i, rel in enumerate(range(-before, after + 1)):
        idx = slot_s + rel
        if 0 <= idx < SPD:
            out[i] = arr[idx]
        elif idx < 0:
            prev_d = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            pidx = idx + SPD
            if prev_d in day_arrays and 0 <= pidx < SPD:
                out[i] = day_arrays[prev_d][pidx]
            else:
                out[i] = arr[0]
        else:
            next_d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()
            nidx = idx - SPD
            if next_d in day_arrays and 0 <= nidx < SPD:
                out[i] = day_arrays[next_d][nidx]
            else:
                out[i] = arr[SPD - 1]
    return out


def _build_residual_mc_pool_96(
    train_days: List, day_targets: Dict, y_mean: float, y_std: float,
) -> np.ndarray:
    rows = []
    for d in train_days:
        if d not in day_targets:
            continue
        y = day_targets[d].astype(np.float64)
        rows.append((y - y_mean) / y_std)
    if not rows:
        return np.array([], dtype=np.float32)
    mat = np.stack(rows, axis=0)
    slot_mean = mat.mean(axis=0, keepdims=True)
    resid = (mat - slot_mean).reshape(-1).astype(np.float32)
    return resid


class Slot96Conv2dDataset(Dataset):
    """每个 (日, 槽) 一个样本 → (C_TOTAL, 12, 7) + 标量目标。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict,
        day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        residual_mc_pool: Optional[np.ndarray] = None,
        residual_mc_prob: float = 0.0,
        residual_mc_scale: float = 0.35,
        residual_mc_npass: int = 1,
        train_oversample: int = 1,
        oversample_resid_scale: float = 0.28,
    ):
        a0 = set(day_lag0.keys())
        a1 = set(day_lag1.keys())
        a2 = set(day_lag2.keys())

        self.items = []
        self.meta = []

        for d in sample_dates:
            if d not in day_targets:
                continue
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

            for slot_s in range(SPD):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    d0, d1, d2 = dates0[k], dates1[k], dates2[k]
                    s0 = _get_slot_window(day_lag0, d0, slot_s)
                    s1 = _get_slot_window(day_lag1, d1, slot_s)
                    s2 = _get_slot_window(day_lag2, d2, slot_s)
                    layer = np.concatenate([s0, s1, s2], axis=1)
                    layers.append(layer)

                grid = np.stack(layers, axis=-1)
                grid = grid.transpose(1, 0, 2)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
                        / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)

                tgt = np.float32((day_targets[d][slot_s] - y_mean) / y_std)
                self.items.append((grid, tgt))
                self.meta.append((d, slot_s))

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
        grid, tgt = self.items[base]
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
        return torch.from_numpy(grid), torch.tensor(tgt)


def _eval_mae(model, loader, y_mean, y_std):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for grid, tgt in loader:
            pred = model(grid.to(DEVICE))
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
):
    epochs = epochs or MAX_EPOCHS
    out_dir = Path(out_dir or V22_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)

    use_resid_mc = V18_RESIDUAL_MC == 1
    need_resid_pool = use_resid_mc or (V18_TRAIN_OVERSAMPLE > 1)
    resid_pool = None
    if need_resid_pool:
        resid_pool = _build_residual_mc_pool_96(train_days, day_targets, y_mean, y_std)
        logger.info(
            "V22 residual pool: size=%d (MC=%s, oversample=%d)",
            len(resid_pool), "on" if use_resid_mc else "off", V18_TRAIN_OVERSAMPLE,
        )

    ds_kw = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        residual_mc_pool=resid_pool if need_resid_pool else None,
        residual_mc_prob=V18_RESIDUAL_MC_P if use_resid_mc else 0.0,
        residual_mc_scale=V18_RESIDUAL_MC_SCALE,
        residual_mc_npass=V18_RESIDUAL_MC_NPASS if use_resid_mc else 1,
        train_oversample=V18_TRAIN_OVERSAMPLE,
        oversample_resid_scale=V18_OVERSAMPLE_RESID_SCALE,
    )
    train_ds = Slot96Conv2dDataset(sample_dates=train_days, **ds_kw)
    val_ds = Slot96Conv2dDataset(
        sample_dates=val_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        residual_mc_pool=None,
        residual_mc_prob=0.0,
        residual_mc_scale=V18_RESIDUAL_MC_SCALE,
        residual_mc_npass=1,
        train_oversample=1,
        oversample_resid_scale=V18_OVERSAMPLE_RESID_SCALE,
    )

    logger.info(
        "V22 Train samples: %d (= %d base × os %d) | Val: %d",
        len(train_ds), train_ds._n_orig, V18_TRAIN_OVERSAMPLE, len(val_ds),
    )

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    model = Conv2dPriceNet(c_in=C_TOTAL, h_slots=WINDOW_SLOTS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("V22 Conv2dPriceNet params: %d  (C_in=%d, h_slots=%d)", n_params, C_TOTAL, WINDOW_SLOTS)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for grid, tgt in tl:
            grid, tgt = grid.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            pred = model(grid)
            loss = F.l1_loss(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        do_log = (ep % LOG_INTERVAL == 0) or ep == epochs - 1
        if do_log:
            train_mae = _eval_mae(
                model,
                DataLoader(train_ds, min(512, len(train_ds)), shuffle=False),
                y_mean, y_std,
            )
            val_mae = _eval_mae(model, val_l, y_mean, y_std)
            logger.info(
                "  ep%3d  loss=%.4f  train_mae=%.1f  val_mae=%.1f  lr=%.1e",
                ep, ep_loss / max(nb, 1), train_mae, val_mae,
                opt.param_groups[0]["lr"],
            )

    ckpt = out_dir / "seed0.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Saved → %s", ckpt)
    return model


def predict_days(model, dates, day_lag0, day_lag1, day_lag2, day_targets,
                 norm_mean, norm_std, y_mean, y_std):
    ds = Slot96Conv2dDataset(
        sample_dates=dates,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    if len(ds) == 0:
        return np.zeros((0, SPD)), np.zeros((0, SPD)), []

    loader = DataLoader(ds, min(512, len(ds)), shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for grid, _ in loader:
            pred = model(grid.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
    preds_flat = np.concatenate(all_preds) * y_std + y_mean

    day_preds: Dict = {}
    for i, (d, slot_s) in enumerate(ds.meta):
        if d not in day_preds:
            day_preds[d] = np.full(SPD, np.nan)
        day_preds[d][slot_s] = preds_flat[i]

    valid_dates = sorted(
        d for d in day_preds
        if d in day_targets
        and not np.isnan(day_preds[d]).any()
        and not np.isnan(day_targets[d]).any()
    )
    p96 = np.array([day_preds[d] for d in valid_dates])
    a96 = np.array([day_targets[d] for d in valid_dates])
    return p96, a96, valid_dates


def _plot_train_last_week_96(p96, a96, dates, plots_dir: Path):
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
    fig, ax = plt.subplots(figsize=(22, 5))
    a_all = np.concatenate([a96[i] for i in range(n)])
    p_all = np.concatenate([p96[i] for i in range(n)])
    x = np.arange(len(a_all))
    ax.plot(x, a_all, "k-", lw=0.8, label="实际(15min)", zorder=3)
    ax.plot(x, p_all, "#E91E63", lw=0.6, alpha=0.85, label="V22训练集")
    pos = 0
    for i in range(n):
        if pos > 0:
            ax.axvline(pos, color="gray", ls="--", alpha=0.25, lw=0.6)
        pos += SPD
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"V22 训练集最后一周 (15min) {dates[0]} ~ {dates[-1]}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "da_train_last_week_96.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_train_last_week_96.png")


def _test_mae_by_seven_day_chunks(
    dates: List,
    p96: np.ndarray,
    a96: np.ndarray,
) -> List[Dict]:
    """
    按测试集日期升序，每连续 7 个自然日为一组（与常见 da_week* 分段一致），
    聚合该组内全部 15min 点计算 MAE/RMSE，并打日志。
    """
    if len(dates) == 0:
        return []
    idx_by_date = {d: i for i, d in enumerate(dates)}
    sdates = sorted(dates)
    out: List[Dict] = []
    wid = 0
    for j in range(0, len(sdates), 7):
        wk = sdates[j : j + 7]
        idxs = [idx_by_date[d] for d in wk]
        p = np.concatenate([p96[i] for i in idxs])
        a = np.concatenate([a96[i] for i in idxs])
        mae_w = float(np.mean(np.abs(p - a)))
        rmse_w = float(np.sqrt(np.mean((p - a) ** 2)))
        wid += 1
        rec = {
            "week": wid,
            "start": str(wk[0]),
            "end": str(wk[-1]),
            "n_days": len(wk),
            "n_points": int(len(p)),
            "mae": mae_w,
            "rmse": rmse_w,
        }
        out.append(rec)
        logger.info(
            "Test MAE by week %2d | %s .. %s | %dd %5d pts | MAE=%7.2f  RMSE=%7.2f",
            wid, wk[0], wk[-1], len(wk), len(p), mae_w, rmse_w,
        )
    return out


def _run_standard_eval(out_dir: Path) -> None:
    root = out_dir.resolve()
    summary = root / "evaluation_summary_appendix_v1.csv"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run_evaluate_all_models.py"),
        "--output-root", str(root),
        "--summary", str(summary),
        "--task", "da",
        "--no-baseline",
    ]
    logger.info("Running standard eval: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


def run_v22(out_dir=None, run_eval: bool = True):
    out_dir = Path(out_dir or V22_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V22 — 96×15min da_clearing_price | window %d+%d+%d | lookback %dd",
                SLOT_CTX_BEFORE, 1, SLOT_CTX_AFTER, LOOKBACK_DAYS)
    logger.info("  input (C,H,W)=(%d, %d, %d)", C_TOTAL, WINDOW_SLOTS, LOOKBACK_DAYS)
    logger.info("  epochs=%d bs=%d lr=%.1e", MAX_EPOCHS, BATCH_SIZE, LR)
    _log_device()

    df = build_feature_matrix()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays_96(df)

    tr_last = TRAIN_END.date()
    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
    train_days = [d for d in valid_dates if d <= tr_last]
    val_days = [d for d in valid_dates if tr_last < d <= val_last]
    test_days = [d for d in valid_dates if ts_first <= d <= ts_last]

    logger.info("Train %d | Val %d | Test %d", len(train_days), len(val_days), len(test_days))

    norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    model = train_model(
        train_days=train_days,
        val_days=val_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        out_dir=out_dir,
    )

    train_last7 = train_days[-7:] if len(train_days) >= 7 else train_days
    p_tr, a_tr, dates_tr = predict_days(
        model, train_last7, day_lag0, day_lag1, day_lag2, day_targets,
        norm_mean, norm_std, y_mean, y_std,
    )
    if len(dates_tr) > 0:
        _plot_train_last_week_96(p_tr, a_tr, dates_tr, out_dir / "plots")

    p96, a96, dates = predict_days(
        model, test_days, day_lag0, day_lag1, day_lag2, day_targets,
        norm_mean, norm_std, y_mean, y_std,
    )

    logger.info("-" * 60)
    logger.info("Test MAE by calendar week (non-overlapping 7-day chunks, 15min pts)")
    weekly = _test_mae_by_seven_day_chunks(dates, p96, a96)
    weekly_path = out_dir / "test_mae_by_week.json"
    with open(weekly_path, "w", encoding="utf-8") as f:
        json.dump(weekly, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", weekly_path.name)

    rows = []
    for i, d in enumerate(dates):
        for s in range(SPD):
            rows.append({
                "ts": pd.Timestamp(d) + pd.Timedelta(minutes=15 * s),
                "actual": float(a96[i, s]),
                "predicted": float(p96[i, s]),
            })
    result = pd.DataFrame(rows).set_index("ts").sort_index()
    result_path = out_dir / "da_result.csv"
    result.to_csv(result_path)
    logger.info("Saved %s (%d rows, %d days)", result_path.name, len(result), len(dates))

    try:
        from price_forecast_eval.viz import run_standard_visualization
        run_standard_visualization(
            result_path,
            out_dir=out_dir / "plots",
            label="V22-96DA",
            actual_col="actual",
            pred_col="predicted",
            mode="appendix",
            weekly=True,
        )
    except Exception as e:
        logger.warning("Standard viz skipped: %s", e)

    af = result["actual"].values
    pf = result["predicted"].values
    mae = float(np.mean(np.abs(af - pf)))
    rmse = float(np.sqrt(np.mean((af - pf) ** 2)))
    shape = quick_shape_report(af, pf, result.index)

    summary_json = out_dir / "v22_metrics_quick.json"
    summary_payload = {
        "mae": mae,
        "rmse": rmse,
        "test_mae_by_week": weekly,
        **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in shape.items()},
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    logger.info("V22 pooled MAE=%.2f RMSE=%.2f", mae, rmse)
    if run_eval:
        _run_standard_eval(out_dir)
    return {"mae": mae, "rmse": rmse, **shape}


if __name__ == "__main__":
    configure_realtime_console_logging()
    run_v22()
