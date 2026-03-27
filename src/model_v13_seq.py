"""
V13 时序模型 — 双输出头(价格 + 差分) + 统计摘要编码器。

核心思想：
  每小时为一个独立样本，模型同时输出两个值：
    ① ŷ_h  — 该小时的绝对价格
    ② Δŷ_h — 该小时与前一小时的价格差分（h0 用 D-1 日 h23 作参考）

  由于数据量有限（~88天），纯 LSTM 编码器难以学到有效模式。
  因此采用 "统计摘要 + 轻量 LSTM" 的混合架构：
    - 从 72h 历史序列提取多尺度统计摘要（均值/std/趋势/最近值等）
    - 轻量 LSTM 只负责捕捉序列中的短期动态
    - 将统计摘要、LSTM 上下文、小时 future 特征拼接后送入双头 MLP

  损失 = w_price * MAE(价格) + w_delta * MAE(差分)

产出：output/v13_seq/
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

from .config import OUTPUT_DIR
from .model_baseline import TRAIN_END, TEST_START
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V13_DIR = OUTPUT_DIR / "v13_seq"
V13_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================================
# 1. 变量分类
# =====================================================================

PAST_OBSERVED = [
    "da_clearing_price",
    "rt_clearing_price",
    "actual_load",
    "total_gen",
    "hydro_gen",
    "non_market_gen",
    "renewable_gen",
    "tie_line_power",
    "settlement_da_price",
    "settlement_rt_price",
    "da_clearing_power",
    "da_clearing_unit_count",
    "rt_clearing_volume",
    "rt_clearing_unit_count",
    "da_reliability_clearing_price",
    "reliability_da_price",
    "reliability_rt_price",
    "actual_load_std",
    "actual_load_range",
    "actual_load_max_ramp",
    "renewable_gen_std",
    "renewable_gen_range",
    "tie_line_power_std",
    "tie_line_power_range",
    "tie_line_power_max_ramp",
    "section_ratio_C-500-洪板断面送板桥-全接线【常】",
    "section_ratio_C-500-资铜断面送铜梁-全接线【常】",
    "da_avg_clearing_price",
    "rt_avg_clearing_price",
    "avg_bid_price",
]

FUTURE_KNOWN = [
    "load_forecast",
    "total_gen_fcst_am",
    "hydro_gen_fcst_am",
    "non_market_gen_fcst_am",
    "renewable_fcst",
    "renewable_fcst_solar_am",
    "renewable_fcst_wind_am",
    "renewable_fcst_total_am",
    "tie_line_fcst_am",
    "total_gen_fcst_pm",
    "hydro_gen_fcst_pm",
    "non_market_gen_fcst_pm",
    "renewable_fcst_solar_pm",
    "renewable_fcst_total_pm",
    "renewable_fcst_wind_pm",
    "tie_line_fcst_pm",
    "maintenance_gen_count",
    "maintenance_grid_count",
]

CALENDAR_FEATURES = ["hour", "day_of_week", "is_weekend", "month"]

TARGET_COL = "da_clearing_price"
LOOKBACK = 72
EFFECTIVE_START = "2025-11-01"
EFFECTIVE_END = "2026-03-10 23:00:00"

# =====================================================================
# 2. 数据加载
# =====================================================================

def _load_raw_hourly() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_hourly_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")

    need_cols = list(set(PAST_OBSERVED + FUTURE_KNOWN) & set(df.columns))
    df = df[need_cols].copy()

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["month"] = df.index.month

    df = df.loc[EFFECTIVE_START:EFFECTIVE_END].copy()
    df = df.sort_index()

    target_valid = df[TARGET_COL].notna().sum()
    logger.info("Raw hourly: %d rows × %d cols (%s ~ %s), target valid: %d",
                len(df), len(df.columns), df.index.min(), df.index.max(), target_valid)
    return df


# =====================================================================
# 3. 统计摘要特征提取
# =====================================================================

def _extract_past_summary(past_arr: np.ndarray, price_col_idx: int) -> np.ndarray:
    """从 (72, n_features) 的标准化历史序列中提取统计摘要。

    输出一个 1-D 向量，包含：
      - 价格列的多尺度统计（24h/48h/72h 的均值、std、趋势斜率、最近值、最小、最大）
      - 最近 3 个价格点
      - 价格差分统计
      - 所有特征的最近 24h 均值
      - 昨天 24h 价格的日内形状偏差（每小时 - 日均值）
      - 前天 24h 价格的日内形状偏差
    """
    n_steps, n_feats = past_arr.shape
    price = past_arr[:, price_col_idx]

    feats = []

    for window in [24, 48, 72]:
        seg = price[-window:]
        feats.append(np.mean(seg))
        feats.append(np.std(seg))
        feats.append(np.min(seg))
        feats.append(np.max(seg))
        if len(seg) > 1:
            x = np.arange(len(seg))
            slope = np.polyfit(x, seg, 1)[0]
            feats.append(slope)
        else:
            feats.append(0.0)

    feats.append(price[-1])
    feats.append(price[-2] if n_steps >= 2 else price[-1])
    feats.append(price[-3] if n_steps >= 3 else price[-1])

    price_diff = np.diff(price)
    feats.append(np.mean(price_diff))
    feats.append(np.std(price_diff))
    feats.append(np.max(price_diff))
    feats.append(np.min(price_diff))

    last24_mean = np.mean(past_arr[-24:], axis=0)
    feats.extend(last24_mean.tolist())

    return np.array(feats, dtype=np.float32)


# =====================================================================
# 4. 数据集 — 每小时一个独立样本，含差分标签 + 统计摘要
# =====================================================================

class HourlySampleDataset(Dataset):
    """每小时一个样本，提供：
      - past_summary: 从 72h 历史提取的统计摘要向量
      - past_seq:     72h 历史序列（给 LSTM）
      - future:       该小时的 future_known 特征
      - target/delta: 价格标签和差分标签
    """

    def __init__(self, df: pd.DataFrame, dates: List,
                 lookback: int,
                 past_obs_cols: List[str],
                 future_known_cols: List[str],
                 target_col: str,
                 scaler_stats: Dict = None):
        self.lookback = lookback
        self.past_obs_cols = past_obs_cols
        self.future_known_cols = future_known_cols
        self.target_col = target_col

        all_feat_cols = list(set(past_obs_cols + future_known_cols))

        if scaler_stats is None:
            valid_mask = df[target_col].notna()
            self.means = df.loc[valid_mask, all_feat_cols].mean()
            self.stds = df.loc[valid_mask, all_feat_cols].std().replace(0, 1)
            self.target_mean = float(df.loc[valid_mask, target_col].mean())
            self.target_std = float(max(df.loc[valid_mask, target_col].std(), 1e-6))
            delta_series = df.loc[valid_mask, target_col].diff()
            self.delta_std = float(max(delta_series.std(), 1e-6))
        else:
            self.means = scaler_stats["means"]
            self.stds = scaler_stats["stds"]
            self.target_mean = scaler_stats["target_mean"]
            self.target_std = scaler_stats["target_std"]
            self.delta_std = scaler_stats["delta_std"]

        self.past_all_cols = past_obs_cols + future_known_cols
        self.price_col_idx = self.past_all_cols.index(target_col) if target_col in self.past_all_cols else 0

        self.samples: List[Dict] = []
        self._build_samples(df, dates)

        if len(self.samples) > 0:
            self.summary_dim = self.samples[0]["past_summary"].shape[0]
        else:
            n_feats = len(self.past_all_cols)
            self.summary_dim = 5 * 3 + 3 + 4 + n_feats

        logger.info("Dataset: %d hourly-samples from %d days (lookback=%d, "
                     "summary_dim=%d, target_mean=%.1f, target_std=%.1f, delta_std=%.1f)",
                     len(self.samples), len(dates), lookback,
                     self.summary_dim,
                     self.target_mean, self.target_std, self.delta_std)

    def _build_samples(self, df: pd.DataFrame, dates: List):
        for d in dates:
            day_mask = df.index.date == d
            day_data = df.loc[day_mask]
            if len(day_data) != 24:
                continue

            targets = day_data[self.target_col].values
            if np.isnan(targets).any():
                continue

            day_start_pos = df.index.get_loc(day_data.index[0])
            if isinstance(day_start_pos, slice):
                day_start_pos = day_start_pos.start
            if day_start_pos < self.lookback:
                continue

            past_slice = df.iloc[day_start_pos - self.lookback: day_start_pos]
            if len(past_slice) != self.lookback:
                continue

            prev_hour_price = past_slice[self.target_col].iloc[-1]
            if np.isnan(prev_hour_price):
                prev_hour_price = self.target_mean

            past_obs = past_slice[self.past_obs_cols].values.astype(np.float64)
            past_fut = past_slice[self.future_known_cols].values.astype(np.float64)
            for ci, col in enumerate(self.past_obs_cols):
                past_obs[:, ci] = (past_obs[:, ci] - self.means.get(col, 0)) / self.stds.get(col, 1)
            for ci, col in enumerate(self.future_known_cols):
                past_fut[:, ci] = (past_fut[:, ci] - self.means.get(col, 0)) / self.stds.get(col, 1)

            past_all = np.concatenate([past_obs, past_fut], axis=1)
            np.nan_to_num(past_all, copy=False, nan=0.0)

            past_summary = _extract_past_summary(past_all, self.price_col_idx)
            np.nan_to_num(past_summary, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            past_summary_t = torch.tensor(past_summary, dtype=torch.float32)
            past_seq_t = torch.tensor(past_all, dtype=torch.float32)

            future_all = day_data[self.future_known_cols].values.astype(np.float64)
            for ci, col in enumerate(self.future_known_cols):
                future_all[:, ci] = (future_all[:, ci] - self.means.get(col, 0)) / self.stds.get(col, 1)
            np.nan_to_num(future_all, copy=False, nan=0.0)

            for h in range(24):
                target_val = float(targets[h])
                target_norm = (target_val - self.target_mean) / self.target_std

                if h == 0:
                    delta_raw = target_val - prev_hour_price
                else:
                    delta_raw = target_val - float(targets[h - 1])
                delta_norm = delta_raw / self.delta_std

                self.samples.append({
                    "past_summary": past_summary_t,
                    "past_seq": past_seq_t,
                    "future": torch.tensor(future_all[h], dtype=torch.float32),
                    "target": torch.tensor(target_norm, dtype=torch.float32),
                    "delta": torch.tensor(delta_norm, dtype=torch.float32),
                    "target_raw": torch.tensor(target_val, dtype=torch.float32),
                    "delta_raw": torch.tensor(delta_raw, dtype=torch.float32),
                    "date": d,
                    "hour": h,
                })

    def get_scaler_stats(self) -> Dict:
        return {
            "means": self.means, "stds": self.stds,
            "target_mean": self.target_mean, "target_std": self.target_std,
            "delta_std": self.delta_std,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["past_summary"], s["past_seq"], s["future"], s["target"], s["delta"], s["target_raw"]


# =====================================================================
# 5. 模型 — 统计摘要 + 轻量 LSTM + 双输出头
# =====================================================================

class SeqPriceModel(nn.Module):
    """
    past_summary (summary_dim) + future_h (future_dim)
    → shared_trunk → price_head → ŷ_h
                    → delta_head → Δŷ_h

    past_seq 仅用于可选的轻量 LSTM 上下文（use_lstm=True 时启用）。
    默认关闭 LSTM，纯靠统计摘要 + future 特征。
    """

    def __init__(self, past_dim: int, future_dim: int, summary_dim: int,
                 hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.2,
                 use_lstm: bool = False):
        super().__init__()
        self.use_lstm = use_lstm

        enc_out = 0
        if use_lstm:
            self.encoder = nn.LSTM(
                input_size=past_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            enc_out = hidden_dim * 2

        head_in = enc_out + summary_dim + future_dim
        self.shared_trunk = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.price_head = nn.Linear(32, 1)
        self.delta_head = nn.Linear(32, 1)

    def forward(self, past_summary: torch.Tensor, past_seq: torch.Tensor,
                future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = [past_summary, future]
        if self.use_lstm:
            _, (h_n, _) = self.encoder(past_seq)
            c_seq = torch.cat([h_n[-2], h_n[-1]], dim=1)
            parts.insert(1, c_seq)

        combined = torch.cat(parts, dim=1)
        trunk = self.shared_trunk(combined)
        price = self.price_head(trunk).squeeze(-1)
        delta = self.delta_head(trunk).squeeze(-1)
        return price, delta


# =====================================================================
# 6. 训练
# =====================================================================

def _compute_val_profile_corr(model, val_ds, device):
    """在验证集上按天计算 profile_corr 均值。"""
    model.eval()
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    preds_raw, actuals_raw = [], []
    with torch.no_grad():
        for past_summary, past_seq, future, _, _, target_raw in loader:
            past_summary = past_summary.to(device)
            past_seq, future = past_seq.to(device), future.to(device)
            pred_price, _ = model(past_summary, past_seq, future)
            preds_raw.append(pred_price.cpu().numpy() * val_ds.target_std + val_ds.target_mean)
            actuals_raw.append(target_raw.numpy())
    preds_raw = np.concatenate(preds_raw)
    actuals_raw = np.concatenate(actuals_raw)

    dates = [s["date"] for s in val_ds.samples]
    hours = [s["hour"] for s in val_ds.samples]

    day_corrs = []
    from collections import defaultdict
    by_day = defaultdict(lambda: {"a": [None]*24, "p": [None]*24})
    for i, (d, h) in enumerate(zip(dates, hours)):
        by_day[d]["a"][h] = actuals_raw[i]
        by_day[d]["p"][h] = preds_raw[i]

    for d, vals in by_day.items():
        a = np.array(vals["a"], dtype=float)
        p = np.array(vals["p"], dtype=float)
        if np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue
        if np.std(a) < 1e-6 or np.std(p) < 1e-6:
            continue
        day_corrs.append(float(np.corrcoef(a, p)[0, 1]))
    return float(np.mean(day_corrs)) if day_corrs else 0.0


def _train_model(model: SeqPriceModel, train_ds: HourlySampleDataset,
                 val_ds: HourlySampleDataset,
                 n_epochs: int = 300, lr: float = 3e-4,
                 patience: int = 60,
                 w_price: float = 0.5, w_delta: float = 0.5) -> SeqPriceModel:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_mae = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss_price = 0.0
        epoch_loss_delta = 0.0
        n_batch = 0
        for past_summary, past_seq, future, target, delta, _ in train_loader:
            past_summary = past_summary.to(DEVICE)
            past_seq = past_seq.to(DEVICE)
            future = future.to(DEVICE)
            target = target.to(DEVICE)
            delta = delta.to(DEVICE)

            pred_price, pred_delta = model(past_summary, past_seq, future)
            loss_price = torch.mean(torch.abs(pred_price - target))
            loss_delta = torch.mean(torch.abs(pred_delta - delta))
            loss = w_price * loss_price + w_delta * loss_delta

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_price += loss_price.item()
            epoch_loss_delta += loss_delta.item()
            n_batch += 1

        scheduler.step()
        avg_price = epoch_loss_price / max(n_batch, 1)
        avg_delta = epoch_loss_delta / max(n_batch, 1)

        model.eval()
        val_price_abs = []
        with torch.no_grad():
            for past_summary, past_seq, future, target, delta, _ in val_loader:
                past_summary = past_summary.to(DEVICE)
                past_seq, future, target = past_seq.to(DEVICE), future.to(DEVICE), target.to(DEVICE)
                pred_price, _ = model(past_summary, past_seq, future)
                val_price_abs.append(torch.abs(pred_price - target).cpu())
        val_mae_norm = float(torch.cat(val_price_abs).mean())
        val_mae_raw = val_mae_norm * train_ds.target_std

        val_corr = _compute_val_profile_corr(model, val_ds, DEVICE)

        if val_mae_raw < best_val_mae:
            best_val_mae = val_mae_raw
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or wait == 0:
            logger.info("Epoch %3d | price=%.4f delta=%.4f | "
                        "val_mae=%.2f val_corr=%.3f | best_mae=%.2f wait=%d",
                        epoch + 1, avg_price, avg_delta,
                        val_mae_raw, val_corr, best_val_mae, wait)

        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Best val MAE (raw): %.2f", best_val_mae)
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model


# =====================================================================
# 7. 预测 + 诊断
# =====================================================================

def _predict_hourly(model: SeqPriceModel, ds: HourlySampleDataset
                    ) -> Tuple[np.ndarray, np.ndarray, List, List]:
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    actuals, preds = [], []
    with torch.no_grad():
        for past_summary, past_seq, future, _, _, target_raw in loader:
            past_summary = past_summary.to(DEVICE)
            past_seq, future = past_seq.to(DEVICE), future.to(DEVICE)
            pred_price, _ = model(past_summary, past_seq, future)
            pred_raw = pred_price.cpu().numpy() * ds.target_std + ds.target_mean
            preds.append(pred_raw)
            actuals.append(target_raw.numpy())

    return np.concatenate(actuals), np.concatenate(preds), \
           [s["date"] for s in ds.samples], [s["hour"] for s in ds.samples]


def _build_result_df(actual, pred, dates, hours):
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=h) for d, h in zip(dates, hours)]
    df = pd.DataFrame({"actual": actual, "pred": pred},
                       index=pd.DatetimeIndex(ts, name="ts"))
    return df.sort_index()


def _daily_diagnostics(results: pd.DataFrame):
    rows = []
    for d in sorted(set(results.index.date)):
        day = results.loc[str(d)]
        if len(day) != 24:
            continue
        a, p = day["actual"].values, day["pred"].values
        corr = float(np.corrcoef(a, p)[0, 1]) if np.std(a) > 1e-6 and np.std(p) > 1e-6 else np.nan
        rows.append({
            "date": str(d),
            "corr_day": round(corr, 4) if np.isfinite(corr) else np.nan,
            "neg_corr_flag": int(corr < 0) if np.isfinite(corr) else 0,
            "peak_true_hour": int(np.argmax(a)),
            "peak_pred_hour": int(np.argmax(p)),
            "valley_true_hour": int(np.argmin(a)),
            "valley_pred_hour": int(np.argmin(p)),
            "amp_true": round(float(np.max(a) - np.min(a)), 2),
            "amp_pred": round(float(np.max(p) - np.min(p)), 2),
            "mae_day": round(float(np.mean(np.abs(a - p))), 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# 8. 主流程
# =====================================================================

def run_v13(n_seeds: int = 5):
    logger.info("=" * 60)
    logger.info("V13 Seq Model (dual-head: price + delta, %d-seed ensemble)", n_seeds)
    logger.info("=" * 60)

    df = _load_raw_hourly()

    past_obs_cols = [c for c in PAST_OBSERVED if c in df.columns]
    future_known_cols = ([c for c in FUTURE_KNOWN if c in df.columns]
                         + [c for c in CALENDAR_FEATURES if c in df.columns])

    logger.info("Past observed: %d cols", len(past_obs_cols))
    logger.info("Future known:  %d cols", len(future_known_cols))

    target_valid_dates = sorted(set(
        df.loc[df[TARGET_COL].notna()].index.date
    ))
    train_end_date = pd.Timestamp(TRAIN_END).date()
    test_start_date = pd.Timestamp(TEST_START).date()

    all_train_dates = [d for d in target_valid_dates if d <= train_end_date]
    test_dates = [d for d in target_valid_dates if d >= test_start_date]

    n_val = max(int(len(all_train_dates) * 0.12), 5)
    val_dates = all_train_dates[-n_val:]
    train_dates = all_train_dates[:-n_val]

    logger.info("Train: %d days (%d samples), Val: %d days (%d samples), Test: %d days",
                len(train_dates), len(train_dates) * 24,
                len(val_dates), len(val_dates) * 24,
                len(test_dates))

    _set_seed(0)
    train_ds = HourlySampleDataset(
        df, train_dates, LOOKBACK,
        past_obs_cols, future_known_cols, TARGET_COL,
    )
    scaler = train_ds.get_scaler_stats()

    val_ds = HourlySampleDataset(
        df, val_dates, LOOKBACK,
        past_obs_cols, future_known_cols, TARGET_COL,
        scaler_stats=scaler,
    )
    test_ds = HourlySampleDataset(
        df, test_dates, LOOKBACK,
        past_obs_cols, future_known_cols, TARGET_COL,
        scaler_stats=scaler,
    )

    past_dim = len(past_obs_cols) + len(future_known_cols)
    future_dim = len(future_known_cols)
    summary_dim = train_ds.summary_dim
    logger.info("Model dims: past_dim=%d, future_dim=%d, summary_dim=%d",
                past_dim, future_dim, summary_dim)

    all_preds = []
    seeds = [42, 123, 2024, 7, 314][:n_seeds]

    for si, seed in enumerate(seeds):
        logger.info("--- Seed %d/%d (seed=%d) ---", si + 1, n_seeds, seed)
        _set_seed(seed)

        model = SeqPriceModel(
            past_dim=past_dim,
            future_dim=future_dim,
            summary_dim=summary_dim,
            hidden_dim=32,
            num_layers=1,
            dropout=0.2,
            use_lstm=True,
        )
        if si == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info("Total parameters: %d (%.1fK)", total_params, total_params / 1000)

        model = _train_model(model, train_ds, val_ds,
                             n_epochs=500, lr=3e-4, patience=60,
                             w_price=0.5, w_delta=0.5)

        _, pred_test, _, _ = _predict_hourly(model, test_ds)
        all_preds.append(pred_test)
        logger.info("Seed %d done, pred range: [%.1f, %.1f]",
                    seed, pred_test.min(), pred_test.max())

    pred_ensemble = np.mean(all_preds, axis=0)
    actual_test = np.array([s["target_raw"].item() for s in test_ds.samples])
    dates_test = [s["date"] for s in test_ds.samples]
    hours_test = [s["hour"] for s in test_ds.samples]

    results = _build_result_df(actual_test, pred_ensemble, dates_test, hours_test)
    results.to_csv(V13_DIR / "da_result.csv")
    logger.info("Saved da_result.csv: %d rows (ensemble of %d seeds)", len(results), n_seeds)

    mae = float(np.mean(np.abs(results["actual"] - results["pred"])))
    rmse = float(np.sqrt(np.mean((results["actual"] - results["pred"]) ** 2)))
    shape_report = compute_shape_report(
        results["actual"].values, results["pred"].values, results.index, include_v7=True)

    diag = _daily_diagnostics(results)
    neg_ratio = float(diag["neg_corr_flag"].mean()) if len(diag) > 0 else np.nan
    diag.to_csv(V13_DIR / "daily_shape_diagnostics.csv", index=False)

    summary = {"model": "V13-Seq", "MAE": round(mae, 2), "RMSE": round(rmse, 2),
               "neg_corr_day_ratio": round(neg_ratio, 4)}
    summary.update(shape_report)

    logger.info("-" * 40)
    logger.info("V13-Seq Ensemble Results:")
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)

    pd.DataFrame([summary]).to_csv(V13_DIR / "summary.csv", index=False)
    return summary, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v13()
