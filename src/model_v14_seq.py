"""
V14 时序模型 — 三路严格不重叠输入 + 独立编码器 + Gate/Attn 融合 + 双输出头。

路径 A: 子小时原始序列 (96 步 × 7 列, 15 分钟粒度 D-1 全天)
        → BiLSTM → c_sub (64,)
路径 B: 日级摘要 (~36 维, 手工统计 + 日历)
        → LayerNorm + MLP → c_summary (64,)
路径 C: 小时点特征 (~10 维, PM 预测 + hour sin/cos + 模板偏差)
        → MLP → c_hour (32,)

融合: c_hist = concat(c_sub, c_summary)
      Gate:  c_attend = sigmoid(W @ c_hour) * c_hist
      Attn:  Cross-Attention(Q=c_hour, KV=c_hist)
      → concat(c_attend, c_hour) → Shared Trunk → price_head + delta_head

产出: output/v14_seq/
"""

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import SOURCE_DIR, OUTPUT_DIR
from .model_baseline import TRAIN_END, TEST_START
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V14_DIR = OUTPUT_DIR / "v14_seq"
V14_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TARGET_COL = "da_clearing_price"
EFFECTIVE_START = "2025-11-01"
EFFECTIVE_END = "2026-03-10 23:00:00"
SUB_HOUR_STEPS = 96  # 24h × 4 (15-min)


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# =====================================================================
# 1. 子小时原始序列加载 (路径 A)
# =====================================================================

_RAW_15MIN_FILES = {
    "da_clearing_price": ("日前市场交易出清结果.csv", "现货日前市场出清电价"),
    "rt_clearing_price": ("实时出清结果.csv", "现货实时出清电价"),
    "actual_load": ("实际负荷.csv", "实际负荷"),
    "renewable_gen": ("新能源总出力.csv", "新能源总出力"),
    "total_gen": ("发电总出力.csv", "发电总出力"),
    "hydro_gen": ("水电（含抽蓄）总出力.csv", "水电（含抽蓄）总出力"),
}

_RAW_5MIN_FILES = {
    "tie_line_power": ("省间联络线输电.csv", "省间联络线输电"),
}

SUB_HOUR_COLS = [
    "da_clearing_price", "rt_clearing_price", "actual_load",
    "renewable_gen", "tie_line_power", "total_gen", "hydro_gen",
]


def _load_sub_hour_15min() -> pd.DataFrame:
    """加载 15 分钟粒度的原始数据并合并为宽表。"""
    parts = []
    for metric, (fname, col_name) in _RAW_15MIN_FILES.items():
        path = SOURCE_DIR / fname
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.rename(columns={"datetime": "ts", col_name: metric})
        df = df[["ts", metric]].dropna(subset=["ts"])
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.set_index("ts").sort_index()
        parts.append(df[[metric]])

    for metric, (fname, col_name) in _RAW_5MIN_FILES.items():
        path = SOURCE_DIR / fname
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.rename(columns={"datetime": "ts", col_name: metric})
        df = df[["ts", metric]].dropna(subset=["ts"])
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.set_index("ts").sort_index()
        resampled = df[[metric]].resample("15min").mean()
        parts.append(resampled)

    result = pd.concat(parts, axis=1, join="outer").sort_index()
    result = result.loc[EFFECTIVE_START:EFFECTIVE_END]
    logger.info("Sub-hour 15min data: %d rows × %d cols (%s ~ %s)",
                len(result), len(result.columns),
                result.index.min(), result.index.max())
    return result


# =====================================================================
# 2. 小时级数据加载 (路径 B/C 共用)
# =====================================================================

def _load_hourly() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_hourly_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    df = df.loc[EFFECTIVE_START:EFFECTIVE_END].sort_index()
    logger.info("Hourly data: %d rows × %d cols", len(df), len(df.columns))
    return df


# =====================================================================
# 3. 日级摘要特征提取 (路径 B, ~36 维)
# =====================================================================

def _build_day_summary(hourly: pd.DataFrame, d, price_col: str = "da_clearing_price") -> Optional[np.ndarray]:
    """为日期 d 构建日级摘要向量。返回 None 如果数据不足。"""
    d_ts = pd.Timestamp(d)

    feats = []

    for offset_days in [1, 2, 3]:
        ref_date = d_ts - pd.Timedelta(days=offset_days)
        mask = hourly.index.date == ref_date.date()
        day_prices = hourly.loc[mask, price_col].dropna()
        if len(day_prices) >= 20:
            feats.append(float(day_prices.mean()))
            feats.append(float(day_prices.max() - day_prices.min()))
            feats.append(float(day_prices.std()))
        else:
            feats.extend([np.nan, np.nan, np.nan])

    d_minus1 = d_ts - pd.Timedelta(days=1)
    d_minus3 = d_ts - pd.Timedelta(days=3)
    h72_mask = (hourly.index >= d_minus3) & (hourly.index < d_ts)
    h72_prices = hourly.loc[h72_mask, price_col].dropna()
    if len(h72_prices) >= 48:
        x = np.arange(len(h72_prices))
        slope = np.polyfit(x, h72_prices.values, 1)[0]
        feats.append(slope)
    else:
        feats.append(0.0)

    d1_mask = hourly.index.date == d_minus1.date()
    d1_prices = hourly.loc[d1_mask, price_col].dropna()
    if len(d1_prices) >= 20:
        diff = np.diff(d1_prices.values)
        feats.append(float(np.mean(diff)))
        feats.append(float(np.std(diff)))
    else:
        feats.extend([0.0, 0.0])

    d1_all = hourly.loc[d1_mask].sort_index()
    if len(d1_all) == 24:
        for h_idx in [21, 22, 23]:
            feats.append(float(d1_all[price_col].iloc[h_idx]))
    else:
        feats.extend([np.nan] * 3)

    for col in ["da_avg_clearing_price", "rt_avg_clearing_price"]:
        vals = hourly.loc[d1_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
        feats.append(float(vals.iloc[0]) if len(vals) > 0 else np.nan)

    d_minus2 = d_ts - pd.Timedelta(days=2)
    d2_mask = hourly.index.date == d_minus2.date()
    col = "avg_bid_price"
    vals = hourly.loc[d2_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
    feats.append(float(vals.iloc[0]) if len(vals) > 0 else np.nan)

    for col in ["da_clearing_power", "da_clearing_unit_count",
                "rt_clearing_volume", "rt_clearing_unit_count"]:
        vals = hourly.loc[d1_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
        feats.append(float(vals.mean()) if len(vals) > 0 else np.nan)

    for col in ["settlement_da_price", "settlement_rt_price"]:
        vals = hourly.loc[d2_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
        feats.append(float(vals.mean()) if len(vals) > 0 else np.nan)

    for col in ["section_ratio_C-500-洪板断面送板桥-全接线【常】",
                "section_ratio_C-500-资铜断面送铜梁-全接线【常】"]:
        vals = hourly.loc[d2_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
        feats.append(float(vals.mean()) if len(vals) > 0 else np.nan)

    for col in ["da_reliability_clearing_price", "reliability_da_price", "reliability_rt_price"]:
        vals = hourly.loc[d1_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
        feats.append(float(vals.mean()) if len(vals) > 0 else np.nan)

    for col in ["maintenance_gen_count", "maintenance_grid_count"]:
        d_mask = hourly.index.date == d_ts.date()
        vals = hourly.loc[d_mask, col].dropna() if col in hourly.columns else pd.Series(dtype=float)
        feats.append(float(vals.iloc[0]) if len(vals) > 0 else 0.0)

    dow = d_ts.dayofweek
    feats.append(math.sin(2 * math.pi * dow / 7))
    feats.append(math.cos(2 * math.pi * dow / 7))
    feats.append(1.0 if dow >= 5 else 0.0)
    month = d_ts.month
    feats.append(math.sin(2 * math.pi * month / 12))
    feats.append(math.cos(2 * math.pi * month / 12))

    return np.array(feats, dtype=np.float32)


# =====================================================================
# 4. 小时点特征 (路径 C, ~10 维)
# =====================================================================

HOUR_FEATURE_COLS = [
    "load_forecast",
    "renewable_fcst",
    "renewable_fcst_total_pm",
    "total_gen_fcst_pm",
    "hydro_gen_fcst_pm",
    "non_market_gen_fcst_pm",
    "tie_line_fcst_pm",
]


def _build_hour_features(hourly: pd.DataFrame, d, h: int,
                         price_col: str = "da_clearing_price") -> np.ndarray:
    """为日期 d 小时 h 构建点特征向量 (~10 维)。"""
    ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
    feats = []
    for col in HOUR_FEATURE_COLS:
        if col in hourly.columns and ts in hourly.index:
            val = hourly.loc[ts, col]
            feats.append(float(val) if pd.notna(val) else np.nan)
        else:
            feats.append(np.nan)

    feats.append(math.sin(2 * math.pi * h / 24))
    feats.append(math.cos(2 * math.pi * h / 24))

    tmpl_col = f"template_dev_h{h}"
    tmpl_val = np.nan
    d_ts = pd.Timestamp(d)
    d_minus1 = d_ts - pd.Timedelta(days=1)
    d1_mask = hourly.index.date == d_minus1.date()
    d1_prices = hourly.loc[d1_mask, price_col].dropna()
    if len(d1_prices) == 24:
        day_mean = d1_prices.mean()
        tmpl_val = float(d1_prices.iloc[h] - day_mean)
    feats.append(tmpl_val if np.isfinite(tmpl_val) else 0.0)

    return np.array(feats, dtype=np.float32)


# =====================================================================
# 5. Dataset
# =====================================================================

class V14Dataset(Dataset):
    def __init__(self, sub_hour_df: pd.DataFrame, hourly_df: pd.DataFrame,
                 dates: List, scaler_stats: Dict = None):
        self.dates = dates

        sub_cols = [c for c in SUB_HOUR_COLS if c in sub_hour_df.columns]
        self.sub_cols = sub_cols

        hour_feat_cols = [c for c in HOUR_FEATURE_COLS if c in hourly_df.columns]

        if scaler_stats is None:
            valid_mask = hourly_df[TARGET_COL].notna()
            self.sub_means = sub_hour_df[sub_cols].mean()
            self.sub_stds = sub_hour_df[sub_cols].std().replace(0, 1)
            self.hour_means = hourly_df.loc[valid_mask, hour_feat_cols].mean()
            self.hour_stds = hourly_df.loc[valid_mask, hour_feat_cols].std().replace(0, 1)
            self.target_mean = float(hourly_df.loc[valid_mask, TARGET_COL].mean())
            self.target_std = float(max(hourly_df.loc[valid_mask, TARGET_COL].std(), 1e-6))
            delta = hourly_df.loc[valid_mask, TARGET_COL].diff()
            self.delta_std = float(max(delta.std(), 1e-6))
        else:
            self.sub_means = scaler_stats["sub_means"]
            self.sub_stds = scaler_stats["sub_stds"]
            self.hour_means = scaler_stats["hour_means"]
            self.hour_stds = scaler_stats["hour_stds"]
            self.target_mean = scaler_stats["target_mean"]
            self.target_std = scaler_stats["target_std"]
            self.delta_std = scaler_stats["delta_std"]

        self.samples: List[Dict] = []
        self._build(sub_hour_df, hourly_df, dates)

        if self.samples:
            self.summary_dim = self.samples[0]["day_summary"].shape[0]
            self.hour_dim = self.samples[0]["hour_feat"].shape[0]
            self.sub_dim = len(sub_cols)
        else:
            self.summary_dim = 36
            self.hour_dim = 10
            self.sub_dim = len(sub_cols)

        logger.info("V14Dataset: %d samples from %d days "
                     "(sub_dim=%d, summary_dim=%d, hour_dim=%d, "
                     "target_mean=%.1f, target_std=%.1f)",
                     len(self.samples), len(dates),
                     self.sub_dim, self.summary_dim, self.hour_dim,
                     self.target_mean, self.target_std)

    def _build(self, sub_df: pd.DataFrame, hourly: pd.DataFrame, dates: List):
        for d in dates:
            d_ts = pd.Timestamp(d)
            day_mask_h = hourly.index.date == d
            day_hourly = hourly.loc[day_mask_h]
            if len(day_hourly) != 24:
                continue
            targets = day_hourly[TARGET_COL].values
            if np.isnan(targets).any():
                continue

            d_minus1 = d_ts - pd.Timedelta(days=1)
            sub_start = d_minus1
            sub_end = d_minus1 + pd.Timedelta(hours=23, minutes=45)
            sub_slice = sub_df.loc[sub_start:sub_end, self.sub_cols]

            expected_idx = pd.date_range(sub_start, sub_end, freq="15min")
            if len(expected_idx) != SUB_HOUR_STEPS:
                continue
            sub_slice = sub_slice.reindex(expected_idx)

            sub_arr = sub_slice.values.astype(np.float64)
            for ci, col in enumerate(self.sub_cols):
                sub_arr[:, ci] = (sub_arr[:, ci] - self.sub_means.get(col, 0)) / self.sub_stds.get(col, 1)
            np.nan_to_num(sub_arr, copy=False, nan=0.0)
            sub_tensor = torch.tensor(sub_arr, dtype=torch.float32)

            day_summary = _build_day_summary(hourly, d)
            if day_summary is None:
                continue
            np.nan_to_num(day_summary, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            summary_tensor = torch.tensor(day_summary, dtype=torch.float32)

            prev_price = np.nan
            d1_mask = hourly.index.date == d_minus1.date()
            d1_h = hourly.loc[d1_mask]
            if len(d1_h) == 24:
                prev_price = float(d1_h[TARGET_COL].iloc[-1])
            if np.isnan(prev_price):
                prev_price = self.target_mean

            for h in range(24):
                target_val = float(targets[h])
                target_norm = (target_val - self.target_mean) / self.target_std

                if h == 0:
                    delta_raw = target_val - prev_price
                else:
                    delta_raw = target_val - float(targets[h - 1])
                delta_norm = delta_raw / self.delta_std

                hour_feat = _build_hour_features(hourly, d, h)
                for ci, col in enumerate(HOUR_FEATURE_COLS):
                    if ci < len(hour_feat) - 3:
                        if col in self.hour_means.index:
                            hour_feat[ci] = (hour_feat[ci] - self.hour_means[col]) / self.hour_stds.get(col, 1)
                np.nan_to_num(hour_feat, copy=False, nan=0.0)
                hour_tensor = torch.tensor(hour_feat, dtype=torch.float32)

                self.samples.append({
                    "sub_hour": sub_tensor,
                    "day_summary": summary_tensor,
                    "hour_feat": hour_tensor,
                    "target": torch.tensor(target_norm, dtype=torch.float32),
                    "delta": torch.tensor(delta_norm, dtype=torch.float32),
                    "target_raw": torch.tensor(target_val, dtype=torch.float32),
                    "date": d,
                    "hour": h,
                })

    def get_scaler_stats(self) -> Dict:
        return {
            "sub_means": self.sub_means, "sub_stds": self.sub_stds,
            "hour_means": self.hour_means, "hour_stds": self.hour_stds,
            "target_mean": self.target_mean, "target_std": self.target_std,
            "delta_std": self.delta_std,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (s["sub_hour"], s["day_summary"], s["hour_feat"],
                s["target"], s["delta"], s["target_raw"])


# =====================================================================
# 6. 模型
# =====================================================================

class V14Model(nn.Module):
    """三路编码 + Gate/Attn 融合 + 双输出头。"""

    def __init__(self, sub_dim: int, summary_dim: int, hour_dim: int,
                 hidden: int = 32, dropout: float = 0.2,
                 fusion_mode: str = "gate",
                 disable_sub_hour: bool = False):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.disable_sub_hour = disable_sub_hour

        enc_out = hidden * 2  # 64
        if not disable_sub_hour:
            self.sub_encoder = nn.LSTM(
                input_size=sub_dim, hidden_size=hidden,
                num_layers=1, batch_first=True, bidirectional=True,
            )

        self.summary_encoder = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.hour_encoder = nn.Sequential(
            nn.Linear(hour_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        hist_dim = (enc_out + 64) if not disable_sub_hour else 64
        c_hour_dim = 32

        if fusion_mode == "gate":
            self.gate_proj = nn.Linear(c_hour_dim, hist_dim)
            fused_dim = hist_dim + c_hour_dim
        elif fusion_mode == "cross_attn":
            self.attn_q = nn.Linear(c_hour_dim, 64)
            self.attn_k = nn.Linear(hist_dim, 64)
            self.attn_v = nn.Linear(hist_dim, 64)
            fused_dim = 64 + c_hour_dim
        else:
            fused_dim = hist_dim + c_hour_dim

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 128),
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

    def forward(self, sub_hour, day_summary, hour_feat):
        if not self.disable_sub_hour:
            _, (h_n, _) = self.sub_encoder(sub_hour)
            c_sub = torch.cat([h_n[-2], h_n[-1]], dim=1)

        c_summary = self.summary_encoder(day_summary)
        c_hour = self.hour_encoder(hour_feat)

        if not self.disable_sub_hour:
            c_hist = torch.cat([c_sub, c_summary], dim=1)
        else:
            c_hist = c_summary

        if self.fusion_mode == "gate":
            gate = torch.sigmoid(self.gate_proj(c_hour))
            c_attend = gate * c_hist
            combined = torch.cat([c_attend, c_hour], dim=1)
        elif self.fusion_mode == "cross_attn":
            Q = self.attn_q(c_hour).unsqueeze(1)
            K = self.attn_k(c_hist).unsqueeze(1)
            V = self.attn_v(c_hist).unsqueeze(1)
            scale = Q.size(-1) ** 0.5
            attn_w = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1)
            c_attend = torch.bmm(attn_w, V).squeeze(1)
            combined = torch.cat([c_attend, c_hour], dim=1)
        else:
            combined = torch.cat([c_hist, c_hour], dim=1)

        trunk_out = self.trunk(combined)
        price = self.price_head(trunk_out).squeeze(-1)
        delta = self.delta_head(trunk_out).squeeze(-1)
        return price, delta


# =====================================================================
# 7. 训练
# =====================================================================

def _compute_val_profile_corr(model, val_ds, device):
    model.eval()
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    preds_raw, actuals_raw = [], []
    with torch.no_grad():
        for sub, summary, hour, _, _, tgt_raw in loader:
            sub, summary, hour = sub.to(device), summary.to(device), hour.to(device)
            p, _ = model(sub, summary, hour)
            preds_raw.append(p.cpu().numpy() * val_ds.target_std + val_ds.target_mean)
            actuals_raw.append(tgt_raw.numpy())
    preds_raw = np.concatenate(preds_raw)
    actuals_raw = np.concatenate(actuals_raw)

    by_day = defaultdict(lambda: {"a": [None]*24, "p": [None]*24})
    for i, s in enumerate(val_ds.samples):
        by_day[s["date"]]["a"][s["hour"]] = actuals_raw[i]
        by_day[s["date"]]["p"][s["hour"]] = preds_raw[i]

    corrs = []
    for vals in by_day.values():
        a = np.array(vals["a"], dtype=float)
        p = np.array(vals["p"], dtype=float)
        if np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue
        if np.std(a) < 1e-6 or np.std(p) < 1e-6:
            continue
        corrs.append(float(np.corrcoef(a, p)[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


def _train_model(model, train_ds, val_ds,
                 n_epochs=500, lr=3e-4, patience=60,
                 w_price=0.5, w_delta=0.5):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    huber = nn.SmoothL1Loss()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_mae = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        ep_price, ep_delta, n_b = 0.0, 0.0, 0
        for sub, summary, hour, target, delta, _ in train_loader:
            sub = sub.to(DEVICE)
            summary = summary.to(DEVICE)
            hour = hour.to(DEVICE)
            target = target.to(DEVICE)
            delta = delta.to(DEVICE)

            pred_p, pred_d = model(sub, summary, hour)
            loss_p = huber(pred_p, target)
            loss_d = huber(pred_d, delta)
            loss = w_price * loss_p + w_delta * loss_d

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_price += loss_p.item()
            ep_delta += loss_d.item()
            n_b += 1

        scheduler.step()

        model.eval()
        val_abs = []
        with torch.no_grad():
            for sub, summary, hour, target, delta, _ in val_loader:
                sub, summary, hour = sub.to(DEVICE), summary.to(DEVICE), hour.to(DEVICE)
                target = target.to(DEVICE)
                pred_p, _ = model(sub, summary, hour)
                val_abs.append(torch.abs(pred_p - target).cpu())
        val_mae_raw = float(torch.cat(val_abs).mean()) * train_ds.target_std

        val_corr = _compute_val_profile_corr(model, val_ds, DEVICE)

        if val_mae_raw < best_val_mae:
            best_val_mae = val_mae_raw
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or wait == 0:
            logger.info("Epoch %3d | price=%.4f delta=%.4f | "
                        "val_mae=%.2f val_corr=%.3f | best=%.2f wait=%d",
                        epoch + 1,
                        ep_price / max(n_b, 1), ep_delta / max(n_b, 1),
                        val_mae_raw, val_corr, best_val_mae, wait)

        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Best val MAE: %.2f", best_val_mae)
    if best_state:
        model.load_state_dict(best_state)
    return model.to(DEVICE)


# =====================================================================
# 8. 预测 + 诊断
# =====================================================================

def _predict(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    actuals, preds = [], []
    with torch.no_grad():
        for sub, summary, hour, _, _, tgt_raw in loader:
            sub, summary, hour = sub.to(DEVICE), summary.to(DEVICE), hour.to(DEVICE)
            p, _ = model(sub, summary, hour)
            preds.append(p.cpu().numpy() * ds.target_std + ds.target_mean)
            actuals.append(tgt_raw.numpy())
    return np.concatenate(actuals), np.concatenate(preds)


def _build_result_df(actual, pred, ds):
    ts = [pd.Timestamp(s["date"]) + pd.Timedelta(hours=s["hour"]) for s in ds.samples]
    df = pd.DataFrame({"actual": actual, "pred": pred},
                       index=pd.DatetimeIndex(ts, name="ts"))
    return df.sort_index()


def _daily_diagnostics(results):
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
            "amp_true": round(float(np.max(a) - np.min(a)), 2),
            "amp_pred": round(float(np.max(p) - np.min(p)), 2),
            "mae_day": round(float(np.mean(np.abs(a - p))), 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# 9. 主流程
# =====================================================================

def run_v14(fusion_mode: str = "gate", n_seeds: int = 5,
            disable_sub_hour: bool = False,
            disable_delta: bool = False):
    label = f"V14-{'Gate' if fusion_mode == 'gate' else 'Attn'}"
    if disable_sub_hour:
        label += "-noSubHour"
    if disable_delta:
        label += "-noDelta"

    logger.info("=" * 60)
    logger.info("%s (%d-seed ensemble)", label, n_seeds)
    logger.info("=" * 60)

    sub_hour_df = _load_sub_hour_15min()
    hourly_df = _load_hourly()

    target_valid_dates = sorted(set(
        hourly_df.loc[hourly_df[TARGET_COL].notna()].index.date
    ))
    train_end_date = pd.Timestamp(TRAIN_END).date()
    test_start_date = pd.Timestamp(TEST_START).date()

    all_train_dates = [d for d in target_valid_dates if d <= train_end_date]
    test_dates = [d for d in target_valid_dates if d >= test_start_date]

    n_val = max(int(len(all_train_dates) * 0.12), 5)
    val_dates = all_train_dates[-n_val:]
    train_dates = all_train_dates[:-n_val]

    logger.info("Train: %d days, Val: %d days, Test: %d days",
                len(train_dates), len(val_dates), len(test_dates))

    _set_seed(0)
    train_ds = V14Dataset(sub_hour_df, hourly_df, train_dates)
    scaler = train_ds.get_scaler_stats()
    val_ds = V14Dataset(sub_hour_df, hourly_df, val_dates, scaler_stats=scaler)
    test_ds = V14Dataset(sub_hour_df, hourly_df, test_dates, scaler_stats=scaler)

    sub_dim = train_ds.sub_dim
    summary_dim = train_ds.summary_dim
    hour_dim = train_ds.hour_dim

    if disable_sub_hour:
        sub_dim_model = sub_dim
    else:
        sub_dim_model = sub_dim

    logger.info("Dims: sub=%d, summary=%d, hour=%d", sub_dim, summary_dim, hour_dim)

    all_preds = []
    seeds = [42, 123, 2024, 7, 314][:n_seeds]

    for si, seed in enumerate(seeds):
        logger.info("--- Seed %d/%d (seed=%d) ---", si + 1, n_seeds, seed)
        _set_seed(seed)

        model = V14Model(
            sub_dim=sub_dim, summary_dim=summary_dim, hour_dim=hour_dim,
            hidden=32, dropout=0.2, fusion_mode=fusion_mode,
            disable_sub_hour=disable_sub_hour,
        )
        if si == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("Parameters: %d (%.1fK)", n_params, n_params / 1000)

        w_d = 0.0 if disable_delta else 0.5
        w_p = 1.0 if disable_delta else 0.5

        model = _train_model(model, train_ds, val_ds,
                             n_epochs=500, lr=3e-4, patience=60,
                             w_price=w_p, w_delta=w_d)

        _, pred_test = _predict(model, test_ds)
        all_preds.append(pred_test)
        logger.info("Seed %d: pred range [%.1f, %.1f]",
                    seed, pred_test.min(), pred_test.max())

    pred_ensemble = np.mean(all_preds, axis=0)
    actual_test = np.array([s["target_raw"].item() for s in test_ds.samples])

    results = _build_result_df(actual_test, pred_ensemble, test_ds)
    safe_label = label.replace(" ", "_")
    result_csv = V14_DIR / f"da_result_{safe_label}.csv"
    results.to_csv(result_csv)

    mae = float(np.mean(np.abs(results["actual"] - results["pred"])))
    rmse = float(np.sqrt(np.mean((results["actual"] - results["pred"]) ** 2)))
    shape_report = compute_shape_report(
        results["actual"].values, results["pred"].values, results.index, include_v7=True)

    diag = _daily_diagnostics(results)
    neg_ratio = float(diag["neg_corr_flag"].mean()) if len(diag) > 0 else np.nan
    diag.to_csv(V14_DIR / "daily_shape_diagnostics.csv", index=False)

    summary = {"model": label, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
               "neg_corr_day_ratio": round(neg_ratio, 4)}
    summary.update(shape_report)

    logger.info("-" * 40)
    logger.info("%s Results:", label)
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)

    pd.DataFrame([summary]).to_csv(V14_DIR / f"summary_{safe_label}.csv", index=False)

    also_as_main = V14_DIR / "da_result.csv"
    results.to_csv(also_as_main)

    return summary, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v14(fusion_mode="gate", n_seeds=5)
