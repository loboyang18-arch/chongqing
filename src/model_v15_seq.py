"""
V15-Lite 验证模型 — 瘦身双路 NN + Gate 融合 + 双输出头。

路径 B: 日级摘要 (~18-20 维, 精简统计 + 日历)
        → LayerNorm + MLP → c_summary (64,)
路径 C: 小时点特征 (~9-10 维, PM 预测 + hour sin/cos, 可选 template_dev)
        → MLP → c_hour (32,)

融合: Gate(c_hour → c_summary) → concat → Trunk → price_head + delta_head

产出: output/v15_verify/
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

from .config import OUTPUT_DIR
from .model_baseline import TRAIN_END, TEST_START
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V15_DIR = OUTPUT_DIR / "v15_verify"
V15_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TARGET_COL = "da_clearing_price"
EFFECTIVE_START = "2025-11-01"
EFFECTIVE_END = "2026-03-10 23:00:00"


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# =====================================================================
# 1. 小时级数据加载
# =====================================================================

def _load_hourly() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_hourly_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    df = df.loc[EFFECTIVE_START:EFFECTIVE_END].sort_index()
    return df


# =====================================================================
# 2. 日级摘要特征 (路径 B, ~18-20 维 — 精简版)
# =====================================================================

def _build_day_summary_slim(hourly: pd.DataFrame, d,
                            price_col: str = "da_clearing_price") -> Optional[np.ndarray]:
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

    dow = d_ts.dayofweek
    feats.append(math.sin(2 * math.pi * dow / 7))
    feats.append(math.cos(2 * math.pi * dow / 7))
    feats.append(1.0 if dow >= 5 else 0.0)
    month = d_ts.month
    feats.append(math.sin(2 * math.pi * month / 12))
    feats.append(math.cos(2 * math.pi * month / 12))

    return np.array(feats, dtype=np.float32)


# =====================================================================
# 3. 小时点特征 (路径 C, ~9 维, 可选 template_dev)
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
                         include_template_dev: bool = False,
                         price_col: str = "da_clearing_price") -> np.ndarray:
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

    if include_template_dev:
        d_ts = pd.Timestamp(d)
        d_minus1 = d_ts - pd.Timedelta(days=1)
        d1_mask = hourly.index.date == d_minus1.date()
        d1_prices = hourly.loc[d1_mask, price_col].dropna()
        if len(d1_prices) == 24:
            day_mean = d1_prices.mean()
            tmpl_val = float(d1_prices.iloc[h] - day_mean)
        else:
            tmpl_val = 0.0
        feats.append(tmpl_val)

    return np.array(feats, dtype=np.float32)


# =====================================================================
# 4. Dataset
# =====================================================================

class V15Dataset(Dataset):
    def __init__(self, hourly_df: pd.DataFrame, dates: List,
                 include_template_dev: bool = False,
                 scaler_stats: Dict = None):
        self.dates = dates
        self.include_template_dev = include_template_dev

        hour_feat_cols = [c for c in HOUR_FEATURE_COLS if c in hourly_df.columns]

        if scaler_stats is None:
            valid_mask = hourly_df[TARGET_COL].notna()
            self.hour_means = hourly_df.loc[valid_mask, hour_feat_cols].mean()
            self.hour_stds = hourly_df.loc[valid_mask, hour_feat_cols].std().replace(0, 1)
            self.target_mean = float(hourly_df.loc[valid_mask, TARGET_COL].mean())
            self.target_std = float(max(hourly_df.loc[valid_mask, TARGET_COL].std(), 1e-6))
            delta = hourly_df.loc[valid_mask, TARGET_COL].diff()
            self.delta_std = float(max(delta.std(), 1e-6))
        else:
            self.hour_means = scaler_stats["hour_means"]
            self.hour_stds = scaler_stats["hour_stds"]
            self.target_mean = scaler_stats["target_mean"]
            self.target_std = scaler_stats["target_std"]
            self.delta_std = scaler_stats["delta_std"]

        self.samples: List[Dict] = []
        self._build(hourly_df, dates)

        if self.samples:
            self.summary_dim = self.samples[0]["day_summary"].shape[0]
            self.hour_dim = self.samples[0]["hour_feat"].shape[0]
        else:
            self.summary_dim = 20
            self.hour_dim = 10 if include_template_dev else 9

        logger.info("V15Dataset: %d samples from %d days "
                     "(summary_dim=%d, hour_dim=%d, tmpl=%s)",
                     len(self.samples), len(dates),
                     self.summary_dim, self.hour_dim, include_template_dev)

    def _build(self, hourly: pd.DataFrame, dates: List):
        for d in dates:
            d_ts = pd.Timestamp(d)
            day_mask = hourly.index.date == d
            day_hourly = hourly.loc[day_mask]
            if len(day_hourly) != 24:
                continue
            targets = day_hourly[TARGET_COL].values
            if np.isnan(targets).any():
                continue

            day_summary = _build_day_summary_slim(hourly, d)
            if day_summary is None:
                continue
            np.nan_to_num(day_summary, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            summary_tensor = torch.tensor(day_summary, dtype=torch.float32)

            d_minus1 = d_ts - pd.Timedelta(days=1)
            d1_mask = hourly.index.date == d_minus1.date()
            d1_h = hourly.loc[d1_mask]
            prev_price = self.target_mean
            if len(d1_h) == 24:
                prev_price = float(d1_h[TARGET_COL].iloc[-1])

            for h in range(24):
                target_val = float(targets[h])
                target_norm = (target_val - self.target_mean) / self.target_std

                delta_raw = target_val - (prev_price if h == 0 else float(targets[h - 1]))
                delta_norm = delta_raw / self.delta_std

                hour_feat = _build_hour_features(
                    hourly, d, h,
                    include_template_dev=self.include_template_dev)
                for ci, col in enumerate(HOUR_FEATURE_COLS):
                    if ci < len(hour_feat) - (3 if self.include_template_dev else 2):
                        if col in self.hour_means.index:
                            hour_feat[ci] = (hour_feat[ci] - self.hour_means[col]) / self.hour_stds.get(col, 1)
                np.nan_to_num(hour_feat, copy=False, nan=0.0)
                hour_tensor = torch.tensor(hour_feat, dtype=torch.float32)

                self.samples.append({
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
            "hour_means": self.hour_means, "hour_stds": self.hour_stds,
            "target_mean": self.target_mean, "target_std": self.target_std,
            "delta_std": self.delta_std,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (s["day_summary"], s["hour_feat"],
                s["target"], s["delta"], s["target_raw"])


# =====================================================================
# 5. 模型
# =====================================================================

class V15Model(nn.Module):
    def __init__(self, summary_dim: int, hour_dim: int,
                 dropout: float = 0.2):
        super().__init__()

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

        hist_dim = 64
        c_hour_dim = 32

        self.gate_proj = nn.Linear(c_hour_dim, hist_dim)
        fused_dim = hist_dim + c_hour_dim  # 96

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.price_head = nn.Linear(32, 1)
        self.delta_head = nn.Linear(32, 1)

    def forward(self, day_summary, hour_feat):
        c_summary = self.summary_encoder(day_summary)
        c_hour = self.hour_encoder(hour_feat)

        gate = torch.sigmoid(self.gate_proj(c_hour))
        c_attend = gate * c_summary
        combined = torch.cat([c_attend, c_hour], dim=1)

        trunk_out = self.trunk(combined)
        price = self.price_head(trunk_out).squeeze(-1)
        delta = self.delta_head(trunk_out).squeeze(-1)
        return price, delta


# =====================================================================
# 6. 训练
# =====================================================================

def _compute_val_profile_corr(model, val_ds, device):
    model.eval()
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    preds_raw, actuals_raw = [], []
    with torch.no_grad():
        for summary, hour, _, _, tgt_raw in loader:
            summary, hour = summary.to(device), hour.to(device)
            p, _ = model(summary, hour)
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
                 w_price=0.7, w_delta=0.3,
                 use_composite_stop=False):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    huber = nn.SmoothL1Loss()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_score = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        ep_price, ep_delta, n_b = 0.0, 0.0, 0
        for summary, hour, target, delta, _ in train_loader:
            summary = summary.to(DEVICE)
            hour = hour.to(DEVICE)
            target = target.to(DEVICE)
            delta = delta.to(DEVICE)

            pred_p, pred_d = model(summary, hour)
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
            for summary, hour, target, delta, _ in val_loader:
                summary, hour = summary.to(DEVICE), hour.to(DEVICE)
                target = target.to(DEVICE)
                pred_p, _ = model(summary, hour)
                val_abs.append(torch.abs(pred_p - target).cpu())
        val_mae_raw = float(torch.cat(val_abs).mean()) * train_ds.target_std

        val_corr = _compute_val_profile_corr(model, val_ds, DEVICE)

        if use_composite_stop:
            mae_norm = val_mae_raw / train_ds.target_std
            score = mae_norm + 0.5 * (1 - val_corr)
        else:
            score = val_mae_raw

        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or wait == 0:
            logger.info("Epoch %3d | price=%.4f delta=%.4f | "
                        "val_mae=%.2f val_corr=%.3f score=%.4f | best=%.4f wait=%d",
                        epoch + 1,
                        ep_price / max(n_b, 1), ep_delta / max(n_b, 1),
                        val_mae_raw, val_corr, score, best_score, wait)

        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Best score: %.4f", best_score)
    if best_state:
        model.load_state_dict(best_state)
    return model.to(DEVICE)


# =====================================================================
# 7. 预测 + 诊断
# =====================================================================

def _predict(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    actuals, preds = [], []
    with torch.no_grad():
        for summary, hour, _, _, tgt_raw in loader:
            summary, hour = summary.to(DEVICE), hour.to(DEVICE)
            p, _ = model(summary, hour)
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
            "date": str(d), "corr_day": round(corr, 4) if np.isfinite(corr) else np.nan,
            "neg_corr_flag": int(corr < 0) if np.isfinite(corr) else 0,
            "mae_day": round(float(np.mean(np.abs(a - p))), 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# 8. 主流程
# =====================================================================

def run_v15(label: str = "V15-Lite",
            n_seeds: int = 5,
            include_template_dev: bool = False,
            w_price: float = 0.7,
            w_delta: float = 0.3,
            use_composite_stop: bool = False,
            ensemble_mode: str = "mean"):
    logger.info("=" * 60)
    logger.info("%s (%d-seed ensemble, tmpl=%s, w=%.1f/%.1f, composite=%s, ens=%s)",
                label, n_seeds, include_template_dev, w_price, w_delta,
                use_composite_stop, ensemble_mode)
    logger.info("=" * 60)

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
    train_ds = V15Dataset(hourly_df, train_dates,
                          include_template_dev=include_template_dev)
    scaler = train_ds.get_scaler_stats()
    val_ds = V15Dataset(hourly_df, val_dates,
                        include_template_dev=include_template_dev,
                        scaler_stats=scaler)
    test_ds = V15Dataset(hourly_df, test_dates,
                         include_template_dev=include_template_dev,
                         scaler_stats=scaler)

    summary_dim = train_ds.summary_dim
    hour_dim = train_ds.hour_dim

    all_preds = []
    val_corrs = []
    seeds = [42, 123, 2024, 7, 314][:n_seeds]

    for si, seed in enumerate(seeds):
        logger.info("--- Seed %d/%d (seed=%d) ---", si + 1, n_seeds, seed)
        _set_seed(seed)

        model = V15Model(summary_dim=summary_dim, hour_dim=hour_dim, dropout=0.2)
        if si == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("Parameters: %d (%.1fK)", n_params, n_params / 1000)

        model = _train_model(model, train_ds, val_ds,
                             n_epochs=500, lr=3e-4, patience=60,
                             w_price=w_price, w_delta=w_delta,
                             use_composite_stop=use_composite_stop)

        vc = _compute_val_profile_corr(model, val_ds, DEVICE)
        val_corrs.append(vc)

        _, pred_test = _predict(model, test_ds)
        all_preds.append(pred_test)
        logger.info("Seed %d: pred range [%.1f, %.1f], val_corr=%.3f",
                    seed, pred_test.min(), pred_test.max(), vc)

    if ensemble_mode == "top3" and len(all_preds) >= 3:
        top_idx = np.argsort(val_corrs)[-3:]
        pred_ensemble = np.mean([all_preds[i] for i in top_idx], axis=0)
        logger.info("Top-3 ensemble (seeds: %s, corrs: %s)",
                    [seeds[i] for i in top_idx],
                    [round(val_corrs[i], 3) for i in top_idx])
    elif ensemble_mode == "weighted":
        weights = np.array(val_corrs)
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(all_preds)) / len(all_preds)
        pred_ensemble = np.average(all_preds, axis=0, weights=weights)
        logger.info("Weighted ensemble (weights: %s)", [round(w, 3) for w in weights])
    else:
        pred_ensemble = np.mean(all_preds, axis=0)

    actual_test = np.array([s["target_raw"].item() for s in test_ds.samples])

    results = _build_result_df(actual_test, pred_ensemble, test_ds)
    safe_label = label.replace(" ", "_")
    result_csv = V15_DIR / f"da_result_{safe_label}.csv"
    results.to_csv(result_csv)

    mae = float(np.mean(np.abs(results["actual"] - results["pred"])))
    rmse = float(np.sqrt(np.mean((results["actual"] - results["pred"]) ** 2)))
    shape_report = compute_shape_report(
        results["actual"].values, results["pred"].values, results.index, include_v7=True)

    diag = _daily_diagnostics(results)
    neg_ratio = float(diag["neg_corr_flag"].mean()) if len(diag) > 0 else np.nan

    summary = {"model": label, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
               "neg_corr_day_ratio": round(neg_ratio, 4)}
    summary.update(shape_report)

    logger.info("-" * 40)
    logger.info("%s Results:", label)
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)

    pd.DataFrame([summary]).to_csv(V15_DIR / f"summary_{safe_label}.csv", index=False)
    return summary, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v15()
