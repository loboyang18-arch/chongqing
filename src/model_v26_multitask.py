"""
V26 — 多任务 ResConv：同时预测日前 + 实时出清（各带 V25 式价格/差分双头）。

输入/骨干与 V25 一致（V24 sql 特征，默认 5+0，Lag2 保留 realtime_clearing_*）。
输出（4 个回归头，无涨跌平分类）：
  - P_da, Δ_da  （与 V25 dual02 相同）
  - P_rt, Δ_rt

损失：
  L = λ_da·(L1(P_da)+δ·L1(Δ_da)) + λ_rt·(L1(P_rt)+δ·L1(Δ_rt))
  δ = V18_DELTA_LAMBDA（默认 0.2）

环境变量：V26_LAMBDA_DA/RT（默认 1/1），继承 V18_* 训练超参。
"""

from __future__ import annotations

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
from .experiment.splits import TRAIN_END, VAL_END, TEST_START, TEST_END
from .model_v18_conv2d import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    DEVICE,
    H_SLOTS,
    LOOKBACK_DAYS,
    MAX_EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    DROPOUT,
    WARMUP_EPOCHS,
    _get_hour_slots,
    compute_norm,
)
from .model_v24_da import (
    _patch_v18_for_v24_direct,
    _restore_v18,
    _snapshot_v18,
    load_sql_feature_matrix,
)
from .model_v25_resconv import _ResBlock
from price_forecast_eval import quick_shape_report

logger = logging.getLogger(__name__)

V26_DIR = OUTPUT_DIR / os.environ.get("V26_OUT_DIR", "v26_multitask").strip()

DA_COL = "market_clearing_price"
RT_COL = "realtime_clearing_price"
LAMBDA_DA = float(os.environ.get("V26_LAMBDA_DA", "1.0"))
LAMBDA_RT = float(os.environ.get("V26_LAMBDA_RT", "1.0"))
DELTA_LAMBDA = float(os.environ.get("V18_DELTA_LAMBDA", "0.2"))


def _make_head(fc_in: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(), nn.Linear(fc_in, 128), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(128, 1),
    )


class QuadDualHeadResConv2dNet(nn.Module):
    """共享 ResConv 骨干 + 日前/实时各一套 price+delta 头（同 V25 DualHead）。"""

    def __init__(self, c_in: int, h_slots: int = H_SLOTS, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.res64 = _ResBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.trans = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.res128a = _ResBlock(128)
        self.res128b = _ResBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.final1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=0), nn.BatchNorm2d(64), nn.GELU())
        self.final2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.GELU())

        h_out = h_slots // 2 // 2 - 2 - 2
        w_out = 7 - 2 - 2
        fc_in = 64 * h_out * w_out

        self.da_price_head = _make_head(fc_in, dropout)
        self.da_delta_head = _make_head(fc_in, dropout)
        self.rt_price_head = _make_head(fc_in, dropout)
        self.rt_delta_head = _make_head(fc_in, dropout)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res64(x)
        x = self.pool1(x)
        x = self.trans(x)
        x = self.res128a(x)
        x = self.res128b(x)
        x = self.pool2(x)
        x = self.final1(x)
        x = self.final2(x)
        return x

    def forward(self, x: torch.Tensor):
        feat = self._backbone(x)
        da_p = self.da_price_head(feat).squeeze(-1)
        da_d = self.da_delta_head(feat).squeeze(-1)
        rt_p = self.rt_price_head(feat).squeeze(-1)
        rt_d = self.rt_delta_head(feat).squeeze(-1)
        return da_p, da_d, rt_p, rt_d


def _hourly_from_96(v: np.ndarray) -> Optional[np.ndarray]:
    if v.size != 96:
        return None
    h = np.nanmean(v.reshape(24, 4), axis=1)
    if h.size == 24 and np.isfinite(h).all():
        return h.astype(np.float32)
    return None


def _build_hourly_deltas(day_targets: Dict) -> Dict:
    day_delta: Dict = {}
    sorted_days = sorted(day_targets.keys())
    for i, d in enumerate(sorted_days):
        abs_vals = day_targets[d]
        prev_d = sorted_days[i - 1] if i > 0 else None
        if prev_d is not None and prev_d in day_targets:
            anchor = float(day_targets[prev_d][-1])
        else:
            anchor = float(abs_vals[0])
        delta = np.empty(24, dtype=np.float32)
        delta[0] = abs_vals[0] - anchor
        delta[1:] = abs_vals[1:] - abs_vals[:-1]
        day_delta[d] = delta
    return day_delta


def _build_multitask_daily(
    df: pd.DataFrame,
    lag0_cols: List[str],
    lag1_cols: List[str],
    lag2_cols: List[str],
) -> Tuple[List, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    feat_cols = lag0_cols + lag1_cols + lag2_cols
    df = df.copy()
    df[feat_cols] = df[feat_cols].ffill()

    start = df.index.min().normalize().date()
    end = df.index.max().date()
    day_lag0, day_lag1, day_lag2 = {}, {}, {}
    day_da, day_rt = {}, {}
    valid = []

    for d_ts in pd.date_range(start, end, freq="D"):
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        raw = df.reindex(grid)
        if raw[feat_cols].isna().all().any():
            continue

        l0 = raw[lag0_cols].values.astype(np.float32)
        steps = np.arange(96, dtype=np.float32)
        dow = float(pd.Timestamp(d).dayofweek)
        te = np.column_stack([
            np.sin(2 * np.pi * steps / 96),
            np.cos(2 * np.pi * steps / 96),
            np.full(96, np.sin(2 * np.pi * dow / 7), dtype=np.float32),
            np.full(96, np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        ])
        day_lag0[d] = np.concatenate([l0, te], axis=1).astype(np.float32)
        day_lag1[d] = raw[lag1_cols].values.astype(np.float32)
        day_lag2[d] = raw[lag2_cols].values.astype(np.float32)

        if DA_COL not in raw.columns or RT_COL not in raw.columns:
            continue
        da_h = _hourly_from_96(raw[DA_COL].to_numpy(dtype=np.float64, copy=False))
        rt_h = _hourly_from_96(raw[RT_COL].to_numpy(dtype=np.float64, copy=False))
        if da_h is None or rt_h is None:
            continue
        day_da[d] = da_h
        day_rt[d] = rt_h
        valid.append(d)

    valid = sorted(valid)
    day_da_delta = _build_hourly_deltas(day_da)
    day_rt_delta = _build_hourly_deltas(day_rt)
    logger.info("V26 daily: %d days with DA+RT (+delta labels)", len(valid))
    return valid, day_lag0, day_lag1, day_lag2, day_da, day_rt, day_da_delta, day_rt_delta


class HourlyMultiTaskDataset(Dataset):
    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict,
        day_lag1: Dict,
        day_lag2: Dict,
        day_da: Dict,
        day_rt: Dict,
        day_da_delta: Dict,
        day_rt_delta: Dict,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        c_total: int,
        da_mean: float,
        da_std: float,
        rt_mean: float,
        rt_std: float,
        da_delta_mean: float,
        da_delta_std: float,
        rt_delta_mean: float,
        rt_delta_std: float,
        ctx_before: int = None,
        ctx_after: int = None,
    ):
        cb = ctx_before if ctx_before is not None else CONTEXT_BEFORE
        ca = ctx_after if ctx_after is not None else CONTEXT_AFTER
        self._c_total = int(c_total)

        a0, a1, a2 = set(day_lag0), set(day_lag1), set(day_lag2)
        self.items = []
        self.meta = []

        for d in sample_dates:
            if d not in day_da or d not in day_rt:
                continue
            dates0 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates1 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS, 0, -1)]
            dates2 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS + 1, 1, -1)]
            if not (all(dd in a0 for dd in dates0)
                    and all(dd in a1 for dd in dates1)
                    and all(dd in a2 for dd in dates2)):
                continue

            for h in range(24):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    d0, d1, d2 = dates0[k], dates1[k], dates2[k]
                    s0 = _get_hour_slots(day_lag0, d0, h, cb, ca)
                    s1 = _get_hour_slots(day_lag1, d1, h, cb, ca)
                    s2 = _get_hour_slots(day_lag2, d2, h, cb, ca)
                    layer = np.concatenate([s0, s1, s2], axis=1)
                    layers.append(layer)
                grid = np.stack(layers, axis=-1).transpose(1, 0, 2)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(self._c_total, 1, 1))
                        / norm_std.reshape(self._c_total, 1, 1)).astype(np.float32)

                da_n = np.float32((day_da[d][h] - da_mean) / da_std)
                rt_n = np.float32((day_rt[d][h] - rt_mean) / rt_std)
                da_dn = np.float32((day_da_delta[d][h] - da_delta_mean) / da_delta_std)
                rt_dn = np.float32((day_rt_delta[d][h] - rt_delta_mean) / rt_delta_std)
                self.items.append((grid, da_n, da_dn, rt_n, rt_dn))
                self.meta.append((d, h))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        grid, da, da_d, rt, rt_d = self.items[idx]
        return (
            torch.from_numpy(np.copy(grid)),
            torch.tensor(da, dtype=torch.float32),
            torch.tensor(da_d, dtype=torch.float32),
            torch.tensor(rt, dtype=torch.float32),
            torch.tensor(rt_d, dtype=torch.float32),
        )


def _delta_stats(day_delta: Dict, train_days: List) -> Tuple[float, float]:
    vals = [day_delta[d] for d in train_days if d in day_delta]
    if not vals:
        return 0.0, 1.0
    stack = np.concatenate(vals)
    return float(stack.mean()), max(float(stack.std()), 1e-8)


def _task_loss(price_p, price_t, delta_p, delta_t) -> torch.Tensor:
    return F.l1_loss(price_p, price_t) + DELTA_LAMBDA * F.l1_loss(delta_p, delta_t)


def _eval_mae_price(
    model: nn.Module,
    loader: DataLoader,
    da_mean: float,
    da_std: float,
    rt_mean: float,
    rt_std: float,
) -> Tuple[float, float]:
    model.eval()
    da_err, rt_err = [], []
    with torch.no_grad():
        for grid, da_t, _, rt_t, _ in loader:
            grid = grid.to(DEVICE)
            da_p, _, rt_p, _ = model(grid)
            da_err.append(np.abs(da_p.cpu().numpy() * da_std + da_mean - da_t.numpy() * da_std - da_mean))
            rt_err.append(np.abs(rt_p.cpu().numpy() * rt_std + rt_mean - rt_t.numpy() * rt_std - rt_mean))
    if not da_err:
        return float("inf"), float("inf")
    return float(np.concatenate(da_err).mean()), float(np.concatenate(rt_err).mean())


def _predict_prices(
    model: nn.Module,
    dates: List,
    day_lag0: Dict,
    day_lag1: Dict,
    day_lag2: Dict,
    day_da: Dict,
    day_rt: Dict,
    day_da_delta: Dict,
    day_rt_delta: Dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    c_total: int,
    da_mean: float,
    da_std: float,
    rt_mean: float,
    rt_std: float,
    da_delta_mean: float,
    da_delta_std: float,
    rt_delta_mean: float,
    rt_delta_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    ds = HourlyMultiTaskDataset(
        dates, day_lag0, day_lag1, day_lag2,
        day_da, day_rt, day_da_delta, day_rt_delta,
        norm_mean, norm_std, c_total,
        da_mean, da_std, rt_mean, rt_std,
        da_delta_mean, da_delta_std, rt_delta_mean, rt_delta_std,
    )
    loader = DataLoader(ds, min(512, max(len(ds), 1)), shuffle=False)
    model.eval()
    da_p, da_a, rt_p, rt_a = [], [], [], []
    with torch.no_grad():
        for grid, d_t, _, r_t, _ in loader:
            grid = grid.to(DEVICE)
            dp, _, rp, _ = model(grid)
            da_p.append(dp.cpu().numpy() * da_std + da_mean)
            rt_p.append(rp.cpu().numpy() * rt_std + rt_mean)
            da_a.append(d_t.numpy() * da_std + da_mean)
            rt_a.append(r_t.numpy() * rt_std + rt_mean)
    if not da_p:
        return np.array([]), np.array([]), np.array([]), np.array([]), []
    return (
        np.concatenate(da_p), np.concatenate(da_a),
        np.concatenate(rt_p), np.concatenate(rt_a),
        list(ds.meta),
    )


def _meta_to_result(
    da_p: np.ndarray, da_a: np.ndarray,
    rt_p: np.ndarray, rt_a: np.ndarray,
    meta: List,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_da, rows_rt = [], []
    for i, (d, h) in enumerate(meta):
        ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
        rows_da.append({"ts": ts, "actual": da_a[i], "predicted": da_p[i]})
        rows_rt.append({"ts": ts, "actual": rt_a[i], "predicted": rt_p[i]})
    return (
        pd.DataFrame(rows_da).set_index("ts").sort_index(),
        pd.DataFrame(rows_rt).set_index("ts").sort_index(),
    )


def run_v26(out_dir: Optional[Path] = None) -> Dict:
    out_dir = Path(out_dir or V26_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cb, ca = CONTEXT_BEFORE, CONTEXT_AFTER
    h_slots = (cb + 1 + ca) * 4
    epochs = MAX_EPOCHS

    logger.info("=" * 60)
    logger.info("V26 QuadDual — DA+RT, each price+delta (no L_dir)")
    logger.info("  ctx %d+%d | δ=%.2f | λ_da=%.2f λ_rt=%.2f", cb, ca, DELTA_LAMBDA, LAMBDA_DA, LAMBDA_RT)

    snap = _snapshot_v18()
    try:
        _patch_v18_for_v24_direct()
        import src.model_v18_conv2d as m18
        lag0, lag1, lag2 = list(m18.LAG0_COLS), list(m18.LAG1_COLS), list(m18.LAG2_COLS)
        c_tot = int(m18.C_TOTAL)

        df = load_sql_feature_matrix()
        (valid, day_lag0, day_lag1, day_lag2,
         day_da, day_rt, day_da_delta, day_rt_delta) = _build_multitask_daily(df, lag0, lag1, lag2)

        tr_last, val_last = TRAIN_END.date(), VAL_END.date()
        ts_first, ts_last = TEST_START.date(), TEST_END.date()
        train_days = [d for d in valid if d <= tr_last]
        val_days = [d for d in valid if tr_last < d <= val_last]
        test_days = [d for d in valid if ts_first <= d <= ts_last]
        logger.info("Train %d | Val %d | Test %d", len(train_days), len(val_days), len(test_days))

        norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
        da_stack = np.stack([day_da[d] for d in train_days])
        rt_stack = np.stack([day_rt[d] for d in train_days])
        da_mean, da_std = float(da_stack.mean()), float(da_stack.std()) + 1e-8
        rt_mean, rt_std = float(rt_stack.mean()), float(rt_stack.std()) + 1e-8
        da_delta_mean, da_delta_std = _delta_stats(day_da_delta, train_days)
        rt_delta_mean, rt_delta_std = _delta_stats(day_rt_delta, train_days)

        ds_kw = dict(
            day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
            day_da=day_da, day_rt=day_rt,
            day_da_delta=day_da_delta, day_rt_delta=day_rt_delta,
            norm_mean=norm_mean, norm_std=norm_std, c_total=c_tot,
            da_mean=da_mean, da_std=da_std, rt_mean=rt_mean, rt_std=rt_std,
            da_delta_mean=da_delta_mean, da_delta_std=da_delta_std,
            rt_delta_mean=rt_delta_mean, rt_delta_std=rt_delta_std,
        )
        train_ds = HourlyMultiTaskDataset(sample_dates=train_days, **ds_kw)
        val_ds = HourlyMultiTaskDataset(sample_dates=val_days, **ds_kw)
        tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
        vl = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

        model = QuadDualHeadResConv2dNet(c_in=c_tot, h_slots=h_slots, dropout=DROPOUT).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("QuadDualHeadResConv2dNet params: %d (C_in=%d)", n_params, c_tot)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - WARMUP_EPOCHS, 1), eta_min=1e-6)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

        for ep in range(epochs):
            model.train()
            ep_loss, nb = 0.0, 0
            for grid, da_t, da_dt, rt_t, rt_dt in tl:
                grid = grid.to(DEVICE)
                da_t, da_dt = da_t.to(DEVICE), da_dt.to(DEVICE)
                rt_t, rt_dt = rt_t.to(DEVICE), rt_dt.to(DEVICE)
                opt.zero_grad()
                da_p, da_dp, rt_p, rt_dp = model(grid)
                loss = (
                    LAMBDA_DA * _task_loss(da_p, da_t, da_dp, da_dt)
                    + LAMBDA_RT * _task_loss(rt_p, rt_t, rt_dp, rt_dt)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
                nb += 1
            sched.step()
            if ep % 10 == 0 or ep == epochs - 1:
                tr_da, tr_rt = _eval_mae_price(model, DataLoader(
                    train_ds, min(512, len(train_ds)), shuffle=False),
                    da_mean, da_std, rt_mean, rt_std)
                va_da, va_rt = _eval_mae_price(model, vl, da_mean, da_std, rt_mean, rt_std)
                logger.info(
                    "  ep%3d loss=%.4f train_da=%.1f train_rt=%.1f val_da=%.1f val_rt=%.1f",
                    ep, ep_loss / max(nb, 1), tr_da, tr_rt, va_da, va_rt,
                )

        torch.save(model.state_dict(), out_dir / "seed0.pt")

        da_p, da_a, rt_p, rt_a, meta = _predict_prices(
            model, test_days, day_lag0, day_lag1, day_lag2,
            day_da, day_rt, day_da_delta, day_rt_delta,
            norm_mean, norm_std, c_tot,
            da_mean, da_std, rt_mean, rt_std,
            da_delta_mean, da_delta_std, rt_delta_mean, rt_delta_std,
        )
        da_df, rt_df = _meta_to_result(da_p, da_a, rt_p, rt_a, meta)
        da_df.to_csv(out_dir / "da_result.csv")
        rt_df.to_csv(out_dir / "rt_result.csv")

        da_mae = float(np.mean(np.abs(da_p - da_a)))
        rt_mae = float(np.mean(np.abs(rt_p - rt_a)))
        da_shape = quick_shape_report(da_a, da_p, da_df.index)
        rt_shape = quick_shape_report(rt_a, rt_p, rt_df.index)

        meta_out = {
            "model": "QuadDualHeadResConv2dNet",
            "V18_CTX_BEFORE": cb, "V18_CTX_AFTER": ca,
            "V18_DELTA_LAMBDA": DELTA_LAMBDA,
            "V26_LAMBDA_DA": LAMBDA_DA, "V26_LAMBDA_RT": LAMBDA_RT,
            "C_TOTAL": c_tot,
            "test_da_mae": da_mae, "test_rt_mae": rt_mae,
            **{f"test_da_{k}": v for k, v in da_shape.items()},
            **{f"test_rt_{k}": v for k, v in rt_shape.items()},
        }
        with (out_dir / "v26_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        logger.info("=" * 60)
        logger.info("V26 TEST — DA MAE=%.2f  RT MAE=%.2f", da_mae, rt_mae)
        logger.info("  DA corr=%.4f  RT corr=%.4f",
                    da_shape.get("profile_corr", 0), rt_shape.get("profile_corr", 0))
        logger.info("=" * 60)
        return meta_out
    finally:
        _restore_v18(snap)
