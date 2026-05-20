"""
V20 — Conv2D 预测 RT-DA 结算价差值

目标 = settlement_rt_price - settlement_da_price
思路：DA 结算价已由其他模型预测（或已知），本模型只学习 RT 相对 DA 的偏差。
最终 RT 预测 = DA 实际/预测 + 本模型预测的 spread。

架构：默认使用与 V20c 同量级的小 Conv2D（32/48/32 + 小 FC）直接回归价差；
  设置环境变量 V20_LARGE=1 可恢复 V18 的 Conv2dPriceNet（大模型）。
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

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
    C_LAG0, C_LAG1, C_LAG2, C_TOTAL,
    LOOKBACK_DAYS, SLOTS_PER_HOUR,
    DEVICE, WARMUP_EPOCHS,
    _log_device, _seed,
    _build_daily_arrays, _get_hour_slots,
    compute_norm,
    HourlyConv2dDataset, Conv2dPriceNet,
    _plot_train_last_week,
)

_V20_CTX_BEFORE = 1
_V20_CTX_AFTER = 1
H_SLOTS = (_V20_CTX_BEFORE + 1 + _V20_CTX_AFTER) * SLOTS_PER_HOUR  # 12
from price_forecast_eval import quick_shape_report

logger = logging.getLogger(__name__)

MAX_EPOCHS = int(os.environ.get("V20_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("V20_BS", "64"))
LR = float(os.environ.get("V20_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("V20_WD", "5e-4"))
USE_LARGE_NET = os.environ.get("V20_LARGE", "0").strip().lower() in ("1", "true", "yes")

CONV1 = int(os.environ.get("V20_CONV1", "32"))
CONV2 = int(os.environ.get("V20_CONV2", "48"))
CONV3 = int(os.environ.get("V20_CONV3", "32"))
FC_HIDDEN = int(os.environ.get("V20_FC", "32"))
DROPOUT = float(os.environ.get("V20_DROPOUT", "0.2"))
DROPOUT2D = float(os.environ.get("V20_DROPOUT2D", "0.06"))

V20_DIR = OUTPUT_DIR / "v20_spread"


class Conv2dSpreadRegNetSmall(nn.Module):
    """(B,C,12,7)→小卷积骨干→标量价差（与 V20c 小网同拓扑，回归头）。"""

    def __init__(
        self,
        c_in: int = C_TOTAL,
        c1: int = CONV1,
        c2: int = CONV2,
        c3: int = CONV3,
        fc_hidden: int = FC_HIDDEN,
        dropout: float = DROPOUT,
        dropout2d: float = DROPOUT2D,
    ):
        super().__init__()
        c1, c2, c3 = max(8, c1), max(8, c2), max(8, c3)
        fc_hidden = max(8, fc_hidden)
        fc_in = c3 * 1 * 5
        d2a = max(0.0, min(dropout2d * 0.55, 0.45))
        d2b = max(0.0, min(dropout2d, 0.45))
        d2c = max(0.0, min(dropout2d * 0.45, 0.35))
        dp = max(0.0, min(dropout, 0.6))

        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Dropout2d(d2a),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Dropout2d(d2b),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=0),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Dropout2d(d2c),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, fc_hidden),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(-1)


def _build_spread_targets(df, valid_dates):
    """构建 RT-DA spread 目标和单独的 DA/RT 日级数组。"""
    spread_targets: Dict = {}
    da_targets: Dict = {}
    rt_targets: Dict = {}
    new_valid = []

    for d in valid_dates:
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        raw = df.reindex(grid)

        da_col, rt_col = "settlement_da_price", "settlement_rt_price"
        if da_col not in raw.columns or rt_col not in raw.columns:
            continue

        da_h = raw[da_col].values[::4][:24]
        rt_h = raw[rt_col].values[::4][:24]

        if (len(da_h) == 24 and len(rt_h) == 24
                and np.isfinite(da_h).all() and np.isfinite(rt_h).all()):
            spread_targets[d] = (rt_h - da_h).astype(np.float32)
            da_targets[d] = da_h.astype(np.float32)
            rt_targets[d] = rt_h.astype(np.float32)
            new_valid.append(d)

    return sorted(new_valid), spread_targets, da_targets, rt_targets


def _eval_mae_hourly(model, loader, y_mean, y_std):
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
    out_dir = Path(out_dir or V20_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)

    ds_kw = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    train_ds = HourlyConv2dDataset(sample_dates=train_days, **ds_kw)
    val_ds = HourlyConv2dDataset(sample_dates=val_days, **ds_kw)

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    if USE_LARGE_NET:
        model = Conv2dPriceNet().to(DEVICE)
        tag = "Conv2dPriceNet(large)"
    else:
        model = Conv2dSpreadRegNetSmall().to(DEVICE)
        tag = "Conv2dSpreadRegNetSmall"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s params: %d  (C_in=%d)", tag, n_params, C_TOTAL)
    if not USE_LARGE_NET:
        logger.info(
            "  small conv %d/%d/%d fc=%d dropout=%.2f dropout2d=%.2f wd=%.2e",
            CONV1, CONV2, CONV3, FC_HIDDEN, DROPOUT, DROPOUT2D, WEIGHT_DECAY,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - WARMUP_EPOCHS, 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS])

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

        do_log = ep % 10 == 0 or ep == epochs - 1
        if do_log:
            train_mae = _eval_mae_hourly(
                model,
                DataLoader(train_ds, min(512, len(train_ds)), shuffle=False),
                y_mean, y_std,
            )
            val_mae = _eval_mae_hourly(model, val_l, y_mean, y_std)
            logger.info(
                "  ep%3d  loss=%.4f  train_mae=%.1f  val_mae=%.1f  lr=%.1e",
                ep, ep_loss / max(nb, 1), train_mae, val_mae,
                opt.param_groups[0]["lr"],
            )

    ckpt = out_dir / "seed0.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Saved last-epoch checkpoint → %s", ckpt)
    return model


def predict_days(model, dates, day_lag0, day_lag1, day_lag2, day_targets,
                 norm_mean, norm_std, y_mean, y_std):
    ds = HourlyConv2dDataset(
        sample_dates=dates,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    if len(ds) == 0:
        return np.zeros((0, 24)), np.zeros((0, 24)), []

    loader = DataLoader(ds, min(512, len(ds)), shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for grid, _ in loader:
            pred = model(grid.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
    preds_flat = np.concatenate(all_preds) * y_std + y_mean

    day_preds: Dict = {}
    for i, (d, h) in enumerate(ds.meta):
        if d not in day_preds:
            day_preds[d] = np.full(24, np.nan)
        day_preds[d][h] = preds_flat[i]

    valid_dates = sorted(
        d for d in day_preds
        if d in day_targets
        and not np.isnan(day_preds[d]).any()
        and not np.isnan(day_targets[d]).any()
    )
    p24 = np.array([day_preds[d] for d in valid_dates])
    a24 = np.array([day_targets[d] for d in valid_dates])
    return p24, a24, valid_dates


def run_v20(out_dir=None):
    out_dir = Path(out_dir) if out_dir else V20_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V20 Conv2D Spread — predict (RT - DA) settlement price")
    if USE_LARGE_NET:
        logger.info("  backbone: Conv2dPriceNet (large) (%d, %d, %d)", C_TOTAL, H_SLOTS, LOOKBACK_DAYS)
    else:
        logger.info(
            "  backbone: small reg net conv=%d/%d/%d fc=%d (%d, %d, %d)",
            CONV1, CONV2, CONV3, FC_HIDDEN, C_TOTAL, H_SLOTS, LOOKBACK_DAYS,
        )
    logger.info("  epochs=%d, bs=%d, lr=%.1e, wd=%.1e, V20_LARGE=%s",
                MAX_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, str(USE_LARGE_NET))
    _log_device()

    df = build_feature_matrix()
    all_valid, day_lag0, day_lag1, day_lag2, *_ = _build_daily_arrays(df)

    valid_dates, spread_targets, da_targets, rt_targets = \
        _build_spread_targets(df, all_valid)
    logger.info("Spread targets: %d valid days", len(valid_dates))

    # 统计训练集 spread 分布
    tr_last = TRAIN_END.date()
    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
    train_days = [d for d in valid_dates if d <= tr_last]
    val_days = [d for d in valid_dates if tr_last < d <= val_last]
    test_days = [d for d in valid_dates if ts_first <= d <= ts_last]

    train_spreads = np.concatenate([spread_targets[d] for d in train_days])
    logger.info("Train spread stats: mean=%.2f std=%.2f min=%.2f max=%.2f",
                train_spreads.mean(), train_spreads.std(),
                train_spreads.min(), train_spreads.max())
    logger.info("Train days: %d (%s ~ %s)", len(train_days),
                train_days[0], train_days[-1])
    logger.info("Val days:   %d (%s ~ %s)", len(val_days),
                val_days[0] if val_days else "?", val_days[-1] if val_days else "?")
    logger.info("Test days:  %d (%s ~ %s)", len(test_days),
                test_days[0], test_days[-1])

    norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)

    tgt_stack = np.stack([spread_targets[d] for d in train_days])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    model = train_model(
        train_days=train_days,
        val_days=val_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=spread_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        out_dir=out_dir,
    )
    logger.info("Using last-epoch weights (ep %d)", MAX_EPOCHS - 1)

    # ── 测试集 spread 预测 ──
    p24_spread, a24_spread, dates = predict_days(
        model, test_days,
        day_lag0, day_lag1, day_lag2, spread_targets,
        norm_mean, norm_std, y_mean, y_std,
    )

    # ── Spread 指标 ──
    spread_flat_p = p24_spread.ravel()
    spread_flat_a = a24_spread.ravel()
    spread_mae = float(np.mean(np.abs(spread_flat_p - spread_flat_a)))
    spread_rmse = float(np.sqrt(np.mean((spread_flat_p - spread_flat_a) ** 2)))
    logger.info("Spread MAE: %.2f  RMSE: %.2f", spread_mae, spread_rmse)

    # ── 重建 RT 价格 = DA_actual + predicted_spread ──
    rt_pred_24 = np.array([da_targets[d] + p24_spread[i] for i, d in enumerate(dates)])
    rt_actual_24 = np.array([rt_targets[d] for d in dates])

    rows_spread = []
    rows_rt = []
    for i, d in enumerate(dates):
        for h in range(24):
            ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
            rows_spread.append({
                "ts": ts,
                "actual": a24_spread[i, h],
                "predicted": p24_spread[i, h],
            })
            rows_rt.append({
                "ts": ts,
                "actual": rt_actual_24[i, h],
                "predicted": rt_pred_24[i, h],
            })

    spread_result = pd.DataFrame(rows_spread).set_index("ts").sort_index()
    spread_result.to_csv(out_dir / "spread_result.csv")

    rt_result = pd.DataFrame(rows_rt).set_index("ts").sort_index()
    rt_result.to_csv(out_dir / "rt_result.csv")
    logger.info("Saved: spread_result.csv, rt_result.csv (%d rows, %d days)",
                len(rt_result), len(dates))

    # ── RT 重建指标 ──
    rt_flat_a = rt_result["actual"].values
    rt_flat_p = rt_result["predicted"].values
    rt_mae = float(np.mean(np.abs(rt_flat_a - rt_flat_p)))
    rt_rmse = float(np.sqrt(np.mean((rt_flat_a - rt_flat_p) ** 2)))
    rt_shape = quick_shape_report(rt_flat_a, rt_flat_p, rt_result.index)

    # ── 可视化 ──
    from price_forecast_eval.viz import run_standard_visualization
    run_standard_visualization(
        out_dir / "rt_result.csv",
        out_dir=out_dir / "plots",
        label="V20-SmallSpread→RT" if not USE_LARGE_NET else "V20-Spread→RT",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    logger.info("=" * 60)
    logger.info("V20 Spread RESULTS")
    logger.info("── Spread (RT-DA) prediction ──")
    logger.info("  Spread MAE:  %.2f", spread_mae)
    logger.info("  Spread RMSE: %.2f", spread_rmse)
    logger.info("── Reconstructed RT = DA_actual + pred_spread ──")
    logger.info("  RT MAE:  %.2f", rt_mae)
    logger.info("  RT RMSE: %.2f", rt_rmse)
    for k, v in rt_shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    return {
        "spread_mae": spread_mae, "spread_rmse": spread_rmse,
        "rt_mae": rt_mae, "rt_rmse": rt_rmse, **rt_shape,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v20()
