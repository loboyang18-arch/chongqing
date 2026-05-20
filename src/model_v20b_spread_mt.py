"""
V20b — Conv2D 多任务：RT-DA 价差回归 + 二分类方向头（Focal Loss）

联合损失: L1(spread) + λ * FocalBCE(sign_logits, 1[spread > threshold])
方向标签: spread > SIGN_THRESHOLD 为 1，否则 0（可设 0 或 2 过滤噪声）。
预测阶段仍用回归头输出 spread（与 V20 一致），另在评估时输出方向头混淆矩阵。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
    C_TOTAL,
    LOOKBACK_DAYS,
    SLOTS_PER_HOUR,
    DEVICE,
    WARMUP_EPOCHS,
    _log_device,
    _seed,
    _build_daily_arrays,
    _get_hour_slots,
    compute_norm,
)

_V20B_CTX_BEFORE = 1
_V20B_CTX_AFTER = 1
H_SLOTS = (_V20B_CTX_BEFORE + 1 + _V20B_CTX_AFTER) * SLOTS_PER_HOUR  # 12
from .model_v20_spread import _build_spread_targets
from price_forecast_eval import quick_shape_report
from price_forecast_eval.viz import run_standard_visualization

logger = logging.getLogger(__name__)

MAX_EPOCHS = int(os.environ.get("V20B_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("V20B_BS", "64"))
LR = float(os.environ.get("V20B_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("V20B_WD", "1e-4"))
LAMBDA_DIR = float(os.environ.get("V20B_LAMBDA", "2.0"))
SIGN_THRESHOLD = float(os.environ.get("V20B_SIGN_THRESHOLD", "0.0"))
FOCAL_ALPHA = float(os.environ.get("V20B_FOCAL_ALPHA", "0.6"))
FOCAL_GAMMA = float(os.environ.get("V20B_FOCAL_GAMMA", "2.0"))

V20B_DIR = OUTPUT_DIR / "v20b_spread_mt"


class FocalLossBCE(nn.Module):
    """Focal loss for binary logits (targets 0/1 float)."""

    def __init__(self, alpha: float = 0.6, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,) or (B,1); targets: (B,) long/float 0/1
        logit = logits.view(-1)
        t = targets.float().view(-1)
        bce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
        p = torch.sigmoid(logit)
        p_t = p * t + (1.0 - p) * (1.0 - t)
        p_t = p_t.clamp(min=1e-7, max=1.0 - 1e-7)
        focal_w = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)
        return (alpha_t * focal_w * bce).mean()


class HourlySpreadMultiTaskDataset(Dataset):
    """每小时 (C,12,7) + 归一化 spread 目标 + 二分类方向标签。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict,
        day_lag1: Dict,
        day_lag2: Dict,
        day_targets: Dict,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        y_mean: float,
        y_std: float,
        sign_threshold: float = SIGN_THRESHOLD,
    ):
        a0 = set(day_lag0.keys())
        a1 = set(day_lag1.keys())
        a2 = set(day_lag2.keys())

        self.items: List[Tuple[np.ndarray, float, float]] = []
        self.meta: List[Tuple] = []

        for d in sample_dates:
            if d not in day_targets:
                continue

            dates0 = [
                (pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                for off in range(LOOKBACK_DAYS - 1, -1, -1)
            ]
            dates1 = [
                (pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                for off in range(LOOKBACK_DAYS, 0, -1)
            ]
            dates2 = [
                (pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                for off in range(LOOKBACK_DAYS + 1, 1, -1)
            ]

            ok = all(dd in a0 for dd in dates0) and all(dd in a1 for dd in dates1) and all(
                dd in a2 for dd in dates2
            )
            if not ok:
                continue

            for h in range(24):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    d0, d1, d2 = dates0[k], dates1[k], dates2[k]
                    s0 = _get_hour_slots(day_lag0, d0, h, _V20B_CTX_BEFORE, _V20B_CTX_AFTER)
                    s1 = _get_hour_slots(day_lag1, d1, h, _V20B_CTX_BEFORE, _V20B_CTX_AFTER)
                    s2 = _get_hour_slots(day_lag2, d2, h, _V20B_CTX_BEFORE, _V20B_CTX_AFTER)
                    layers.append(np.concatenate([s0, s1, s2], axis=1))

                grid = np.stack(layers, axis=-1).transpose(1, 0, 2)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = (
                    (grid - norm_mean.reshape(C_TOTAL, 1, 1))
                    / norm_std.reshape(C_TOTAL, 1, 1)
                ).astype(np.float32)

                spread_h = float(day_targets[d][h])
                tgt = np.float32((spread_h - y_mean) / y_std)
                sign_label = 1.0 if spread_h > sign_threshold else 0.0

                self.items.append((grid, tgt, sign_label))
                self.meta.append((d, h))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        grid, tgt, sl = self.items[idx]
        return (
            torch.from_numpy(grid),
            torch.tensor(tgt),
            torch.tensor(sl, dtype=torch.float32),
        )


class Conv2dSpreadMultiTaskNet(nn.Module):
    """共享 Conv2D 骨干 + spread 回归头 + 二分类方向头（单 logit）。"""

    def __init__(self, c_in: int = C_TOTAL):
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
        self.flatten = nn.Flatten()

        self.reg_head = nn.Sequential(
            nn.Linear(320, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(320, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.flatten(x)
        spread = self.reg_head(feat).squeeze(-1)
        sign_logit = self.dir_head(feat).squeeze(-1)
        return spread, sign_logit


def _eval_mae_hourly(model, loader, y_mean: float, y_std: float) -> float:
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for batch in loader:
            grid, tgt = batch[0], batch[1]
            spread, _ = model(grid.to(DEVICE))
            ps.append(spread.cpu().numpy())
            ts.append(tgt.numpy())
    p = np.concatenate(ps) * y_std + y_mean
    t = np.concatenate(ts) * y_std + y_mean
    return float(np.mean(np.abs(p - t)))


def _eval_sign_acc_from_regression(model, loader, y_mean: float, y_std: float, threshold: float) -> float:
    """(pred > th) == (actual > th)"""
    model.eval()
    ok, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            grid, tgt = batch[0], batch[1]
            spread, _ = model(grid.to(DEVICE))
            pred = spread.cpu().numpy() * y_std + y_mean
            act = tgt.numpy() * y_std + y_mean
            pa = pred > threshold
            aa = act > threshold
            ok += (pa == aa).sum()
            n += len(act)
    return ok / max(n, 1)


def _eval_dir_head_acc(model, loader) -> float:
    model.eval()
    ok, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            grid, _, sign_lab = batch
            _, logit = model(grid.to(DEVICE))
            pred = (torch.sigmoid(logit) > 0.5).long()
            lab = sign_lab.long()
            ok += (pred.cpu() == lab).sum().item()
            n += lab.numel()
    return ok / max(n, 1)


def train_model(
    train_days: List,
    val_days: List,
    day_lag0: Dict,
    day_lag1: Dict,
    day_lag2: Dict,
    day_targets: Dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    y_mean: float,
    y_std: float,
    epochs: int | None = None,
    out_dir: Path | None = None,
    sign_threshold: float = SIGN_THRESHOLD,
) -> Conv2dSpreadMultiTaskNet:
    epochs = epochs or MAX_EPOCHS
    out_dir = Path(out_dir or V20B_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)

    ds_kw = dict(
        day_lag0=day_lag0,
        day_lag1=day_lag1,
        day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean,
        norm_std=norm_std,
        y_mean=y_mean,
        y_std=y_std,
        sign_threshold=sign_threshold,
    )
    train_ds = HourlySpreadMultiTaskDataset(sample_dates=train_days, **ds_kw)
    val_ds = HourlySpreadMultiTaskDataset(sample_dates=val_days, **ds_kw)

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    model = Conv2dSpreadMultiTaskNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Conv2dSpreadMultiTaskNet params: %d  (C_in=%d)", n_params, C_TOTAL)

    focal = FocalLossBCE(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - WARMUP_EPOCHS, 1), eta_min=1e-6
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )

    for ep in range(epochs):
        model.train()
        ep_l1, ep_fl, ep_dir_ok, ep_dir_n, nb = 0.0, 0.0, 0, 0, 0

        for grid, tgt, sign_lab in tl:
            grid = grid.to(DEVICE)
            tgt = tgt.to(DEVICE)
            sign_lab = sign_lab.to(DEVICE)

            opt.zero_grad()
            spread, sign_logit = model(grid)

            l1 = F.l1_loss(spread, tgt)
            fl = focal(sign_logit, sign_lab)
            loss = l1 + LAMBDA_DIR * fl

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_l1 += l1.item()
            ep_fl += fl.item()
            pred_bin = (torch.sigmoid(sign_logit) > 0.5).long()
            ep_dir_ok += (pred_bin == sign_lab.long()).sum().item()
            ep_dir_n += sign_lab.numel()
            nb += 1

        sched.step()

        do_log = ep % 10 == 0 or ep == epochs - 1
        if do_log:
            train_mae = _eval_mae_hourly(
                model,
                DataLoader(train_ds, min(512, len(train_ds)), shuffle=False),
                y_mean,
                y_std,
            )
            val_mae = _eval_mae_hourly(model, val_l, y_mean, y_std)
            val_sign_reg = _eval_sign_acc_from_regression(model, val_l, y_mean, y_std, sign_threshold)
            val_dir_head = _eval_dir_head_acc(model, val_l)
            batch_dir = ep_dir_ok / max(ep_dir_n, 1)
            logger.info(
                "  ep%3d  L1=%.4f FL=%.4f dir_b=%.3f | tr_mae=%.1f val_mae=%.1f "
                "val_sign(reg)=%.3f val_sign(head)=%.3f lr=%.1e",
                ep,
                ep_l1 / max(nb, 1),
                ep_fl / max(nb, 1),
                batch_dir,
                train_mae,
                val_mae,
                val_sign_reg,
                val_dir_head,
                opt.param_groups[0]["lr"],
            )

    ckpt = out_dir / "seed0.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Saved last-epoch checkpoint → %s", ckpt)
    return model


def predict_days(
    model: Conv2dSpreadMultiTaskNet,
    dates: List,
    day_lag0: Dict,
    day_lag1: Dict,
    day_lag2: Dict,
    day_targets: Dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    y_mean: float,
    y_std: float,
    sign_threshold: float = SIGN_THRESHOLD,
):
    """仅用回归头得到 spread 预测。"""
    ds = HourlySpreadMultiTaskDataset(
        sample_dates=dates,
        day_lag0=day_lag0,
        day_lag1=day_lag1,
        day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean,
        norm_std=norm_std,
        y_mean=y_mean,
        y_std=y_std,
        sign_threshold=sign_threshold,
    )
    if len(ds) == 0:
        return np.zeros((0, 24)), np.zeros((0, 24)), []

    loader = DataLoader(ds, min(512, len(ds)), shuffle=False)
    model.eval()
    all_spread, all_logit = [], []
    with torch.no_grad():
        for batch in loader:
            grid = batch[0].to(DEVICE)
            spr, logit = model(grid)
            all_spread.append(spr.cpu().numpy())
            all_logit.append(logit.cpu().numpy())

    preds_flat = np.concatenate(all_spread) * y_std + y_mean
    logits_flat = np.concatenate(all_logit)

    day_preds: Dict = {}
    day_logits: Dict = {}
    for i, (d, h) in enumerate(ds.meta):
        if d not in day_preds:
            day_preds[d] = np.full(24, np.nan)
            day_logits[d] = np.full(24, np.nan)
        day_preds[d][h] = preds_flat[i]
        day_logits[d][h] = logits_flat[i]

    valid_dates = sorted(
        d
        for d in day_preds
        if d in day_targets
        and not np.isnan(day_preds[d]).any()
        and not np.isnan(day_targets[d]).any()
    )
    p24 = np.array([day_preds[d] for d in valid_dates])
    a24 = np.array([day_targets[d] for d in valid_dates])
    log24 = np.array([day_logits[d] for d in valid_dates])
    return p24, a24, valid_dates, log24


def _confusion_and_pr(
    actual: np.ndarray,
    pred_reg: np.ndarray,
    pred_head_prob: np.ndarray | None,
    threshold: float,
) -> Dict:
    """actual/pred_reg: flat arrays."""
    act_pos = actual > threshold
    reg_pos = pred_reg > threshold

    tp = int(np.sum(act_pos & reg_pos))
    tn = int(np.sum(~act_pos & ~reg_pos))
    fp = int(np.sum(~act_pos & reg_pos))
    fn = int(np.sum(act_pos & ~reg_pos))

    prec_pos = tp / max(tp + fp, 1)
    rec_pos = tp / max(tp + fn, 1)
    prec_neg = tn / max(tn + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    sign_acc = (tp + tn) / max(len(actual), 1)

    out = {
        "threshold": threshold,
        "sign_acc_regression": sign_acc,
        "confusion_reg_tp": tp,
        "confusion_reg_tn": tn,
        "confusion_reg_fp": fp,
        "confusion_reg_fn": fn,
        "precision_positive": prec_pos,
        "recall_positive": rec_pos,
        "precision_negative": prec_neg,
        "recall_negative": rec_neg,
    }

    if pred_head_prob is not None:
        head_pos = pred_head_prob > 0.5
        tp_h = int(np.sum(act_pos & head_pos))
        tn_h = int(np.sum(~act_pos & ~head_pos))
        fp_h = int(np.sum(~act_pos & head_pos))
        fn_h = int(np.sum(act_pos & ~head_pos))
        out["sign_acc_head"] = (tp_h + tn_h) / max(len(actual), 1)
        out["confusion_head_tp"] = tp_h
        out["confusion_head_tn"] = tn_h
        out["confusion_head_fp"] = fp_h
        out["confusion_head_fn"] = fn_h

    return out


def run_v20b(out_dir: Path | None = None) -> Dict:
    out_dir = Path(out_dir) if out_dir else V20B_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V20b Spread Multi-Task — L1(spread) + λ·FocalBCE(sign)")
    logger.info(
        "  λ_dir=%.2f threshold=%.2f focal_alpha=%.2f focal_gamma=%.1f",
        LAMBDA_DIR,
        SIGN_THRESHOLD,
        FOCAL_ALPHA,
        FOCAL_GAMMA,
    )
    logger.info("  epochs=%d bs=%d lr=%.1e wd=%.1e", MAX_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY)
    _log_device()

    df = build_feature_matrix()
    all_valid, day_lag0, day_lag1, day_lag2, *_ = _build_daily_arrays(df)
    valid_dates, spread_targets, da_targets, rt_targets = _build_spread_targets(df, all_valid)

    tr_last = TRAIN_END.date()
    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
    train_days = [d for d in valid_dates if d <= tr_last]
    val_days = [d for d in valid_dates if tr_last < d <= val_last]
    test_days = [d for d in valid_dates if ts_first <= d <= ts_last]

    norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([spread_targets[d] for d in train_days])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    model = train_model(
        train_days=train_days,
        val_days=val_days,
        day_lag0=day_lag0,
        day_lag1=day_lag1,
        day_lag2=day_lag2,
        day_targets=spread_targets,
        norm_mean=norm_mean,
        norm_std=norm_std,
        y_mean=y_mean,
        y_std=y_std,
        out_dir=out_dir,
        sign_threshold=SIGN_THRESHOLD,
    )
    logger.info("Using last-epoch weights (ep %d)", MAX_EPOCHS - 1)

    p24, a24, dates, log24 = predict_days(
        model,
        test_days,
        day_lag0,
        day_lag1,
        day_lag2,
        spread_targets,
        norm_mean,
        norm_std,
        y_mean,
        y_std,
        sign_threshold=SIGN_THRESHOLD,
    )

    spread_flat_p = p24.ravel()
    spread_flat_a = a24.ravel()
    head_prob_flat = 1.0 / (1.0 + np.exp(-log24.ravel()))

    spread_mae = float(np.mean(np.abs(spread_flat_p - spread_flat_a)))
    spread_rmse = float(np.sqrt(np.mean((spread_flat_p - spread_flat_a) ** 2)))

    dir_metrics = _confusion_and_pr(
        spread_flat_a, spread_flat_p, head_prob_flat, SIGN_THRESHOLD
    )
    def _json_safe(x):
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        return x

    with open(out_dir / "direction_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: _json_safe(v) for k, v in dir_metrics.items()}, f, indent=2)

    cm_path = out_dir / "confusion_regression_vs_actual.csv"
    # 保存 2x2 表
    tp, tn, fp, fn = (
        dir_metrics["confusion_reg_tp"],
        dir_metrics["confusion_reg_tn"],
        dir_metrics["confusion_reg_fp"],
        dir_metrics["confusion_reg_fn"],
    )
    pd.DataFrame(
        [
            ["actual_neg_pred_neg", tn],
            ["actual_neg_pred_pos", fp],
            ["actual_pos_pred_neg", fn],
            ["actual_pos_pred_pos", tp],
        ],
        columns=["cell", "count"],
    ).to_csv(cm_path, index=False)

    rt_pred_24 = np.array([da_targets[d] + p24[i] for i, d in enumerate(dates)])
    rt_actual_24 = np.array([rt_targets[d] for d in dates])

    rows_spread = []
    rows_rt = []
    for i, d in enumerate(dates):
        for h in range(24):
            ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
            rows_spread.append(
                {
                    "ts": ts,
                    "actual": a24[i, h],
                    "predicted": p24[i, h],
                    "sign_logit": log24[i, h],
                    "sign_prob": float(head_prob_flat[i * 24 + h]),
                }
            )
            rows_rt.append(
                {
                    "ts": ts,
                    "actual": rt_actual_24[i, h],
                    "predicted": rt_pred_24[i, h],
                }
            )

    pd.DataFrame(rows_spread).set_index("ts").sort_index().to_csv(out_dir / "spread_result.csv")
    rt_df = pd.DataFrame(rows_rt).set_index("ts").sort_index()
    rt_df.to_csv(out_dir / "rt_result.csv")

    rt_mae = float(np.mean(np.abs(rt_df["actual"] - rt_df["predicted"])))
    rt_rmse = float(np.sqrt(np.mean((rt_df["actual"] - rt_df["predicted"]) ** 2)))
    rt_shape = quick_shape_report(rt_df["actual"].values, rt_df["predicted"].values, rt_df.index)

    run_standard_visualization(
        out_dir / "rt_result.csv",
        out_dir=out_dir / "plots",
        label="V20b-SpreadMT→RT",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    logger.info("=" * 60)
    logger.info("V20b RESULTS")
    logger.info("  Spread MAE: %.2f  RMSE: %.2f", spread_mae, spread_rmse)
    logger.info(
        "  Sign acc (regression vs actual, th=%.2f): %.4f",
        SIGN_THRESHOLD,
        dir_metrics["sign_acc_regression"],
    )
    logger.info(
        "  Sign acc (dir head vs actual): %.4f",
        dir_metrics.get("sign_acc_head", float("nan")),
    )
    logger.info("  Precision pos/neg: %.3f / %.3f", dir_metrics["precision_positive"], dir_metrics["precision_negative"])
    logger.info("  Recall pos/neg: %.3f / %.3f", dir_metrics["recall_positive"], dir_metrics["recall_negative"])
    logger.info("  RT MAE: %.2f  RMSE: %.2f", rt_mae, rt_rmse)
    for k, v in rt_shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    return {
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "rt_mae": rt_mae,
        "rt_rmse": rt_rmse,
        **dir_metrics,
        **{k: float(v) for k, v in rt_shape.items()},
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v20b()
