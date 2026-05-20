"""
V19 — Conv2D Multi-Task: 小时级回归 + 方向分类辅助头

借鉴内蒙 V8 Multi-Task，基于 V18 Conv2D 骨干增加方向头：
  - 回归头：预测目标价（默认与 V18 相同：da_clearing_price）
  - 方向头：预测相比前一小时的涨/跌/平（3 分类）
  - 联合损失 = L1 + λ·CrossEntropy(direction)
  - 直接对齐 direction_acc 指标

输入: (C, 12, 7) — 同 V18
训练配置: 200ep, Warmup+Cosine, AdamW
"""

import logging
import os
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
    EFFECTIVE_START, EFFECTIVE_END, TRAIN_END, VAL_END, TEST_END, TEST_START,
    HIST_COLS, FUTR_COLS, N_LAG1, N_HIST, N_FUTR,
)
from .model_v18_conv2d import (
    LAG0_COLS, LAG1_COLS, LAG2_COLS,
    C_LAG0, C_LAG1, C_LAG2, C_TOTAL,
    LOOKBACK_DAYS, SPD, SLOTS_PER_HOUR,
    TARGET_COL, DEVICE, WARMUP_EPOCHS,
    _log_device, _seed,
    _build_daily_arrays, _get_hour_slots,
    compute_norm, predict_days as _predict_days_v18,
    _plot_train_last_week,
)

_V19_CTX_BEFORE = 1
_V19_CTX_AFTER = 1
H_SLOTS = (_V19_CTX_BEFORE + 1 + _V19_CTX_AFTER) * SLOTS_PER_HOUR  # 12
from price_forecast_eval import quick_shape_report

logger = logging.getLogger(__name__)

MAX_EPOCHS = int(os.environ.get("V19_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("V19_BS", "64"))
LR = float(os.environ.get("V19_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("V19_WD", "1e-4"))
LAMBDA_DIR = float(os.environ.get("V19_LAMBDA", "0.3"))
DIR_CLASSES = 3  # down=0, flat=1, up=2

V19_DIR = OUTPUT_DIR / "v19_multitask"


def _rebuild_targets(df, valid_dates, target_col):
    """用指定的 target_col 替换 day_targets，返回新的 (valid_dates, day_targets)。"""
    day_targets: Dict = {}
    new_valid = []
    for d in valid_dates:
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        raw = df.reindex(grid)
        if target_col not in raw.columns:
            continue
        hourly_vals = raw[target_col].values[::4][:24]
        if len(hourly_vals) == 24 and np.isfinite(hourly_vals).all():
            day_targets[d] = hourly_vals.astype(np.float32)
            new_valid.append(d)
    return sorted(new_valid), day_targets


# ── Dataset ────────────────────────────────────────────────────────
class HourlyMultiTaskDataset(Dataset):
    """每小时一个样本 → (C, 12, 7) 输入 + 价格标量 + 方向标签(0/1/2)。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict,
        day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
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

            d_prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()

            for h in range(24):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    d0, d1, d2 = dates0[k], dates1[k], dates2[k]
                    s0 = _get_hour_slots(day_lag0, d0, h, _V19_CTX_BEFORE, _V19_CTX_AFTER)
                    s1 = _get_hour_slots(day_lag1, d1, h, _V19_CTX_BEFORE, _V19_CTX_AFTER)
                    s2 = _get_hour_slots(day_lag2, d2, h, _V19_CTX_BEFORE, _V19_CTX_AFTER)
                    layers.append(np.concatenate([s0, s1, s2], axis=1))

                grid = np.stack(layers, axis=-1).transpose(1, 0, 2)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
                        / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)

                tgt = np.float32((day_targets[d][h] - y_mean) / y_std)

                if h > 0:
                    diff = day_targets[d][h] - day_targets[d][h - 1]
                elif d_prev in day_targets:
                    diff = day_targets[d][0] - day_targets[d_prev][23]
                else:
                    diff = 0.0

                if diff > 0:
                    dir_label = 2
                elif diff < 0:
                    dir_label = 0
                else:
                    dir_label = 1

                self.items.append((grid, tgt, dir_label))
                self.meta.append((d, h))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        grid, tgt, dl = self.items[idx]
        return (torch.from_numpy(grid),
                torch.tensor(tgt),
                torch.tensor(dl, dtype=torch.long))


# ── Model ──────────────────────────────────────────────────────────
class Conv2dMultiTaskNet(nn.Module):
    """
    (B, C, 12, 7) → Conv2d×3 → 共享特征(320)
      ├→ 回归头 → (B,)  价格预测
      └→ 方向头 → (B,3) 涨/平/跌分类
    """

    def __init__(self, c_in=C_TOTAL):
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
        self.flatten = nn.Flatten()  # 64*1*5 = 320

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
            nn.Linear(32, DIR_CLASSES),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.flatten(x)
        price = self.reg_head(feat).squeeze(-1)
        direction = self.dir_head(feat)
        return price, direction


# ── Evaluation helpers ─────────────────────────────────────────────
def _eval_mae_hourly(model, loader, y_mean, y_std):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for batch in loader:
            grid, tgt = batch[0], batch[1]
            price, _ = model(grid.to(DEVICE))
            ps.append(price.cpu().numpy())
            ts.append(tgt.numpy())
    p = np.concatenate(ps) * y_std + y_mean
    t = np.concatenate(ts) * y_std + y_mean
    return float(np.mean(np.abs(p - t)))


def _eval_dir_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            grid, dir_label = batch[0], batch[2]
            _, dir_logits = model(grid.to(DEVICE))
            pred_cls = dir_logits.argmax(dim=1)
            correct += (pred_cls.cpu() == dir_label).sum().item()
            total += dir_label.numel()
    return correct / max(total, 1)


# ── Training ──────────────────────────────────────────────────────
def train_model(
    train_days, val_days,
    day_lag0, day_lag1, day_lag2, day_targets,
    norm_mean, norm_std, y_mean, y_std,
    epochs=None, out_dir=None,
):
    epochs = epochs or MAX_EPOCHS
    out_dir = Path(out_dir or V19_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)

    ds_kw = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    train_ds = HourlyMultiTaskDataset(sample_dates=train_days, **ds_kw)
    val_ds = HourlyMultiTaskDataset(sample_dates=val_days, **ds_kw)

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    model = Conv2dMultiTaskNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Conv2dMultiTaskNet params: %d  (C_in=%d)", n_params, C_TOTAL)

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
        ep_l1, ep_ce, ep_dir_ok, ep_dir_n, nb = 0., 0., 0, 0, 0

        for grid, tgt, dir_label in tl:
            grid = grid.to(DEVICE)
            tgt = tgt.to(DEVICE)
            dir_label = dir_label.to(DEVICE)

            opt.zero_grad()
            price, dir_logits = model(grid)

            l1 = F.l1_loss(price, tgt)
            ce = F.cross_entropy(dir_logits, dir_label)
            loss = l1 + LAMBDA_DIR * ce

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_l1 += l1.item()
            ep_ce += ce.item()
            ep_dir_ok += (dir_logits.argmax(1) == dir_label).sum().item()
            ep_dir_n += dir_label.numel()
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
            val_dir = _eval_dir_acc(model, val_l)
            batch_dir_acc = ep_dir_ok / max(ep_dir_n, 1)
            logger.info(
                "  ep%3d  L1=%.4f CE=%.3f dir=%.3f"
                " | train=%.1f val=%.1f | v_dir=%.3f lr=%.1e",
                ep, ep_l1 / max(nb, 1), ep_ce / max(nb, 1),
                batch_dir_acc,
                train_mae, val_mae, val_dir,
                opt.param_groups[0]["lr"],
            )

    ckpt = out_dir / "seed0.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Saved last-epoch checkpoint → %s", ckpt)
    return model


# ── Prediction (wraps V18 predict_days for the multi-task model) ──
def predict_days(model, dates, day_lag0, day_lag1, day_lag2, day_targets,
                 norm_mean, norm_std, y_mean, y_std):
    """Multi-task prediction: use regression head only."""
    ds = HourlyMultiTaskDataset(
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
        for batch in loader:
            grid = batch[0]
            price, _ = model(grid.to(DEVICE))
            all_preds.append(price.cpu().numpy())
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


# ── Main ───────────────────────────────────────────────────────────
def run_v19(out_dir=None, target_col: Optional[str] = None, task="da"):
    default_dir = V19_DIR if task == "da" else OUTPUT_DIR / "v19_multitask_rt"
    out_dir = Path(out_dir) if out_dir else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_col is None:
        target_col = TARGET_COL

    logger.info("=" * 60)
    logger.info("V19 Conv2D Multi-Task — regression + direction")
    logger.info("  target: %s  (task=%s)", target_col, task)
    logger.info("  backbone: Conv2d (%d, %d, %d) → 320-d shared features",
                C_TOTAL, H_SLOTS, LOOKBACK_DAYS)
    logger.info("  reg_head: 320→64→1,  dir_head: 320→32→3")
    logger.info("  λ_dir=%.2f, epochs=%d, bs=%d, lr=%.1e, wd=%.1e",
                LAMBDA_DIR, MAX_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY)
    _log_device()

    df = build_feature_matrix()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets, *_ = _build_daily_arrays(df)

    if target_col != TARGET_COL:
        logger.info("Rebuilding targets with %s ...", target_col)
        valid_dates, day_targets = _rebuild_targets(df, valid_dates, target_col)
        logger.info("  %d valid days after target rebuild", len(valid_dates))

    tr_last = TRAIN_END.date()
    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
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

    model = train_model(
        train_days=train_days,
        val_days=val_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        out_dir=out_dir,
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

    # ── 测试集预测 ──
    p24, a24, dates = predict_days(
        model, test_days,
        day_lag0, day_lag1, day_lag2, day_targets,
        norm_mean, norm_std, y_mean, y_std,
    )

    rows = []
    for i, d in enumerate(dates):
        for h in range(24):
            rows.append({
                "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                "actual": a24[i, h],
                "predicted": p24[i, h],
            })
    result = pd.DataFrame(rows).set_index("ts").sort_index()
    result_csv = out_dir / f"{task}_result.csv"
    result.to_csv(result_csv)
    logger.info("Saved: %s (%d rows, %d days)", result_csv.name, len(result), len(dates))

    # ── 方向准确率 ──
    test_ds = HourlyMultiTaskDataset(
        sample_dates=test_days,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    test_dir_acc = _eval_dir_acc(
        model, DataLoader(test_ds, min(512, max(len(test_ds), 1)), shuffle=False))
    logger.info("  Test direction accuracy (aux head): %.3f", test_dir_acc)

    viz_label = f"V19-MT-{task.upper()}"
    from price_forecast_eval.viz import run_standard_visualization
    run_standard_visualization(
        result_csv,
        out_dir=out_dir / "plots",
        label=viz_label,
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
    logger.info("V19 Multi-Task RESULTS")
    logger.info("  MAE:  %.2f", mae)
    logger.info("  RMSE: %.2f", rmse)
    logger.info("  test_dir_acc (aux): %.4f", test_dir_acc)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    return {"mae": mae, "rmse": rmse, "test_dir_acc": test_dir_acc, **shape}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v19()
