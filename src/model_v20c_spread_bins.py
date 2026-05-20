"""
V20c — 价差纯分桶分类 → 用类别代表值重构 spread → 叠加 DA 得 RT

流程：
  1. 用环境变量 V20C_BIN_EDGES 定义递增边界（逗号分隔），得到 K=len(edges)+1 个区间类（默认 -5,5 → 3 类）。
  2. 仅在训练集上统计每个类的 spread 均值作为该类代表值（用于重构）；空类则退回区间中点。
  3. Conv2D 骨干 + 多类 logits，损失为加权 CrossEntropy（类频逆比 + 可选 label_smoothing）。
  4. 预测：argmax → rep_spread[k]；RT_hat = DA_actual + rep_spread[k]。

正则与容量（环境变量，均有默认）：
  V20C_WD, V20C_DROPOUT, V20C_DROPOUT2D, V20C_LABEL_SMOOTH
  V20C_CONV1, V20C_CONV2, V20C_CONV3, V20C_FC（默认小模型 32/48/32 + fc=32）
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

_V20C_CTX_BEFORE = 1
_V20C_CTX_AFTER = 1
H_SLOTS = (_V20C_CTX_BEFORE + 1 + _V20C_CTX_AFTER) * SLOTS_PER_HOUR  # 12
from .model_v20_spread import _build_spread_targets
from price_forecast_eval import quick_shape_report
from price_forecast_eval.viz import run_standard_visualization

logger = logging.getLogger(__name__)

MAX_EPOCHS = int(os.environ.get("V20C_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("V20C_BS", "64"))
LR = float(os.environ.get("V20C_LR", "1e-3"))
# 默认略加强正则，缓解价差分桶上的过拟合（可用 V20C_WD 覆盖）
WEIGHT_DECAY = float(os.environ.get("V20C_WD", "5e-4"))
DROPOUT = float(os.environ.get("V20C_DROPOUT", "0.35"))
DROPOUT2D = float(os.environ.get("V20C_DROPOUT2D", "0.12"))
LABEL_SMOOTH = float(os.environ.get("V20C_LABEL_SMOOTH", "0.08"))
# 小模型默认通道（可 V20C_CONV1/2/3、V20C_FC 覆盖以恢复大网络）
CONV1 = int(os.environ.get("V20C_CONV1", "32"))
CONV2 = int(os.environ.get("V20C_CONV2", "48"))
CONV3 = int(os.environ.get("V20C_CONV3", "32"))
FC_HIDDEN = int(os.environ.get("V20C_FC", "32"))

# 默认 2 个边界 → 3 类：spread < lo（偏负）、[lo, hi)（近零）、≥ hi（偏正）
_DEFAULT_EDGES = "-5,5"

V20C_DIR = OUTPUT_DIR / "v20c_spread_bins"


def _parse_bin_edges(raw: str | None) -> np.ndarray:
    s = (raw or os.environ.get("V20C_BIN_EDGES", _DEFAULT_EDGES)).strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 1:
        raise ValueError("V20C_BIN_EDGES 至少需要 1 个边界")
    edges = np.array([float(x) for x in parts], dtype=np.float64)
    edges.sort()
    if not np.all(np.diff(edges) > 0):
        raise ValueError("V20C_BIN_EDGES 必须严格递增")
    return edges


def spread_to_class(spread: float, edges: np.ndarray) -> int:
    """edges 长度 m → 类 0..m。"""
    return int(np.searchsorted(edges, spread, side="right"))


def _midpoint_fallback(k: int, edges: np.ndarray, train_min: float, train_max: float) -> float:
    m = len(edges)
    if k == 0:
        left = train_min
        right = edges[0]
    elif k == m:
        left = edges[-1]
        right = train_max
    else:
        left = edges[k - 1]
        right = edges[k]
    return float(0.5 * (left + right))


def compute_class_centers(
    spreads: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    edges: np.ndarray,
) -> np.ndarray:
    """按训练样本 spread 与标签求各类均值；空类用区间中点。"""
    train_min = float(np.min(spreads))
    train_max = float(np.max(spreads))
    centers = np.zeros(num_classes, dtype=np.float64)
    for k in range(num_classes):
        mask = labels == k
        if mask.any():
            centers[k] = float(np.mean(spreads[mask]))
        else:
            centers[k] = _midpoint_fallback(k, edges, train_min, train_max)
    return centers.astype(np.float32)


def _class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


class HourlySpreadBinDataset(Dataset):
    """每小时 (C,12,7) + 整数类标签。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict,
        day_lag1: Dict,
        day_lag2: Dict,
        day_spread: Dict,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        edges: np.ndarray,
    ):
        a0 = set(day_lag0.keys())
        a1 = set(day_lag1.keys())
        a2 = set(day_lag2.keys())

        self.items: List[Tuple[np.ndarray, int]] = []
        self.meta: List[Tuple] = []

        for d in sample_dates:
            if d not in day_spread:
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

            if not (
                all(dd in a0 for dd in dates0)
                and all(dd in a1 for dd in dates1)
                and all(dd in a2 for dd in dates2)
            ):
                continue

            for h in range(24):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    d0, d1, d2 = dates0[k], dates1[k], dates2[k]
                    s0 = _get_hour_slots(day_lag0, d0, h, _V20C_CTX_BEFORE, _V20C_CTX_AFTER)
                    s1 = _get_hour_slots(day_lag1, d1, h, _V20C_CTX_BEFORE, _V20C_CTX_AFTER)
                    s2 = _get_hour_slots(day_lag2, d2, h, _V20C_CTX_BEFORE, _V20C_CTX_AFTER)
                    layers.append(np.concatenate([s0, s1, s2], axis=1))

                grid = np.stack(layers, axis=-1).transpose(1, 0, 2)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = (
                    (grid - norm_mean.reshape(C_TOTAL, 1, 1))
                    / norm_std.reshape(C_TOTAL, 1, 1)
                ).astype(np.float32)

                sp = float(day_spread[d][h])
                cls = spread_to_class(sp, edges)
                self.items.append((grid, cls))
                self.meta.append((d, h))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        grid, cls = self.items[idx]
        return torch.from_numpy(grid), torch.tensor(cls, dtype=torch.long)


class Conv2dSpreadBinNet(nn.Module):
    """Conv2D 骨干 + 多类 spread 分桶头（Dropout2d + 分类头 Dropout）。

    空间尺寸与 V18 一致：(C,12,7)→…→(c3,1,5)，故 fc_in = c3*5。
    """

    def __init__(
        self,
        num_classes: int,
        c_in: int = C_TOTAL,
        c1: int = CONV1,
        c2: int = CONV2,
        c3: int = CONV3,
        fc_hidden: int = FC_HIDDEN,
        dropout: float = DROPOUT,
        dropout2d: float = DROPOUT2D,
    ):
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3 = max(8, c1), max(8, c2), max(8, c3)
        fc_hidden = max(8, fc_hidden)
        fc_in = c3 * 1 * 5

        d2a = max(0.0, min(dropout2d * 0.55, 0.5))
        d2b = max(0.0, min(dropout2d, 0.5))
        d2c = max(0.0, min(dropout2d * 0.45, 0.4))
        dp = max(0.0, min(dropout, 0.7))

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
        self.flatten = nn.Flatten()
        self.cls_head = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.flatten(x)
        return self.cls_head(feat)


def _eval_cls_acc(model: Conv2dSpreadBinNet, loader: DataLoader) -> float:
    model.eval()
    ok, n = 0, 0
    with torch.no_grad():
        for grid, lab in loader:
            logits = model(grid.to(DEVICE))
            pred = logits.argmax(dim=1)
            ok += (pred.cpu() == lab).sum().item()
            n += lab.numel()
    return ok / max(n, 1)


def train_model(
    train_days: List,
    val_days: List,
    day_lag0: Dict,
    day_lag1: Dict,
    day_lag2: Dict,
    spread_targets: Dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    edges: np.ndarray,
    class_weights: torch.Tensor,
    epochs: int | None = None,
    out_dir: Path | None = None,
) -> Conv2dSpreadBinNet:
    epochs = epochs or MAX_EPOCHS
    out_dir = Path(out_dir or V20C_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(edges) + 1
    _seed(42)

    ds_kw = dict(
        day_lag0=day_lag0,
        day_lag1=day_lag1,
        day_lag2=day_lag2,
        day_spread=spread_targets,
        norm_mean=norm_mean,
        norm_std=norm_std,
        edges=edges,
    )
    train_ds = HourlySpreadBinDataset(sample_dates=train_days, **ds_kw)
    val_ds = HourlySpreadBinDataset(sample_dates=val_days, **ds_kw)

    logger.info(
        "Train samples: %d | Val samples: %d | num_classes=%d",
        len(train_ds),
        len(val_ds),
        num_classes,
    )

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    model = Conv2dSpreadBinNet(
        num_classes=num_classes,
        c1=CONV1,
        c2=CONV2,
        c3=CONV3,
        fc_hidden=FC_HIDDEN,
        dropout=DROPOUT,
        dropout2d=DROPOUT2D,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Conv2dSpreadBinNet params: %d | conv=%d/%d/%d fc=%d | dropout=%.2f dropout2d=%.2f "
        "label_smooth=%.3f wd=%.2e",
        n_params,
        CONV1,
        CONV2,
        CONV3,
        FC_HIDDEN,
        DROPOUT,
        DROPOUT2D,
        LABEL_SMOOTH,
        WEIGHT_DECAY,
    )

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
        ep_loss, nb = 0.0, 0
        for grid, lab in tl:
            grid, lab = grid.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            logits = model(grid)
            loss = F.cross_entropy(
                logits,
                lab,
                weight=class_weights,
                label_smoothing=min(LABEL_SMOOTH, 0.33),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        if ep % 10 == 0 or ep == epochs - 1:
            tr_acc = _eval_cls_acc(model, DataLoader(train_ds, min(512, len(train_ds)), shuffle=False))
            va_acc = _eval_cls_acc(model, val_l)
            logger.info(
                "  ep%3d  loss=%.4f  train_cls=%.3f  val_cls=%.3f  lr=%.1e",
                ep,
                ep_loss / max(nb, 1),
                tr_acc,
                va_acc,
                opt.param_groups[0]["lr"],
            )

    ckpt = out_dir / "seed0.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Saved last-epoch checkpoint → %s", ckpt)
    return model


def predict_bins(
    model: Conv2dSpreadBinNet,
    dates: List,
    day_lag0: Dict,
    day_lag1: Dict,
    day_lag2: Dict,
    spread_targets: Dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray]:
    """pred_class (N,24)、actual spread (N,24)、dates、max_prob (N,24)。"""
    ds = HourlySpreadBinDataset(
        sample_dates=dates,
        day_lag0=day_lag0,
        day_lag1=day_lag1,
        day_lag2=day_lag2,
        day_spread=spread_targets,
        norm_mean=norm_mean,
        norm_std=norm_std,
        edges=edges,
    )
    if len(ds) == 0:
        return (
            np.zeros((0, 24)),
            np.zeros((0, 24)),
            [],
            np.zeros((0, 24)),
        )

    loader = DataLoader(ds, min(512, len(ds)), shuffle=False)
    model.eval()
    all_cls, all_prob = [], []
    with torch.no_grad():
        for grid, _ in loader:
            logits = model(grid.to(DEVICE))
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            all_cls.append(pred.cpu().numpy())
            all_prob.append(conf.cpu().numpy())

    cls_flat = np.concatenate(all_cls)
    prob_flat = np.concatenate(all_prob)

    day_cls: Dict = {}
    day_pr: Dict = {}
    for i, (d, h) in enumerate(ds.meta):
        if d not in day_cls:
            day_cls[d] = np.full(24, -1, dtype=np.int32)
            day_pr[d] = np.full(24, np.nan, dtype=np.float32)
        day_cls[d][h] = int(cls_flat[i])
        day_pr[d][h] = float(prob_flat[i])

    valid_dates = sorted(
        d
        for d in day_cls
        if d in spread_targets
        and (day_cls[d] >= 0).all()
        and not np.isnan(spread_targets[d]).any()
    )
    k_cls = np.array([day_cls[d] for d in valid_dates])
    k_pr = np.array([day_pr[d] for d in valid_dates])
    a24 = np.array([spread_targets[d] for d in valid_dates])
    return k_cls, a24, valid_dates, k_pr


def run_v20c(out_dir: Path | None = None) -> Dict[str, Any]:
    out_dir = Path(out_dir) if out_dir else V20C_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = _parse_bin_edges(os.environ.get("V20C_BIN_EDGES"))
    num_classes = len(edges) + 1

    logger.info("=" * 60)
    logger.info("V20c Spread bin classification → reconstruct RT")
    logger.info("  bin edges: %s  →  K=%d classes", edges.tolist(), num_classes)
    logger.info(
        "  epochs=%d bs=%d lr=%.1e wd=%.1e conv=%d/%d/%d fc=%d dropout=%.2f dropout2d=%.2f label_smooth=%.3f",
        MAX_EPOCHS,
        BATCH_SIZE,
        LR,
        WEIGHT_DECAY,
        CONV1,
        CONV2,
        CONV3,
        FC_HIDDEN,
        DROPOUT,
        DROPOUT2D,
        LABEL_SMOOTH,
    )
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

    # 训练集上打标签并求类中心
    train_spreads_list: List[float] = []
    train_labels_list: List[int] = []
    for d in train_days:
        if d not in spread_targets:
            continue
        for h in range(24):
            sp = float(spread_targets[d][h])
            train_spreads_list.append(sp)
            train_labels_list.append(spread_to_class(sp, edges))
    train_spreads = np.array(train_spreads_list, dtype=np.float64)
    train_labels = np.array(train_labels_list, dtype=np.int64)
    class_centers = compute_class_centers(train_spreads, train_labels, num_classes, edges)
    class_weights = _class_weights_from_labels(train_labels, num_classes)

    bin_cfg = {
        "edges": edges.tolist(),
        "num_classes": num_classes,
        "class_centers": class_centers.tolist(),
        "class_counts": np.bincount(train_labels, minlength=num_classes).tolist(),
    }
    with open(out_dir / "bin_config.json", "w", encoding="utf-8") as f:
        json.dump(bin_cfg, f, indent=2)
    logger.info("Class centers (train mean per bin): %s", class_centers.round(2).tolist())

    model = train_model(
        train_days=train_days,
        val_days=val_days,
        day_lag0=day_lag0,
        day_lag1=day_lag1,
        day_lag2=day_lag2,
        spread_targets=spread_targets,
        norm_mean=norm_mean,
        norm_std=norm_std,
        edges=edges,
        class_weights=class_weights,
        out_dir=out_dir,
    )
    logger.info("Using last-epoch weights (ep %d)", MAX_EPOCHS - 1)

    pred_cls, a24_spread, dates, pred_conf = predict_bins(
        model,
        test_days,
        day_lag0,
        day_lag1,
        day_lag2,
        spread_targets,
        norm_mean,
        norm_std,
        edges,
    )
    p24_spread = class_centers[pred_cls].astype(np.float32)

    spread_flat_a = a24_spread.ravel()
    spread_flat_p = p24_spread.ravel()
    spread_mae = float(np.mean(np.abs(spread_flat_p - spread_flat_a)))
    spread_rmse = float(np.sqrt(np.mean((spread_flat_p - spread_flat_a) ** 2)))

    vf = np.vectorize(lambda s: spread_to_class(float(s), edges))
    act_cls_2d = vf(a24_spread).astype(np.int64)
    cls_acc_strict = float(np.mean(pred_cls.ravel() == act_cls_2d.ravel()))

    sign_ok = float(((spread_flat_p > 0) == (spread_flat_a > 0)).mean())

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, p in zip(act_cls_2d.ravel(), pred_cls.ravel()):
        cm[int(a), int(p)] += 1
    pd.DataFrame(cm).to_csv(out_dir / "confusion_class.csv")

    metrics = {
        "class_accuracy": cls_acc_strict,
        "sign_accuracy_reconstructed": float(sign_ok),
        "spread_mae_recon": spread_mae,
        "spread_rmse_recon": spread_rmse,
    }
    with open(out_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rt_pred_24 = np.array([da_targets[d] + p24_spread[i] for i, d in enumerate(dates)])
    rt_actual_24 = np.array([rt_targets[d] for d in dates])

    rows_spread = []
    rows_rt = []
    for i, d in enumerate(dates):
        for h in range(24):
            ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
            ac = int(act_cls_2d[i, h])
            pc = int(pred_cls[i, h])
            rows_spread.append(
                {
                    "ts": ts,
                    "actual_spread": a24_spread[i, h],
                    "pred_class": pc,
                    "actual_class": ac,
                    "pred_spread_from_class": p24_spread[i, h],
                    "class_center": float(class_centers[pc]),
                    "max_prob": pred_conf[i, h],
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
        label="V20c-Bins→RT",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    logger.info("=" * 60)
    logger.info("V20c RESULTS")
    logger.info("  Class accuracy (exact bin): %.4f", cls_acc_strict)
    logger.info("  Sign acc (recon spread vs actual): %.4f", sign_ok)
    logger.info("  Recon spread MAE: %.2f  RMSE: %.2f", spread_mae, spread_rmse)
    logger.info("  RT MAE: %.2f  RMSE: %.2f", rt_mae, rt_rmse)
    for k, v in rt_shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    return {
        **metrics,
        "rt_mae": rt_mae,
        "rt_rmse": rt_rmse,
        **{k: float(v) for k, v in rt_shape.items()},
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v20c()
