"""
V8：日级曲线模型 — PyTorch MLP 一次输出 24 维日前价；
训练损失 = SmoothL1(逐点) + w_amp·|Δ日振幅| + w_corr·(1 - 日内Pearson)。

与逐小时 LGB 不同：每个完整自然日为一个样本，shape 项显式参与反传。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OUTPUT_DIR, PARAMS_DIR
from .model_baseline import (
    EARLY_STOPPING_ROUNDS,
    NUM_BOOST_ROUND,
    TEST_START,
    TRAIN_END,
    _compute_metrics,
    _load_dataset,
)
from .model_v5_profile import _build_day_level_features, _composite_score, _plot_all_generic
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V8_DA_DIR = OUTPUT_DIR / "v8_da_day_shape"
V8_DA_DIR.mkdir(exist_ok=True)


def _load_tuned_params(name: str) -> Dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


def _dates_with_full_day(df: pd.DataFrame) -> List:
    dates = sorted(set(df.index.date))
    out = []
    for d in dates:
        if (df.index.date == d).sum() == 24:
            out.append(d)
    return out


def _hourly_matrix_for_dates(
    df: pd.DataFrame, target_col: str, dates: List,
) -> Tuple[np.ndarray, List]:
    """每个日期一行 24 维，按 hour=0..23 排序。"""
    rows = []
    kept = []
    for d in dates:
        sub = df[df.index.date == d].sort_index()
        if len(sub) != 24:
            continue
        hrs = sub.index.hour.values
        if not np.array_equal(hrs, np.arange(24)):
            sub = sub.iloc[np.argsort(hrs)]
        y = sub[target_col].values.astype(np.float64)
        if np.any(~np.isfinite(y)):
            continue
        rows.append(y)
        kept.append(d)
    return np.stack(rows, axis=0), kept


def _build_daily_X(
    daily: pd.DataFrame,
    dates: List,
    exclude: Tuple[str, ...] = ("target_daily_mean",),
) -> Tuple[np.ndarray, List[str]]:
    use_dates = [d for d in dates if d in daily.index]
    sub = daily.loc[use_dates]
    feat_cols = [
        c
        for c in sub.columns
        if c not in exclude
        and sub[c].dtype != object
        and np.issubdtype(sub[c].dtype, np.number)
    ]
    X = sub[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_cols


def _daily_level_lag1(daily: pd.DataFrame, dates: List) -> np.ndarray:
    """日前一日均价水平，作当日 24h 基准（无同日目标泄漏）。"""
    use_dates = [d for d in dates if d in daily.index]
    v = daily.loc[use_dates, "daily_mean_lag1"].values.astype(np.float64)
    return np.nan_to_num(v, nan=0.0)


def _standardize(
    X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-6
    return (
        (X_tr - mu) / sig,
        (X_va - mu) / sig,
        (X_te - mu) / sig,
        mu,
        sig,
    )


class DayCurveMLP(nn.Module):
    """日级特征 -> 24 维相对「lag1 日均价」的偏移；推理时再加回 level。"""

    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 24),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    cov = (pred * target).mean(dim=1)
    std_p = pred.std(dim=1, unbiased=False) + 1e-5
    std_t = target.std(dim=1, unbiased=False) + 1e-5
    r = cov / (std_p * std_t)
    return torch.clamp(r, -1.0, 1.0)


def _v8_loss(
    pred_price: torch.Tensor,
    target_price: torch.Tensor,
    w_amp: float = 0.3,
    w_corr: float = 0.35,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """pred/target 均为还原后的绝对电价 (B,24)。"""
    loss_pt = F.smooth_l1_loss(pred_price, target_price, beta=15.0)
    amp_p = pred_price.max(dim=1)[0] - pred_price.min(dim=1)[0]
    amp_t = target_price.max(dim=1)[0] - target_price.min(dim=1)[0]
    loss_amp = F.l1_loss(amp_p, amp_t)
    r = _pearson_batch(pred_price, target_price)
    loss_corr = (1.0 - r).mean()
    total = loss_pt + w_amp * loss_amp + w_corr * loss_corr
    dbg = {
        "L_pt": float(loss_pt.detach()),
        "L_amp": float(loss_amp.detach()),
        "L_corr": float(loss_corr.detach()),
    }
    return total, dbg


def _pred_matrix_to_test_series(
    df_test: pd.DataFrame,
    test_dates: List,
    pred_mat: np.ndarray,
    level_vec: np.ndarray,
) -> np.ndarray:
    """pred_mat 为偏移，加 level 后得到绝对价。"""
    pred_map: Dict = {}
    for i, d in enumerate(test_dates):
        lv = level_vec[i]
        for h in range(24):
            pred_map[(d, h)] = pred_mat[i, h] + lv
    out = []
    for ts in df_test.index:
        out.append(pred_map.get((ts.date(), ts.hour), np.nan))
    return np.array(out, dtype=float)


def train_v8_da(
    epochs: int = 400,
    batch_size: int = 16,
    lr: float = 5e-4,
    w_amp: float = 0.3,
    w_corr: float = 0.35,
    patience: int = 70,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = "da"
    target_col = "target_da_clearing_price"
    logger.info("=" * 60)
    logger.info("V8 DA DAY-SHAPE (PyTorch): %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    df = df.sort_index()
    daily = _build_day_level_features(df, target_col)

    all_dates = _dates_with_full_day(df)
    train_end_d = pd.Timestamp(TRAIN_END).date()
    test_start_d = pd.Timestamp(TEST_START).date()

    train_dates_all = [d for d in all_dates if d <= train_end_d]
    test_dates = [d for d in all_dates if d >= test_start_d]
    if len(train_dates_all) < 15 or len(test_dates) < 3:
        raise ValueError("Not enough full days for V8 train/test split.")

    n_val = max(int(len(train_dates_all) * 0.18), 5)
    train_dates = train_dates_all[:-n_val]
    val_dates = train_dates_all[-n_val:]

    Y_tr, d_tr = _hourly_matrix_for_dates(df, target_col, train_dates)
    Y_va, d_va = _hourly_matrix_for_dates(df, target_col, val_dates)
    Y_te, d_te = _hourly_matrix_for_dates(df, target_col, test_dates)

    X_tr, feat_names = _build_daily_X(daily, d_tr)
    X_va, _ = _build_daily_X(daily, d_va)
    X_te, _ = _build_daily_X(daily, d_te)

    lv_tr = _daily_level_lag1(daily, d_tr)
    lv_va = _daily_level_lag1(daily, d_va)
    lv_te = _daily_level_lag1(daily, d_te)

    assert X_tr.shape[0] == Y_tr.shape[0]
    X_tr, X_va, X_te, _, _ = _standardize(X_tr, X_va, X_te)

    device = torch.device("cpu")
    model = DayCurveMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    R_tr = Y_tr - lv_tr[:, np.newaxis]
    R_va = Y_va - lv_va[:, np.newaxis]
    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(R_tr).float(),
        torch.from_numpy(lv_tr).float(),
        torch.from_numpy(Y_tr).float(),
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_va).float(),
        torch.from_numpy(R_va).float(),
        torch.from_numpy(lv_va).float(),
        torch.from_numpy(Y_va).float(),
    )
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    def _forward_prices(rb_hat: torch.Tensor, lv_b: torch.Tensor) -> torch.Tensor:
        return rb_hat + lv_b.unsqueeze(1)

    best_state = None
    best_val = float("inf")
    bad = 0

    for ep in range(epochs):
        model.train()
        tr_losses = []
        for xb, _rb_tgt, lv_b, y_price in tr_loader:
            xb = xb.to(device)
            lv_b = lv_b.to(device)
            y_price = y_price.to(device)
            opt.zero_grad()
            rb_hat = model(xb)
            pred_p = _forward_prices(rb_hat, lv_b)
            loss, _ = _v8_loss(pred_p, y_price, w_amp=w_amp, w_corr=w_corr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, _rb_tgt, lv_b, y_price in va_loader:
                xb = xb.to(device)
                lv_b = lv_b.to(device)
                y_price = y_price.to(device)
                rb_hat = model(xb)
                pred_p = _forward_prices(rb_hat, lv_b)
                loss, _ = _v8_loss(pred_p, y_price, w_amp=w_amp, w_corr=w_corr)
                va_losses.append(loss.item())
        va_m = float(np.mean(va_losses))
        sched.step(va_m)

        if va_m < best_val - 1e-5:
            best_val = va_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if (ep + 1) % 50 == 0:
            logger.info(
                "  epoch %d | train=%.4f val=%.4f best_val=%.4f",
                ep + 1,
                float(np.mean(tr_losses)),
                va_m,
                best_val,
            )
        if bad >= patience:
            logger.info("  early stop at epoch %d", ep + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        rb_te = model(torch.from_numpy(X_te).float().to(device)).cpu().numpy()

    test_df = df.loc[TEST_START:].copy()
    pred_v8 = _pred_matrix_to_test_series(test_df, d_te, rb_te, lv_te)

    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]
    train_df = df.loc[:TRAIN_END]
    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    pred_lgb = lgb_model.predict(test_df[feature_cols])

    results = pd.DataFrame(
        {
            "actual": test_df[target_col].values,
            "pred_lgb": pred_lgb,
            "pred_v8": pred_v8,
        },
        index=test_df.index,
    )
    results.index.name = "ts"
    results.to_csv(V8_DA_DIR / "da_result.csv")

    rows = []
    y_actual = test_df[target_col].values
    for label, col in [("LGB_Huber", "pred_lgb"), ("V8_DA_DayShape", "pred_v8")]:
        m = _compute_metrics(y_actual, results[col].values)
        sr = compute_shape_report(
            y_actual,
            results[col].values,
            test_df.index,
            include_v7=True,
        )
        score = _composite_score({**m, **sr}, name)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)
        logger.info(
            "  %-18s MAE=%.2f RMSE=%.2f | corr=%.3f amp_err=%.1f | score=%.4f",
            label,
            m["MAE"],
            m["RMSE"],
            sr["profile_corr"],
            sr["amplitude_err"],
            score,
        )

    pd.DataFrame(rows).to_csv(V8_DA_DIR / "summary.csv", index=False)
    meta = {
        "w_amp": w_amp,
        "w_corr": w_corr,
        "n_feat": len(feat_names),
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
    }
    pd.Series(meta).to_csv(V8_DA_DIR / "v8_meta.csv", header=False)

    _plot_all_generic(results, name, "pred_v8", "V8 DA Day", V8_DA_DIR)
    return results, rows


def run_v8():
    train_v8_da()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_v8()
