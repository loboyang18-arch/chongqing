"""
V9：在 V8 日级 MLP 上提高 Pearson、略降振幅权重（形状优先）；逐点 SmoothL1 主锚与 V8 一致。

默认不在损失中加峰谷时刻（避免拉低 profile_corr）；时刻进入 composite_priority 与 shape 报告。

输出：output/v9_da_day_shape/，并与 V8 summary 对比。
"""

from __future__ import annotations

import json
import logging
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
from .model_v5_profile import (
    _build_day_level_features,
    _composite_score,
    _plot_all_generic,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V9_DA_DIR = OUTPUT_DIR / "v9_da_day_shape"
V9_DA_DIR.mkdir(exist_ok=True)


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
    use_dates = [d for d in dates if d in daily.index]
    v = daily.loc[use_dates, "daily_mean_lag1"].values.astype(np.float64)
    return np.nan_to_num(v, nan=0.0)


def _regime_vec(daily: pd.DataFrame, dates: List) -> np.ndarray:
    use_dates = [d for d in dates if d in daily.index]
    r = daily.loc[use_dates, "regime_shift_flag"].fillna(0).astype(np.float64).values
    return r


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


def _build_hourly_X(
    df: pd.DataFrame,
    dates: List,
    target_col: str,
    hour_cols: List[str] | None = None,
) -> Tuple[np.ndarray, List[str], List]:
    """生成小时级输入张量：(N_day, 24, F_hour)。"""
    if hour_cols is None:
        cand = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
        sampled_days = dates[: min(len(dates), 20)]
        keep = []
        for c in cand:
            varies = False
            for d in sampled_days:
                sub = df[df.index.date == d]
                if len(sub) != 24:
                    continue
                if float(sub[c].nunique(dropna=True)) > 1:
                    varies = True
                    break
            if varies:
                keep.append(c)
        core_order = [
            "hour",
            "load_forecast",
            "renewable_fcst",
            "net_load_forecast",
            "renewable_ratio",
            "load_x_hour",
            "load_ramp_1h",
            "renewable_ramp_1h",
            "net_load_ramp_1h",
            "load_ramp_24h",
            "renewable_ramp_24h",
            "net_load_ramp_24h",
            "day_of_week",
        ]
        keep_set = set(keep)
        hour_cols = [c for c in core_order if c in keep_set]

        # 兜底：若核心特征缺失过多，则补充少量小时变化特征
        if len(hour_cols) < 8:
            extra = [c for c in sorted(keep) if c not in set(hour_cols)]
            hour_cols.extend(extra[: max(0, 12 - len(hour_cols))])

    rows = []
    kept = []
    for d in dates:
        sub = df[df.index.date == d].sort_index()
        if len(sub) != 24:
            continue
        hrs = sub.index.hour.values
        if not np.array_equal(hrs, np.arange(24)):
            sub = sub.iloc[np.argsort(hrs)]
        xh = sub[hour_cols].values.astype(np.float64)
        xh = np.nan_to_num(xh, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(xh)
        kept.append(d)
    return np.stack(rows, axis=0), hour_cols, kept


def _standardize_hourly(
    Xh_tr: np.ndarray, Xh_va: np.ndarray, Xh_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按小时特征维标准化（在 day×hour 维度上估计均值方差）。"""
    flat = Xh_tr.reshape(-1, Xh_tr.shape[-1])
    mu = flat.mean(axis=0)
    sig = flat.std(axis=0) + 1e-6
    return (
        (Xh_tr - mu.reshape(1, 1, -1)) / sig.reshape(1, 1, -1),
        (Xh_va - mu.reshape(1, 1, -1)) / sig.reshape(1, 1, -1),
        (Xh_te - mu.reshape(1, 1, -1)) / sig.reshape(1, 1, -1),
    )


class DayCurveMLP(nn.Module):
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


class DayHourFusionNet(nn.Module):
    """日级 + 小时级双分支融合，逐小时输出残差。"""

    def __init__(
        self,
        day_in_dim: int,
        hour_in_dim: int,
        day_hidden: int = 128,
        hour_hidden: int = 64,
        fuse_hidden: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.day_encoder = nn.Sequential(
            nn.Linear(day_in_dim, day_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(day_hidden, day_hidden),
            nn.ReLU(),
        )
        self.hour_encoder = nn.Sequential(
            nn.Linear(hour_in_dim, hour_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hour_hidden, hour_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(day_hidden + hour_hidden, fuse_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_hidden, 1),
        )

    def forward(self, x_day: torch.Tensor, x_hour: torch.Tensor) -> torch.Tensor:
        z_day = self.day_encoder(x_day)
        z_hour = self.hour_encoder(x_hour)
        z_day_24 = z_day.unsqueeze(1).expand(-1, z_hour.shape[1], -1)
        z = torch.cat([z_day_24, z_hour], dim=-1)
        return self.head(z).squeeze(-1)


def _pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    cov = (pred * target).mean(dim=1)
    std_p = pred.std(dim=1, unbiased=False) + 1e-5
    std_t = target.std(dim=1, unbiased=False) + 1e-5
    r = cov / (std_p * std_t)
    return torch.clamp(r, -1.0, 1.0)


def _soft_expected_peak_hour(
    price_24: torch.Tensor, temperature: float = 6.0, peak: bool = True
) -> torch.Tensor:
    """可微近似：峰/谷所在期望小时 ∈ [0,23]。"""
    B, H = price_24.shape
    hrs = torch.arange(H, device=price_24.device, dtype=price_24.dtype).unsqueeze(0).expand(B, -1)
    if peak:
        logits = price_24 * temperature
    else:
        logits = (-price_24) * temperature
    p = F.softmax(logits, dim=1)
    return (p * hrs).sum(dim=1)


def _v9_loss(
    pred_price: torch.Tensor,
    target_price: torch.Tensor,
    amp_extreme_thresh: float,
    w_amp: float = 0.26,
    w_corr: float = 0.42,
    w_time: float = 0.0,
    soft_temp: float = 6.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    在 V8 损失（逐点 + 振幅 + Pearson）上可选峰谷时刻项；默认 w_time=0（时刻仅在评估 composite_priority 中体现，
    因试验中较小 w_time 会拉低 profile_corr）。

    业务优先级：w_corr ≥ w_amp；逐点系数 1。amp_extreme_thresh 仅占位（与 meta 一致）。
    """
    _ = amp_extreme_thresh

    loss_pt = F.smooth_l1_loss(pred_price, target_price, beta=15.0)

    amp_p = pred_price.max(dim=1)[0] - pred_price.min(dim=1)[0]
    amp_t = target_price.max(dim=1)[0] - target_price.min(dim=1)[0]
    loss_amp = F.l1_loss(amp_p, amp_t)

    r = _pearson_batch(pred_price, target_price)
    loss_corr = (1.0 - r).mean()

    loss_time = torch.zeros((), device=pred_price.device, dtype=pred_price.dtype)
    if w_time > 1e-12:
        ep_p = _soft_expected_peak_hour(pred_price, soft_temp, peak=True)
        ep_t = _soft_expected_peak_hour(target_price, soft_temp, peak=True)
        ev_p = _soft_expected_peak_hour(pred_price, soft_temp, peak=False)
        ev_t = _soft_expected_peak_hour(target_price, soft_temp, peak=False)
        l_time_each = torch.abs(ep_p - ep_t) + torch.abs(ev_p - ev_t)
        loss_time = l_time_each.mean()

    total = loss_pt + w_amp * loss_amp + w_corr * loss_corr + w_time * loss_time

    dbg = {
        "L_pt": float(loss_pt.detach()),
        "L_amp": float(loss_amp.detach()),
        "L_corr": float(loss_corr.detach()),
        "L_time": float(loss_time.detach()),
    }
    return total, dbg


def _composite_priority_da(metrics: Dict) -> float:
    """选模分数（越低越好）：形状 > 振幅 > 峰谷时刻 > MAE。"""
    corr = metrics.get("profile_corr", 0)
    amp = metrics.get("amplitude_err", 100)
    ph = metrics.get("peak_hour_err", 12.0)
    vh = metrics.get("valley_hour_err", 12.0)
    mae = metrics.get("MAE", 50)
    corr_n = 1.0 - max(min(corr, 1.0), -1.0)
    amp_n = amp / 150.0
    time_n = ((ph + vh) / 2.0) / 12.0
    mae_n = mae / 50.0
    return 0.40 * corr_n + 0.30 * amp_n + 0.20 * time_n + 0.10 * mae_n


def _pred_matrix_to_test_series(
    df_test: pd.DataFrame,
    test_dates: List,
    pred_mat: np.ndarray,
    level_vec: np.ndarray,
) -> np.ndarray:
    pred_map: Dict = {}
    for i, d in enumerate(test_dates):
        lv = level_vec[i]
        for h in range(24):
            pred_map[(d, h)] = pred_mat[i, h] + lv
    out = []
    for ts in df_test.index:
        out.append(pred_map.get((ts.date(), ts.hour), np.nan))
    return np.array(out, dtype=float)


def _metrics_by_regime(
    y: np.ndarray,
    pred: np.ndarray,
    index: pd.DatetimeIndex,
    daily: pd.DataFrame,
) -> Dict[str, float]:
    """按日聚合 regime_shift，分块 MAE / profile_corr。"""
    dfm = pd.DataFrame({"y": y, "p": pred, "d": index.date})
    rows = []
    for d, g in dfm.groupby("d"):
        if len(g) != 24:
            continue
        rs = int(daily.loc[d, "regime_shift_flag"]) if d in daily.index else 0
        mae_d = float(np.mean(np.abs(g["y"].values - g["p"].values)))
        rows.append((rs, mae_d))
    if not rows:
        return {}
    reg_list = [m for r, m in rows if r == 1]
    stab_list = [m for r, m in rows if r == 0]
    reg_mae = float(np.mean(reg_list)) if reg_list else float("nan")
    stab_mae = float(np.mean(stab_list)) if stab_list else float("nan")
    return {
        "mae_regime_days": round(float(reg_mae), 3) if np.isfinite(reg_mae) else np.nan,
        "mae_stable_days": round(float(stab_mae), 3) if np.isfinite(stab_mae) else np.nan,
    }


def train_v9_da(
    epochs: int = 400,
    batch_size: int = 16,
    lr: float = 5e-4,
    patience: int = 70,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = "da"
    target_col = "target_da_clearing_price"
    logger.info("=" * 60)
    logger.info("V9 DA DAY-SHAPE (day+hour fusion, compact hourly features)")
    logger.info("=" * 60)

    df = _load_dataset(name)
    df = df.sort_index()
    daily = _build_day_level_features(df, target_col)

    all_dates = _dates_with_full_day(df)
    train_end_d = pd.Timestamp(TRAIN_END).date()

    train_dates_all = [d for d in all_dates if d <= train_end_d]
    test_dates = [d for d in all_dates if d >= pd.Timestamp(TEST_START).date()]
    if len(train_dates_all) < 15 or len(test_dates) < 3:
        raise ValueError("Not enough full days for V9.")

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

    amp_train_days = Y_tr.max(axis=1) - Y_tr.min(axis=1)
    amp_extreme_thresh = float(np.percentile(amp_train_days, 80))

    Xh_tr, hour_feat_names, kh_tr = _build_hourly_X(df, d_tr, target_col)
    Xh_va, _, kh_va = _build_hourly_X(df, d_va, target_col, hour_cols=hour_feat_names)
    Xh_te, _, kh_te = _build_hourly_X(df, d_te, target_col, hour_cols=hour_feat_names)
    if kh_tr != d_tr or kh_va != d_va or kh_te != d_te:
        raise ValueError("Date alignment failed for hourly feature tensor.")

    X_tr, X_va, X_te, _, _ = _standardize(X_tr, X_va, X_te)
    Xh_tr, Xh_va, Xh_te = _standardize_hourly(Xh_tr, Xh_va, Xh_te)

    device = torch.device("cpu")
    model = DayHourFusionNet(X_tr.shape[1], Xh_tr.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(Xh_tr).float(),
        torch.from_numpy(lv_tr).float(),
        torch.from_numpy(Y_tr).float(),
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_va).float(),
        torch.from_numpy(Xh_va).float(),
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
        for xb_day, xb_hour, lv_b, y_price in tr_loader:
            xb_day = xb_day.to(device)
            xb_hour = xb_hour.to(device)
            lv_b = lv_b.to(device)
            y_price = y_price.to(device)
            opt.zero_grad()
            rb_hat = model(xb_day, xb_hour)
            pred_p = _forward_prices(rb_hat, lv_b)
            loss, _ = _v9_loss(pred_p, y_price, amp_extreme_thresh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb_day, xb_hour, lv_b, y_price in va_loader:
                xb_day = xb_day.to(device)
                xb_hour = xb_hour.to(device)
                lv_b = lv_b.to(device)
                y_price = y_price.to(device)
                rb_hat = model(xb_day, xb_hour)
                pred_p = _forward_prices(rb_hat, lv_b)
                loss, _ = _v9_loss(pred_p, y_price, amp_extreme_thresh)
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
                "  epoch %d | train=%.4f val=%.4f best=%.4f | amp80_th=%.1f",
                ep + 1,
                float(np.mean(tr_losses)),
                va_m,
                best_val,
                amp_extreme_thresh,
            )
        if bad >= patience:
            logger.info("  early stop at epoch %d", ep + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        rb_te = model(
            torch.from_numpy(X_te).float().to(device),
            torch.from_numpy(Xh_te).float().to(device),
        ).cpu().numpy()

    test_df = df.loc[TEST_START:].copy()
    pred_v9 = _pred_matrix_to_test_series(test_df, d_te, rb_te, lv_te)

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

    y_actual = test_df[target_col].values
    results = pd.DataFrame(
        {
            "actual": y_actual,
            "pred_lgb": pred_lgb,
            "pred_v9": pred_v9,
        },
        index=test_df.index,
    )
    results.index.name = "ts"
    results.to_csv(V9_DA_DIR / "da_result.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V9_DA_DayShape", "pred_v9")]:
        m = _compute_metrics(y_actual, results[col].values)
        sr = compute_shape_report(
            y_actual,
            results[col].values,
            test_df.index,
            include_v7=True,
        )
        merged = {**m, **sr}
        score = _composite_score(merged, name)
        score_pri = _composite_priority_da(merged)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        row["composite_priority"] = round(score_pri, 4)
        rows.append(row)
        logger.info(
            "  %-18s MAE=%.2f RMSE=%.2f | corr=%.3f peak_h=%.2f valley_h=%.2f amp=%.1f | comp=%.4f pri=%.4f",
            label,
            m["MAE"],
            m["RMSE"],
            sr["profile_corr"],
            sr["peak_hour_err"],
            sr["valley_hour_err"],
            sr["amplitude_err"],
            score,
            score_pri,
        )

    br = _metrics_by_regime(y_actual, pred_v9, test_df.index, daily)
    if br:
        logger.info("  V9 by regime: %s", br)

    pd.DataFrame(rows).to_csv(V9_DA_DIR / "summary.csv", index=False)
    meta = {
        "amp_extreme_thresh": amp_extreme_thresh,
        "loss_w_amp": 0.26,
        "loss_w_corr": 0.42,
        "loss_w_time": 0.0,
        "n_feat": len(feat_names),
        "n_hour_feat": len(hour_feat_names),
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
    }
    meta.update(br)
    pd.Series(meta).to_csv(V9_DA_DIR / "v9_meta.csv", header=False)

    _plot_all_generic(results, name, "pred_v9", "V9 DA Day", V9_DA_DIR)

    return results, rows, meta


def _write_comparison_and_note(rows: List[Dict], meta: Dict) -> None:
    """V8 vs V9 对比表 + 简短实验说明。"""
    v8_path = OUTPUT_DIR / "v8_da_day_shape" / "summary.csv"
    lines = []
    if v8_path.exists():
        v8 = pd.read_csv(v8_path)
        v9 = pd.DataFrame(rows)
        cmp_rows = []
        for metric in [
            "MAE",
            "RMSE",
            "profile_corr",
            "peak_hour_err",
            "valley_hour_err",
            "amplitude_err",
            "composite_score",
            "composite_priority",
        ]:
            r8 = v8[v8["method"] == "V8_DA_DayShape"]
            r9 = v9[v9["method"] == "V9_DA_DayShape"]
            if len(r8) and len(r9) and metric in r9.columns:
                b = float(r9.iloc[0][metric])
                a = float(r8.iloc[0][metric]) if metric in r8.columns else float("nan")
                cmp_rows.append(
                    {
                        "metric": metric,
                        "v8": a,
                        "v9": b,
                        "delta_v9_minus_v8": b - a
                        if np.isfinite(a) and np.isfinite(b)
                        else np.nan,
                    }
                )
        pd.DataFrame(cmp_rows).to_csv(V9_DA_DIR / "v9_vs_v8.csv", index=False)
        lines.append("对比表已写入 v9_vs_v8.csv。")

    note = """# V9 实验说明（相对 V8）

## 输入组织（day + hour）

- 日级分支：使用 daily features（原 V8/V9 逻辑）编码当天全局状态。
- 小时级分支：使用每小时特征张量 `(N_day,24,F_hour)` 编码日内条件。
- 融合头：逐小时拼接 day/hour 隐表示后输出 24 点残差，再加 `daily_mean_lag1` 还原绝对价格。

## 训练损失

默认 `loss_pt + 0.26·L1(振幅) + 0.42·mean(1−r)`（无时刻损失，避免与 profile_corr 冲突）；峰谷时刻仅在 **composite_priority** 与 shape 报告中体现。可将 `w_time>0` 打开 soft 时刻正则。

## 如何读结果

- **composite_priority**（越低越好）：0.4×(1−corr) + 0.3×amp + 0.2×时刻 + 0.1×MAE；**composite_score** 仍为 V5 联合分数。
- `mae_regime_days` / `mae_stable_days`：按日 regime 的 MAE 分解（仅评估，不参与训练）。

## 参数

- `amp_extreme_thresh`：训练集日振幅 80% 分位，见 v9_meta.csv。
"""
    note_path = V9_DA_DIR / "experiment_note.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)
        f.write("\n## 本次 run meta\n\n")
        f.write(json.dumps(meta, indent=2, ensure_ascii=False))
        f.write("\n")
    lines.append(str(note_path))
    logger.info("  %s", " | ".join(lines))


def run_v9():
    _, rows, meta = train_v9_da()
    _write_comparison_and_note(rows, meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_v9()
