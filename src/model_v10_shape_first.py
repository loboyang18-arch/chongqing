"""
V10（第一性原理版）：

1) 先学形状：预测去均值、单位尺度的日内曲线 shape(24)
2) 再学尺度：单独预测当天残差尺度 std_day
3) 最后还原价格：pred = level_lag1 + std_pred * shape_pred

输出目录：output/v10_da_shape_first/
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
from .model_v5_profile import _build_day_level_features, _composite_score, _plot_all_generic
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V10_DA_DIR = OUTPUT_DIR / "v10_da_shape_first"
V10_DA_DIR.mkdir(exist_ok=True)


def _load_tuned_params(name: str) -> Dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


def _dates_with_full_day(df: pd.DataFrame) -> List:
    dates = sorted(set(df.index.date))
    return [d for d in dates if (df.index.date == d).sum() == 24]


def _hourly_matrix_for_dates(df: pd.DataFrame, target_col: str, dates: List) -> Tuple[np.ndarray, List]:
    rows, kept = [], []
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
        if c not in exclude and sub[c].dtype != object and np.issubdtype(sub[c].dtype, np.number)
    ]
    X = sub[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_cols


def _daily_level_lag1(daily: pd.DataFrame, dates: List) -> np.ndarray:
    use_dates = [d for d in dates if d in daily.index]
    v = daily.loc[use_dates, "daily_mean_lag1"].values.astype(np.float64)
    return np.nan_to_num(v, nan=0.0)


def _standardize(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-6
    return (X_tr - mu) / sig, (X_va - mu) / sig, (X_te - mu) / sig


def _build_hourly_core_X(
    df: pd.DataFrame,
    dates: List,
    target_col: str,
    hour_cols: List[str] | None = None,
) -> Tuple[np.ndarray, List[str], List]:
    # 第一性原理：仅保留少量“决定形状”的核心小时特征
    if hour_cols is None:
        preferred = [
            "net_load_forecast",
            "load_forecast",
            "renewable_fcst",
            "renewable_ratio",
            "load_ramp_1h",
            "renewable_ramp_1h",
            "net_load_ramp_1h",
            "day_of_week",
        ]
        hour_cols = [c for c in preferred if c in df.columns and c != target_col]

    rows, kept = [], []
    for d in dates:
        sub = df[df.index.date == d].sort_index()
        if len(sub) != 24:
            continue
        hrs = sub.index.hour.values
        if not np.array_equal(hrs, np.arange(24)):
            sub = sub.iloc[np.argsort(hrs)]
            hrs = sub.index.hour.values

        base = sub[hour_cols].values.astype(np.float64) if hour_cols else np.zeros((24, 0))
        hsin = np.sin(2 * np.pi * hrs / 24.0).reshape(-1, 1)
        hcos = np.cos(2 * np.pi * hrs / 24.0).reshape(-1, 1)
        xh = np.concatenate([base, hsin, hcos], axis=1)
        xh = np.nan_to_num(xh, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(xh)
        kept.append(d)
    final_cols = hour_cols + ["hour_sin", "hour_cos"]
    return np.stack(rows, axis=0), final_cols, kept


def _standardize_hourly(Xh_tr: np.ndarray, Xh_va: np.ndarray, Xh_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = Xh_tr.reshape(-1, Xh_tr.shape[-1])
    mu = flat.mean(axis=0)
    sig = flat.std(axis=0) + 1e-6
    return (
        (Xh_tr - mu.reshape(1, 1, -1)) / sig.reshape(1, 1, -1),
        (Xh_va - mu.reshape(1, 1, -1)) / sig.reshape(1, 1, -1),
        (Xh_te - mu.reshape(1, 1, -1)) / sig.reshape(1, 1, -1),
    )


def _pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    cov = (pred * target).mean(dim=1)
    std_p = pred.std(dim=1, unbiased=False) + 1e-5
    std_t = target.std(dim=1, unbiased=False) + 1e-5
    r = cov / (std_p * std_t)
    return torch.clamp(r, -1.0, 1.0)


class ShapeFirstNet(nn.Module):
    def __init__(self, day_in_dim: int, hour_in_dim: int, day_hidden: int = 128, hour_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.day_enc = nn.Sequential(
            nn.Linear(day_in_dim, day_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(day_hidden, day_hidden),
            nn.ReLU(),
        )
        self.hour_enc = nn.Sequential(
            nn.Linear(hour_in_dim, hour_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hour_hidden, hour_hidden),
            nn.ReLU(),
        )
        self.shape_head = nn.Sequential(
            nn.Linear(day_hidden + hour_hidden, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(day_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_day: torch.Tensor, x_hour: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_day = self.day_enc(x_day)  # (B,D)
        z_hour = self.hour_enc(x_hour)  # (B,24,H)
        z_day_24 = z_day.unsqueeze(1).expand(-1, z_hour.shape[1], -1)
        z = torch.cat([z_day_24, z_hour], dim=-1)
        shape_raw = self.shape_head(z).squeeze(-1)  # (B,24)
        shape = shape_raw - shape_raw.mean(dim=1, keepdim=True)
        log_std = self.log_std_head(z_day).squeeze(-1)  # (B,)
        std_pred = F.softplus(log_std) + 1e-3
        return shape, std_pred


def _shape_first_loss(shape_pred: torch.Tensor, shape_true: torch.Tensor, std_pred: torch.Tensor, std_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_shape_pt = F.smooth_l1_loss(shape_pred, shape_true, beta=0.5)
    r_each = _pearson_batch(shape_pred, shape_true)
    l_shape_corr = (1.0 - r_each).mean()
    # 反相惩罚：仅在 r<0 时生效，推动同相。
    l_anti_phase = F.relu(-r_each).mean()
    l_std = F.smooth_l1_loss(torch.log(std_pred + 1e-6), torch.log(std_true + 1e-6), beta=0.2)
    total = 0.70 * l_shape_corr + 0.12 * l_shape_pt + 0.10 * l_std + 0.25 * l_anti_phase
    return total, {
        "L_shape_corr": float(l_shape_corr.detach()),
        "L_shape_pt": float(l_shape_pt.detach()),
        "L_std": float(l_std.detach()),
        "L_anti_phase": float(l_anti_phase.detach()),
    }


def _daily_shape_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    dfm = pd.DataFrame({"y": y_true, "p": y_pred}, index=index)
    rows = []
    for d, g in dfm.groupby(dfm.index.date):
        if len(g) != 24:
            continue
        y = g["y"].values.astype(float)
        p = g["p"].values.astype(float)
        r = np.corrcoef(y, p)[0, 1] if np.std(y) > 1e-8 and np.std(p) > 1e-8 else np.nan
        peak_true = int(np.argmax(y))
        peak_pred = int(np.argmax(p))
        valley_true = int(np.argmin(y))
        valley_pred = int(np.argmin(p))
        rows.append(
            {
                "date": str(d),
                "corr_day": float(r) if np.isfinite(r) else np.nan,
                "neg_corr_flag": int(np.isfinite(r) and r < 0.0),
                "peak_true_hour": peak_true,
                "peak_pred_hour": peak_pred,
                "peak_hour_diff": peak_pred - peak_true,
                "valley_true_hour": valley_true,
                "valley_pred_hour": valley_pred,
                "valley_hour_diff": valley_pred - valley_true,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["neg_corr_flag"] = out["neg_corr_flag"].astype(int)
    return out


def _pred_matrix_to_test_series(df_test: pd.DataFrame, test_dates: List, pred_mat: np.ndarray) -> np.ndarray:
    pred_map: Dict = {}
    for i, d in enumerate(test_dates):
        for h in range(24):
            pred_map[(d, h)] = pred_mat[i, h]
    out = [pred_map.get((ts.date(), ts.hour), np.nan) for ts in df_test.index]
    return np.array(out, dtype=float)


def _metrics_by_regime(y: np.ndarray, pred: np.ndarray, index: pd.DatetimeIndex, daily: pd.DataFrame) -> Dict[str, float]:
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


def train_v10_da(epochs: int = 420, batch_size: int = 16, lr: float = 3e-4, patience: int = 60, seed: int = 42) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = "da"
    target_col = "target_da_clearing_price"
    logger.info("=" * 60)
    logger.info("V10 DA SHAPE-FIRST (shape->scale->level)")
    logger.info("=" * 60)

    df = _load_dataset(name).sort_index()
    daily = _build_day_level_features(df, target_col)

    all_dates = _dates_with_full_day(df)
    train_end_d = pd.Timestamp(TRAIN_END).date()
    train_dates_all = [d for d in all_dates if d <= train_end_d]
    test_dates = [d for d in all_dates if d >= pd.Timestamp(TEST_START).date()]
    if len(train_dates_all) < 15 or len(test_dates) < 3:
        raise ValueError("Not enough full days for V10.")

    n_val = max(int(len(train_dates_all) * 0.18), 5)
    train_dates, val_dates = train_dates_all[:-n_val], train_dates_all[-n_val:]

    Y_tr, d_tr = _hourly_matrix_for_dates(df, target_col, train_dates)
    Y_va, d_va = _hourly_matrix_for_dates(df, target_col, val_dates)
    Y_te, d_te = _hourly_matrix_for_dates(df, target_col, test_dates)

    Xd_tr, day_feat_names = _build_daily_X(daily, d_tr)
    Xd_va, _ = _build_daily_X(daily, d_va)
    Xd_te, _ = _build_daily_X(daily, d_te)

    Xh_tr, hour_feat_names, kh_tr = _build_hourly_core_X(df, d_tr, target_col)
    Xh_va, _, kh_va = _build_hourly_core_X(df, d_va, target_col, hour_cols=hour_feat_names[:-2])
    Xh_te, _, kh_te = _build_hourly_core_X(df, d_te, target_col, hour_cols=hour_feat_names[:-2])
    if kh_tr != d_tr or kh_va != d_va or kh_te != d_te:
        raise ValueError("Date alignment failed for hourly tensor.")

    lv_tr = _daily_level_lag1(daily, d_tr)
    lv_va = _daily_level_lag1(daily, d_va)
    lv_te = _daily_level_lag1(daily, d_te)

    R_tr = Y_tr - lv_tr[:, np.newaxis]
    R_va = Y_va - lv_va[:, np.newaxis]
    std_tr = np.std(R_tr, axis=1) + 1e-3
    std_va = np.std(R_va, axis=1) + 1e-3
    shape_tr = R_tr / std_tr[:, np.newaxis]
    shape_va = R_va / std_va[:, np.newaxis]
    shape_tr = shape_tr - shape_tr.mean(axis=1, keepdims=True)
    shape_va = shape_va - shape_va.mean(axis=1, keepdims=True)

    Xd_tr, Xd_va, Xd_te = _standardize(Xd_tr, Xd_va, Xd_te)
    Xh_tr, Xh_va, Xh_te = _standardize_hourly(Xh_tr, Xh_va, Xh_te)

    device = torch.device("cpu")
    model = ShapeFirstNet(Xd_tr.shape[1], Xh_tr.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15, min_lr=1e-5)

    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xd_tr).float(),
        torch.from_numpy(Xh_tr).float(),
        torch.from_numpy(shape_tr).float(),
        torch.from_numpy(std_tr).float(),
        torch.from_numpy(lv_tr).float(),
        torch.from_numpy(Y_tr).float(),
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xd_va).float(),
        torch.from_numpy(Xh_va).float(),
        torch.from_numpy(shape_va).float(),
        torch.from_numpy(std_va).float(),
        torch.from_numpy(lv_va).float(),
        torch.from_numpy(Y_va).float(),
    )
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    bad = 0
    for ep in range(epochs):
        model.train()
        tr_losses = []
        for xb_day, xb_hour, sb_true, std_true, _lv_b, _y_b in tr_loader:
            xb_day = xb_day.to(device)
            xb_hour = xb_hour.to(device)
            sb_true = sb_true.to(device)
            std_true = std_true.to(device)
            opt.zero_grad()
            sb_pred, std_pred = model(xb_day, xb_hour)
            loss, _ = _shape_first_loss(sb_pred, sb_true, std_pred, std_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb_day, xb_hour, sb_true, std_true, _lv_b, _y_b in va_loader:
                xb_day = xb_day.to(device)
                xb_hour = xb_hour.to(device)
                sb_true = sb_true.to(device)
                std_true = std_true.to(device)
                sb_pred, std_pred = model(xb_day, xb_hour)
                loss, _ = _shape_first_loss(sb_pred, sb_true, std_pred, std_true)
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
            logger.info("  epoch %d | train=%.4f val=%.4f best=%.4f", ep + 1, float(np.mean(tr_losses)), va_m, best_val)
        if bad >= patience:
            logger.info("  early stop at epoch %d", ep + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        sb_te, std_te_pred = model(
            torch.from_numpy(Xd_te).float().to(device),
            torch.from_numpy(Xh_te).float().to(device),
        )
        sb_te = sb_te.cpu().numpy()
        std_te_pred = std_te_pred.cpu().numpy()

    pred_mat = lv_te[:, np.newaxis] + std_te_pred[:, np.newaxis] * sb_te
    test_df = df.loc[TEST_START:].copy()
    pred_v10 = _pred_matrix_to_test_series(test_df, d_te, pred_mat)

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
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    pred_lgb = lgb_model.predict(test_df[feature_cols])

    y_actual = test_df[target_col].values
    results = pd.DataFrame({"actual": y_actual, "pred_lgb": pred_lgb, "pred_v10": pred_v10}, index=test_df.index)
    results.index.name = "ts"
    results.to_csv(V10_DA_DIR / "da_result.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V10_DA_ShapeFirst", "pred_v10")]:
        m = _compute_metrics(y_actual, results[col].values)
        sr = compute_shape_report(y_actual, results[col].values, test_df.index, include_v7=True)
        score = _composite_score({**m, **sr}, name)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)
        logger.info(
            "  %-18s MAE=%.2f RMSE=%.2f | corr=%.3f peak_h=%.2f valley_h=%.2f amp=%.1f | score=%.4f",
            label,
            m["MAE"],
            m["RMSE"],
            sr["profile_corr"],
            sr["peak_hour_err"],
            sr["valley_hour_err"],
            sr["amplitude_err"],
            score,
        )

    br = _metrics_by_regime(y_actual, pred_v10, test_df.index, daily)
    if br:
        logger.info("  V10 by regime: %s", br)

    diag = _daily_shape_diagnostics(y_actual, pred_v10, test_df.index)
    diag.to_csv(V10_DA_DIR / "daily_shape_diagnostics.csv", index=False)
    neg_ratio = float(diag["neg_corr_flag"].mean()) if len(diag) else float("nan")
    logger.info("  V10 neg_corr_day_ratio=%.3f", neg_ratio if np.isfinite(neg_ratio) else float("nan"))

    pd.DataFrame(rows).to_csv(V10_DA_DIR / "summary.csv", index=False)
    meta = {
        "n_day_feat": len(day_feat_names),
        "n_hour_feat": len(hour_feat_names),
        "hour_feat_names": "|".join(hour_feat_names),
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
        "neg_corr_day_ratio": round(neg_ratio, 4) if np.isfinite(neg_ratio) else np.nan,
    }
    meta.update(br)
    pd.Series(meta).to_csv(V10_DA_DIR / "v10_meta.csv", header=False)
    _plot_all_generic(results, name, "pred_v10", "V10 DA ShapeFirst", V10_DA_DIR)
    return results, rows, meta


def _write_comparison(rows: List[Dict], meta: Dict) -> None:
    v10 = pd.DataFrame(rows)
    for tag, base_path, base_method in [
        ("v8", OUTPUT_DIR / "v8_da_day_shape" / "summary.csv", "V8_DA_DayShape"),
        ("v9", OUTPUT_DIR / "v9_da_day_shape" / "summary.csv", "V9_DA_DayShape"),
    ]:
        if not base_path.exists():
            continue
        base = pd.read_csv(base_path)
        cmp_rows = []
        for metric in ["MAE", "RMSE", "profile_corr", "peak_hour_err", "valley_hour_err", "amplitude_err", "composite_score"]:
            r0 = base[base["method"] == base_method]
            r1 = v10[v10["method"] == "V10_DA_ShapeFirst"]
            if len(r0) and len(r1) and metric in r0.columns and metric in r1.columns:
                a = float(r0.iloc[0][metric])
                b = float(r1.iloc[0][metric])
                cmp_rows.append({"metric": metric, tag: a, "v10": b, "delta_v10_minus_base": b - a})
        pd.DataFrame(cmp_rows).to_csv(V10_DA_DIR / f"v10_vs_{tag}.csv", index=False)

    note = """# V10 实验说明（第一性原理）

- 任务拆分：先 shape（零均值单位尺度）→ 再 std_day（尺度）→ 最后加 level_lag1 还原价格。
- 小时输入只用核心驱动（net_load/load/renewable/ramp + hour sin/cos）。
- 目标：减少反相，优先保证形状方向一致性。
- 诊断：输出 daily_shape_diagnostics.csv，并在 meta 记录 neg_corr_day_ratio。
"""
    with open(V10_DA_DIR / "experiment_note.md", "w", encoding="utf-8") as f:
        f.write(note)
        f.write("\n## 本次 run meta\n\n")
        f.write(json.dumps(meta, indent=2, ensure_ascii=False))
        f.write("\n")


def run_v10():
    _, rows, meta = train_v10_da()
    _write_comparison(rows, meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_v10()

