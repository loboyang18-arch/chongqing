"""
V7b 结构化曲线预测 — DA：自适应 Level + S1/S2/S3 shape 与 template_blend 的 λ 融合；
RT：模板+残差、连续风险分与方向/振幅门控解耦。

输出: output/v7_da_structural/, output/v7_rt_structural/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

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
    _train_day_level,
    _composite_score,
    _compute_regime_flag,
    _blend_day_level_signals,
    _search_da_v6_level_params,
    _search_rt_v6_level_params,
    _plot_all_generic,
    _compute_shape_sources,
    _shape_corr_arrays,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V7_DA_DIR = OUTPUT_DIR / "v7_da_structural"
V7_RT_DIR = OUTPUT_DIR / "v7_rt_structural"
V7_DA_DIR.mkdir(exist_ok=True)
V7_RT_DIR.mkdir(exist_ok=True)


def _load_tuned_params(name: str) -> Dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def _gamma_smooth(
    recent_errors: np.ndarray,
    regime_flag: int,
    gamma_min: float,
    gamma_max: float,
) -> float:
    """连续 gamma，避免阶跃。"""
    valid = recent_errors[np.isfinite(recent_errors)]
    if len(valid) == 0:
        return gamma_min
    mag = float(np.clip(np.mean(np.abs(valid[-3:])) / 22.0, 0.0, 1.0))
    sig = _sigmoid(6.0 * (mag - 0.45))
    reg = 0.25 * float(regime_flag)
    g = gamma_min + (gamma_max - gamma_min) * np.clip(sig + reg, 0.0, 1.0)
    return float(np.clip(g, gamma_min, gamma_max))


def _simulate_adaptive_level_v7(
    actual: np.ndarray,
    model_pred: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    regime_flags: np.ndarray,
    regime_confirmed: np.ndarray,
    weights: Tuple[float, float, float],
    gamma_min: float,
    gamma_max: float,
    initial_prev_err: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """双速 EMA 混合 beta 平滑；gamma 连续化。"""
    corrected = np.full(len(actual), np.nan)
    gamma_trace = np.full(len(actual), np.nan)
    prev_err = initial_prev_err if np.isfinite(initial_prev_err) else 0.0
    recent_errors: List[float] = []

    for i in range(len(actual)):
        reg = int(regime_flags[i])
        reg_c = int(regime_confirmed[i]) if i < len(regime_confirmed) else 0
        eff_reg = max(reg, reg_c)
        # _blend_day_level_signals 在 model_pred 为 nan 时仍可仅用 EMA（与 V6 RT 一致）
        blended = _blend_day_level_signals(
            model_pred[i], ema_fast[i], ema_slow[i], eff_reg, weights
        )
        gamma_t = _gamma_smooth(
            np.array(recent_errors, dtype=float),
            eff_reg,
            gamma_min,
            gamma_max,
        )
        gamma_trace[i] = gamma_t
        corrected[i] = blended + gamma_t * prev_err if np.isfinite(blended) else np.nan

        if np.isfinite(actual[i]) and np.isfinite(corrected[i]):
            prev_err = actual[i] - corrected[i]
            recent_errors.append(prev_err)

    return corrected, gamma_trace


def _compute_da_template_lambda_v7b(
    regime_shift: int,
    regime_confirmed: int,
    n7: float,
    n14: float,
    n_global: float,
    val_shape_corr: float,
) -> float:
    """按日 λ_t：平稳/覆盖高 → 更信模板；regime shift / 缺测 / shape 强 → 更信模型。"""
    lam = 0.48
    steady = 1.0 if (not regime_shift and not regime_confirmed) else 0.0
    lam += 0.14 * steady
    lam -= 0.20 * int(bool(regime_shift)) + 0.10 * int(bool(regime_confirmed))
    n_eff = min(float(n7) if np.isfinite(n7) else 0.0, 7.0) / 7.0
    n14_eff = min(float(n14) if np.isfinite(n14) else 0.0, 14.0) / 14.0
    cover = float(np.sqrt(max(n_eff * n14_eff, 0.0)))
    lam *= 0.32 + 0.68 * cover
    if float(n_global) < 3.0:
        lam *= 0.45
    if val_shape_corr > 0.45:
        lam -= 0.09
    if val_shape_corr > 0.52:
        lam -= 0.05
    return float(np.clip(lam, 0.10, 0.82))


def _risk_score_continuous(df: pd.DataFrame) -> np.ndarray:
    """连续风险分 [0,1]。"""
    valley = df.get("valley_flag", pd.Series(0, index=df.index)).astype(float)
    lo = df.get("low_price_risk_flag", pd.Series(0, index=df.index)).astype(float)
    hi = df.get("high_renewable_flag", pd.Series(0, index=df.index)).astype(float)
    ext = df.get("extreme_gap_flag", pd.Series(0, index=df.index)).astype(float)
    if "rt_price_roll24h_std" in df.columns:
        vol = (df["rt_price_roll24h_std"].fillna(0) / 50.0).clip(0, 1)
    else:
        vol = pd.Series(0.0, index=df.index)
    risk = (
        0.25 * valley
        + 0.20 * lo
        + 0.15 * hi
        + 0.15 * ext
        + 0.25 * vol
    )
    return np.clip(risk.values, 0.0, 1.0)


def _gate_dir_amp(shape_24h: np.ndarray, alpha_dir: float, alpha_amp: float) -> np.ndarray:
    """方向 + 振幅门控解耦。"""
    s = np.array(shape_24h, dtype=float)
    m = float(np.mean(s))
    c = s - m
    ptp = float(np.ptp(c))
    if ptp < 1e-6:
        return s
    c_norm = c / ptp
    out = m + alpha_dir * c + (alpha_amp - 1.0) * (c_norm * ptp)
    return out


def train_task_v7_da() -> Tuple[pd.DataFrame, List[Dict]]:
    name = "da"
    target_col = "target_da_clearing_price"
    logger.info("=" * 60)
    logger.info("V7b DA STRUCTURAL (template blend): %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    base_feats = [c for c in df.columns if c != target_col]
    feature_cols = [c for c in base_feats if c in df.columns]

    df["date"] = df.index.date
    daily_mean = df.groupby("date")[target_col].transform("mean")
    df["target_daily_mean"] = daily_mean
    df["target_hourly_dev"] = df[target_col] - daily_mean

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()
    train_dates = sorted(set(train_df["date"]))
    test_dates = sorted(set(test_df["date"]))

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    pred_lgb = lgb_model.predict(test_df[feature_cols])

    daily = _build_day_level_features(df, target_col)
    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["target_daily_amp"] = daily_amp

    rs = daily["regime_shift_flag"].fillna(0).astype(int)
    daily["regime_confirmed"] = (rs.shift(1).fillna(0).astype(int) & rs).astype(int)

    _, level_pred_train, level_pred_test, _ = _train_day_level(daily, train_dates, test_dates, params)
    actual_daily_test = daily.loc[test_dates, "target_daily_mean"].values

    n_val = max(int(len(train_dates) * 0.2), 5)
    val_dates_lev = train_dates[-n_val:]
    val_actual = daily.loc[val_dates_lev, "target_daily_mean"].values
    val_model = level_pred_train.loc[val_dates_lev].values
    val_fast = daily.loc[val_dates_lev, "daily_mean_ema2"].values
    val_slow = daily.loc[val_dates_lev, "daily_mean_ema5"].values
    val_regime = _compute_regime_flag(daily.loc[val_dates_lev]).values

    best_weights, gamma_min, gamma_max, _ = _search_da_v6_level_params(
        val_actual, val_model, val_fast, val_slow, val_regime
    )

    train_actual_daily = daily.loc[train_dates, "target_daily_mean"].values
    train_model = level_pred_train.loc[train_dates].values
    train_fast = daily.loc[train_dates, "daily_mean_ema2"].values
    train_slow = daily.loc[train_dates, "daily_mean_ema5"].values
    train_regime = _compute_regime_flag(daily.loc[train_dates]).values
    train_conf = daily.loc[train_dates, "regime_confirmed"].values
    train_level_corr, _ = _simulate_adaptive_level_v7(
        train_actual_daily,
        train_model,
        train_fast,
        train_slow,
        train_regime,
        train_conf,
        best_weights,
        gamma_min,
        gamma_max,
    )

    last_train_actual = train_actual_daily[-1]
    last_train_pred = train_level_corr[-1]
    prev_err = last_train_actual - last_train_pred if np.isfinite(last_train_pred) else 0.0

    test_model = level_pred_test.values
    test_fast = daily.loc[test_dates, "daily_mean_ema2"].values
    test_slow = daily.loc[test_dates, "daily_mean_ema5"].values
    test_regime = _compute_regime_flag(daily.loc[test_dates]).values
    test_conf = daily.loc[test_dates, "regime_confirmed"].values
    level_corrected_test, gamma_trace = _simulate_adaptive_level_v7(
        actual_daily_test,
        test_model,
        test_fast,
        test_slow,
        test_regime,
        test_conf,
        best_weights,
        gamma_min,
        gamma_max,
        initial_prev_err=prev_err,
    )
    level_map = dict(zip(test_dates, level_corrected_test))
    level_per_hour = np.array([level_map.get(d, np.nan) for d in test_df["date"]])

    params_shape = params.copy()
    params_shape["objective"] = "huber"
    ds_t = lgb.Dataset(train_df[feature_cols], label=train_df["target_hourly_dev"])
    ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_hourly_dev"], reference=ds_t)
    shape_model = lgb.train(
        params_shape,
        ds_t,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_t, ds_v],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    shape_pred_train = shape_model.predict(train_df[feature_cols])
    actual_dev_std = train_df["target_hourly_dev"].std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = float(min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0))

    full_df = pd.concat([train_df, test_df])
    s1_test, s2_test, s3_test = _compute_shape_sources(
        test_df, full_df, target_col, shape_model, feature_cols, scale_factor
    )

    n_train = len(train_df)
    val_h_start = int(n_train * 0.75)
    pre_val_df = train_df.iloc[:val_h_start]
    val_df = train_df.iloc[val_h_start:]

    params_oof = params.copy()
    params_oof["objective"] = "huber"
    ds_pre = lgb.Dataset(pre_val_df[feature_cols], label=pre_val_df["target_hourly_dev"])
    ds_oof = lgb.Dataset(val_df[feature_cols], label=val_df["target_hourly_dev"], reference=ds_pre)
    shape_oof = lgb.train(
        params_oof,
        ds_pre,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_pre, ds_oof],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_pred_std = np.std(shape_oof.predict(pre_val_df[feature_cols]))
    oof_scale = min(pre_val_df["target_hourly_dev"].std() / max(oof_pred_std, 1e-6), 5.0)
    s2_val_oof = shape_oof.predict(val_df[feature_cols]) * oof_scale

    s1_train, _, s3_train = _compute_shape_sources(
        train_df, full_df, target_col, shape_model, feature_cols, scale_factor
    )
    train_level_map = dict(zip(train_dates, train_level_corr))
    train_level_h = np.array([train_level_map.get(d, np.nan) for d in train_df["date"]])

    val_actual_h = train_df[target_col].values[val_h_start:]
    val_level_h = train_level_h[val_h_start:]
    val_s1 = s1_train[val_h_start:]
    val_s3 = s3_train[val_h_start:]
    val_dates_h = train_df.index.date[val_h_start:]

    best_sw, best_sr = (1.0, 0.0, 0.0), -1.0
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -1e-6:
                continue
            blend = w1 * val_s1 + w2 * s2_val_oof + w3 * val_s3
            pred = val_level_h + blend
            fin = np.isfinite(pred)
            if fin.sum() < 48:
                continue
            sr = _shape_corr_arrays(val_actual_h[fin], pred[fin], val_dates_h[fin])
            if sr > best_sr:
                best_sr = sr
                best_sw = (round(w1, 2), round(w2, 2), w3)
    w1, w2, w3 = best_sw
    logger.info(
        "  V7b shape weights: S1=%.2f S2=%.2f S3=%.2f | val_shape_r=%.3f",
        w1,
        w2,
        w3,
        best_sr,
    )

    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_blend = np.where(
        can_mix,
        w1 * s1_test + w2 * s2_test + w3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, s2_test),
    )

    tmpl_blend = np.zeros(len(test_df))
    for j, idx in enumerate(test_df.index):
        hh = idx.hour
        c = f"template_blend_h{hh}"
        if c in test_df.columns:
            tmpl_blend[j] = float(test_df.iloc[j][c])
        else:
            c2 = f"template_dev_h{hh}"
            tmpl_blend[j] = float(test_df.iloc[j][c2]) if c2 in test_df.columns else 0.0

    shape_final = np.zeros(len(test_df), dtype=float)
    level_diag_rows = []
    for i, d in enumerate(test_dates):
        mask = test_df["date"].values == d
        if mask.sum() != 24:
            continue
        row0 = test_df.loc[mask].iloc[0]
        rs_d = int(daily.loc[d, "regime_shift_flag"]) if d in daily.index else 0
        rc_d = int(daily.loc[d, "regime_confirmed"]) if d in daily.index else 0
        n7 = float(row0.get("template_n_7d", np.nan))
        n14 = float(row0.get("template_n_14d", np.nan))
        ng = float(row0.get("template_n_global", np.nan))
        lam_t = _compute_da_template_lambda_v7b(rs_d, rc_d, n7, n14, ng, best_sr)
        shape_final[mask] = lam_t * tmpl_blend[mask] + (1.0 - lam_t) * shape_blend[mask]
        level_diag_rows.append(
            {
                "date": d,
                "lambda_template": lam_t,
                "gamma_used": gamma_trace[i],
                "regime_shift": rs_d,
                "regime_confirmed": rc_d,
            }
        )

    pred_v7 = level_per_hour + shape_final
    actual_test = test_df[target_col].values
    results = pd.DataFrame(
        {
            "actual": actual_test,
            "pred_lgb": pred_lgb,
            "pred_v7": pred_v7,
        },
        index=test_df.index,
    )
    results.index.name = "ts"
    results.to_csv(V7_DA_DIR / "da_result.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V7b_DA_Structural", "pred_v7")]:
        m = _compute_metrics(actual_test, results[col].values)
        sr = compute_shape_report(actual_test, results[col].values, test_df.index, include_v7=True)
        score = _composite_score({**m, **sr}, name)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)
        logger.info(
            "  %-22s MAE=%.2f RMSE=%.2f | corr=%.3f amp_err=%.1f | score=%.4f",
            label,
            m["MAE"],
            m["RMSE"],
            sr["profile_corr"],
            sr["amplitude_err"],
            score,
        )

    pd.DataFrame(rows).to_csv(V7_DA_DIR / "summary.csv", index=False)
    pd.DataFrame(level_diag_rows).set_index("date").to_csv(V7_DA_DIR / "da_level_diag.csv")
    _plot_all_generic(results, name, "pred_v7", "V7b DA", V7_DA_DIR)
    return results, rows


def train_task_v7_rt() -> Tuple[pd.DataFrame, List[Dict]]:
    name = "rt"
    target_col = "target_rt_clearing_price"
    logger.info("=" * 60)
    logger.info("V7 RT STRUCTURAL: %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    df["date"] = df.index.date
    daily_mean = df.groupby("date")[target_col].transform("mean")
    df["target_daily_mean"] = daily_mean
    df["target_hourly_dev"] = df[target_col] - daily_mean

    tmpl_h = np.zeros(len(df))
    for j, idx in enumerate(df.index):
        hh = idx.hour
        c = f"template_dev_h{hh}"
        tmpl_h[j] = float(df.iloc[j][c]) if c in df.columns else 0.0
    df["template_at_h"] = tmpl_h
    df["target_residual"] = df["target_hourly_dev"] - df["template_at_h"]

    # 正常样本过滤：训练 shape 时排除极端小时
    dm = df.groupby("date")[target_col].transform("median")
    dstd = df.groupby("date")[target_col].transform(
        lambda x: float(x.std()) * 2.0 + 1e-6 if len(x) > 1 else 1e-6
    )
    normal_mask = (df[target_col] - dm).abs() <= dstd

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()
    train_dates = sorted(set(train_df["date"]))
    test_dates = sorted(set(test_df["date"]))

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    pred_lgb = lgb_model.predict(test_df[feature_cols])

    daily = _build_day_level_features(df, target_col)
    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["target_daily_amp"] = daily_amp
    rs = daily["regime_shift_flag"].fillna(0).astype(int)
    daily["regime_confirmed"] = (rs.shift(1).fillna(0).astype(int) & rs).astype(int)

    n_val = max(int(len(train_dates) * 0.2), 5)
    val_dates_lev = train_dates[-n_val:]
    val_actual = daily.loc[val_dates_lev, "target_daily_mean"].values
    val_fast = daily.loc[val_dates_lev, "daily_mean_ema2"].values
    val_slow = daily.loc[val_dates_lev, "daily_mean_ema5"].values
    val_regime = _compute_regime_flag(daily.loc[val_dates_lev]).values

    best_weights, gamma_min, gamma_max, _ = _search_rt_v6_level_params(
        val_actual, val_fast, val_slow, val_regime
    )

    train_actual_daily = daily.loc[train_dates, "target_daily_mean"].values
    train_fast = daily.loc[train_dates, "daily_mean_ema2"].values
    train_slow = daily.loc[train_dates, "daily_mean_ema5"].values
    train_regime = _compute_regime_flag(daily.loc[train_dates]).values
    train_conf = daily.loc[train_dates, "regime_confirmed"].values
    zero_model = np.full(len(train_actual_daily), np.nan)
    train_level_corr, _ = _simulate_adaptive_level_v7(
        train_actual_daily,
        zero_model,
        train_fast,
        train_slow,
        train_regime,
        train_conf,
        best_weights,
        gamma_min,
        gamma_max,
    )

    last_train_actual = train_actual_daily[-1]
    last_train_pred = train_level_corr[-1]
    prev_err = last_train_actual - last_train_pred if np.isfinite(last_train_pred) else 0.0

    actual_daily_test = daily.loc[test_dates, "target_daily_mean"].values
    test_fast = daily.loc[test_dates, "daily_mean_ema2"].values
    test_slow = daily.loc[test_dates, "daily_mean_ema5"].values
    test_regime = _compute_regime_flag(daily.loc[test_dates]).values
    test_conf = daily.loc[test_dates, "regime_confirmed"].values
    zero_t = np.full(len(actual_daily_test), np.nan)
    level_corrected_test, _ = _simulate_adaptive_level_v7(
        actual_daily_test,
        zero_t,
        test_fast,
        test_slow,
        test_regime,
        test_conf,
        best_weights,
        gamma_min,
        gamma_max,
        initial_prev_err=prev_err,
    )
    level_map = dict(zip(test_dates, level_corrected_test))
    level_per_hour = np.array([level_map.get(d, np.nan) for d in test_df["date"]])

    nm = normal_mask.reindex(train_df.index).fillna(False)
    train_nr = train_df.loc[nm]
    if len(train_nr) < 500:
        train_nr = train_df
    params_shape = params.copy()
    params_shape["objective"] = "huber"
    ds_t = lgb.Dataset(train_nr[feature_cols], label=train_nr["target_residual"])
    ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_residual"], reference=ds_t)
    shape_model = lgb.train(
        params_shape,
        ds_t,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_t, ds_v],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    res_pred = shape_model.predict(test_df[feature_cols])
    tmpl_h_test = np.zeros(len(test_df))
    for j, idx in enumerate(test_df.index):
        hh = idx.hour
        c = f"template_dev_h{hh}"
        tmpl_h_test[j] = float(test_df.iloc[j][c]) if c in test_df.columns else 0.0
    shape_raw = tmpl_h_test + res_pred

    risk = _risk_score_continuous(test_df)
    if "da_clearing_price_lag24h" in test_df.columns:
        naive = test_df["da_clearing_price_lag24h"].values
    else:
        naive = test_df.get("rt_clearing_price_lag24h", pd.Series(np.nan, index=test_df.index)).values

    pred_v7 = np.zeros(len(test_df), dtype=float)
    for d in sorted(test_df["date"].unique()):
        mask = (test_df["date"] == d).values
        if mask.sum() != 24:
            continue
        day_shape = shape_raw[mask]
        r = float(np.mean(risk[mask]))
        alpha_dir = float(np.clip(1.1 - 0.9 * r, 0.15, 1.0))
        alpha_amp = float(np.clip(0.85 + 0.35 * (1.0 - r), 0.2, 1.3))
        gated = _gate_dir_amp(day_shape, alpha_dir, alpha_amp)
        lvl = level_per_hour[mask]
        pred_h = lvl + gated
        if r > 0.55 and np.isfinite(naive[mask]).any():
            pred_h = 0.55 * pred_h + 0.45 * naive[mask]
        pred_v7[mask] = pred_h
    actual_test = test_df[target_col].values
    results = pd.DataFrame(
        {
            "actual": actual_test,
            "pred_lgb": pred_lgb,
            "pred_v7": pred_v7,
        },
        index=test_df.index,
    )
    results.to_csv(V7_RT_DIR / "rt_result.csv")

    diag = pd.DataFrame(index=test_df.index)
    diag["risk_score"] = risk
    diag["alpha_dir"] = np.nan
    diag["alpha_amp"] = np.nan
    for idx in test_df.index:
        m = test_df.index.date == idx.date()
        r = float(np.mean(risk[m]))
        diag.loc[idx, "alpha_dir"] = float(np.clip(1.1 - 0.9 * r, 0.15, 1.0))
        diag.loc[idx, "alpha_amp"] = float(np.clip(0.85 + 0.35 * (1.0 - r), 0.2, 1.3))
    diag["pred_v7"] = pred_v7
    diag.to_csv(V7_RT_DIR / "alpha_diag.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V7_RT_Structural", "pred_v7")]:
        m = _compute_metrics(actual_test, results[col].values)
        sr = compute_shape_report(actual_test, results[col].values, test_df.index, include_v7=True)
        score = _composite_score({**m, **sr}, name)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)
        logger.info(
            "  %-22s MAE=%.2f RMSE=%.2f | corr=%.3f amp_err=%.1f | score=%.4f",
            label,
            m["MAE"],
            m["RMSE"],
            sr["profile_corr"],
            sr["amplitude_err"],
            score,
        )

    pd.DataFrame(rows).to_csv(V7_RT_DIR / "summary.csv", index=False)
    _plot_all_generic(results, name, "pred_v7", "V7 RT", V7_RT_DIR)
    return results, rows


def run_v7_all():
    _, ra = train_task_v7_da()
    _, rb = train_task_v7_rt()
    all_rows = ra + rb
    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "v7_summary.csv", index=False)
    v7df = pd.DataFrame(all_rows)
    try:
        v6ref = pd.read_csv(OUTPUT_DIR / "v6_vs_v5_summary.csv")
        merged = []
        for task in ("da", "rt"):
            r7 = v7df[(v7df["task"] == task) & (v7df["method"].str.contains("V7", na=False))]
            if len(r7) == 0:
                continue
            a = r7.iloc[0]
            sub = v6ref[v6ref["task"] == task]
            for _, row in sub.iterrows():
                mname = row["metric"]
                v6 = row["v6"]
                v7 = a.get(mname, np.nan)
                merged.append({
                    "task": task,
                    "metric": mname,
                    "v6": v6,
                    "v7": v7,
                    "delta_v7_minus_v6": v7 - v6 if np.isfinite(v7) and np.isfinite(v6) else np.nan,
                })
        if merged:
            pd.DataFrame(merged).to_csv(OUTPUT_DIR / "v7_vs_v6_summary.csv", index=False)
    except Exception as ex:
        logger.warning("Could not write v7_vs_v6_summary: %s", ex)
    logger.info("V7 done. Outputs in %s and %s", V7_DA_DIR, V7_RT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_v7_all()
