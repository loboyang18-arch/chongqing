"""
V12 Shape Optimization — V11 增强特征 + V5/V6 两阶段 Level+Shape 框架。

核心改进：
  1. 日级特征扩展：PM 预测、出清信号、子时段波动的日聚合
  2. Shape LGB 使用增强后 feature_da.csv 全部 225 列
  3. S4 shape 源：V11 逐点 LGB 去均值
  4. 可选：形状感知自定义 LGB 目标函数

变体：
  A: 增强特征 + V5 框架（标准 Huber shape LGB）
  B: A + shape_aware_huber 自定义损失
  C: A + S4 多路集成

输出：output/v12_shape_opt/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, PARAMS_DIR
from .model_baseline import (
    EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND, PEAK_HOURS, VALLEY_HOURS,
    TEST_START, TRAIN_END, _compute_metrics, _load_dataset,
)
from .shape_metrics import compute_shape_report, _split_daily

logger = logging.getLogger(__name__)

V12_DIR = OUTPUT_DIR / "v12_shape_opt"
V12_DIR.mkdir(exist_ok=True)


def _load_tuned_params(name: str) -> Dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


# =====================================================================
# 1. Day-Level Features: V12 扩展版
# =====================================================================

def _build_day_level_features_v12(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """V12 日级特征：在 V5 基础上加入 PM 预测、出清信号、子时段统计的日聚合。"""
    df = df.copy()
    df["date"] = df.index.date
    daily_target = df.groupby("date")[target_col].mean()

    agg = {}

    for col in ["load_forecast", "net_load_forecast", "supply_demand_gap",
                 "renewable_fcst_total_am", "tie_line_fcst_am",
                 "total_gen_fcst_am", "hydro_gen_fcst_am"]:
        if col in df.columns:
            agg[f"{col}_dmean"] = (col, "mean")
            agg[f"{col}_dmax"] = (col, "max")
            agg[f"{col}_dmin"] = (col, "min")

    for col in ["total_gen_fcst_pm", "hydro_gen_fcst_pm", "non_market_gen_fcst_pm",
                 "renewable_fcst_total_pm", "tie_line_fcst_pm"]:
        if col in df.columns:
            agg[f"{col}_dmean"] = (col, "mean")
            agg[f"{col}_dmax"] = (col, "max")
            agg[f"{col}_dmin"] = (col, "min")

    for col in ["da_clearing_power_lag24h", "da_clearing_unit_count_lag24h",
                 "rt_clearing_volume_lag24h", "rt_clearing_unit_count_lag24h"]:
        if col in df.columns:
            agg[f"{col}_dmean"] = (col, "mean")

    for col in ["actual_load_std_lag48h", "actual_load_range_lag48h",
                 "actual_load_max_ramp_lag48h",
                 "renewable_gen_std_lag48h", "renewable_gen_range_lag48h",
                 "tie_line_power_std_lag48h", "tie_line_power_range_lag48h",
                 "tie_line_power_max_ramp_lag48h"]:
        if col in df.columns:
            agg[f"{col}_dmean"] = (col, "mean")

    for col in ["maintenance_gen_count", "maintenance_grid_count"]:
        if col in df.columns:
            agg[f"{col}_dmax"] = (col, "max")

    for col in ["renewable_ratio"]:
        if col in df.columns:
            agg[f"{col}_dmean"] = (col, "mean")

    daily = df.groupby("date").agg(**agg)

    for col in ["load_forecast", "net_load_forecast", "renewable_fcst_total_am"]:
        dmax = f"{col}_dmax"
        dmin = f"{col}_dmin"
        if dmax in daily.columns and dmin in daily.columns:
            daily[f"{col}_drange"] = daily[dmax] - daily[dmin]

    for col in ["total_gen_fcst_pm", "tie_line_fcst_pm", "renewable_fcst_total_pm"]:
        dmax = f"{col}_dmax"
        dmin = f"{col}_dmin"
        if dmax in daily.columns and dmin in daily.columns:
            daily[f"{col}_drange"] = daily[dmax] - daily[dmin]

    for cal in ["day_of_week", "is_weekend", "month"]:
        if cal in df.columns:
            daily[cal] = df.groupby("date")[cal].first()

    daily["target_daily_mean"] = daily_target

    daily["daily_mean_lag1"] = daily_target.shift(1)
    daily["daily_mean_lag3_avg"] = daily_target.rolling(3, min_periods=1).mean().shift(1)
    daily["daily_mean_lag7_avg"] = daily_target.rolling(7, min_periods=1).mean().shift(1)
    daily["daily_mean_ema2"] = daily_target.ewm(span=2, adjust=False).mean().shift(1)
    daily["daily_mean_ema3"] = daily_target.ewm(span=3, adjust=False).mean().shift(1)
    daily["daily_mean_ema5"] = daily_target.ewm(span=5, adjust=False).mean().shift(1)
    daily["daily_mean_roll14"] = daily_target.rolling(14, min_periods=5).mean().shift(1)
    daily["daily_mean_roll14_std"] = daily_target.rolling(14, min_periods=5).std().shift(1)
    daily["daily_mean_recent3"] = daily_target.rolling(3, min_periods=2).mean().shift(1)
    daily["daily_mean_diff1"] = daily_target.shift(1) - daily_target.shift(2)
    daily["daily_mean_diff3"] = daily_target.shift(1) - daily_target.shift(4)

    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["daily_amp_lag1"] = daily_amp.shift(1)
    daily["daily_amp_lag3_avg"] = daily_amp.rolling(3, min_periods=1).mean().shift(1)

    peak_mean = df[df.index.hour.isin(PEAK_HOURS)].groupby("date")[target_col].mean()
    valley_mean = df[df.index.hour.isin(VALLEY_HOURS)].groupby("date")[target_col].mean()
    daily["daily_peak_mean_lag1"] = peak_mean.shift(1)
    daily["daily_valley_mean_lag1"] = valley_mean.shift(1)

    diff_threshold = np.maximum(10.0, daily["daily_mean_roll14_std"].fillna(0.0) * 0.6)
    daily["regime_shift_flag"] = (
        ((daily["daily_mean_roll14"] - daily["daily_mean_recent3"]) > diff_threshold)
        | (daily["daily_mean_diff3"] < -12.0)
    ).astype(int)
    daily["regime_strength"] = np.maximum(
        daily["daily_mean_roll14"] - daily["daily_mean_recent3"], 0.0
    )

    return daily


# =====================================================================
# 2. Level 层辅助函数（复用 V6 逻辑）
# =====================================================================

def _train_day_level(daily: pd.DataFrame, train_dates, test_dates, params):
    target = "target_daily_mean"
    feat_cols = [c for c in daily.columns if c != target and c != "target_daily_amp"]

    train = daily.loc[train_dates]
    test = daily.loc[test_dates]

    n_val = max(int(len(train) * 0.2), 5)
    train_sub = train.iloc[:-n_val]
    val_sub = train.iloc[-n_val:]

    p = params.copy()
    p["objective"] = "huber"
    p["num_leaves"] = min(p.get("num_leaves", 31), 15)
    p["min_child_samples"] = 3
    p["learning_rate"] = 0.05

    dt = lgb.Dataset(train_sub[feat_cols], label=train_sub[target])
    dv = lgb.Dataset(val_sub[feat_cols], label=val_sub[target], reference=dt)
    model = lgb.train(
        p, dt, num_boost_round=500,
        valid_sets=[dt, dv], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    logger.info("  Day-level model best iter: %d", model.best_iteration)

    pred_train = pd.Series(model.predict(train[feat_cols]), index=train.index)
    pred_test = pd.Series(model.predict(test[feat_cols]), index=test.index)
    return model, pred_train, pred_test, feat_cols


def _compute_regime_flag(daily: pd.DataFrame) -> pd.Series:
    if "regime_shift_flag" in daily.columns:
        return daily["regime_shift_flag"].fillna(0).astype(int)
    return pd.Series(0, index=daily.index, dtype=int)


def _blend_day_level_signals(model_pred, ema_fast, ema_slow, regime_flag, weights):
    signals = np.array([model_pred, ema_fast, ema_slow], dtype=float)
    base_weights = np.array(weights, dtype=float)
    valid = np.isfinite(signals)
    if not valid.any():
        return np.nan
    base_weights[~valid] = 0.0
    if base_weights.sum() <= 0:
        base_weights = valid.astype(float)
    if regime_flag:
        base_weights[1] += 0.20
        base_weights[0] = max(0.0, base_weights[0] - 0.12)
        base_weights[2] = max(0.0, base_weights[2] - 0.08)
    base_weights = base_weights / base_weights.sum()
    return float(np.nansum(signals * base_weights))


def _compute_adaptive_gamma(recent_errors, regime_flag, gamma_min, gamma_max):
    if gamma_max < gamma_min:
        gamma_max = gamma_min
    valid = recent_errors[np.isfinite(recent_errors)]
    if len(valid) == 0:
        return gamma_min
    recent = valid[-3:]
    bias = float(np.mean(recent))
    if abs(bias) < 1e-6:
        consistency = 0.0
    else:
        consistency = float(np.mean(np.sign(recent) == np.sign(bias)))
    magnitude = np.clip(abs(bias) / 20.0, 0.0, 1.0)
    signal = 0.55 * magnitude + 0.25 * consistency + 0.20 * int(regime_flag)
    gamma = gamma_min + (gamma_max - gamma_min) * signal
    return float(np.clip(gamma, gamma_min, gamma_max))


def _simulate_adaptive_level(actual, model_pred, ema_fast, ema_slow,
                             regime_flags, weights, gamma_min, gamma_max,
                             initial_prev_err=0.0):
    corrected = np.full(len(actual), np.nan)
    gamma_trace = np.full(len(actual), np.nan)
    prev_err = initial_prev_err if np.isfinite(initial_prev_err) else 0.0
    recent_errors: List[float] = []
    for i in range(len(actual)):
        blended = _blend_day_level_signals(
            model_pred[i], ema_fast[i], ema_slow[i], int(regime_flags[i]), weights
        )
        gamma_t = _compute_adaptive_gamma(
            np.array(recent_errors, dtype=float), int(regime_flags[i]),
            gamma_min, gamma_max,
        )
        gamma_trace[i] = gamma_t
        corrected[i] = blended + gamma_t * prev_err if np.isfinite(blended) else np.nan
        if np.isfinite(actual[i]) and np.isfinite(corrected[i]):
            prev_err = actual[i] - corrected[i]
            recent_errors.append(prev_err)
    return corrected, gamma_trace


def _search_level_params(val_actual, val_model, val_fast, val_slow, val_regime):
    best = ((0.4, 0.35, 0.25), 0.03, 0.20)
    best_mae = np.inf
    for w_model in np.arange(0.2, 0.81, 0.1):
        for w_fast in np.arange(0.1, 0.71, 0.1):
            w_slow = round(1.0 - w_model - w_fast, 2)
            if w_slow < -1e-6:
                continue
            weights = (round(w_model, 2), round(w_fast, 2), max(0.0, w_slow))
            for gamma_min in [0.02, 0.04, 0.06, 0.08]:
                for gamma_max in [0.16, 0.22, 0.28, 0.32]:
                    corrected, _ = _simulate_adaptive_level(
                        val_actual, val_model, val_fast, val_slow,
                        val_regime, weights, gamma_min, gamma_max
                    )
                    valid = np.isfinite(corrected) & np.isfinite(val_actual)
                    if valid.sum() < 3:
                        continue
                    mae = np.mean(np.abs(val_actual[valid] - corrected[valid]))
                    if mae < best_mae:
                        best_mae = mae
                        best = (weights, gamma_min, gamma_max)
    weights, gamma_min, gamma_max = best
    return weights, gamma_min, gamma_max, best_mae


# =====================================================================
# 3. Shape 层
# =====================================================================

def _safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa, bb = a[mask], b[mask]
    if np.std(aa) < 1e-6 or np.std(bb) < 1e-6:
        return np.nan
    corr = np.corrcoef(aa, bb)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def _shape_corr_arrays(actual, pred, dates):
    corrs = []
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() == 24:
            a, p = actual[mask], pred[mask]
            if np.std(a) > 1e-6 and np.std(p) > 1e-6:
                c = np.corrcoef(a, p)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
    return np.mean(corrs) if corrs else -1.0


def _compute_shape_sources(target_df, full_df, target_col, shape_model,
                           feat_cols, scale_factor):
    dates = target_df.index.date
    target_dates = sorted(set(dates))
    all_dates = sorted(set(full_df.index.date))
    all_date_idx = {d: i for i, d in enumerate(all_dates)}
    daily_mean = full_df.groupby(full_df.index.date)[target_col].mean()

    hourly_dev = {}
    for d in all_dates:
        mask = full_df.index.date == d
        hourly_dev[d] = full_df.loc[mask, target_col].values - daily_mean.get(d, np.nan)

    s1 = np.full(len(target_df), np.nan)
    for d in target_dates:
        idx = all_date_idx[d]
        prev = [all_dates[idx - k] for k in range(1, 4) if idx - k >= 0]
        if not prev:
            continue
        mask = dates == d
        avg = np.zeros(mask.sum())
        cnt = 0
        for pd_ in prev:
            dev = hourly_dev.get(pd_)
            if dev is not None and len(dev) == mask.sum():
                avg += dev
                cnt += 1
        if cnt > 0:
            s1[mask] = avg / cnt

    s2 = shape_model.predict(target_df[feat_cols]) * scale_factor

    s3 = np.full(len(target_df), np.nan)
    for d in target_dates:
        idx = all_date_idx[d]
        if idx == 0:
            continue
        prev_d = all_dates[idx - 1]
        mask = dates == d
        dev = hourly_dev.get(prev_d)
        if dev is not None and len(dev) == mask.sum():
            s3[mask] = dev

    return s1, s2, s3


def _train_amplitude_model(daily, train_dates, test_dates, params):
    target = "target_daily_amp"
    feat_cols = [c for c in daily.columns
                 if c not in (target, "target_daily_mean")]
    train = daily.loc[train_dates]
    test = daily.loc[test_dates]
    n_val = max(int(len(train) * 0.2), 5)
    train_sub = train.iloc[:-n_val]
    val_sub = train.iloc[-n_val:]
    p = params.copy()
    p["objective"] = "huber"
    p["num_leaves"] = min(p.get("num_leaves", 31), 15)
    p["min_child_samples"] = 3
    p["learning_rate"] = 0.05
    dt = lgb.Dataset(train_sub[feat_cols], label=train_sub[target])
    dv = lgb.Dataset(val_sub[feat_cols], label=val_sub[target], reference=dt)
    model = lgb.train(
        p, dt, num_boost_round=500,
        valid_sets=[dt, dv], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    pred_test = pd.Series(model.predict(test[feat_cols]), index=test.index)
    return model, pred_test


def _scale_shape(shape_24h, target_amp):
    current_amp = np.max(shape_24h) - np.min(shape_24h)
    if current_amp < 1e-6:
        return shape_24h
    scale = max(target_amp, 1.0) / current_amp
    scale = np.clip(scale, 0.2, 10.0)
    return shape_24h * scale


# =====================================================================
# 4. Shape-Aware Custom Objective for LGB
# =====================================================================

def _shape_aware_huber_obj(y_pred, dtrain):
    """Huber 损失 + 日内方差匹配正则。

    在 Huber 基础上增加：鼓励预测的日内方差接近真实值的日内方差。
    假设数据按时间排序，每 24 行为一天。
    """
    y_true = dtrain.get_label()
    residual = y_pred - y_true
    delta = 10.0
    mask = np.abs(residual) <= delta
    grad = np.where(mask, residual, delta * np.sign(residual))
    hess = np.where(mask, 1.0, 1e-3)

    n = len(y_true)
    n_full_days = n // 24
    for i in range(n_full_days):
        sl = slice(i * 24, (i + 1) * 24)
        y_t = y_true[sl]
        y_p = y_pred[sl]
        std_t = np.std(y_t)
        std_p = np.std(y_p)
        if std_t > 1e-6 and std_p > 1e-6:
            var_diff = (std_p - std_t) / std_t
            grad[sl] += 0.05 * var_diff * (y_p - np.mean(y_p)) / std_p

    return grad, hess


def _shape_aware_mae_eval(y_pred, dtrain):
    """自定义 eval metric (MAE)，配合自定义 objective 使用。"""
    y_true = dtrain.get_label()
    mae = np.mean(np.abs(y_true - y_pred))
    return "mae", mae, False


# =====================================================================
# 5. Daily Diagnostics
# =====================================================================

def _daily_diagnostics(actual, pred, index):
    rows = []
    dates = index.date
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() != 24:
            continue
        a, p = actual[mask], pred[mask]
        corr = float(np.corrcoef(a, p)[0, 1]) if np.std(a) > 1e-6 and np.std(p) > 1e-6 else np.nan
        rows.append({
            "date": str(d),
            "corr_day": round(corr, 4) if np.isfinite(corr) else np.nan,
            "neg_corr_flag": int(corr < 0) if np.isfinite(corr) else 0,
            "peak_true_hour": int(np.argmax(a)),
            "peak_pred_hour": int(np.argmax(p)),
            "valley_true_hour": int(np.argmin(a)),
            "valley_pred_hour": int(np.argmin(p)),
            "amp_true": round(float(np.max(a) - np.min(a)), 2),
            "amp_pred": round(float(np.max(p) - np.min(p)), 2),
            "mae_day": round(float(np.mean(np.abs(a - p))), 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# 6. Main Training Pipeline
# =====================================================================

def run_v12_variant(variant: str = "A"):
    """运行 V12 的某个变体。

    variant:
      "A" - 增强特征 + V5 框架（标准 Huber shape LGB）
      "B" - A + shape_aware_huber 自定义损失
      "C" - A + S4 多路集成
    """
    name = "da"
    target_col = "target_da_clearing_price"
    logger.info("=" * 60)
    logger.info("V12-%s: %s", variant, name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    df["date"] = df.index.date
    daily_mean = df.groupby("date")[target_col].transform("mean")
    df["target_daily_mean"] = daily_mean
    df["target_hourly_dev"] = df[target_col] - daily_mean

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()
    train_dates = sorted(set(train_df["date"]))
    test_dates = sorted(set(test_df["date"]))

    # ── [A] Point LGB baseline ──────────────────────
    logger.info("--- [A] Point LGB baseline ---")
    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)
    lgb_model = lgb.train(
        params, dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
                   lgb.log_evaluation(200)],
    )
    pred_lgb = lgb_model.predict(test_df[feature_cols])

    # ── [B] Adaptive Day-Level ──────────────────────
    logger.info("--- [B] Adaptive Day-Level (V12 features) ---")
    daily = _build_day_level_features_v12(df, target_col)
    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["target_daily_amp"] = daily_amp

    _, level_pred_train, level_pred_test, _ = _train_day_level(
        daily, train_dates, test_dates, params
    )
    actual_daily_test = daily.loc[test_dates, "target_daily_mean"].values

    n_val = max(int(len(train_dates) * 0.2), 5)
    val_dates_lev = train_dates[-n_val:]
    val_actual = daily.loc[val_dates_lev, "target_daily_mean"].values
    val_model = level_pred_train.loc[val_dates_lev].values
    val_fast = daily.loc[val_dates_lev, "daily_mean_ema2"].values
    val_slow = daily.loc[val_dates_lev, "daily_mean_ema5"].values
    val_regime = _compute_regime_flag(daily.loc[val_dates_lev]).values

    best_weights, gamma_min, gamma_max, val_mae = _search_level_params(
        val_actual, val_model, val_fast, val_slow, val_regime
    )
    logger.info(
        "  Level weights: model=%.2f fast=%.2f slow=%.2f | gamma=[%.2f, %.2f] | val MAE=%.2f",
        best_weights[0], best_weights[1], best_weights[2], gamma_min, gamma_max, val_mae,
    )

    train_actual_daily = daily.loc[train_dates, "target_daily_mean"].values
    train_model = level_pred_train.loc[train_dates].values
    train_fast = daily.loc[train_dates, "daily_mean_ema2"].values
    train_slow = daily.loc[train_dates, "daily_mean_ema5"].values
    train_regime = _compute_regime_flag(daily.loc[train_dates]).values
    train_level_corr, _ = _simulate_adaptive_level(
        train_actual_daily, train_model, train_fast, train_slow,
        train_regime, best_weights, gamma_min, gamma_max,
    )

    last_train_actual = train_actual_daily[-1]
    last_train_pred = train_level_corr[-1]
    prev_err = last_train_actual - last_train_pred if np.isfinite(last_train_pred) else 0.0

    test_model = level_pred_test.values
    test_fast = daily.loc[test_dates, "daily_mean_ema2"].values
    test_slow = daily.loc[test_dates, "daily_mean_ema5"].values
    test_regime = _compute_regime_flag(daily.loc[test_dates]).values
    level_corrected_test, gamma_trace = _simulate_adaptive_level(
        actual_daily_test, test_model, test_fast, test_slow,
        test_regime, best_weights, gamma_min, gamma_max, initial_prev_err=prev_err
    )
    level_map = dict(zip(test_dates, level_corrected_test))
    level_per_hour = np.array([level_map.get(d, np.nan) for d in test_df["date"]])

    # ── [C] Shape Layer ─────────────────────────────
    logger.info("--- [C] Shape Layer (variant=%s) ---", variant)
    params_shape = params.copy()

    if variant == "B":
        logger.info("  Using shape-aware custom objective")
        params_shape["objective"] = _shape_aware_huber_obj
        params_shape.pop("metric", None)
        ds_t = lgb.Dataset(train_df[feature_cols], label=train_df["target_hourly_dev"])
        ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_hourly_dev"], reference=ds_t)
        shape_model = lgb.train(
            params_shape, ds_t, num_boost_round=NUM_BOOST_ROUND,
            feval=_shape_aware_mae_eval,
            valid_sets=[ds_t, ds_v], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(200)],
        )
    else:
        params_shape["objective"] = "huber"
        ds_t = lgb.Dataset(train_df[feature_cols], label=train_df["target_hourly_dev"])
        ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_hourly_dev"], reference=ds_t)
        shape_model = lgb.train(
            params_shape, ds_t, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_t, ds_v], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(200)],
        )

    shape_pred_train = shape_model.predict(train_df[feature_cols])
    actual_dev_std = train_df["target_hourly_dev"].std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0)
    logger.info("  Shape scale_factor: %.3f", scale_factor)

    full_df = pd.concat([train_df, test_df])
    s1_test, s2_test, s3_test = _compute_shape_sources(
        test_df, full_df, target_col, shape_model, feature_cols, scale_factor
    )

    # S4: V11 逐点 LGB 去均值
    s4_test = None
    if variant == "C":
        logger.info("  Building S4: point LGB de-meaned shape")
        pred_lgb_daily_mean = {}
        for d in test_dates:
            mask = test_df["date"].values == d
            pred_lgb_daily_mean[d] = np.mean(pred_lgb[mask])
        s4_test = np.array([
            pred_lgb[i] - pred_lgb_daily_mean.get(test_df["date"].values[i], 0)
            for i in range(len(test_df))
        ])

    # ── Shape weight search on validation ──────────
    n_train = len(train_df)
    val_h_start = int(n_train * 0.75)
    pre_val_df = train_df.iloc[:val_h_start]
    val_df = train_df.iloc[val_h_start:]

    if variant == "B":
        params_oof = params.copy()
        params_oof["objective"] = _shape_aware_huber_obj
        params_oof.pop("metric", None)
        ds_pre = lgb.Dataset(pre_val_df[feature_cols], label=pre_val_df["target_hourly_dev"])
        ds_oof = lgb.Dataset(val_df[feature_cols], label=val_df["target_hourly_dev"], reference=ds_pre)
        shape_oof = lgb.train(
            params_oof, ds_pre, num_boost_round=NUM_BOOST_ROUND,
            feval=_shape_aware_mae_eval,
            valid_sets=[ds_pre, ds_oof], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
    else:
        params_oof = params.copy()
        params_oof["objective"] = "huber"
        ds_pre = lgb.Dataset(pre_val_df[feature_cols], label=pre_val_df["target_hourly_dev"])
        ds_oof = lgb.Dataset(val_df[feature_cols], label=val_df["target_hourly_dev"], reference=ds_pre)
        shape_oof = lgb.train(
            params_oof, ds_pre, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_pre, ds_oof], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
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

    # S4 for validation (if variant C)
    s4_val = None
    if variant == "C":
        lgb_pred_train = lgb_model.predict(train_df[feature_cols])
        lgb_daily_mean_train = {}
        for d in train_dates:
            mask_d = train_df["date"].values == d
            lgb_daily_mean_train[d] = np.mean(lgb_pred_train[mask_d])
        s4_all_train = np.array([
            lgb_pred_train[i] - lgb_daily_mean_train.get(train_df["date"].values[i], 0)
            for i in range(len(train_df))
        ])
        s4_val = s4_all_train[val_h_start:]

    if variant == "C":
        best_sw, best_sr = (0.25, 0.25, 0.25, 0.25), -1.0
        for w1 in np.arange(0, 1.01, 0.1):
            for w2 in np.arange(0, 1.01 - w1, 0.1):
                for w3 in np.arange(0, 1.01 - w1 - w2, 0.1):
                    w4 = round(1.0 - w1 - w2 - w3, 2)
                    if w4 < -1e-6:
                        continue
                    blend = w1 * val_s1 + w2 * s2_val_oof + w3 * val_s3 + w4 * s4_val
                    pred = val_level_h + blend
                    fin = np.isfinite(pred)
                    if fin.sum() < 48:
                        continue
                    sr = _shape_corr_arrays(val_actual_h[fin], pred[fin], val_dates_h[fin])
                    if sr > best_sr:
                        best_sr = sr
                        best_sw = (round(w1, 2), round(w2, 2), round(w3, 2), round(w4, 2))
        w1, w2, w3, w4 = best_sw
        logger.info("  Shape weights: S1=%.2f S2=%.2f S3=%.2f S4=%.2f | val_shape_r=%.3f",
                    w1, w2, w3, w4, best_sr)
    else:
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
        logger.info("  Shape weights: S1=%.2f S2=%.2f S3=%.2f | val_shape_r=%.3f",
                    w1, w2, w3, best_sr)

    # ── Apply on test ──────────────────────────────
    if variant == "C":
        w1, w2, w3, w4 = best_sw
        can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
        shape_raw = np.where(
            can_mix,
            w1 * s1_test + w2 * s2_test + w3 * s3_test + w4 * s4_test,
            np.where(np.isfinite(s1_test), s1_test, s2_test),
        )
    else:
        w1, w2, w3 = best_sw
        can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
        shape_raw = np.where(
            can_mix,
            w1 * s1_test + w2 * s2_test + w3 * s3_test,
            np.where(np.isfinite(s1_test), s1_test, s2_test),
        )

    # ── [D] Amplitude Scaling ──────────────────────
    logger.info("--- [D] Amplitude Scaling ---")
    _, amp_pred_test = _train_amplitude_model(daily, train_dates, test_dates, params)
    amp_pred_map = dict(zip(test_dates, amp_pred_test.values))

    shape_scaled = np.copy(shape_raw)
    for d in test_dates:
        mask = test_df["date"].values == d
        if mask.sum() != 24:
            continue
        target_amp = amp_pred_map.get(d, np.nan)
        if np.isfinite(target_amp):
            shape_scaled[mask] = _scale_shape(shape_raw[mask], target_amp)

    pred_v12 = level_per_hour + shape_scaled

    # ── Results ────────────────────────────────────
    actual_test = test_df[target_col].values
    results = pd.DataFrame({
        "actual": actual_test,
        "pred": pred_v12,
    }, index=test_df.index)
    results.index.name = "ts"

    suffix = f"_{variant}" if variant != "A" else ""
    results.to_csv(V12_DIR / f"da_result{suffix}.csv")

    mae = float(np.mean(np.abs(actual_test - pred_v12)))
    rmse = float(np.sqrt(np.mean((actual_test - pred_v12) ** 2)))
    shape_report = compute_shape_report(actual_test, pred_v12, test_df.index, include_v7=True)
    diag = _daily_diagnostics(actual_test, pred_v12, test_df.index)
    neg_ratio = float(diag["neg_corr_flag"].mean())

    summary = {
        "model": f"V12-{variant}",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "neg_corr_day_ratio": round(neg_ratio, 4),
    }
    summary.update(shape_report)

    logger.info("-" * 40)
    logger.info("V12-%s Results:", variant)
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)

    pd.DataFrame([summary]).to_csv(V12_DIR / f"summary{suffix}.csv", index=False)
    diag.to_csv(V12_DIR / f"daily_shape_diagnostics{suffix}.csv", index=False)

    return summary, results


def run_all_variants():
    """运行 V12 全部三个变体并汇总。"""
    all_summaries = []
    for v in ["A", "B", "C"]:
        logger.info("\n" + "=" * 70)
        logger.info("Running V12-%s", v)
        logger.info("=" * 70)
        s, _ = run_v12_variant(v)
        all_summaries.append(s)

    combined = pd.DataFrame(all_summaries)
    combined.to_csv(V12_DIR / "summary_all_variants.csv", index=False)
    logger.info("\n=== V12 All Variants Summary ===")
    logger.info("\n%s", combined.to_string(index=False))
    return combined


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_variants()
