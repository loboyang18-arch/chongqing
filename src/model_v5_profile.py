"""
V5 Profile Forecast — 日曲线结构预测优化

改进点（相对 V4）:
  1. DA/RT: Day-level 专属模型预测日均价，替换原 Level 层
  2. DA/RT: 日振幅预测 + shape 向量缩放修正
  3. RT:    Spike 风险规则标签 + spike 时段 fallback
  4. 联合选模分数 (composite score)

输出到 output/v5_profile/
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, PARAMS_DIR
from .model_baseline import (
    EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND, PEAK_HOURS, VALLEY_HOURS,
    TEST_START, TRAIN_END, _compute_metrics, _load_dataset,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)
V5_DIR = OUTPUT_DIR / "v5_profile"
V5_DIR.mkdir(exist_ok=True)
V6_DA_DIR = OUTPUT_DIR / "v6_da_level"
V6_RT_DIR = OUTPUT_DIR / "v6_rt_gating"
V6_DA_DIR.mkdir(exist_ok=True)
V6_RT_DIR.mkdir(exist_ok=True)


def _setup_cn_font():
    for p in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Medium.ttc"]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["mathtext.fontset"] = "cm"


def _load_tuned_params(name: str) -> Dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


# =====================================================================
# 1. Day-Level Model: 日均价专属预测
# =====================================================================

def _build_day_level_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """从小时级数据构建日级特征 (每天一行)。

    特征类别:
      - 负荷预测摘要: 均值/峰值/谷值/峰谷差
      - 新能源预测摘要: 均值/峰值/谷值/ramp
      - Net load 摘要
      - 检修 / 日历
      - 历史日均价 lag1/3/7
      - 历史日振幅 lag1/3
    """
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


def _train_day_level(daily: pd.DataFrame, train_dates, test_dates, params):
    """训练日均价模型 (LGB, huber)。"""
    target = "target_daily_mean"
    feat_cols = [c for c in daily.columns if c != target]

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
        callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)],
    )
    logger.info("  Day-level model best iter: %d", model.best_iteration)

    pred_train = pd.Series(model.predict(train[feat_cols]), index=train.index)
    pred_test = pd.Series(model.predict(test[feat_cols]), index=test.index)

    return model, pred_train, pred_test, feat_cols


# =====================================================================
# 2. Amplitude Model: 日振幅预测 + shape 缩放
# =====================================================================

def _train_amplitude_model(daily: pd.DataFrame, train_dates, test_dates, params):
    """训练日振幅预测模型。"""
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
    logger.info("  Amplitude model best iter: %d", model.best_iteration)

    pred_test = pd.Series(model.predict(test[feat_cols]), index=test.index)
    return model, pred_test


def _scale_shape(shape_24h: np.ndarray, target_amp: float) -> np.ndarray:
    """将 shape 向量缩放到目标振幅。"""
    current_amp = np.max(shape_24h) - np.min(shape_24h)
    if current_amp < 1e-6:
        return shape_24h
    scale = max(target_amp, 1.0) / current_amp
    scale = np.clip(scale, 0.2, 10.0)
    return shape_24h * scale


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-6 or np.std(bb) < 1e-6:
        return np.nan
    corr = np.corrcoef(aa, bb)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def _compute_regime_flag(daily: pd.DataFrame) -> pd.Series:
    if "regime_shift_flag" in daily.columns:
        return daily["regime_shift_flag"].fillna(0).astype(int)
    return pd.Series(0, index=daily.index, dtype=int)


def _blend_day_level_signals(model_pred: float, ema_fast: float, ema_slow: float,
                             regime_flag: int, weights: Tuple[float, float, float]) -> float:
    signals = np.array([model_pred, ema_fast, ema_slow], dtype=float)
    base_weights = np.array(weights, dtype=float)
    valid = np.isfinite(signals)
    if not valid.any():
        return np.nan

    base_weights[~valid] = 0.0
    if base_weights.sum() <= 0:
        base_weights = valid.astype(float)

    if regime_flag:
        # 切换期更信快 EMA，适度降低 model / slow EMA 的锚定作用。
        base_weights[1] += 0.20
        base_weights[0] = max(0.0, base_weights[0] - 0.12)
        base_weights[2] = max(0.0, base_weights[2] - 0.08)

    base_weights = base_weights / base_weights.sum()
    return float(np.nansum(signals * base_weights))


def _compute_adaptive_gamma(recent_errors: np.ndarray, regime_flag: int,
                            gamma_min: float, gamma_max: float) -> float:
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


def _simulate_adaptive_level(
    actual: np.ndarray,
    model_pred: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    regime_flags: np.ndarray,
    weights: Tuple[float, float, float],
    gamma_min: float,
    gamma_max: float,
    initial_prev_err: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    corrected = np.full(len(actual), np.nan)
    gamma_trace = np.full(len(actual), np.nan)
    prev_err = initial_prev_err if np.isfinite(initial_prev_err) else 0.0
    recent_errors: List[float] = []

    for i in range(len(actual)):
        blended = _blend_day_level_signals(
            model_pred[i], ema_fast[i], ema_slow[i], int(regime_flags[i]), weights
        )
        gamma_t = _compute_adaptive_gamma(
            np.array(recent_errors, dtype=float),
            int(regime_flags[i]),
            gamma_min,
            gamma_max,
        )
        gamma_trace[i] = gamma_t
        corrected[i] = blended + gamma_t * prev_err if np.isfinite(blended) else np.nan

        if np.isfinite(actual[i]) and np.isfinite(corrected[i]):
            prev_err = actual[i] - corrected[i]
            recent_errors.append(prev_err)

    return corrected, gamma_trace


def _search_da_v6_level_params(
    val_actual: np.ndarray,
    val_model: np.ndarray,
    val_ema_fast: np.ndarray,
    val_ema_slow: np.ndarray,
    val_regime: np.ndarray,
) -> Tuple[Tuple[float, float, float], float, float, float]:
    best = ((0.4, 0.35, 0.25), 0.03, 0.20)
    best_mae = np.inf

    gamma_mins = [0.02, 0.04, 0.06, 0.08]
    gamma_maxs = [0.16, 0.22, 0.28, 0.32]

    for w_model in np.arange(0.2, 0.81, 0.1):
        for w_fast in np.arange(0.1, 0.71, 0.1):
            w_slow = round(1.0 - w_model - w_fast, 2)
            if w_slow < -1e-6:
                continue
            weights = (round(w_model, 2), round(w_fast, 2), max(0.0, w_slow))
            for gamma_min in gamma_mins:
                for gamma_max in gamma_maxs:
                    corrected, _ = _simulate_adaptive_level(
                        val_actual, val_model, val_ema_fast, val_ema_slow,
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


def _search_rt_v6_level_params(
    val_actual: np.ndarray,
    val_ema_fast: np.ndarray,
    val_ema_slow: np.ndarray,
    val_regime: np.ndarray,
) -> Tuple[Tuple[float, float, float], float, float, float]:
    best = ((0.0, 0.65, 0.35), 0.08, 0.25)
    best_mae = np.inf

    gamma_mins = [0.08, 0.10, 0.12]
    gamma_maxs = [0.20, 0.25, 0.30, 0.35]
    zero_model = np.full(len(val_actual), np.nan)

    for w_fast in np.arange(0.3, 1.01, 0.05):
        w_slow = round(1.0 - w_fast, 2)
        weights = (0.0, round(w_fast, 2), max(0.0, w_slow))
        for gamma_min in gamma_mins:
            for gamma_max in gamma_maxs:
                corrected, _ = _simulate_adaptive_level(
                    val_actual, zero_model, val_ema_fast, val_ema_slow,
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


def _compute_daily_shape_confidence(target_df: pd.DataFrame, s1: np.ndarray,
                                    s2: np.ndarray, s3: np.ndarray) -> Dict:
    dates = target_df["date"].values
    conf_map = {}
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() != 24:
            conf_map[d] = 0.5
            continue
        scores = []
        for a, b in [(s1, s2), (s1, s3), (s2, s3)]:
            corr = _safe_corr(a[mask], b[mask])
            if np.isfinite(corr):
                scores.append((corr + 1.0) / 2.0)
        conf_map[d] = float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.5
    return conf_map


def _conditional_scale_shape(shape_24h: np.ndarray, target_amp: float,
                             level_confidence: float, shape_confidence: float,
                             regime_flag: int) -> np.ndarray:
    current_amp = np.max(shape_24h) - np.min(shape_24h)
    if current_amp < 1e-6 or not np.isfinite(target_amp):
        return shape_24h

    base_scale = np.clip(max(target_amp, 1.0) / current_amp, 0.2, 10.0)
    trust = 0.35 + 0.35 * np.clip(level_confidence, 0.0, 1.0) \
        + 0.30 * np.clip(shape_confidence, 0.0, 1.0)
    if regime_flag:
        trust = min(1.0, trust + 0.10)
    trust = np.clip(trust, 0.25, 1.0)
    adjusted_scale = 1.0 + (base_scale - 1.0) * trust
    adjusted_scale = np.clip(adjusted_scale, 0.5, 4.0)
    return shape_24h * adjusted_scale


def _compute_shape_gate_alpha(shape_24h: np.ndarray, spike_risk_24h: np.ndarray,
                              shape_confidence: float, recent_amp: float) -> np.ndarray:
    raw_amp = np.max(shape_24h) - np.min(shape_24h)
    base_alpha = np.clip(0.30 + 0.80 * np.clip(shape_confidence, 0.0, 1.0), 0.25, 1.10)

    if np.isfinite(recent_amp) and recent_amp > 1e-6:
        amp_ratio = raw_amp / recent_amp
        if amp_ratio > 1.6:
            base_alpha *= 0.85
        elif amp_ratio > 1.25:
            base_alpha *= 0.92
        elif amp_ratio < 0.5:
            base_alpha *= 0.95

    alpha = np.full(len(shape_24h), base_alpha, dtype=float)
    risk_mask = spike_risk_24h.astype(bool)
    alpha[risk_mask] = np.minimum(alpha[risk_mask], 0.25)
    return np.clip(alpha, 0.0, 1.10)


def _apply_shape_gate(shape_24h: np.ndarray, alpha_24h: np.ndarray) -> np.ndarray:
    return shape_24h * alpha_24h


def _apply_rt_risk_adjustment(level: np.ndarray, gated_shape: np.ndarray,
                              naive_lag24: np.ndarray, spike_risk: np.ndarray) -> np.ndarray:
    pred = level + gated_shape
    return _apply_spike_fallback(pred, naive_lag24, spike_risk, blend_weight=0.65)


# =====================================================================
# 3. Spike Risk: RT 异常风险规则
# =====================================================================

def _compute_spike_risk(df: pd.DataFrame) -> pd.Series:
    """规则型 spike 风险标签。需要多条件联合触发，避免覆盖率过高。

    触发规则（满足任一）:
      1. 谷段 + 近期低价风险 (valley AND low_price_risk)
      2. 谷段 + 高新能源 (valley AND high_renewable)
      3. 极端供需缺口 (extreme_gap, 独立触发)
      4. 近期低价风险 + 高新能源 (联合触发)
    """
    risk = pd.Series(0, index=df.index, dtype=int)
    valley = df.get("valley_flag", pd.Series(0, index=df.index)).astype(bool)
    hi_renew = df.get("high_renewable_flag", pd.Series(0, index=df.index)).astype(bool)
    lo_price = df.get("low_price_risk_flag", pd.Series(0, index=df.index)).astype(bool)
    ext_gap = df.get("extreme_gap_flag", pd.Series(0, index=df.index)).astype(bool)

    risk |= (valley & lo_price).astype(int)
    risk |= (valley & hi_renew).astype(int)
    risk |= ext_gap.astype(int)
    risk |= (lo_price & hi_renew).astype(int)

    return risk


def _apply_spike_fallback(pred: np.ndarray, naive_lag24: np.ndarray,
                          spike_risk: np.ndarray, blend_weight: float = 0.5) -> np.ndarray:
    """高风险时段: 将预测与 naive (lag24h) 做加权混合，减少极端偏差。"""
    out = pred.copy()
    risk_mask = spike_risk == 1
    valid = risk_mask & np.isfinite(naive_lag24)
    out[valid] = blend_weight * pred[valid] + (1 - blend_weight) * naive_lag24[valid]
    return out


# =====================================================================
# 4. Shape 层: S1/S2/S3 混合 (复用 V4 逻辑)
# =====================================================================

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


# =====================================================================
# 5. Composite Score
# =====================================================================

def _composite_score(metrics: Dict, task: str) -> float:
    """联合选模分数 (越低越好)。"""
    mae = metrics.get("MAE", 50)
    rmse = metrics.get("RMSE", 80)
    corr = metrics.get("profile_corr", 0)
    amp = metrics.get("amplitude_err", 100)
    dacc = metrics.get("direction_acc", 0.5)

    mae_n = mae / 50.0
    rmse_n = rmse / 80.0
    corr_n = 1.0 - max(min(corr, 1.0), -1.0)
    amp_n = amp / 150.0
    dacc_n = 1.0 - max(min(dacc, 1.0), 0.0)

    if task == "da":
        return 0.35 * mae_n + 0.15 * rmse_n + 0.25 * corr_n + 0.15 * amp_n + 0.10 * dacc_n
    else:
        return 0.45 * mae_n + 0.20 * rmse_n + 0.15 * corr_n + 0.10 * amp_n + 0.10 * dacc_n


def _plot_day_overlay_generic(results: pd.DataFrame, days: List, title: str,
                              filename: str, name: str, pred_col: str,
                              pred_label: str, out_dir: Path):
    _setup_cn_font()
    n = len(days)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    hours = list(range(24))
    last_i = 0
    for i, d in enumerate(sorted(days)):
        ax = axes[i // cols][i % cols]
        day_data = results.loc[str(d)]
        if len(day_data) != 24:
            continue
        last_i = i

        ax.plot(hours, day_data["actual"].values, "k-", linewidth=2.0, label="实际", zorder=3)
        ax.plot(hours, day_data["pred_lgb"].values, "#1976D2", linewidth=1.3, alpha=0.8, label="LGB")
        ax.plot(hours, day_data[pred_col].values, "#E91E63", linewidth=1.5, alpha=0.9, label=pred_label)

        ax.set_title(str(d), fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for j in range(last_i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"{name.upper()} — {title}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out_dir / filename)


def _plot_all_generic(results: pd.DataFrame, name: str, pred_col: str,
                      pred_label: str, out_dir: Path):
    for cat, title in [("typical", "典型日曲线叠图"),
                       ("high", "极端高价日曲线叠图"),
                       ("low", "极端低价日曲线叠图")]:
        days = _select_days(results.rename(columns={pred_col: "pred_v5"}), cat, 6)
        _plot_day_overlay_generic(
            results, days, title, f"{name}_{cat}_days.png",
            name, pred_col, pred_label, out_dir,
        )


def _train_task_v6_da():
    name = "da"
    target_col = "target_da_clearing_price"
    logger.info("=" * 60)
    logger.info("V6 DA LEVEL: %s", name.upper())
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

    logger.info("--- [B] Adaptive Day-Level ---")
    daily = _build_day_level_features(df, target_col)
    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["target_daily_amp"] = daily_amp

    _, level_pred_train, level_pred_test, _ = _train_day_level(daily, train_dates, test_dates, params)
    actual_daily_test = daily.loc[test_dates, "target_daily_mean"].values

    n_val = max(int(len(train_dates) * 0.2), 5)
    val_dates_lev = train_dates[-n_val:]
    val_actual = daily.loc[val_dates_lev, "target_daily_mean"].values
    val_model = level_pred_train.loc[val_dates_lev].values
    val_fast = daily.loc[val_dates_lev, "daily_mean_ema2"].values
    val_slow = daily.loc[val_dates_lev, "daily_mean_ema5"].values
    val_regime = _compute_regime_flag(daily.loc[val_dates_lev]).values

    best_weights, gamma_min, gamma_max, val_mae = _search_da_v6_level_params(
        val_actual, val_model, val_fast, val_slow, val_regime
    )
    logger.info(
        "  DA V6 level weights: model=%.2f fast=%.2f slow=%.2f | gamma=[%.2f, %.2f] | val MAE=%.2f",
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

    logger.info("--- [C] Shape Layer ---")
    params_shape = params.copy()
    params_shape["objective"] = "huber"
    ds_t = lgb.Dataset(train_df[feature_cols], label=train_df["target_hourly_dev"])
    ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_hourly_dev"], reference=ds_t)
    shape_model = lgb.train(
        params_shape, ds_t, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_t, ds_v], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    shape_pred_train = shape_model.predict(train_df[feature_cols])
    actual_dev_std = train_df["target_hourly_dev"].std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0)

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
    logger.info("  Shape weights: S1=%.2f S2=%.2f S3=%.2f | val_shape_r=%.3f", w1, w2, w3, best_sr)

    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_raw = np.where(
        can_mix,
        w1 * s1_test + w2 * s2_test + w3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, s2_test),
    )

    logger.info("--- [D] Conditional Amplitude Scaling ---")
    _, amp_pred_test = _train_amplitude_model(daily, train_dates, test_dates, params)
    amp_pred_map = dict(zip(test_dates, amp_pred_test.values))
    shape_conf_map = _compute_daily_shape_confidence(test_df, s1_test, s2_test, s3_test)

    level_diag_rows = []
    shape_scaled = np.copy(shape_raw)
    for i, d in enumerate(test_dates):
        mask = test_df["date"].values == d
        if mask.sum() != 24:
            continue
        signal_spread = np.nanmax([
            abs(test_model[i] - test_fast[i]),
            abs(test_model[i] - test_slow[i]),
            abs(test_fast[i] - test_slow[i]),
        ])
        level_conf = float(np.clip(1.0 - signal_spread / 30.0, 0.0, 1.0))
        shape_conf = shape_conf_map.get(d, 0.5)
        target_amp = amp_pred_map.get(d, np.nan)
        shape_scaled[mask] = _conditional_scale_shape(
            shape_raw[mask], target_amp, level_conf, shape_conf, int(test_regime[i])
        )
        level_diag_rows.append({
            "date": d,
            "actual_mean": actual_daily_test[i],
            "level_pred": level_corrected_test[i],
            "level_err": level_corrected_test[i] - actual_daily_test[i],
            "amp_actual": daily_amp.loc[d],
            "amp_pred": target_amp,
            "regime_flag": int(test_regime[i]),
            "gamma_used": gamma_trace[i],
            "level_conf": level_conf,
            "shape_conf": shape_conf,
        })

    pred_v6 = level_per_hour + shape_scaled
    actual_test = test_df[target_col].values
    results = pd.DataFrame({
        "actual": actual_test,
        "pred_lgb": pred_lgb,
        "pred_v6": pred_v6,
    }, index=test_df.index)
    results.index.name = "ts"
    results.to_csv(V6_DA_DIR / "da_result.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V6_DA_Level", "pred_v6")]:
        m = _compute_metrics(actual_test, results[col].values)
        sr = compute_shape_report(actual_test, results[col].values, test_df.index)
        score = _composite_score({**m, **sr}, name)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)
        logger.info(
            "  %-18s MAE=%.2f RMSE=%.2f | corr=%.3f amp_err=%.1f dir_acc=%.3f | score=%.4f",
            label, m["MAE"], m["RMSE"], sr["profile_corr"], sr["amplitude_err"],
            sr["direction_acc"], score,
        )

    pd.DataFrame(rows).to_csv(V6_DA_DIR / "summary.csv", index=False)
    pd.DataFrame(level_diag_rows).set_index("date").to_csv(V6_DA_DIR / "da_level_diag.csv")
    _plot_all_generic(results, name, "pred_v6", "V6 DA", V6_DA_DIR)
    return results, rows


def _train_task_v6_rt():
    name = "rt"
    target_col = "target_rt_clearing_price"
    logger.info("=" * 60)
    logger.info("V6 RT GATED SHAPE: %s", name.upper())
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

    logger.info("--- [B] Fast Level Tracking ---")
    daily = _build_day_level_features(df, target_col)
    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["target_daily_amp"] = daily_amp

    n_val = max(int(len(train_dates) * 0.2), 5)
    val_dates_lev = train_dates[-n_val:]
    val_actual = daily.loc[val_dates_lev, "target_daily_mean"].values
    val_fast = daily.loc[val_dates_lev, "daily_mean_ema2"].values
    val_slow = daily.loc[val_dates_lev, "daily_mean_ema5"].values
    val_regime = _compute_regime_flag(daily.loc[val_dates_lev]).values

    best_weights, gamma_min, gamma_max, val_mae = _search_rt_v6_level_params(
        val_actual, val_fast, val_slow, val_regime
    )
    logger.info(
        "  RT V6 level weights: fast=%.2f slow=%.2f | gamma=[%.2f, %.2f] | val MAE=%.2f",
        best_weights[1], best_weights[2], gamma_min, gamma_max, val_mae,
    )

    train_actual_daily = daily.loc[train_dates, "target_daily_mean"].values
    nan_model_train = np.full(len(train_dates), np.nan)
    train_fast = daily.loc[train_dates, "daily_mean_ema2"].values
    train_slow = daily.loc[train_dates, "daily_mean_ema5"].values
    train_regime = _compute_regime_flag(daily.loc[train_dates]).values
    train_level_corr, _ = _simulate_adaptive_level(
        train_actual_daily, nan_model_train, train_fast, train_slow,
        train_regime, best_weights, gamma_min, gamma_max,
    )

    actual_daily_test = daily.loc[test_dates, "target_daily_mean"].values
    prev_err = train_actual_daily[-1] - train_level_corr[-1] if np.isfinite(train_level_corr[-1]) else 0.0
    nan_model_test = np.full(len(test_dates), np.nan)
    test_fast = daily.loc[test_dates, "daily_mean_ema2"].values
    test_slow = daily.loc[test_dates, "daily_mean_ema5"].values
    test_regime = _compute_regime_flag(daily.loc[test_dates]).values
    level_corrected_test, gamma_trace = _simulate_adaptive_level(
        actual_daily_test, nan_model_test, test_fast, test_slow,
        test_regime, best_weights, gamma_min, gamma_max, initial_prev_err=prev_err
    )
    level_map = dict(zip(test_dates, level_corrected_test))
    level_per_hour = np.array([level_map.get(d, np.nan) for d in test_df["date"]])

    logger.info("--- [C] Shape Layer ---")
    params_shape = params.copy()
    params_shape["objective"] = "huber"
    ds_t = lgb.Dataset(train_df[feature_cols], label=train_df["target_hourly_dev"])
    ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_hourly_dev"], reference=ds_t)
    shape_model = lgb.train(
        params_shape, ds_t, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_t, ds_v], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    shape_pred_train = shape_model.predict(train_df[feature_cols])
    actual_dev_std = train_df["target_hourly_dev"].std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0)

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

    best_sw, best_sr = (0.45, 0.35, 0.20), -1.0
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
    logger.info("  Shape weights: S1=%.2f S2=%.2f S3=%.2f | val_shape_r=%.3f", w1, w2, w3, best_sr)

    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_raw = np.where(
        can_mix,
        w1 * s1_test + w2 * s2_test + w3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, s2_test),
    )

    logger.info("--- [D] Shape Gating + Risk Adjustment ---")
    spike_risk = _compute_spike_risk(test_df).values
    naive_col = "rt_clearing_price_lag24h" if "rt_clearing_price_lag24h" in test_df.columns \
        else "da_clearing_price_d0"
    naive_vals = test_df[naive_col].values if naive_col in test_df.columns \
        else np.full(len(test_df), np.nan)

    shape_conf_map = _compute_daily_shape_confidence(test_df, s1_test, s2_test, s3_test)
    gated_shape = np.copy(shape_raw)
    alpha_trace = np.full(len(test_df), np.nan)
    diag_rows = []

    for i, d in enumerate(test_dates):
        mask = test_df["date"].values == d
        if mask.sum() != 24:
            continue
        recent_amp = daily.loc[d, "daily_amp_lag3_avg"]
        risk_day = spike_risk[mask]
        shape_conf = shape_conf_map.get(d, 0.5)

        day_shape = np.copy(shape_raw[mask])
        risk_ratio = float(np.mean(risk_day))
        if risk_ratio < 0.15 and shape_conf > 0.55 and np.isfinite(recent_amp) and recent_amp > 0:
            mildly_scaled = _scale_shape(day_shape, recent_amp)
            day_shape = 0.2 * day_shape + 0.8 * mildly_scaled

        alpha_day = _compute_shape_gate_alpha(day_shape, risk_day, shape_conf, recent_amp)
        gated_shape[mask] = _apply_shape_gate(day_shape, alpha_day)
        alpha_trace[mask] = alpha_day

        diag_rows.append({
            "date": d,
            "actual_mean": actual_daily_test[i],
            "level_pred": level_corrected_test[i],
            "level_err": level_corrected_test[i] - actual_daily_test[i],
            "gamma_used": gamma_trace[i],
            "regime_flag": int(test_regime[i]),
            "shape_conf": shape_conf,
            "alpha_mean": float(np.mean(alpha_day)),
            "spike_risk_ratio": risk_ratio,
            "recent_amp": recent_amp,
        })

    pred_v6 = _apply_rt_risk_adjustment(level_per_hour, gated_shape, naive_vals, spike_risk)

    actual_test = test_df[target_col].values
    results = pd.DataFrame({
        "actual": actual_test,
        "pred_lgb": pred_lgb,
        "pred_v6": pred_v6,
        "alpha_t": alpha_trace,
        "spike_risk": spike_risk,
    }, index=test_df.index)
    results.index.name = "ts"
    results.to_csv(V6_RT_DIR / "rt_result.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V6_RT_GatedShape", "pred_v6")]:
        m = _compute_metrics(actual_test, results[col].values)
        sr = compute_shape_report(actual_test, results[col].values, test_df.index)
        score = _composite_score({**m, **sr}, name)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)
        logger.info(
            "  %-18s MAE=%.2f RMSE=%.2f | corr=%.3f amp_err=%.1f dir_acc=%.3f | score=%.4f",
            label, m["MAE"], m["RMSE"], sr["profile_corr"], sr["amplitude_err"],
            sr["direction_acc"], score,
        )

    pd.DataFrame(rows).to_csv(V6_RT_DIR / "summary.csv", index=False)
    pd.DataFrame(diag_rows).set_index("date").to_csv(V6_RT_DIR / "rt_level_diag.csv")
    results[["alpha_t", "spike_risk", "pred_v6"]].to_csv(V6_RT_DIR / "alpha_diag.csv")
    _plot_all_generic(results, name, "pred_v6", "V6 RT", V6_RT_DIR)
    return results, rows


def run_v6_all():
    all_rows = []
    _, da_rows = _train_task_v6_da()
    all_rows.extend(da_rows)
    _, rt_rows = _train_task_v6_rt()
    all_rows.extend(rt_rows)

    summary = pd.DataFrame(all_rows).sort_values(["task", "composite_score"])
    summary.to_csv(OUTPUT_DIR / "v6_compare_summary.csv", index=False)
    logger.info("\n=== V6 Summary ===")
    logger.info("\n%s", summary.to_string(index=False))
    return summary


# =====================================================================
# 主训练流程
# =====================================================================

def _train_task(name: str, target_col: str):
    logger.info("=" * 60)
    logger.info("V5 PROFILE FORECAST: %s", name.upper())
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

    # ── A. 逐点 LGB baseline (与 V4 相同) ─────────────
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
    logger.info("  LGB best iter: %d", lgb_model.best_iteration)

    # ── B. Day-Level Model ─────────────────────────────
    logger.info("--- [B] Day-Level Model ---")
    daily = _build_day_level_features(df, target_col)

    daily_amp = df.groupby("date")[target_col].apply(lambda x: x.max() - x.min())
    daily["target_daily_amp"] = daily_amp

    level_model, level_pred_train, level_pred_test, level_feat_cols = \
        _train_day_level(daily, train_dates, test_dates, params)

    level_mae = np.mean(np.abs(
        daily.loc[test_dates, "target_daily_mean"].values - level_pred_test.values))
    logger.info("  Day-level MAE: %.2f", level_mae)

    naive_ema = daily["daily_mean_ema3"]
    naive_ema_test = naive_ema.loc[test_dates].values
    actual_daily_test = daily.loc[test_dates, "target_daily_mean"].values

    # 混合: day-level model + naive_ema (在验证集搜索权重)
    n_val = max(int(len(train_dates) * 0.2), 5)
    val_dates_lev = train_dates[-n_val:]
    val_actual = daily.loc[val_dates_lev, "target_daily_mean"].values
    val_lgb_lev = level_pred_train.loc[val_dates_lev].values
    val_ema = naive_ema.loc[val_dates_lev].values

    best_w, best_lev_mae = 1.0, np.inf
    for w in np.arange(0, 1.01, 0.05):
        blended = w * val_lgb_lev + (1 - w) * val_ema
        valid = np.isfinite(blended) & np.isfinite(val_actual)
        if valid.sum() < 3:
            continue
        mae = np.mean(np.abs(val_actual[valid] - blended[valid]))
        if mae < best_lev_mae:
            best_lev_mae = mae
            best_w = round(w, 2)
    logger.info("  Level blend: w_model=%.2f, w_ema=%.2f", best_w, 1 - best_w)

    level_final_test = best_w * level_pred_test.values + (1 - best_w) * naive_ema_test

    # 在线误差校正
    best_gamma, best_gamma_mae = 0.0, np.inf
    for gamma in np.arange(0, 1.01, 0.05):
        corrected = np.copy(best_w * val_lgb_lev + (1 - best_w) * val_ema)
        for i in range(1, len(corrected)):
            if np.isfinite(val_actual[i - 1]):
                corrected[i] += gamma * (val_actual[i - 1] - corrected[i - 1])
        valid = np.isfinite(corrected) & np.isfinite(val_actual)
        mae = np.mean(np.abs(val_actual[valid] - corrected[valid]))
        if mae < best_gamma_mae:
            best_gamma_mae = mae
            best_gamma = round(gamma, 2)
    logger.info("  Error correction gamma: %.2f", best_gamma)

    # 应用到测试集
    last_train_actual = daily.loc[train_dates[-1], "target_daily_mean"]
    last_train_pred = best_w * level_pred_train.loc[train_dates[-1]] + \
                      (1 - best_w) * naive_ema.loc[train_dates[-1]]
    prev_err = last_train_actual - last_train_pred

    level_corrected_test = np.copy(level_final_test)
    for i in range(len(level_corrected_test)):
        if np.isfinite(prev_err):
            level_corrected_test[i] += best_gamma * prev_err
        if np.isfinite(actual_daily_test[i]) and np.isfinite(level_corrected_test[i]):
            prev_err = actual_daily_test[i] - level_corrected_test[i]

    level_corr_mae = np.nanmean(np.abs(actual_daily_test - level_corrected_test))
    logger.info("  Corrected level MAE: %.2f (raw: %.2f)", level_corr_mae, level_mae)

    level_map = dict(zip(test_dates, level_corrected_test))
    level_per_hour = np.array([level_map.get(d, np.nan) for d in test_df["date"]])

    # ── C. Shape 层 ────────────────────────────────────
    logger.info("--- [C] Shape Layer ---")
    params_shape = params.copy()
    params_shape["objective"] = "huber"
    ds_t = lgb.Dataset(train_df[feature_cols], label=train_df["target_hourly_dev"])
    ds_v = lgb.Dataset(test_df[feature_cols], label=test_df["target_hourly_dev"], reference=ds_t)
    shape_model = lgb.train(
        params_shape, ds_t, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_t, ds_v], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    shape_pred_train = shape_model.predict(train_df[feature_cols])
    actual_dev_std = train_df["target_hourly_dev"].std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0)

    full_df = pd.concat([train_df, test_df])
    s1_test, s2_test, s3_test = _compute_shape_sources(
        test_df, full_df, target_col, shape_model, feature_cols, scale_factor)

    # OOF shape 用于 weight search
    n_train = len(train_df)
    val_h_start = int(n_train * 0.75)
    pre_val_df = train_df.iloc[:val_h_start]
    val_df = train_df.iloc[val_h_start:]

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
        train_df, full_df, target_col, shape_model, feature_cols, scale_factor)

    # 用 corrected level 在验证集搜索 shape weights
    # 重建训练期 corrected level
    train_level_blend = best_w * level_pred_train.values + (1 - best_w) * naive_ema.loc[train_dates].values
    train_level_corr = np.copy(train_level_blend)
    pe = 0.0
    train_actual_daily = daily.loc[train_dates, "target_daily_mean"].values
    for i in range(1, len(train_level_corr)):
        if np.isfinite(pe):
            train_level_corr[i] += best_gamma * pe
        if np.isfinite(train_actual_daily[i]):
            pe = train_actual_daily[i] - train_level_corr[i]

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
    logger.info("  Shape weights: S1=%.2f S2=%.2f S3=%.2f  val_shape_r=%.3f", w1, w2, w3, best_sr)

    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_raw = np.where(can_mix,
                         w1 * s1_test + w2 * s2_test + w3 * s3_test,
                         np.where(np.isfinite(s1_test), s1_test, s2_test))

    # ── D. Amplitude Scaler ────────────────────────────
    logger.info("--- [D] Amplitude Scaler ---")
    amp_model, amp_pred_test = _train_amplitude_model(daily, train_dates, test_dates, params)

    amp_pred_map = dict(zip(test_dates, amp_pred_test.values))
    shape_scaled = np.copy(shape_raw)
    for d in test_dates:
        mask = test_df["date"].values == d
        if mask.sum() != 24:
            continue
        target_amp = amp_pred_map.get(d, np.nan)
        if np.isfinite(target_amp) and target_amp > 0:
            shape_scaled[mask] = _scale_shape(shape_raw[mask], target_amp)

    pred_v5_raw = level_per_hour + shape_raw
    pred_v5_scaled = level_per_hour + shape_scaled

    # ── E. RT Spike Fallback ───────────────────────────
    if name == "rt":
        logger.info("--- [E] RT Spike Fallback ---")
        spike_risk = _compute_spike_risk(test_df)
        n_risk = spike_risk.sum()
        logger.info("  Spike risk hours: %d / %d (%.1f%%)",
                    n_risk, len(spike_risk), n_risk / len(spike_risk) * 100)

        naive_col = "rt_clearing_price_lag24h" if "rt_clearing_price_lag24h" in test_df.columns \
            else "da_clearing_price_d0"
        naive_vals = test_df[naive_col].values if naive_col in test_df.columns \
            else np.full(len(test_df), np.nan)

        pred_v5_scaled = _apply_spike_fallback(
            pred_v5_scaled, naive_vals, spike_risk.values, blend_weight=0.6)

    # ── 汇总 ──────────────────────────────────────────
    actual_test = test_df[target_col].values
    results = pd.DataFrame({
        "actual": actual_test,
        "pred_lgb": pred_lgb,
        "pred_v5": pred_v5_scaled,
    }, index=test_df.index)
    results.index.name = "ts"
    results.to_csv(V5_DIR / f"{name}_result.csv")

    rows = []
    for label, col in [("LGB_Huber", "pred_lgb"), ("V5_Profile", "pred_v5")]:
        m = _compute_metrics(actual_test, results[col].values)
        sr = compute_shape_report(actual_test, results[col].values, test_df.index)
        score = _composite_score({**m, **sr}, name)
        logger.info("  %-18s MAE=%.2f RMSE=%.2f | corr=%.3f amp_err=%.1f dir_acc=%.3f | score=%.4f",
                    label, m["MAE"], m["RMSE"],
                    sr["profile_corr"], sr["amplitude_err"], sr["direction_acc"], score)
        row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
        row.update(sr)
        row["composite_score"] = round(score, 4)
        rows.append(row)

    # Level 跟踪诊断
    level_diag = pd.DataFrame({
        "actual_mean": actual_daily_test,
        "level_pred": level_corrected_test,
        "level_err": level_corrected_test - actual_daily_test,
        "amp_actual": daily_amp.loc[test_dates].values,
        "amp_pred": amp_pred_test.values,
    }, index=test_dates)
    level_diag.to_csv(V5_DIR / f"{name}_level_diag.csv")

    return results, rows


# =====================================================================
# 可视化
# =====================================================================

def _select_days(results, category, n=6):
    daily = results.groupby(results.index.date).agg(
        actual_mean=("actual", "mean"),
        mae_v5=("actual", lambda x: np.nan),
    )
    daily["mae_v5"] = results.groupby(results.index.date).apply(
        lambda g: np.mean(np.abs(g["actual"] - g["pred_v5"])))

    if category == "typical":
        med = daily["mae_v5"].median()
        daily["dist"] = (daily["mae_v5"] - med).abs()
        return list(daily.nsmallest(n, "dist").index)
    elif category == "high":
        return list(daily.nlargest(n, "actual_mean").index)
    elif category == "low":
        return list(daily.nsmallest(n, "actual_mean").index)
    return []


def _plot_day_overlay(results, days, title, filename, name):
    _setup_cn_font()
    n = len(days)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    hours = list(range(24))
    last_i = 0
    for i, d in enumerate(sorted(days)):
        ax = axes[i // cols][i % cols]
        day_data = results.loc[str(d)]
        if len(day_data) != 24:
            continue
        last_i = i

        ax.plot(hours, day_data["actual"].values, "k-", linewidth=2.0, label="实际", zorder=3)
        ax.plot(hours, day_data["pred_lgb"].values, "#1976D2", linewidth=1.3, alpha=0.8, label="LGB")
        ax.plot(hours, day_data["pred_v5"].values, "#E91E63", linewidth=1.5, alpha=0.9, label="V5 Profile")

        ax.set_title(str(d), fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for j in range(last_i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"{name.upper()} — {title}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(V5_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", filename)


def _plot_all(results, name):
    for cat, title in [("typical", "典型日曲线叠图"),
                       ("high", "极端高价日曲线叠图"),
                       ("low", "极端低价日曲线叠图")]:
        days = _select_days(results, cat, 6)
        _plot_day_overlay(results, days, title, f"{name}_{cat}_days.png", name)


# =====================================================================
# 入口
# =====================================================================

def run_all():
    all_rows = []
    for name, target_col in [("da", "target_da_clearing_price"),
                              ("rt", "target_rt_clearing_price")]:
        results, rows = _train_task(name, target_col)
        _plot_all(results, name)
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    summary = summary.sort_values(["task", "composite_score"])
    summary.to_csv(V5_DIR / "summary.csv", index=False)
    logger.info("\n=== V5 Summary ===")
    logger.info("\n%s", summary.to_string(index=False))

    # 追加: 与历史模型的联合评分对比
    _compare_with_history(summary)

    return summary


def _compare_with_history(v5_summary):
    """读取 shape_evaluation_summary, 追加 composite_score, 与 V5 合并排名。"""
    eval_path = OUTPUT_DIR / "shape_evaluation_summary.csv"
    if not eval_path.exists():
        return

    hist = pd.read_csv(eval_path)
    for _, row in hist.iterrows():
        task = row["task"]
        m = {
            "MAE": row.get("MAE", 50),
            "RMSE": row.get("RMSE", 80),
            "profile_corr": row.get("profile_corr", 0),
            "amplitude_err": row.get("amplitude_err", 100),
            "direction_acc": row.get("direction_acc", 0.5),
        }
        hist.loc[_, "composite_score"] = round(_composite_score(m, task), 4)

    hist = hist[["source", "task", "method", "MAE", "RMSE",
                 "profile_corr", "amplitude_err", "direction_acc", "composite_score"]]

    v5_rows = []
    for _, row in v5_summary.iterrows():
        v5_rows.append({
            "source": "v5_profile",
            "task": row["task"],
            "method": row["method"],
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "profile_corr": row.get("profile_corr", 0),
            "amplitude_err": row.get("amplitude_err", 0),
            "direction_acc": row.get("direction_acc", 0),
            "composite_score": row["composite_score"],
        })

    combined = pd.concat([hist, pd.DataFrame(v5_rows)], ignore_index=True)
    combined = combined.sort_values(["task", "composite_score"])

    da_top = combined[combined["task"] == "da"].head(10)
    rt_top = combined[combined["task"] == "rt"].head(10)

    logger.info("\n=== DA Top-10 (by composite score) ===")
    logger.info("\n%s", da_top.to_string(index=False))
    logger.info("\n=== RT Top-10 (by composite score) ===")
    logger.info("\n%s", rt_top.to_string(index=False))

    combined.to_csv(V5_DIR / "all_models_ranked.csv", index=False)
    logger.info("  Saved: all_models_ranked.csv")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
