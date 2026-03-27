"""
自适应 Level 跟踪 + 形状混合预测模型 V2

改进点（相比 model_shape_final.py 的 V1）：
  - Level 层：LGB 预测 + Naive 跟踪信号多源混合 + 在线误差校正
  - 解决 V1 中 Level LGB 锚定训练均值、无法跟随测试期价格趋势的问题

架构：
  Level_corrected = ErrorCorrect( blend(LGB, Naive_1d, Naive_3d, Naive_EMA) )
  Shape_blend = w1*S1 + w2*S2 + w3*S3    (与 V1 完全一致)
  pred = Level_corrected + Shape_blend

Naive Level 信号（均使用预测时间点可获取的数据，无泄漏）：
  Naive_1d  : 昨日实际均价
  Naive_3d  : 近 3 日实际均价的简单平均
  Naive_EMA : 近期实际均价的指数加权移动平均 (span=3)

可视化结果输出到 output/viz/v2_*
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

logger = logging.getLogger(__name__)
VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities (shared with V1)
# ---------------------------------------------------------------------------

def _setup_cn_font():
    paths = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]
    for p in paths:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["mathtext.fontset"] = "cm"


def _load_tuned_params(name: str) -> Dict:
    path = PARAMS_DIR / f"tuning_{name}_best_params.json"
    with open(path) as f:
        return json.load(f)


def _build_daily_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df_feat = df[feature_cols].copy()
    df_feat["date"] = df_feat.index.date
    agg_funcs = {}
    for col in feature_cols:
        if col in ("hour", "day_of_week", "is_weekend", "month"):
            continue
        agg_funcs[col] = ["mean"]
    daily = df_feat.groupby("date").agg(agg_funcs)
    daily.columns = [f"{c[0]}_dmean" for c in daily.columns]
    for cal_col in ("day_of_week", "is_weekend", "month"):
        if cal_col in feature_cols:
            daily[cal_col] = df_feat.groupby("date")[cal_col].first()
    return daily


def _compute_shape_sources(
    target_df: pd.DataFrame,
    full_history_df: pd.DataFrame,
    target_col: str, naive_col: str,
    shape_model: lgb.Booster, shape_feat_cols: List[str],
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = target_df.index.date
    target_dates = sorted(set(dates))
    all_dates = sorted(set(full_history_df.index.date))
    all_date_idx = {d: i for i, d in enumerate(all_dates)}
    daily_mean_actual = full_history_df.groupby(full_history_df.index.date)[target_col].mean()

    hourly_dev_by_date = {}
    for d in all_dates:
        mask = full_history_df.index.date == d
        day_data = full_history_df.loc[mask, target_col].values
        day_mean = daily_mean_actual.get(d, np.nan)
        hourly_dev_by_date[d] = day_data - day_mean

    s1 = np.full(len(target_df), np.nan)
    for d in target_dates:
        idx = all_date_idx[d]
        prev_days = [all_dates[idx - k] for k in range(1, 4) if idx - k >= 0]
        if not prev_days:
            continue
        mask = dates == d
        n_hours = mask.sum()
        avg_dev = np.zeros(n_hours)
        count = 0
        for pd_ in prev_days:
            dev = hourly_dev_by_date.get(pd_, None)
            if dev is not None and len(dev) == n_hours:
                avg_dev += dev
                count += 1
        if count > 0:
            avg_dev /= count
            s1[mask] = avg_dev

    s2 = shape_model.predict(target_df[shape_feat_cols]) * scale_factor

    s3 = np.full(len(target_df), np.nan)
    for d in target_dates:
        idx = all_date_idx[d]
        if idx == 0:
            continue
        prev_d = all_dates[idx - 1]
        mask = dates == d
        n_hours = mask.sum()
        dev = hourly_dev_by_date.get(prev_d, None)
        if dev is not None and len(dev) == n_hours:
            s3[mask] = dev

    return s1, s2, s3


def _compute_shape_corr(results_df, actual_col, pred_col):
    corr_daily = []
    for d in results_df.index.normalize().unique():
        day = results_df.loc[str(d.date())]
        if len(day) == 24:
            c = np.corrcoef(day[actual_col].values, day[pred_col].values)[0, 1]
            if np.isfinite(c):
                corr_daily.append(c)
    return np.mean(corr_daily) if corr_daily else np.nan


def _compute_shape_corr_arrays(actual: np.ndarray, pred: np.ndarray, dates: np.ndarray) -> float:
    corr_daily = []
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() == 24:
            a = actual[mask]
            p = pred[mask]
            if np.std(a) > 1e-6 and np.std(p) > 1e-6:
                c = np.corrcoef(a, p)[0, 1]
                if np.isfinite(c):
                    corr_daily.append(c)
    return np.mean(corr_daily) if corr_daily else -1.0


# ---------------------------------------------------------------------------
# V2 新增: 自适应 Level 信号与校正
# ---------------------------------------------------------------------------

def _compute_naive_level_signals(daily_actual_mean: pd.Series) -> Dict[str, pd.Series]:
    """计算三种 Naive Level 跟踪信号 (均 shift(1) 确保无泄漏)。"""
    naive_1d = daily_actual_mean.shift(1)
    naive_3d = daily_actual_mean.rolling(3, min_periods=1).mean().shift(1)
    naive_ema = daily_actual_mean.ewm(span=3, adjust=False).mean().shift(1)
    return {"naive_1d": naive_1d, "naive_3d": naive_3d, "naive_ema": naive_ema}


def _grid_search_level_weights(
    level_lgb: np.ndarray,
    naive_signals: Dict[str, np.ndarray],
    actual: np.ndarray,
    step: float = 0.05,
) -> Tuple[Dict[str, float], float]:
    """在验证集上搜索 Level 混合权重，最小化日均价 MAE。

    权重约束: w_lgb + w_1d + w_3d + w_ema = 1, 各 >= 0
    """
    signal_names = ["lgb", "naive_1d", "naive_3d", "naive_ema"]
    signals = [level_lgb, naive_signals["naive_1d"],
               naive_signals["naive_3d"], naive_signals["naive_ema"]]

    best_mae = np.inf
    best_w = {n: 0.0 for n in signal_names}
    best_w["lgb"] = 1.0

    for w_lgb in np.arange(0, 1.01, step):
        for w_1d in np.arange(0, 1.01 - w_lgb, step):
            for w_3d in np.arange(0, 1.01 - w_lgb - w_1d, step):
                w_ema = round(1.0 - w_lgb - w_1d - w_3d, 2)
                if w_ema < -1e-6:
                    continue
                blended = (w_lgb * signals[0] + w_1d * signals[1]
                           + w_3d * signals[2] + w_ema * signals[3])
                valid = np.isfinite(blended) & np.isfinite(actual)
                if valid.sum() < 5:
                    continue
                mae = np.mean(np.abs(actual[valid] - blended[valid]))
                if mae < best_mae:
                    best_mae = mae
                    best_w = {"lgb": round(w_lgb, 2), "naive_1d": round(w_1d, 2),
                              "naive_3d": round(w_3d, 2), "naive_ema": round(w_ema, 2)}

    return best_w, best_mae


def _search_gamma(
    level_blended: np.ndarray,
    actual: np.ndarray,
    step: float = 0.05,
) -> Tuple[float, float]:
    """搜索在线误差校正系数 gamma。

    模拟在线过程: corrected[i] = blended[i] + gamma * (actual[i-1] - blended[i-1])
    """
    best_gamma = 0.0
    best_mae = np.inf

    for gamma in np.arange(0, 1.01, step):
        corrected = np.copy(level_blended)
        for i in range(1, len(corrected)):
            if np.isfinite(actual[i - 1]) and np.isfinite(level_blended[i - 1]):
                error_prev = actual[i - 1] - corrected[i - 1]
                corrected[i] = level_blended[i] + gamma * error_prev
        valid = np.isfinite(corrected) & np.isfinite(actual)
        if valid.sum() < 5:
            continue
        mae = np.mean(np.abs(actual[valid] - corrected[valid]))
        if mae < best_mae:
            best_mae = mae
            best_gamma = round(gamma, 2)

    return best_gamma, best_mae


def _apply_online_error_correction(
    level_blended_test: np.ndarray,
    actual_test: np.ndarray,
    gamma: float,
    last_train_error: float,
) -> np.ndarray:
    """在测试集上顺序应用在线误差校正。

    对于第 i 天: corrected[i] = blended[i] + gamma * error[i-1]
    第 0 天使用训练末尾的残差。
    """
    corrected = np.copy(level_blended_test)
    prev_error = last_train_error
    for i in range(len(corrected)):
        if np.isfinite(prev_error):
            corrected[i] = level_blended_test[i] + gamma * prev_error
        if np.isfinite(actual_test[i]) and np.isfinite(corrected[i]):
            prev_error = actual_test[i] - corrected[i]
    return corrected


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

def run_shape_v2(name: str, target_col: str, naive_col: str):
    logger.info("=" * 60)
    logger.info("SHAPE V2 MODEL (Adaptive Level): %s", name.upper())
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

    # ==================================================================
    # Stage 1: Level LGB (与 V1 相同)
    # ==================================================================
    logger.info("--- Stage 1: Level LGB ---")

    daily_train = _build_daily_features(train_df, feature_cols)
    daily_test = _build_daily_features(test_df, feature_cols)

    daily_target_train = train_df.groupby("date")[target_col].mean()
    daily_target_test = test_df.groupby("date")[target_col].mean()

    daily_train = daily_train.loc[daily_target_train.index]
    daily_test = daily_test.loc[daily_target_test.index]
    level_feat_cols = list(daily_train.columns)

    params_level = params.copy()
    params_level["objective"] = "regression"
    params_level["num_leaves"] = min(params_level.get("num_leaves", 31), 31)
    params_level["min_child_samples"] = 5

    dt_train = lgb.Dataset(daily_train[level_feat_cols], label=daily_target_train)
    dt_val = lgb.Dataset(daily_test[level_feat_cols], label=daily_target_test, reference=dt_train)
    level_model = lgb.train(
        params_level, dt_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dt_train, dt_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )
    logger.info("  Level LGB best iteration: %d", level_model.best_iteration)

    level_lgb_train = pd.Series(
        level_model.predict(daily_train[level_feat_cols]),
        index=daily_target_train.index,
    )
    level_lgb_test = pd.Series(
        level_model.predict(daily_test[level_feat_cols]),
        index=daily_target_test.index,
    )
    level_lgb_mae = np.mean(np.abs(daily_target_test.values - level_lgb_test.values))
    logger.info("  Level LGB MAE (test): %.2f", level_lgb_mae)

    # ==================================================================
    # Stage 1b: 自适应 Level — Naive 信号混合 + 误差校正
    # ==================================================================
    logger.info("--- Stage 1b: Adaptive Level (Naive blend + error correction) ---")

    all_daily_actual = df.groupby("date")[target_col].mean()
    naive_signals_all = _compute_naive_level_signals(all_daily_actual)

    # -- 验证集: 训练集后 25% 的日期 --
    train_dates = sorted(daily_target_train.index)
    n_train_days = len(train_dates)
    val_day_start = int(n_train_days * 0.75)
    val_dates = train_dates[val_day_start:]
    pre_val_dates = train_dates[:val_day_start]

    val_actual = daily_target_train.loc[val_dates].values
    val_lgb = level_lgb_train.loc[val_dates].values
    val_naive = {k: v.loc[val_dates].values for k, v in naive_signals_all.items()}

    # -- Grid search Level weights --
    level_weights, val_level_mae = _grid_search_level_weights(
        val_lgb, val_naive, val_actual, step=0.05,
    )
    logger.info("  Level weights: %s  val_MAE=%.2f", level_weights, val_level_mae)

    # -- 构建验证集的 blended level 用于搜索 gamma --
    val_blended = (level_weights["lgb"] * val_lgb
                   + level_weights["naive_1d"] * val_naive["naive_1d"]
                   + level_weights["naive_3d"] * val_naive["naive_3d"]
                   + level_weights["naive_ema"] * val_naive["naive_ema"])

    gamma, val_corrected_mae = _search_gamma(val_blended, val_actual, step=0.05)
    logger.info("  Error correction gamma=%.2f  val_corrected_MAE=%.2f", gamma, val_corrected_mae)

    # -- 在测试集上应用 --
    test_dates = sorted(daily_target_test.index)
    test_lgb = level_lgb_test.loc[test_dates].values
    test_naive = {k: v.reindex(test_dates).values for k, v in naive_signals_all.items()}

    test_blended = (level_weights["lgb"] * test_lgb
                    + level_weights["naive_1d"] * test_naive["naive_1d"]
                    + level_weights["naive_3d"] * test_naive["naive_3d"]
                    + level_weights["naive_ema"] * test_naive["naive_ema"])

    # 训练集末尾的误差 (用于测试集第一天的校正)
    last_train_day = train_dates[-1]
    last_train_blended = (level_weights["lgb"] * level_lgb_train.loc[last_train_day]
                          + level_weights["naive_1d"] * naive_signals_all["naive_1d"].loc[last_train_day]
                          + level_weights["naive_3d"] * naive_signals_all["naive_3d"].loc[last_train_day]
                          + level_weights["naive_ema"] * naive_signals_all["naive_ema"].loc[last_train_day])
    last_train_error = daily_target_train.loc[last_train_day] - last_train_blended

    test_actual_daily = daily_target_test.loc[test_dates].values
    level_corrected_test = _apply_online_error_correction(
        test_blended, test_actual_daily, gamma, last_train_error,
    )

    corrected_mae = np.mean(np.abs(test_actual_daily - level_corrected_test))
    logger.info("  Adaptive Level MAE (test): %.2f  (LGB-only: %.2f, improvement: %.1f%%)",
                corrected_mae, level_lgb_mae,
                (level_lgb_mae - corrected_mae) / level_lgb_mae * 100)

    # 构建日期→校正Level映射
    level_corrected_map = dict(zip(test_dates, level_corrected_test))
    level_corrected_per_hour = np.array([level_corrected_map.get(d, np.nan)
                                         for d in test_df["date"]])

    # 也保留纯 LGB level 用于对比
    level_lgb_map = dict(zip(test_dates, test_lgb))
    level_lgb_per_hour = np.array([level_lgb_map.get(d, np.nan) for d in test_df["date"]])

    # ==================================================================
    # Stage 2: Shape LGB (与 V1 相同)
    # ==================================================================
    logger.info("--- Stage 2: Shape LGB (for S2 signal) ---")

    shape_target_train = train_df["target_hourly_dev"]
    shape_feat_cols = list(feature_cols)

    params_shape = params.copy()
    params_shape["objective"] = "huber"

    ds_train = lgb.Dataset(train_df[shape_feat_cols], label=shape_target_train)
    ds_val = lgb.Dataset(test_df[shape_feat_cols], label=test_df["target_hourly_dev"], reference=ds_train)
    shape_model = lgb.train(
        params_shape, ds_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(200)],
    )
    logger.info("  Shape model best iteration: %d", shape_model.best_iteration)

    shape_pred_train = shape_model.predict(train_df[shape_feat_cols])
    actual_dev_std = shape_target_train.std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0)
    logger.info("  Shape amplitude scale: actual_std=%.2f, pred_std=%.2f, factor=%.2f",
                actual_dev_std, pred_dev_std, scale_factor)

    # ==================================================================
    # Stage 2b: Shape 权重 Grid Search (S1/S2/S3, 与 V1 相同逻辑)
    # ==================================================================
    logger.info("--- Stage 2b: Shape mixing weights (shape_r target, OOF) ---")

    full_df = pd.concat([train_df, test_df])
    s1_train, _, s3_train = _compute_shape_sources(
        train_df, full_df, target_col, naive_col, shape_model, shape_feat_cols, scale_factor)
    s1_test, s2_test, s3_test = _compute_shape_sources(
        test_df, full_df, target_col, naive_col, shape_model, shape_feat_cols, scale_factor)

    n_train = len(train_df)
    val_hour_start = int(n_train * 0.75)
    val_h_slice = slice(val_hour_start, n_train)
    pre_val_df = train_df.iloc[:val_hour_start]
    val_df = train_df.iloc[val_hour_start:]

    params_oof = params.copy()
    params_oof["objective"] = "huber"
    ds_pre = lgb.Dataset(pre_val_df[shape_feat_cols], label=pre_val_df["target_hourly_dev"])
    ds_oof_v = lgb.Dataset(val_df[shape_feat_cols], label=val_df["target_hourly_dev"], reference=ds_pre)
    shape_oof = lgb.train(
        params_oof, ds_pre,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_pre, ds_oof_v], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    s2_val_oof_raw = shape_oof.predict(val_df[shape_feat_cols])
    oof_pred_std = np.std(shape_oof.predict(pre_val_df[shape_feat_cols]))
    oof_scale = min(pre_val_df["target_hourly_dev"].std() / max(oof_pred_std, 1e-6), 5.0)
    s2_val_oof = s2_val_oof_raw * oof_scale
    logger.info("  OOF shape model: iter=%d, scale=%.2f", shape_oof.best_iteration, oof_scale)

    # 验证集用自适应 Level (而非纯 LGB)
    level_train_corrected = _build_train_level_corrected(
        level_lgb_train, naive_signals_all, daily_target_train,
        level_weights, gamma, train_dates,
    )
    level_train_corrected_per_hour = np.array(
        [level_train_corrected.get(d, np.nan) for d in train_df["date"]])

    val_actual_h = train_df[target_col].values[val_h_slice]
    val_level_h = level_train_corrected_per_hour[val_h_slice]
    val_s1 = s1_train[val_h_slice]
    val_s2 = s2_val_oof
    val_s3 = s3_train[val_h_slice]
    val_dates_h = train_df.index.date[val_hour_start:n_train]

    best_w = (1.0, 0.0, 0.0)
    best_shape_r = -1.0
    grid_step = 0.05
    for w1 in np.arange(0, 1.01, grid_step):
        for w2 in np.arange(0, 1.01 - w1, grid_step):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -1e-6:
                continue
            shape_blend = w1 * val_s1 + w2 * val_s2 + w3 * val_s3
            pred_blend = val_level_h + shape_blend
            fin = np.isfinite(pred_blend)
            if fin.sum() < 48:
                continue
            sr = _compute_shape_corr_arrays(val_actual_h[fin], pred_blend[fin], val_dates_h[fin])
            if sr > best_shape_r:
                best_shape_r = sr
                best_w = (round(w1, 2), round(w2, 2), w3)

    w1, w2, w3 = best_w
    logger.info("  Shape weights: S1=%.2f, S2=%.2f, S3=%.2f  val_shape_r=%.3f",
                w1, w2, w3, best_shape_r)

    # -- 测试集 shape 混合 --
    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_mix_test = np.where(
        can_mix,
        w1 * s1_test + w2 * s2_test + w3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, s2_test),
    )

    # ==================================================================
    # 合成最终预测
    # ==================================================================
    actual_test = test_df[target_col].values
    pred_naive = test_df[naive_col].values

    # V2: Adaptive Level + Shape
    pred_v2 = level_corrected_per_hour + shape_mix_test

    # V1 baseline: LGB Level + Shape (使用相同 shape 权重)
    pred_v1 = level_lgb_per_hour + shape_mix_test

    results = pd.DataFrame({
        "actual": actual_test,
        "naive": pred_naive,
        "pred_v1_lgb_level": pred_v1,
        "pred_v2_adaptive": pred_v2,
    }, index=test_df.index)
    results.index.name = "ts"

    methods = {
        "Naive": "naive",
        "V1_LGB_Level": "pred_v1_lgb_level",
        "V2_Adaptive_Level": "pred_v2_adaptive",
    }

    logger.info("\n=== %s V2 自适应 Level 模型结果 ===", name.upper())
    summary_rows = []
    for label, col in methods.items():
        m = _compute_metrics(actual_test, results[col].values)
        shape_corr = _compute_shape_corr(results, "actual", col)
        logger.info("  %-30s MAE=%6.2f  RMSE=%6.2f  shape_r=%.3f",
                    label, m["MAE"], m["RMSE"], shape_corr)
        summary_rows.append({
            "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"],
            "sMAPE(%)": m["sMAPE(%)"], "wMAPE(%)": m["wMAPE(%)"],
            "shape_corr": round(shape_corr, 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(VIZ_DIR / f"v2_{name}_summary.csv", index=False)
    results.to_csv(VIZ_DIR / f"v2_{name}_result.csv")

    # Level 跟踪详情
    level_tracking = pd.DataFrame({
        "actual_daily_mean": test_actual_daily,
        "level_lgb": test_lgb,
        "level_blended": test_blended,
        "level_corrected": level_corrected_test,
    }, index=test_dates)
    level_tracking["lgb_error"] = level_tracking["actual_daily_mean"] - level_tracking["level_lgb"]
    level_tracking["corrected_error"] = level_tracking["actual_daily_mean"] - level_tracking["level_corrected"]
    level_tracking.to_csv(VIZ_DIR / f"v2_{name}_level_tracking.csv")
    logger.info("  Level tracking saved to v2_%s_level_tracking.csv", name)

    extra_info = {
        "level_weights": level_weights,
        "gamma": gamma,
        "shape_weights": best_w,
        "level_lgb_mae": round(level_lgb_mae, 2),
        "level_corrected_mae": round(corrected_mae, 2),
        "val_shape_r": round(best_shape_r, 4),
    }

    return results, summary, methods, level_tracking, extra_info


def _build_train_level_corrected(
    level_lgb_train: pd.Series,
    naive_signals_all: Dict[str, pd.Series],
    daily_target_train: pd.Series,
    level_weights: Dict[str, float],
    gamma: float,
    train_dates: list,
) -> Dict:
    """在训练集上重建 corrected level (用于 shape grid search 验证集)。"""
    result = {}
    prev_error = 0.0
    for i, d in enumerate(train_dates):
        lgb_val = level_lgb_train.get(d, np.nan)
        n1 = naive_signals_all["naive_1d"].get(d, np.nan)
        n3 = naive_signals_all["naive_3d"].get(d, np.nan)
        ne = naive_signals_all["naive_ema"].get(d, np.nan)
        if any(np.isnan(x) for x in [lgb_val, n1, n3, ne]):
            result[d] = lgb_val
            continue
        blended = (level_weights["lgb"] * lgb_val + level_weights["naive_1d"] * n1
                   + level_weights["naive_3d"] * n3 + level_weights["naive_ema"] * ne)
        corrected = blended + gamma * prev_error if i > 0 else blended
        result[d] = corrected
        actual = daily_target_train.get(d, np.nan)
        if np.isfinite(actual) and np.isfinite(corrected):
            prev_error = actual - corrected
    return result


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_shape_v2(results, summary, methods, level_tracking, extra_info, name):
    _setup_cn_font()

    lw = extra_info["level_weights"]
    gamma = extra_info["gamma"]

    # ── 1. Level 跟踪对比图 ─────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    dates = level_tracking.index
    ax.plot(dates, level_tracking["actual_daily_mean"], "o-", color="black",
            linewidth=1.8, markersize=4, label="实际日均价", zorder=5)
    ax.plot(dates, level_tracking["level_lgb"], "s--", color="#90CAF9",
            linewidth=1.2, markersize=3, alpha=0.8, label="V1: Level LGB")
    ax.plot(dates, level_tracking["level_corrected"], "D-", color="#E91E63",
            linewidth=1.5, markersize=3, alpha=0.9,
            label=f"V2: Adaptive Level (w_lgb={lw['lgb']}, gamma={gamma})")
    ax.set_ylabel("日均价 (元/MWh)")
    improve_pct = ((extra_info['level_lgb_mae'] - extra_info['level_corrected_mae'])
                    / extra_info['level_lgb_mae'] * 100)
    ax.set_title(f"{name.upper()} - Level 跟踪对比: LGB vs 自适应校正\n"
                 f"LGB MAE={extra_info['level_lgb_mae']:.1f}  "
                 f"Adaptive MAE={extra_info['level_corrected_mae']:.1f}  "
                 f"改善={improve_pct:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(dates, level_tracking["lgb_error"], width=0.8, alpha=0.4,
            color="#90CAF9", label="V1 LGB 误差")
    ax2.bar(dates, level_tracking["corrected_error"], width=0.4, alpha=0.8,
            color="#E91E63", label="V2 Adaptive 误差")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("误差 (实际-预测)")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"v2_{name}_level_tracking.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. 全时段预测对比 ─────────────────────────────────
    colors = {"Naive": "#BDBDBD", "V1_LGB_Level": "#90CAF9", "V2_Adaptive_Level": "#E91E63"}

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(results.index, results["actual"], color="black", linewidth=1.2,
            label="实际", zorder=5)
    for label, col in methods.items():
        s = summary[summary["method"] == label]
        ls = "--" if label == "Naive" else "-"
        lw_line = 0.6 if label == "Naive" else 1.0
        alpha = 0.4 if label == "Naive" else 0.85
        ax.plot(results.index, results[col], ls, color=colors[label],
                linewidth=lw_line, alpha=alpha,
                label=f"{label} (MAE={s['MAE'].values[0]:.1f}, r={s['shape_corr'].values[0]:.3f})")
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} - V2 自适应 Level 全时段对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"v2_{name}_full.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. 缩放周 03-01~03-07 ────────────────────────────
    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2026-03-07 23:00:00")
    week = results.loc[start:end]
    if len(week) > 0:
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(week.index, week["actual"], "o-", color="black", markersize=2,
                linewidth=1.5, label="实际", zorder=5)
        for label, col in methods.items():
            if label == "Naive":
                continue
            ax.plot(week.index, week[col], "-", color=colors[label], linewidth=1.2,
                    alpha=0.85, label=label)
        ax.set_ylabel("电价 (元/MWh)")
        ax.set_title(f"{name.upper()} - 缩放周 03-01~03-07 (V1 vs V2)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))
        plt.tight_layout()
        plt.savefig(VIZ_DIR / f"v2_{name}_zoom_week.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ── 4. 典型日对比 (6 天: 3天来自下滑周 + 3天典型) ───────
    results_plot = results.copy()
    results_plot["date"] = results_plot.index.date
    daily_std = results_plot.groupby("date")["actual"].std()

    target_days = []
    # 下滑周的 3 天
    for d_str in ["2026-03-01", "2026-03-04", "2026-03-06"]:
        d = pd.Timestamp(d_str).date()
        if d in daily_std.index:
            target_days.append((d, f"下滑期 {str(d)[5:]}"))
    # 典型 3 天
    volatile_day = daily_std.idxmax()
    stable_day = daily_std.idxmin()
    med_idx = (daily_std - daily_std.median()).abs().argsort().iloc[0]
    medium_day = daily_std.index[med_idx]
    for d, lbl in [(stable_day, "平稳日"), (medium_day, "中等日"), (volatile_day, "高波动日")]:
        if d not in [x[0] for x in target_days]:
            target_days.append((d, f"{lbl} {str(d)[5:]}"))

    n_panels = min(len(target_days), 6)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes_flat = axes.flatten() if n_panels > 1 else [axes]
    for i, (day, title) in enumerate(target_days[:n_panels]):
        ax = axes_flat[i]
        day_data = results_plot[results_plot["date"] == day]
        hours = day_data.index.hour
        ax.plot(hours, day_data["actual"], "o-", color="black", linewidth=2,
                markersize=4, label="实际", zorder=5)
        for label, col in methods.items():
            if label == "Naive":
                continue
            ax.plot(hours, day_data[col], "-", color=colors[label],
                    linewidth=1.4, alpha=0.85, label=label)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
        if i % ncols == 0:
            ax.set_ylabel("电价 (元/MWh)")
        if i == 0:
            ax.legend(fontsize=7, loc="best")
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"{name.upper()} - V1 vs V2 典型日逐小时对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"v2_{name}_days.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 5. 逐日 MAE 对比 ─────────────────────────────────
    daily_mae_v1 = []
    daily_mae_v2 = []
    day_labels = []
    for d in sorted(results_plot["date"].unique()):
        day_data = results_plot[results_plot["date"] == d]
        if len(day_data) == 24:
            mae_v1 = np.mean(np.abs(day_data["actual"].values - day_data["pred_v1_lgb_level"].values))
            mae_v2 = np.mean(np.abs(day_data["actual"].values - day_data["pred_v2_adaptive"].values))
            daily_mae_v1.append(mae_v1)
            daily_mae_v2.append(mae_v2)
            day_labels.append(str(d)[5:])

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(day_labels))
    w = 0.35
    ax.bar(x - w / 2, daily_mae_v1, w, color="#90CAF9", alpha=0.8, label="V1 LGB Level")
    ax.bar(x + w / 2, daily_mae_v2, w, color="#E91E63", alpha=0.8, label="V2 Adaptive Level")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("日 MAE (元/MWh)")
    ax.set_title(f"{name.upper()} - 逐日 MAE 对比: V1 vs V2", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"v2_{name}_daily_mae.png", dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved 5 plots to output/viz/v2_%s_*.png", name)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_all():
    _setup_cn_font()
    all_results = {}
    for name, target_col, naive_col in [
        ("da", "target_da_clearing_price", "da_clearing_price_lag24h"),
        ("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h"),
    ]:
        results, summary, methods, level_tracking, extra_info = run_shape_v2(
            name, target_col, naive_col)
        plot_shape_v2(results, summary, methods, level_tracking, extra_info, name)
        all_results[name] = {
            "results": results, "summary": summary, "methods": methods,
            "extra_info": extra_info, "level_tracking": level_tracking,
        }

    logger.info("=" * 60)
    logger.info("ALL SHAPE V2 EXPERIMENTS COMPLETE")
    logger.info("Results and plots in output/viz/v2_*")
    logger.info("=" * 60)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
