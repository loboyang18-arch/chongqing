"""
V4 Shape-Aware 重训练与可视化

使用更新后的特征集（含 20 个新特征）+ Huber 损失 + Level+Shape 架构，
重新训练 DA/RT 模型，输出到 output/v4_shape/。

生成三种日曲线叠图：
  1. 典型日（MAE 接近中位数的 6 天）
  2. 极端高价日（日均价最高的 6 天）
  3. 极端低价日（日均价最低的 6 天）
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
    EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND,
    TEST_START, TRAIN_END, _compute_metrics, _load_dataset,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)
V4_DIR = OUTPUT_DIR / "v4_shape"
V4_DIR.mkdir(exist_ok=True)


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


# ── Level+Shape 核心逻辑 (复用 V2) ─────────────────────────

def _build_daily_features(df, feature_cols):
    df_feat = df[feature_cols].copy()
    df_feat["date"] = df_feat.index.date
    agg_funcs = {}
    for col in feature_cols:
        if col in ("hour", "day_of_week", "is_weekend", "month",
                    "peak_flag", "valley_flag"):
            continue
        agg_funcs[col] = ["mean"]
    daily = df_feat.groupby("date").agg(agg_funcs)
    daily.columns = [f"{c[0]}_dmean" for c in daily.columns]
    for cal_col in ("day_of_week", "is_weekend", "month"):
        if cal_col in feature_cols:
            daily[cal_col] = df_feat.groupby("date")[cal_col].first()
    return daily


def _compute_naive_level_signals(daily_actual_mean):
    naive_1d = daily_actual_mean.shift(1)
    naive_3d = daily_actual_mean.rolling(3, min_periods=1).mean().shift(1)
    naive_ema = daily_actual_mean.ewm(span=3, adjust=False).mean().shift(1)
    return {"naive_1d": naive_1d, "naive_3d": naive_3d, "naive_ema": naive_ema}


def _grid_search_level_weights(level_lgb, naive_signals, actual, step=0.05):
    signals = [level_lgb, naive_signals["naive_1d"],
               naive_signals["naive_3d"], naive_signals["naive_ema"]]
    best_mae, best_w = np.inf, {"lgb": 1.0, "naive_1d": 0.0, "naive_3d": 0.0, "naive_ema": 0.0}
    for w_lgb in np.arange(0, 1.01, step):
        for w_1d in np.arange(0, 1.01 - w_lgb, step):
            for w_3d in np.arange(0, 1.01 - w_lgb - w_1d, step):
                w_ema = round(1.0 - w_lgb - w_1d - w_3d, 2)
                if w_ema < -1e-6:
                    continue
                blended = w_lgb * signals[0] + w_1d * signals[1] + w_3d * signals[2] + w_ema * signals[3]
                valid = np.isfinite(blended) & np.isfinite(actual)
                if valid.sum() < 5:
                    continue
                mae = np.mean(np.abs(actual[valid] - blended[valid]))
                if mae < best_mae:
                    best_mae = mae
                    best_w = {"lgb": round(w_lgb, 2), "naive_1d": round(w_1d, 2),
                              "naive_3d": round(w_3d, 2), "naive_ema": round(w_ema, 2)}
    return best_w, best_mae


def _search_gamma(level_blended, actual, step=0.05):
    best_gamma, best_mae = 0.0, np.inf
    for gamma in np.arange(0, 1.01, step):
        corrected = np.copy(level_blended)
        for i in range(1, len(corrected)):
            if np.isfinite(actual[i - 1]) and np.isfinite(level_blended[i - 1]):
                corrected[i] = level_blended[i] + gamma * (actual[i - 1] - corrected[i - 1])
        valid = np.isfinite(corrected) & np.isfinite(actual)
        if valid.sum() < 5:
            continue
        mae = np.mean(np.abs(actual[valid] - corrected[valid]))
        if mae < best_mae:
            best_mae = mae
            best_gamma = round(gamma, 2)
    return best_gamma, best_mae


def _apply_online_correction(blended, actual, gamma, last_error):
    corrected = np.copy(blended)
    prev_error = last_error
    for i in range(len(corrected)):
        if np.isfinite(prev_error):
            corrected[i] = blended[i] + gamma * prev_error
        if np.isfinite(actual[i]) and np.isfinite(corrected[i]):
            prev_error = actual[i] - corrected[i]
    return corrected


def _compute_shape_sources(target_df, full_df, target_col, shape_model,
                           shape_feat_cols, scale_factor):
    dates = target_df.index.date
    target_dates = sorted(set(dates))
    all_dates = sorted(set(full_df.index.date))
    all_date_idx = {d: i for i, d in enumerate(all_dates)}
    daily_mean_actual = full_df.groupby(full_df.index.date)[target_col].mean()

    hourly_dev = {}
    for d in all_dates:
        mask = full_df.index.date == d
        day_data = full_df.loc[mask, target_col].values
        hourly_dev[d] = day_data - daily_mean_actual.get(d, np.nan)

    s1 = np.full(len(target_df), np.nan)
    for d in target_dates:
        idx = all_date_idx[d]
        prev_days = [all_dates[idx - k] for k in range(1, 4) if idx - k >= 0]
        if not prev_days:
            continue
        mask = dates == d
        n_h = mask.sum()
        avg = np.zeros(n_h)
        cnt = 0
        for pd_ in prev_days:
            dev = hourly_dev.get(pd_)
            if dev is not None and len(dev) == n_h:
                avg += dev
                cnt += 1
        if cnt > 0:
            s1[mask] = avg / cnt

    s2 = shape_model.predict(target_df[shape_feat_cols]) * scale_factor

    s3 = np.full(len(target_df), np.nan)
    for d in target_dates:
        idx = all_date_idx[d]
        if idx == 0:
            continue
        prev_d = all_dates[idx - 1]
        mask = dates == d
        n_h = mask.sum()
        dev = hourly_dev.get(prev_d)
        if dev is not None and len(dev) == n_h:
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


# ── 训练主流程 ─────────────────────────────────────────────

def _train_task(name: str, target_col: str):
    """训练单任务 (DA 或 RT) 的 LGB baseline + Level+Shape V2。"""
    logger.info("=" * 60)
    logger.info("V4 TRAINING: %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()

    # ── A. 逐点 LGB baseline ──────────────────────
    logger.info("--- [A] Point LGB (Huber, new features) ---")
    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)
    lgb_model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
                   lgb.log_evaluation(200)],
    )
    pred_lgb = lgb_model.predict(test_df[feature_cols])
    logger.info("  LGB best iter: %d", lgb_model.best_iteration)

    # ── B. Level+Shape (V2 架构) ──────────────────
    logger.info("--- [B] Level+Shape V2 (Adaptive Level) ---")

    df["date"] = df.index.date
    daily_mean = df.groupby("date")[target_col].transform("mean")
    df["target_daily_mean"] = daily_mean
    df["target_hourly_dev"] = df[target_col] - daily_mean

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()

    # B.1 Level LGB
    daily_train = _build_daily_features(train_df, feature_cols)
    daily_test = _build_daily_features(test_df, feature_cols)
    daily_target_train = train_df.groupby("date")[target_col].mean()
    daily_target_test = test_df.groupby("date")[target_col].mean()
    daily_train = daily_train.loc[daily_target_train.index]
    daily_test = daily_test.loc[daily_target_test.index]
    level_feat_cols = list(daily_train.columns)

    params_level = params.copy()
    params_level["objective"] = "huber"
    params_level["num_leaves"] = min(params_level.get("num_leaves", 31), 31)
    params_level["min_child_samples"] = 5

    dt_l = lgb.Dataset(daily_train[level_feat_cols], label=daily_target_train)
    dv_l = lgb.Dataset(daily_test[level_feat_cols], label=daily_target_test, reference=dt_l)
    level_model = lgb.train(
        params_level, dt_l, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dt_l, dv_l], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    level_lgb_train = pd.Series(level_model.predict(daily_train[level_feat_cols]),
                                index=daily_target_train.index)
    level_lgb_test = pd.Series(level_model.predict(daily_test[level_feat_cols]),
                               index=daily_target_test.index)

    # B.2 Adaptive Level
    all_daily_actual = df.groupby("date")[target_col].mean()
    naive_signals_all = _compute_naive_level_signals(all_daily_actual)

    train_dates = sorted(daily_target_train.index)
    n_train_days = len(train_dates)
    val_day_start = int(n_train_days * 0.75)
    val_dates = train_dates[val_day_start:]

    val_actual = daily_target_train.loc[val_dates].values
    val_lgb = level_lgb_train.loc[val_dates].values
    val_naive = {k: v.loc[val_dates].values for k, v in naive_signals_all.items()}

    level_weights, _ = _grid_search_level_weights(val_lgb, val_naive, val_actual)
    logger.info("  Level weights: %s", level_weights)

    val_blended = (level_weights["lgb"] * val_lgb
                   + level_weights["naive_1d"] * val_naive["naive_1d"]
                   + level_weights["naive_3d"] * val_naive["naive_3d"]
                   + level_weights["naive_ema"] * val_naive["naive_ema"])
    gamma, _ = _search_gamma(val_blended, val_actual)
    logger.info("  Gamma: %.2f", gamma)

    test_dates = sorted(daily_target_test.index)
    test_lgb_arr = level_lgb_test.loc[test_dates].values
    test_naive = {k: v.reindex(test_dates).values for k, v in naive_signals_all.items()}
    test_blended = (level_weights["lgb"] * test_lgb_arr
                    + level_weights["naive_1d"] * test_naive["naive_1d"]
                    + level_weights["naive_3d"] * test_naive["naive_3d"]
                    + level_weights["naive_ema"] * test_naive["naive_ema"])

    last_day = train_dates[-1]
    last_blended = (level_weights["lgb"] * level_lgb_train.loc[last_day]
                    + level_weights["naive_1d"] * naive_signals_all["naive_1d"].loc[last_day]
                    + level_weights["naive_3d"] * naive_signals_all["naive_3d"].loc[last_day]
                    + level_weights["naive_ema"] * naive_signals_all["naive_ema"].loc[last_day])
    last_error = daily_target_train.loc[last_day] - last_blended
    test_actual_daily = daily_target_test.loc[test_dates].values
    level_corrected = _apply_online_correction(test_blended, test_actual_daily, gamma, last_error)

    level_map = dict(zip(test_dates, level_corrected))
    level_per_hour = np.array([level_map.get(d, np.nan) for d in test_df["date"]])

    # B.3 Shape 混合
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

    n_train = len(train_df)
    val_h_start = int(n_train * 0.75)
    val_df = train_df.iloc[val_h_start:]
    pre_val_df = train_df.iloc[:val_h_start]

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

    # 用 adaptive level 在验证集做 shape weight search
    # 重建训练期的 corrected level
    train_corrected = {}
    prev_err = 0.0
    for i, d in enumerate(train_dates):
        lgb_v = level_lgb_train.loc[d]
        n1 = naive_signals_all["naive_1d"].get(d, np.nan)
        n3 = naive_signals_all["naive_3d"].get(d, np.nan)
        ne = naive_signals_all["naive_ema"].get(d, np.nan)
        bl = level_weights["lgb"] * lgb_v
        for k, v in [("naive_1d", n1), ("naive_3d", n3), ("naive_ema", ne)]:
            if np.isfinite(v):
                bl += level_weights[k] * v
        if i > 0 and np.isfinite(prev_err):
            bl += gamma * prev_err
        train_corrected[d] = bl
        act = daily_target_train.get(d, np.nan)
        prev_err = act - bl if np.isfinite(act) and np.isfinite(bl) else 0.0

    level_train_corr_h = np.array([train_corrected.get(d, np.nan) for d in train_df["date"]])

    val_actual_h = train_df[target_col].values[val_h_start:]
    val_level_h = level_train_corr_h[val_h_start:]
    val_s1 = s1_train[val_h_start:]
    val_s3 = s3_train[val_h_start:]
    val_dates_h = train_df.index.date[val_h_start:]

    best_w, best_sr = (1.0, 0.0, 0.0), -1.0
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
                best_w = (round(w1, 2), round(w2, 2), w3)

    w1, w2, w3 = best_w
    logger.info("  Shape weights: S1=%.2f S2=%.2f S3=%.2f  val_shape_r=%.3f", w1, w2, w3, best_sr)

    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_test = np.where(can_mix,
                          w1 * s1_test + w2 * s2_test + w3 * s3_test,
                          np.where(np.isfinite(s1_test), s1_test, s2_test))
    pred_v2 = level_per_hour + shape_test

    # ── 汇总结果 ──────────────────────────────────
    actual_test = test_df[target_col].values
    results = pd.DataFrame({
        "actual": actual_test,
        "pred_lgb": pred_lgb,
        "pred_level_shape": pred_v2,
    }, index=test_df.index)
    results.index.name = "ts"
    results.to_csv(V4_DIR / f"{name}_result.csv")

    for label, col in [("LGB_Huber", "pred_lgb"), ("Level+Shape_V2", "pred_level_shape")]:
        m = _compute_metrics(actual_test, results[col].values)
        sr = compute_shape_report(actual_test, results[col].values, test_df.index)
        logger.info("  %-20s MAE=%.2f RMSE=%.2f | corr=%.3f norm_mae=%.3f "
                    "peak_err=%.1f valley_err=%.1f amp_err=%.1f dir_acc=%.3f",
                    label, m["MAE"], m["RMSE"],
                    sr["profile_corr"], sr["norm_profile_mae"],
                    sr["peak_hour_err"], sr["valley_hour_err"],
                    sr["amplitude_err"], sr["direction_acc"])

    return results


# ── 可视化 ─────────────────────────────────────────────────

def _select_days(results: pd.DataFrame, category: str, n: int = 6) -> List:
    """选取典型日/极端高价日/极端低价日。"""
    daily = results.groupby(results.index.date).agg(
        actual_mean=("actual", "mean"),
        actual_max=("actual", "max"),
        actual_min=("actual", "min"),
    )
    daily["mae_lgb"] = results.groupby(results.index.date).apply(
        lambda g: np.mean(np.abs(g["actual"] - g["pred_lgb"]))
    )

    if category == "typical":
        median_mae = daily["mae_lgb"].median()
        daily["dist_to_median"] = (daily["mae_lgb"] - median_mae).abs()
        return list(daily.nsmallest(n, "dist_to_median").index)
    elif category == "high":
        return list(daily.nlargest(n, "actual_mean").index)
    elif category == "low":
        return list(daily.nsmallest(n, "actual_mean").index)
    return []


def _plot_day_overlay(results: pd.DataFrame, days: List, title: str,
                      filename: str, name: str):
    """单类日曲线叠图：每天一个子图，实际 vs 各模型预测。"""
    _setup_cn_font()
    n = len(days)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    hours = list(range(24))
    for i, d in enumerate(sorted(days)):
        ax = axes[i // cols][i % cols]
        day_data = results.loc[str(d)]
        if len(day_data) != 24:
            continue

        ax.plot(hours, day_data["actual"].values, "k-", linewidth=2.0,
                label="实际", zorder=3)
        ax.plot(hours, day_data["pred_lgb"].values, "#1976D2", linewidth=1.4,
                alpha=0.85, label="LGB")
        ax.plot(hours, day_data["pred_level_shape"].values, "#E91E63",
                linewidth=1.4, alpha=0.85, label="Level+Shape")

        ax.set_title(str(d), fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"{name.upper()} — {title}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(V4_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", filename)


def _plot_all_overlays(results: pd.DataFrame, name: str):
    """生成三种叠图。"""
    typical = _select_days(results, "typical", 6)
    high = _select_days(results, "high", 6)
    low = _select_days(results, "low", 6)

    _plot_day_overlay(results, typical, "典型日曲线叠图",
                      f"{name}_typical_days.png", name)
    _plot_day_overlay(results, high, "极端高价日曲线叠图",
                      f"{name}_high_price_days.png", name)
    _plot_day_overlay(results, low, "极端低价日曲线叠图",
                      f"{name}_low_price_days.png", name)


# ── 入口 ───────────────────────────────────────────────────

def run_all():
    summary_rows = []
    for name, target_col in [("da", "target_da_clearing_price"),
                              ("rt", "target_rt_clearing_price")]:
        results = _train_task(name, target_col)
        _plot_all_overlays(results, name)

        actual = results["actual"].values
        idx = results.index
        for label, col in [("LGB_Huber", "pred_lgb"),
                           ("Level+Shape_V2", "pred_level_shape")]:
            m = _compute_metrics(actual, results[col].values)
            sr = compute_shape_report(actual, results[col].values, idx)
            row = {"task": name, "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"]}
            row.update(sr)
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(V4_DIR / "summary.csv", index=False)
    logger.info("\n=== V4 Summary ===")
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
