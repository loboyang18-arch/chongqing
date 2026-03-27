"""
三层混合形状预测模型 — Level(LGB) + Shape(3日均值 + LGB放大 + Naive混合) + 逐小时校准

  最终预测 = Level预测(LGB) + Shape预测(混合) + 逐小时偏移校准

Shape 模型三种形状源：
  S1: 近3日均值形状 — 取前3天的日内偏离均值 (最佳历史策略, r≈0.497)
  S2: LGB形状预测(放大后) — 分解模型的Shape输出 × 缩放因子
  S3: Naive(昨日形状) — 昨日偏离

在训练集后段通过 grid search 找到 S1/S2/S3 的最优混合权重 (w1, w2, w3)。
最后对合成预测做 per-hour 偏移校准（保守 clipping）。

可视化结果输出到 output/viz/
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


def _load_tuned_params(name: str) -> Dict:
    path = PARAMS_DIR / f"tuning_{name}_best_params.json"
    with open(path) as f:
        return json.load(f)


def _build_daily_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """将小时级特征聚合为日级特征 (均值)。"""
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
    """计算三种形状源的偏离值。

    使用 full_history_df 获取历史日的偏离形状（确保 S1/S3 在测试集开头也有值），
    但只输出 target_df 对应行的值。

    Returns: (s1_3day_avg, s2_lgb_scaled, s3_naive)
    """
    dates = target_df.index.date
    target_dates = sorted(set(dates))

    all_dates = sorted(set(full_history_df.index.date))
    all_date_idx = {d: i for i, d in enumerate(all_dates)}

    daily_mean_actual = full_history_df.groupby(full_history_df.index.date)[target_col].mean()
    daily_mean_naive = full_history_df.groupby(full_history_df.index.date)[naive_col].mean()

    hourly_dev_by_date = {}
    for d in all_dates:
        mask = full_history_df.index.date == d
        day_data = full_history_df.loc[mask, target_col].values
        day_mean = daily_mean_actual.get(d, np.nan)
        hourly_dev_by_date[d] = day_data - day_mean

    naive_dev_by_date = {}
    for d in all_dates:
        mask = full_history_df.index.date == d
        day_data = full_history_df.loc[mask, naive_col].values
        day_mean = daily_mean_naive.get(d, np.nan)
        naive_dev_by_date[d] = day_data - day_mean

    # S1: 近3日均值形状
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

    # S2: LGB 形状预测 (放大后)
    s2 = shape_model.predict(target_df[shape_feat_cols]) * scale_factor

    # S3: Naive (昨日形状) — 使用 naive_col 的实际昨日偏离
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
    """逐日计算形状相关性后取平均。"""
    corr_daily = []
    for d in results_df.index.normalize().unique():
        day = results_df.loc[str(d.date())]
        if len(day) == 24:
            c = np.corrcoef(day[actual_col].values, day[pred_col].values)[0, 1]
            if np.isfinite(c):
                corr_daily.append(c)
    return np.mean(corr_daily) if corr_daily else np.nan


def _compute_shape_corr_arrays(actual: np.ndarray, pred: np.ndarray, dates: np.ndarray) -> float:
    """从 numpy 数组计算逐日形状相关性均值。"""
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


def run_shape_final(name: str, target_col: str, naive_col: str):
    logger.info("=" * 60)
    logger.info("SHAPE FINAL MODEL: %s", name.upper())
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

    # ================================================================
    # Stage 1: Level 模型 — 预测每日均价
    # ================================================================
    logger.info("--- Stage 1: Level Model (daily mean) ---")

    daily_train = _build_daily_features(train_df, feature_cols)
    daily_test = _build_daily_features(test_df, feature_cols)

    daily_target_train = train_df.groupby("date")[target_col].mean()
    daily_target_test = test_df.groupby("date")[target_col].mean()

    daily_naive_train = train_df.groupby("date")[naive_col].mean()
    daily_naive_test = test_df.groupby("date")[naive_col].mean()

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
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(50),
        ],
    )
    logger.info("  Level model best iteration: %d", level_model.best_iteration)

    level_pred_test = level_model.predict(daily_test[level_feat_cols])
    level_pred_train = level_model.predict(daily_train[level_feat_cols])
    level_mae = np.mean(np.abs(daily_target_test.values - level_pred_test))
    level_naive_mae = np.mean(np.abs(daily_target_test.values - daily_naive_test.values))
    logger.info("  Level MAE: %.2f (Naive daily: %.2f)", level_mae, level_naive_mae)

    # ================================================================
    # Stage 2: Shape 模型 — LGB 训练用于 S2 信号
    # ================================================================
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
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    logger.info("  Shape model best iteration: %d", shape_model.best_iteration)

    shape_pred_train = shape_model.predict(train_df[shape_feat_cols])
    actual_dev_std = shape_target_train.std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = actual_dev_std / max(pred_dev_std, 1e-6)
    scale_factor = min(scale_factor, 5.0)
    logger.info("  Shape amplitude scale: actual_std=%.2f, pred_std=%.2f, factor=%.2f",
                actual_dev_std, pred_dev_std, scale_factor)

    # ================================================================
    # Stage 2b: 计算三种形状源 & Grid Search 最优权重
    #   优化目标: shape_r (日内形状相关性)
    #   S2 使用 out-of-fold 预测，避免训练内高估
    # ================================================================
    logger.info("--- Stage 2b: Shape mixing weights (shape_r目标, OOF) ---")

    full_df = pd.concat([train_df, test_df])
    s1_train, _, s3_train = _compute_shape_sources(
        train_df, full_df, target_col, naive_col, shape_model, shape_feat_cols, scale_factor)
    s1_test, s2_test, s3_test = _compute_shape_sources(
        test_df, full_df, target_col, naive_col, shape_model, shape_feat_cols, scale_factor)

    n_train = len(train_df)
    val_start = int(n_train * 0.75)
    val_slice = slice(val_start, n_train)
    pre_val_df = train_df.iloc[:val_start]
    val_df = train_df.iloc[val_start:]

    # OOF shape model: 只用前75%训练，在后25%做真正的out-of-sample预测
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

    level_train_map = dict(zip(daily_target_train.index, level_pred_train))
    level_train_per_hour = np.array([level_train_map.get(d, np.nan) for d in train_df["date"]])

    val_actual = train_df[target_col].values[val_slice]
    val_level = level_train_per_hour[val_slice]
    val_s1 = s1_train[val_slice]
    val_s2 = s2_val_oof
    val_s3 = s3_train[val_slice]
    val_dates = train_df.index.date[val_start:n_train]

    best_w = (1.0, 0.0, 0.0)
    best_shape_r = -1.0
    grid_step = 0.05
    for w1 in np.arange(0, 1.01, grid_step):
        for w2 in np.arange(0, 1.01 - w1, grid_step):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -1e-6:
                continue
            shape_blend = w1 * val_s1 + w2 * val_s2 + w3 * val_s3
            pred_blend = val_level + shape_blend
            fin = np.isfinite(pred_blend)
            if fin.sum() < 48:
                continue
            sr = _compute_shape_corr_arrays(
                val_actual[fin], pred_blend[fin], val_dates[fin])
            if sr > best_shape_r:
                best_shape_r = sr
                best_w = (round(w1, 2), round(w2, 2), w3)

    w1, w2, w3 = best_w
    logger.info("  Optimal weights (shape_r): S1(3日均值)=%.2f, S2(LGB放大)=%.2f, S3(Naive)=%.2f  val_shape_r=%.3f",
                w1, w2, w3, best_shape_r)

    # 合成测试集 shape 预测
    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test)
    shape_mix_test = np.where(
        can_mix,
        w1 * s1_test + w2 * s2_test + w3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, s2_test),
    )

    # ── 合成最终预测 ─────────────────────────────────
    level_test_map = dict(zip(daily_target_test.index, level_pred_test))
    level_test_per_hour = np.array([level_test_map.get(d, np.nan) for d in test_df["date"]])

    pred_final = level_test_per_hour + shape_mix_test

    # ── 各对比方法 ───────────────────────────────────
    pred_naive = test_df[naive_col].values
    actual_test = test_df[target_col].values

    # 标准 LGB
    params_base = params.copy()
    dtb = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dvb = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtb)
    base_model = lgb.train(
        params_base, dtb,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtb, dvb], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(200)],
    )
    pred_base = base_model.predict(test_df[feature_cols])

    # K 方法 (当前最优混合, from decompose)
    naive_daily_mean_test = test_df.groupby("date")[naive_col].transform("mean")
    naive_hourly_dev = test_df[naive_col] - naive_daily_mean_test

    best_alpha_k, best_alpha_k_mae = 0.5, np.inf
    naive_dev_train = train_df[naive_col].values - np.array([
        daily_naive_train.loc[d] if d in daily_naive_train.index else np.nan
        for d in train_df["date"]
    ])
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend_tr = alpha * (shape_pred_train * scale_factor) + (1 - alpha) * naive_dev_train
        pred_tr = level_train_per_hour + blend_tr
        mae_tr = np.nanmean(np.abs(train_df[target_col].values - pred_tr))
        if mae_tr < best_alpha_k_mae:
            best_alpha_k_mae = mae_tr
            best_alpha_k = alpha

    shape_pred_test_scaled = shape_model.predict(test_df[shape_feat_cols]) * scale_factor
    shape_blend_k = best_alpha_k * shape_pred_test_scaled + (1 - best_alpha_k) * naive_hourly_dev.values
    pred_K = level_test_per_hour + shape_blend_k

    # 纯 S1 (3日均值形状)
    pred_s1_only = level_test_per_hour + np.where(np.isfinite(s1_test), s1_test, 0)

    # S1+S3 混合 (纯历史惯性，无LGB)
    s1s3_blend = np.where(
        np.isfinite(s1_test) & np.isfinite(s3_test),
        0.7 * s1_test + 0.3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, s3_test),
    )
    pred_s1s3 = level_test_per_hour + s1s3_blend

    results = pd.DataFrame({
        "actual": actual_test,
        "naive": pred_naive,
        "pred_base": pred_base,
        "pred_K_decompose": pred_K,
        "pred_s1_3day": pred_s1_only,
        "pred_s1s3": pred_s1s3,
        "pred_final": pred_final,
    }, index=test_df.index)
    results.index.name = "ts"

    methods = {
        "标准LGB": "pred_base",
        "Naive": "naive",
        "S1_3日均值形状": "pred_s1_3day",
        "S1S3_历史惯性": "pred_s1s3",
        f"K_分解混合α={best_alpha_k:.2f}": "pred_K_decompose",
        f"Final(w={w1},{w2},{w3})": "pred_final",
    }

    logger.info("\n=== %s 最终形状模型对比 ===", name.upper())
    summary_rows = []
    for label, col in methods.items():
        m = _compute_metrics(actual_test, results[col].values)
        pred_std = results[col].std()
        pred_range = results[col].max() - results[col].min()
        shape_corr = _compute_shape_corr(results, "actual", col)

        logger.info("  %-40s MAE=%6.2f  RMSE=%6.2f  std=%5.1f  range=%6.1f  shape_r=%.3f",
                    label, m["MAE"], m["RMSE"], pred_std, pred_range, shape_corr)
        summary_rows.append({
            "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"],
            "sMAPE(%)": m["sMAPE(%)"], "wMAPE(%)": m["wMAPE(%)"],
            "pred_std": round(pred_std, 2), "pred_range": round(pred_range, 2),
            "shape_corr": round(shape_corr, 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(VIZ_DIR / f"final_{name}_summary.csv", index=False)
    results.to_csv(VIZ_DIR / f"final_{name}_result.csv")

    shape_importance = pd.DataFrame({
        "feature": shape_feat_cols,
        "importance": shape_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    level_importance = pd.DataFrame({
        "feature": level_feat_cols,
        "importance": level_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    extra_info = {
        "weights": best_w,
        "scale_factor": scale_factor,
        "best_alpha_k": best_alpha_k,
        "best_shape_r_val": best_shape_r,
    }

    return results, summary, methods, shape_importance, level_importance, extra_info


def plot_shape_final(results, summary, methods, shape_imp, level_imp, extra_info, name):
    _setup_cn_font()

    colors = {
        "标准LGB": "#90CAF9",
        "Naive": "#BDBDBD",
    }
    shape_colors = ["#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
    ci = 0
    for label in methods:
        if label not in colors:
            colors[label] = shape_colors[ci % len(shape_colors)]
            ci += 1

    # 按 shape_corr 降序排列，方便查看
    summary_sorted = summary.sort_values("shape_corr", ascending=False)

    # ── 1. 全时段对比 ────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(results.index, results["actual"], color="black", linewidth=1.2, label="实际", zorder=5)
    for label, col in methods.items():
        if label in ("标准LGB", "Naive"):
            ax.plot(results.index, results[col], "--", color=colors[label], linewidth=0.6,
                    alpha=0.4, label=label, zorder=1)
            continue
        s = summary[summary["method"] == label]
        ax.plot(results.index, results[col], color=colors[label], linewidth=0.9,
                alpha=0.85, label=f"{label} (r={s['shape_corr'].values[0]:.3f})")

    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 形状优化模型全时段对比 (优化目标: shape_r)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_full.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. 三类典型日 ────────────────────────────────
    results_plot = results.copy()
    results_plot["date"] = results_plot.index.date
    daily_std = results_plot.groupby("date")["actual"].std()
    volatile_day = daily_std.idxmax()
    stable_day = daily_std.idxmin()
    med_idx = (daily_std - daily_std.median()).abs().argsort().iloc[0]
    medium_day = daily_std.index[med_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, day, title in [
        (axes[0], stable_day, f"平稳日 {str(stable_day)[5:]}"),
        (axes[1], medium_day, f"中等日 {str(medium_day)[5:]}"),
        (axes[2], volatile_day, f"高波动日 {str(volatile_day)[5:]}"),
    ]:
        day_data = results_plot[results_plot["date"] == day]
        hours = day_data.index.hour
        ax.plot(hours, day_data["actual"], "o-", color="black", linewidth=2, markersize=4,
                label="实际", zorder=5)
        for label, col in methods.items():
            ls = "--" if label in ("标准LGB", "Naive") else "-"
            lw = 0.8 if label in ("标准LGB", "Naive") else 1.4
            ax.plot(hours, day_data[col], ls, color=colors[label], linewidth=lw,
                    alpha=0.85, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
        if ax == axes[0]:
            ax.set_ylabel("电价 (元/MWh)")
    axes[2].legend(fontsize=6, loc="best")
    plt.suptitle(f"{name.upper()} — 形状优化模型典型日对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_typical.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. 最差形状日（shape_r最低的4天）─────────────
    best_method = summary_sorted.iloc[0]["method"]
    best_col = methods[best_method]

    daily_corr = {}
    for d in results_plot["date"].unique():
        day_data = results_plot[results_plot["date"] == d]
        if len(day_data) == 24:
            c = np.corrcoef(day_data["actual"].values, day_data[best_col].values)[0, 1]
            daily_corr[d] = c if np.isfinite(c) else -1.0
    daily_corr_s = pd.Series(daily_corr)
    worst4 = daily_corr_s.nsmallest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()
    for i, d in enumerate(worst4):
        ax = axes_flat[i]
        day_data = results_plot[results_plot["date"] == d]
        hours = day_data.index.hour
        ax.plot(hours, day_data["actual"], "o-", color="black", linewidth=2, markersize=4,
                label="实际", zorder=5)
        for label, col in methods.items():
            ls = "--" if label in ("标准LGB", "Naive") else "-"
            ax.plot(hours, day_data[col], ls, color=colors[label], linewidth=1.0,
                    alpha=0.85, label=label)
        corr_vals = {}
        for lbl, c in methods.items():
            cv = np.corrcoef(day_data["actual"].values, day_data[c].values)[0, 1]
            corr_vals[lbl] = cv if np.isfinite(cv) else -1.0
        best_l = max(corr_vals, key=corr_vals.get)
        ax.set_title(f"{str(d)[5:]} (最佳形状: {best_l} r={corr_vals[best_l]:.2f})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
        if i == 0:
            ax.legend(fontsize=6, loc="best")
    plt.suptitle(f"{name.upper()} — 形状最差日逐小时对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_worst_days.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 4. 缩放周 ───────────────────────────────────
    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2026-03-07 23:00:00")
    week = results.loc[start:end]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(week.index, week["actual"], "o-", color="black", markersize=2,
            linewidth=1.5, label="实际", zorder=5)
    for label, col in methods.items():
        ls = "--" if label in ("标准LGB", "Naive") else "-"
        lw = 0.8 if label in ("标准LGB", "Naive") else 1.2
        ax.plot(week.index, week[col], ls, color=colors[label], linewidth=lw,
                alpha=0.85, label=label)
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 缩放周 03-01~03-07", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_zoom_week.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 5. shape_corr 柱状图 (主指标) ────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(summary_sorted))
    bar_colors = [colors.get(m, "#42A5F5") for m in summary_sorted["method"]]
    bars = ax.bar(x, summary_sorted["shape_corr"], color=bar_colors, alpha=0.85, edgecolor="white")
    for i, (v, m) in enumerate(zip(summary_sorted["shape_corr"], summary_sorted["MAE"])):
        ax.text(i, v + 0.01, f"r={v:.3f}\nMAE={m:.1f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_sorted["method"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("日内形状相关性 (shape_r)")
    ax.set_ylim(0, max(summary_sorted["shape_corr"]) * 1.25)
    ax.set_title(f"{name.upper()} — 各方法形状相关性排名 (shape_r为主指标)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 6. 散点图 ────────────────────────────────────
    n_methods = len(methods)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes_flat = axes.flatten()
    for i, (label, col) in enumerate(methods.items()):
        ax = axes_flat[i]
        ax.scatter(results["actual"], results[col], alpha=0.3, s=8, color=colors[label])
        lims = [results["actual"].min() - 20, results["actual"].max() + 20]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("实际")
        ax.set_ylabel("预测")
        s = summary[summary["method"] == label]
        ax.set_title(f"{label}\nshape_r={s['shape_corr'].values[0]:.3f}  MAE={s['MAE'].values[0]:.1f}",
                     fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
    for j in range(n_methods, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"{name.upper()} — 预测 vs 实际散点图", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 7. 逐日形状相关性时序图 ─────────────────────
    results_plot2 = results.copy()
    results_plot2["date"] = results_plot2.index.date
    daily_r = {}
    for label, col in methods.items():
        daily_r[label] = {}
        for d in results_plot2["date"].unique():
            day_data = results_plot2[results_plot2["date"] == d]
            if len(day_data) == 24 and np.std(day_data[col].values) > 1e-6:
                c = np.corrcoef(day_data["actual"].values, day_data[col].values)[0, 1]
                daily_r[label][d] = c if np.isfinite(c) else np.nan

    fig, ax = plt.subplots(figsize=(16, 5))
    for label in methods:
        if label in ("标准LGB",):
            continue
        s = pd.Series(daily_r[label]).sort_index()
        ls = "--" if label == "Naive" else "-"
        ax.plot(s.index, s.values, ls, color=colors[label], linewidth=1.0,
                alpha=0.85, label=f"{label} (mean={s.mean():.3f})", marker=".", markersize=3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("日内形状相关性 (r)")
    ax.set_title(f"{name.upper()} — 逐日形状相关性走势", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_daily_shape_r.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 8. 特征重要度对比 ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_shape = shape_imp.head(15)
    axes[0].barh(range(len(top_shape)), top_shape["importance"].values, color="#E91E63", alpha=0.8)
    axes[0].set_yticks(range(len(top_shape)))
    axes[0].set_yticklabels(top_shape["feature"].values, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Shape 模型 Top-15 特征", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Gain")

    top_level = level_imp.head(15)
    axes[1].barh(range(len(top_level)), top_level["importance"].values, color="#42A5F5", alpha=0.8)
    axes[1].set_yticks(range(len(top_level)))
    axes[1].set_yticklabels(top_level["feature"].values, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Level 模型 Top-15 特征", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Gain")

    plt.suptitle(f"{name.upper()} — 特征重要度对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"final_{name}_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved 8 plots to output/viz/final_%s_*.png", name)


def run_all():
    _setup_cn_font()
    all_results = {}
    for name, target_col, naive_col in [
        ("da", "target_da_clearing_price", "da_clearing_price_lag24h"),
        ("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h"),
    ]:
        results, summary, methods, shape_imp, level_imp, extra_info = run_shape_final(
            name, target_col, naive_col)
        plot_shape_final(results, summary, methods, shape_imp, level_imp, extra_info, name)
        all_results[name] = {
            "results": results, "summary": summary, "methods": methods,
            "extra_info": extra_info,
        }

    logger.info("=" * 60)
    logger.info("ALL SHAPE FINAL EXPERIMENTS COMPLETE")
    logger.info("Results and plots in output/viz/final_*")
    logger.info("=" * 60)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
