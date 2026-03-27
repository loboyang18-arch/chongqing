"""
形状-水平分解模型 — 将电价预测拆为两个独立子任务。

  Level 模型: 预测每日均价 (日级, 每天 1 个值)
  Shape 模型: 预测每小时偏离当日均价的量 (小时级)
  合成: pred_h = level_pred_d + shape_pred_h

Shape 模型主要利用 lag0 的逐小时供需特征 (net_load, solar, wind 等),
学习 "什么样的供需结构对应什么方向的价格偏离"。

可视化结果输出到 output/viz/
"""

import json
import logging
import os
from typing import Dict, List

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
    """将小时级特征聚合为日级特征 (均值 + 标准差 + min/max)。"""
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


def run_decompose(name: str, target_col: str, naive_col: str):
    logger.info("=" * 60)
    logger.info("DECOMPOSE MODEL: %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    train_df = df.loc[:TRAIN_END]
    test_df = df.loc[TEST_START:]

    # ── 计算每日均价和小时偏离 ────────────────────────
    df["date"] = df.index.date
    daily_mean = df.groupby("date")[target_col].transform("mean")
    df["target_daily_mean"] = daily_mean
    df["target_hourly_dev"] = df[target_col] - daily_mean

    train_df = df.loc[:TRAIN_END]
    test_df = df.loc[TEST_START:]

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
    level_mae = np.mean(np.abs(daily_target_test.values - level_pred_test))
    level_naive_mae = np.mean(np.abs(daily_target_test.values - daily_naive_test.values))
    logger.info("  Level MAE: %.2f (Naive daily: %.2f)", level_mae, level_naive_mae)

    level_importance = pd.DataFrame({
        "feature": level_feat_cols,
        "importance": level_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info("  Level Top-10 features:")
    for _, row in level_importance.head(10).iterrows():
        logger.info("    %-50s %.1f", row["feature"], row["importance"])

    # ================================================================
    # Stage 2: Shape 模型 — 预测每小时偏离当日均价
    # ================================================================
    logger.info("--- Stage 2: Shape Model (hourly deviation) ---")

    shape_target_train = train_df["target_hourly_dev"]
    shape_target_test = test_df["target_hourly_dev"]

    shape_feat_cols = [c for c in feature_cols]

    params_shape = params.copy()
    params_shape["objective"] = "huber"

    ds_train = lgb.Dataset(train_df[shape_feat_cols], label=shape_target_train)
    ds_val = lgb.Dataset(test_df[shape_feat_cols], label=shape_target_test, reference=ds_train)
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

    shape_pred_test = shape_model.predict(test_df[shape_feat_cols])
    shape_mae = np.mean(np.abs(shape_target_test.values - shape_pred_test))
    logger.info("  Shape MAE (deviation): %.2f", shape_mae)

    shape_importance = pd.DataFrame({
        "feature": shape_feat_cols,
        "importance": shape_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info("  Shape Top-10 features:")
    for _, row in shape_importance.head(10).iterrows():
        logger.info("    %-50s %.1f", row["feature"], row["importance"])

    # ── Naive 的日内偏离（基准对比）──────────────────
    naive_daily_mean_test = test_df.groupby("date")[naive_col].transform("mean")
    naive_hourly_dev = test_df[naive_col] - naive_daily_mean_test

    shape_naive_mae = np.mean(np.abs(shape_target_test.values - naive_hourly_dev.values))
    logger.info("  Shape Naive MAE (deviation): %.2f", shape_naive_mae)

    # ================================================================
    # 合成预测
    # ================================================================
    logger.info("--- Synthesis ---")

    level_pred_map = dict(zip(daily_target_test.index, level_pred_test))
    level_per_hour = np.array([level_pred_map[d] for d in test_df["date"]])

    naive_level_per_hour = np.array([
        daily_naive_test.loc[d] if d in daily_naive_test.index else np.nan
        for d in test_df["date"]
    ])
    pred_I = test_df[naive_col].values
    actual_test = test_df[target_col].values

    # ── 幅度放大：用训练集标准差比值缩放 shape 预测 ──
    shape_pred_train = shape_model.predict(train_df[shape_feat_cols])
    actual_dev_std = shape_target_train.std()
    pred_dev_std = np.std(shape_pred_train)
    scale_factor = actual_dev_std / max(pred_dev_std, 1e-6)
    scale_factor = min(scale_factor, 5.0)
    logger.info("  Shape amplitude scale: actual_std=%.2f, pred_std=%.2f, factor=%.2f",
                actual_dev_std, pred_dev_std, scale_factor)

    shape_pred_scaled = shape_pred_test * scale_factor

    # ── 各方法 ───────────────────────────────────────
    # F: Level(LGB) + Shape(LGB) 原始
    pred_F = level_per_hour + shape_pred_test
    # Fs: Level(LGB) + Shape(LGB) 放大
    pred_Fs = level_per_hour + shape_pred_scaled
    # G: Level(LGB) + Shape(Naive)
    pred_G = level_per_hour + naive_hourly_dev.values
    # Hs: Level(Naive) + Shape(LGB) 放大
    pred_Hs = naive_level_per_hour + shape_pred_scaled
    # J: Level(LGB) + 混合Shape (0.5*LGB放大 + 0.5*Naive)
    shape_blend = 0.5 * shape_pred_scaled + 0.5 * naive_hourly_dev.values
    pred_J = level_per_hour + shape_blend

    # ── 在训练集上搜索最优混合比 α ─────────────────
    level_pred_train_map = dict(zip(daily_target_train.index, level_model.predict(daily_train[level_feat_cols])))
    level_train_per_hour = np.array([level_pred_train_map.get(d, np.nan) for d in train_df["date"]])
    naive_dev_train = train_df[naive_col].values - np.array([
        daily_naive_train.loc[d] if d in daily_naive_train.index else np.nan
        for d in train_df["date"]
    ])

    best_alpha, best_alpha_mae = 0.5, np.inf
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend_train = alpha * (shape_pred_train * scale_factor) + (1 - alpha) * naive_dev_train
        pred_train_blend = level_train_per_hour + blend_train
        mae_blend = np.nanmean(np.abs(train_df[target_col].values - pred_train_blend))
        if mae_blend < best_alpha_mae:
            best_alpha_mae = mae_blend
            best_alpha = alpha
    logger.info("  Optimal blend α=%.2f (LGB shape weight)", best_alpha)

    shape_blend_opt = best_alpha * shape_pred_scaled + (1 - best_alpha) * naive_hourly_dev.values
    pred_K = level_per_hour + shape_blend_opt

    # 基准: 标准 LGB
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

    results = pd.DataFrame({
        "actual": actual_test,
        "naive": pred_I,
        "pred_base": pred_base,
        "pred_F_lgb_lgb": pred_F,
        "pred_Fs_scaled": pred_Fs,
        "pred_G_lgb_naive": pred_G,
        "pred_J_blend50": pred_J,
        "pred_K_blend_opt": pred_K,
    }, index=test_df.index)
    results.index.name = "ts"

    methods = {
        "标准LGB": "pred_base",
        "F_LGB+Shape原始": "pred_F_lgb_lgb",
        "Fs_LGB+Shape放大": "pred_Fs_scaled",
        "G_LGB+Naive形状": "pred_G_lgb_naive",
        f"K_LGB+混合α={best_alpha:.2f}": "pred_K_blend_opt",
        "Naive": "naive",
    }

    logger.info("\n=== %s 分解模型对比 ===", name.upper())
    summary_rows = []
    for label, col in methods.items():
        m = _compute_metrics(actual_test, results[col].values)
        pred_std = results[col].std()
        pred_range = results[col].max() - results[col].min()

        corr_daily = []
        for d in results.index.normalize().unique():
            day = results.loc[str(d.date())]
            if len(day) == 24:
                c = np.corrcoef(day["actual"].values, day[col].values)[0, 1]
                if np.isfinite(c):
                    corr_daily.append(c)
        shape_corr = np.mean(corr_daily) if corr_daily else np.nan

        logger.info("  %-35s MAE=%6.2f  RMSE=%6.2f  std=%5.1f  shape_r=%.3f",
                    label, m["MAE"], m["RMSE"], pred_std, shape_corr)
        summary_rows.append({
            "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"],
            "sMAPE(%)": m["sMAPE(%)"], "wMAPE(%)": m["wMAPE(%)"],
            "pred_std": round(pred_std, 2), "pred_range": round(pred_range, 2),
            "shape_corr": round(shape_corr, 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(VIZ_DIR / f"decompose_{name}_summary.csv", index=False)
    results.to_csv(VIZ_DIR / f"decompose_{name}_result.csv")

    return results, summary, methods, shape_importance, level_importance


def plot_decompose(results, summary, methods, shape_imp, level_imp, name):
    _setup_cn_font()

    color_pool = ["#90CAF9", "#E91E63", "#FF9800", "#4CAF50", "#9C27B0", "#BDBDBD"]
    colors = {}
    for i, label in enumerate(methods.keys()):
        if label == "Naive":
            colors[label] = "#BDBDBD"
        else:
            colors[label] = color_pool[i % len(color_pool)]

    # ── 1. 全时段对比 ────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(results.index, results["actual"], color="black", linewidth=1.2, label="实际", zorder=5)
    for label, col in methods.items():
        if label == "Naive":
            ax.plot(results.index, results[col], "--", color=colors[label], linewidth=0.6,
                    alpha=0.4, label=f"Naive", zorder=1)
            continue
        s = summary[summary["method"] == label]
        ax.plot(results.index, results[col], color=colors[label], linewidth=0.9,
                alpha=0.85, label=f"{label} (MAE={s['MAE'].values[0]:.1f}, r={s['shape_corr'].values[0]:.3f})")

    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 形状-水平分解模型全时段对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"decompose_{name}_full.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. 三类典型日 ────────────────────────────────
    results["date"] = results.index.date
    daily_std = results.groupby("date")["actual"].std()
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
        day_data = results[results["date"] == day]
        hours = day_data.index.hour
        ax.plot(hours, day_data["actual"], "o-", color="black", linewidth=2, markersize=4,
                label="实际", zorder=5)
        for label, col in methods.items():
            ls = "--" if label == "Naive" else "-"
            ax.plot(hours, day_data[col], ls, color=colors[label], linewidth=1.2,
                    alpha=0.85, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
        if ax == axes[0]:
            ax.set_ylabel("电价 (元/MWh)")
    axes[2].legend(fontsize=7, loc="best")
    plt.suptitle(f"{name.upper()} — 分解模型典型日对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"decompose_{name}_typical.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. 最差周缩放 ────────────────────────────────
    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2026-03-07 23:00:00")
    week = results.loc[start:end]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(week.index, week["actual"], "o-", color="black", markersize=2,
            linewidth=1.5, label="实际", zorder=5)
    for label, col in methods.items():
        ls = "--" if label == "Naive" else "-"
        lw = 0.8 if label == "Naive" else 1.2
        ax.plot(week.index, week[col], ls, color=colors[label], linewidth=lw,
                alpha=0.85, label=label)
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 缩放周 03-01~03-07", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"decompose_{name}_zoom_week.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 4. MAE vs shape_corr 柱状图 ──────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    w = 0.35
    ax1.bar(x - w / 2, summary["MAE"], w, color="#42A5F5", label="MAE", alpha=0.85)
    ax1.set_ylabel("MAE (元/MWh)", color="#42A5F5")
    ax1.tick_params(axis="y", labelcolor="#42A5F5")

    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, summary["shape_corr"], w, color="#EF5350", label="形状相关性", alpha=0.85)
    ax2.set_ylabel("日内形状相关性 (r)", color="#EF5350")
    ax2.tick_params(axis="y", labelcolor="#EF5350")
    ax2.set_ylim(0, 1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["method"], rotation=20, ha="right", fontsize=8)
    ax1.set_title(f"{name.upper()} — 分解模型 MAE vs 形状相关性", fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"decompose_{name}_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 5. Shape 模型特征重要度 ──────────────────────
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
    plt.savefig(VIZ_DIR / f"decompose_{name}_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 6. 散点图 ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, (label, col) in enumerate(methods.items()):
        ax = axes[i]
        ax.scatter(results["actual"], results[col], alpha=0.3, s=8, color=colors[label])
        lims = [results["actual"].min() - 20, results["actual"].max() + 20]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("实际")
        ax.set_ylabel("预测")
        s = summary[summary["method"] == label]
        ax.set_title(f"{label}\nMAE={s['MAE'].values[0]:.1f} r={s['shape_corr'].values[0]:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_aspect("equal")
    if len(methods) < 6:
        axes[-1].set_visible(False)
    plt.suptitle(f"{name.upper()} — 预测 vs 实际散点图", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"decompose_{name}_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved 6 plots to output/viz/decompose_%s_*.png", name)


def run_all():
    _setup_cn_font()
    for name, target_col, naive_col in [
        ("da", "target_da_clearing_price", "da_clearing_price_lag24h"),
        ("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h"),
    ]:
        results, summary, methods, shape_imp, level_imp = run_decompose(name, target_col, naive_col)
        plot_decompose(results, summary, methods, shape_imp, level_imp, name)

    logger.info("=" * 60)
    logger.info("ALL DECOMPOSE EXPERIMENTS COMPLETE")
    logger.info("Results and plots in output/viz/decompose_*")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
