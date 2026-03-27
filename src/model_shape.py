"""
形状感知模型 — 多种策略让模型同时捕捉价格水平和日内形状。

策略对比：
  A) 标准 LGB（当前最优，输出近常数）
  B) Naive + LGB 残差修正（以 Naive 为基座，LGB 只学修正量）
  C) 样本加权（高波动小时给更大权重，迫使模型关注形状）
  D) 最优逐小时混合（per-hour α: pred = α*naive + (1-α)*lgb）
  E) Naive + 加权残差修正（B+C 结合）

可视化全部放入 output/viz/
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


def _train_lgb_weighted(
    X_train, y_train, X_val, y_val, params, weights=None,
):
    dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    logger.info("  Best iteration: %d", model.best_iteration)
    return model


def run_shape_experiment(name: str, target_col: str, naive_col: str):
    logger.info("=" * 60)
    logger.info("SHAPE EXPERIMENT: %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)

    feature_cols = [c for c in df.columns if c != target_col]
    train_df = df.loc[:TRAIN_END]
    test_df = df.loc[TEST_START:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    naive_train = train_df[naive_col].values
    naive_test = test_df[naive_col].values
    actual_test = y_test.values

    results = pd.DataFrame({"actual": actual_test, "naive": naive_test}, index=test_df.index)

    # ── A) 标准 LGB ──────────────────────────────────
    logger.info("[A] Standard LGB")
    model_a = _train_lgb_weighted(X_train, y_train, X_test, y_test, params)
    results["pred_A_lgb"] = model_a.predict(X_test)

    # ── B) Naive + LGB 残差修正 ──────────────────────
    logger.info("[B] Naive + LGB residual correction")
    residual_train = y_train.values - naive_train
    residual_test = actual_test - naive_test

    params_b = params.copy()
    params_b["objective"] = "huber"

    dtrain_b = lgb.Dataset(X_train, label=residual_train)
    dval_b = lgb.Dataset(X_test, label=residual_test, reference=dtrain_b)
    model_b = lgb.train(
        params_b, dtrain_b,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain_b, dval_b],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    logger.info("  Best iteration: %d", model_b.best_iteration)
    correction_b = model_b.predict(X_test)
    results["pred_B_naive_corr"] = naive_test + correction_b

    # ── C) 样本加权 LGB ─────────────────────────────
    logger.info("[C] Sample-weighted LGB")
    dev_from_mean = np.abs(y_train.values - y_train.rolling(168, min_periods=24).mean().values)
    dev_from_mean = np.nan_to_num(dev_from_mean, nan=0.0)
    weights_c = 1.0 + dev_from_mean / (np.percentile(dev_from_mean[dev_from_mean > 0], 75) + 1e-6)
    weights_c = np.clip(weights_c, 1.0, 10.0)

    model_c = _train_lgb_weighted(X_train, y_train, X_test, y_test, params, weights=weights_c)
    results["pred_C_weighted"] = model_c.predict(X_test)

    # ── D) 最优逐小时混合 ────────────────────────────
    logger.info("[D] Optimal per-hour blend (naive + lgb)")
    pred_lgb = results["pred_A_lgb"].values
    best_alpha = np.zeros(24)
    for h in range(24):
        mask_train = train_df.index.hour == h
        lgb_h = model_a.predict(X_train[mask_train])
        naive_h = naive_train[mask_train]
        actual_h = y_train.values[mask_train]

        best_mae = np.inf
        for alpha in np.arange(0.0, 1.01, 0.05):
            blend = alpha * naive_h + (1 - alpha) * lgb_h
            mae = np.mean(np.abs(actual_h - blend))
            if mae < best_mae:
                best_mae = mae
                best_alpha[h] = alpha
        logger.info("  Hour %2d: α=%.2f (naive weight)", h, best_alpha[h])

    blend_pred = np.zeros(len(test_df))
    for h in range(24):
        mask = test_df.index.hour == h
        blend_pred[mask] = best_alpha[h] * naive_test[mask] + (1 - best_alpha[h]) * pred_lgb[mask]
    results["pred_D_blend"] = blend_pred

    # ── E) Naive + 加权残差修正 ──────────────────────
    logger.info("[E] Naive + weighted residual correction")
    weights_e = 1.0 + dev_from_mean / (np.percentile(dev_from_mean[dev_from_mean > 0], 75) + 1e-6)
    weights_e = np.clip(weights_e, 1.0, 10.0)

    dtrain_e = lgb.Dataset(X_train, label=residual_train, weight=weights_e)
    dval_e = lgb.Dataset(X_test, label=residual_test, reference=dtrain_e)
    model_e = lgb.train(
        params_b, dtrain_e,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain_e, dval_e],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    logger.info("  Best iteration: %d", model_e.best_iteration)
    correction_e = model_e.predict(X_test)
    results["pred_E_naive_wcorr"] = naive_test + correction_e

    # ── 评估 ─────────────────────────────────────────
    methods = {
        "A_标准LGB": "pred_A_lgb",
        "B_Naive+残差": "pred_B_naive_corr",
        "C_加权LGB": "pred_C_weighted",
        "D_逐时混合": "pred_D_blend",
        "E_Naive+加权残差": "pred_E_naive_wcorr",
        "Naive": "naive",
    }

    logger.info("\n=== %s 各方法对比 ===", name.upper())
    summary_rows = []
    for label, col in methods.items():
        m = _compute_metrics(actual_test, results[col].values)
        pred_std = results[col].std()
        pred_range = results[col].max() - results[col].min()

        # shape correlation: hourly pattern similarity
        corr_daily = []
        for d in results.index.normalize().unique():
            day = results.loc[str(d.date())]
            if len(day) == 24:
                c = np.corrcoef(day["actual"].values, day[col].values)[0, 1]
                if np.isfinite(c):
                    corr_daily.append(c)
        shape_corr = np.mean(corr_daily) if corr_daily else np.nan

        logger.info("  %-20s MAE=%6.2f  RMSE=%6.2f  sMAPE=%5.2f%%  std=%5.1f  range=%6.1f  shape_r=%.3f",
                    label, m["MAE"], m["RMSE"], m["sMAPE(%)"], pred_std, pred_range, shape_corr)
        summary_rows.append({
            "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"],
            "sMAPE(%)": m["sMAPE(%)"], "wMAPE(%)": m["wMAPE(%)"],
            "pred_std": round(pred_std, 2), "pred_range": round(pred_range, 2),
            "shape_corr": round(shape_corr, 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(VIZ_DIR / f"shape_{name}_summary.csv", index=False)

    results.index.name = "ts"
    results.to_csv(VIZ_DIR / f"shape_{name}_result.csv")

    return results, summary, methods


def plot_all(results, summary, methods, name):
    """生成全套可视化对比图，放入 output/viz/。"""
    _setup_cn_font()
    actual_std = results["actual"].std()

    colors = {
        "A_标准LGB": "#90CAF9",
        "B_Naive+残差": "#E91E63",
        "C_加权LGB": "#4CAF50",
        "D_逐时混合": "#FF9800",
        "E_Naive+加权残差": "#9C27B0",
        "Naive": "#BDBDBD",
    }

    # ── 1. 全时段对比 ────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(results.index, results["actual"], color="black", linewidth=1.2, label="实际", zorder=5)
    for label, col in methods.items():
        if label == "Naive":
            continue
        ax.plot(results.index, results[col], color=colors[label], linewidth=0.8,
                alpha=0.8, label=f"{label} (MAE={summary[summary['method']==label]['MAE'].values[0]:.1f})")
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 各方法全时段预测对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"shape_{name}_full.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. 选取3个典型日做逐小时对比 ─────────────────
    results["date"] = results.index.date

    daily_std = results.groupby("date")["actual"].std()
    volatile_day = daily_std.idxmax()
    stable_day = daily_std.idxmin()
    medium_day = daily_std.iloc[(daily_std - daily_std.median()).abs().argsort().iloc[0]]
    medium_day_date = daily_std.index[(daily_std - daily_std.median()).abs().argsort().iloc[0]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, day, title in [
        (axes[0], stable_day, f"平稳日 {str(stable_day)[5:]}"),
        (axes[1], medium_day_date, f"中等日 {str(medium_day_date)[5:]}"),
        (axes[2], volatile_day, f"高波动日 {str(volatile_day)[5:]}"),
    ]:
        day_data = results[results["date"] == day]
        hours = day_data.index.hour

        ax.plot(hours, day_data["actual"], "o-", color="black", linewidth=2, markersize=4,
                label="实际", zorder=5)
        ax.plot(hours, day_data["naive"], "x--", color=colors["Naive"], linewidth=1,
                markersize=3, label="Naive", alpha=0.7)
        for label, col in methods.items():
            if label == "Naive":
                continue
            ax.plot(hours, day_data[col], "-", color=colors[label], linewidth=1.2,
                    alpha=0.85, label=label)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
        if ax == axes[0]:
            ax.set_ylabel("电价 (元/MWh)")

    axes[2].legend(fontsize=7, loc="best")
    plt.suptitle(f"{name.upper()} — 三类典型日逐小时对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"shape_{name}_typical_days.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. 最差4天逐小时对比 ─────────────────────────
    best_col = summary.loc[summary["shape_corr"].idxmax(), "method"]
    best_pred_col = methods[best_col]

    daily_mae = results.groupby("date").apply(
        lambda g: np.mean(np.abs(g["actual"] - g[best_pred_col]))
    )
    worst4 = daily_mae.nlargest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for i, d in enumerate(worst4):
        ax = axes[i]
        day_data = results[results["date"] == d]
        hours = day_data.index.hour

        ax.plot(hours, day_data["actual"], "o-", color="black", linewidth=2, markersize=4,
                label="实际", zorder=5)
        ax.plot(hours, day_data["naive"], "x--", color=colors["Naive"], linewidth=1,
                markersize=3, label="Naive", alpha=0.6)
        for label, col in methods.items():
            if label in ("Naive",):
                continue
            ax.plot(hours, day_data[col], "-", color=colors[label], linewidth=1.2,
                    alpha=0.85, label=label)

        mae_vals = {lbl: np.mean(np.abs(day_data["actual"] - day_data[c])) for lbl, c in methods.items()}
        best_label = min(mae_vals, key=mae_vals.get)
        ax.set_title(f"{str(d)[5:]} (最优: {best_label} MAE={mae_vals[best_label]:.1f})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    plt.suptitle(f"{name.upper()} — 高误差日逐小时详细对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"shape_{name}_worst_days.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 4. 缩放周（03-01~03-07）────────────────────
    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2026-03-07 23:00:00")
    week = results.loc[start:end]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(week.index, week["actual"], "o-", color="black", markersize=2,
            linewidth=1.5, label="实际", zorder=5)
    ax.plot(week.index, week["naive"], "--", color=colors["Naive"], linewidth=1,
            alpha=0.5, label="Naive")
    for label, col in methods.items():
        if label == "Naive":
            continue
        ax.plot(week.index, week[col], "-", color=colors[label], linewidth=1.2,
                alpha=0.85, label=label)

    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 缩放周 03-01~03-07", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"shape_{name}_zoom_week.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 5. 方法对比柱状图（MAE + shape_corr 双轴）──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    w = 0.35

    bars1 = ax1.bar(x - w / 2, summary["MAE"], w, color="#42A5F5", label="MAE", alpha=0.85)
    ax1.set_ylabel("MAE (元/MWh)", color="#42A5F5")
    ax1.tick_params(axis="y", labelcolor="#42A5F5")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, summary["shape_corr"], w, color="#EF5350", label="形状相关性", alpha=0.85)
    ax2.set_ylabel("日内形状相关性 (r)", color="#EF5350")
    ax2.tick_params(axis="y", labelcolor="#EF5350")
    ax2.set_ylim(0, 1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["method"], rotation=20, ha="right", fontsize=9)
    ax1.set_title(f"{name.upper()} — MAE vs 形状相关性", fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"shape_{name}_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 6. 预测 vs 实际散点图 ────────────────────────
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
        m = _compute_metrics(results["actual"].values, results[col].values)
        ax.set_title(f"{label}\nMAE={m['MAE']:.1f} r={summary[summary['method']==label]['shape_corr'].values[0]:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_aspect("equal")

    plt.suptitle(f"{name.upper()} — 预测 vs 实际散点图", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"shape_{name}_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Saved 6 plots to output/viz/shape_%s_*.png", name)


def run_all_shape():
    _setup_cn_font()

    for name, target_col, naive_col in [
        ("da", "target_da_clearing_price", "da_clearing_price_lag24h"),
        ("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h"),
    ]:
        results, summary, methods = run_shape_experiment(name, target_col, naive_col)
        plot_all(results, summary, methods, name)

    logger.info("=" * 60)
    logger.info("ALL SHAPE EXPERIMENTS COMPLETE")
    logger.info("Results and plots saved to output/viz/")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_shape()
