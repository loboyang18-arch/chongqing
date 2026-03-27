"""
变化率预测模型 — 预测相对于昨日同小时的价格变化率，而非绝对价格。

核心思想：
  target = (P_t - P_{t-24}) / max(|P_{t-24}|, 50)
  final_price = P_{t-24} + target_pred * max(|P_{t-24}|, 50)

分母 clip 到 50 避免低价时变化率爆炸。

产出：
  - output/delta_da_result.csv
  - output/delta_rt_result.csv
  - output/delta_comparison.png (与绝对值模型曲线对比)
"""

import json
import logging
import os
from typing import Dict, Tuple

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
    _time_split,
)

logger = logging.getLogger(__name__)

DENOM_FLOOR = 50.0


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


def run_delta_model(
    name: str,
    target_col: str,
    naive_col: str,
    price_col: str,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("DELTA MODEL: %s", name.upper())
    logger.info("=" * 60)

    df = _load_dataset(name)

    # ── 构建变化率目标 ────────────────────────────
    base_price = df[naive_col]  # = lag24h price = P_{t-24}
    actual_price = df[target_col]
    denom = base_price.abs().clip(lower=DENOM_FLOOR)
    delta_target = (actual_price - base_price) / denom

    df["target_delta"] = delta_target
    df["_base_price"] = base_price
    df["_denom"] = denom

    feature_cols = [c for c in df.columns
                    if c not in (target_col, "target_delta", "_base_price", "_denom")]

    train_df = df.loc[:TRAIN_END]
    test_df = df.loc[TEST_START:]

    valid_train = train_df.dropna(subset=["target_delta"])
    valid_test = test_df.dropna(subset=["target_delta"])

    logger.info("Train: %d rows, Test: %d rows", len(valid_train), len(valid_test))

    X_train = valid_train[feature_cols]
    y_train = valid_train["target_delta"]
    X_test = valid_test[feature_cols]
    y_test = valid_test["target_delta"]

    # ── 使用 Huber 损失训练（对变化率中的极端值更稳健）──
    params = _load_tuned_params(name)
    params["objective"] = "huber"

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
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
    logger.info("Best iteration: %d", model.best_iteration)

    # ── 预测变化率 → 反变换为绝对价格 ────────────
    delta_pred = model.predict(X_test)
    base = valid_test["_base_price"].values
    denom_vals = valid_test["_denom"].values
    price_pred = base + delta_pred * denom_vals

    actual_vals = valid_test[target_col].values
    naive_vals = valid_test[naive_col].values

    results = pd.DataFrame({
        "actual": actual_vals,
        "pred_delta": price_pred,
        "pred_naive": naive_vals,
        "delta_actual": y_test.values,
        "delta_pred": delta_pred,
        "base_price": base,
    }, index=valid_test.index)
    results.index.name = "ts"
    results.to_csv(OUTPUT_DIR / f"delta_{name}_result.csv")

    m_delta = _compute_metrics(actual_vals, price_pred)
    m_naive = _compute_metrics(actual_vals, naive_vals)
    logger.info("  Delta model: MAE=%.2f  RMSE=%.2f  sMAPE=%.2f%%  wMAPE=%.2f%%",
                m_delta["MAE"], m_delta["RMSE"], m_delta["sMAPE(%)"], m_delta["wMAPE(%)"])
    logger.info("  Naive:       MAE=%.2f  RMSE=%.2f", m_naive["MAE"], m_naive["RMSE"])

    # ── 特征重要度 ────────────────────────────────
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info("Top-10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info("  %-50s %.1f", row["feature"], row["importance"])

    return results


def run_all_delta():
    _setup_cn_font()

    da = run_delta_model("da", "target_da_clearing_price", "da_clearing_price_lag24h",
                         "da_clearing_price")
    rt = run_delta_model("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h",
                         "rt_clearing_price")

    # ── 加载旧模型结果对比 ────────────────────────
    old_da = pd.read_csv(OUTPUT_DIR / "tuning_da_result.csv", parse_dates=["ts"], index_col="ts")
    old_rt = pd.read_csv(OUTPUT_DIR / "ensemble_rt_result.csv", parse_dates=["ts"], index_col="ts")

    _plot_comparison(da, old_da, "DA", "pred_lgb")
    _plot_comparison(rt, old_rt, "RT", "pred_wavg")

    # ── 合并绘制总对比图 ─────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    for ax, new, old, old_col, name, color_new, color_old in [
        (axes[0], da, old_da, "pred_lgb", "DA 日前电价", "#E91E63", "#90CAF9"),
        (axes[1], rt, old_rt, "pred_wavg", "RT 实时电价", "#E91E63", "#90CAF9"),
    ]:
        ax.plot(new.index, new["actual"], color="black", linewidth=1, label="实际", zorder=3)
        ax.plot(new.index, old[old_col].reindex(new.index), color=color_old, linewidth=0.9,
                alpha=0.7, label="绝对值模型", zorder=2)
        ax.plot(new.index, new["pred_delta"], color=color_new, linewidth=0.9,
                alpha=0.8, label="变化率模型", zorder=2)

        mae_old = np.mean(np.abs(new["actual"].values - old[old_col].reindex(new.index).values))
        mae_new = np.mean(np.abs(new["actual"].values - new["pred_delta"].values))

        ax.set_ylabel("电价 (元/MWh)")
        ax.set_title(f"{name}  |  绝对值模型 MAE={mae_old:.1f} → 变化率模型 MAE={mae_new:.1f}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delta_comparison.png", dpi=120, bbox_inches="tight")
    logger.info("Saved delta_comparison.png")
    plt.close()

    # ── 缩放到最差周 ─────────────────────────────
    _plot_worst_week(da, old_da, "DA", "pred_lgb")
    _plot_worst_week(rt, old_rt, "RT", "pred_wavg")

    logger.info("=" * 60)
    logger.info("DELTA MODEL SUMMARY")
    logger.info("=" * 60)
    for name, new_df, old_df, old_col in [
        ("DA", da, old_da, "pred_lgb"),
        ("RT", rt, old_rt, "pred_wavg"),
    ]:
        m_new = _compute_metrics(new_df["actual"].values, new_df["pred_delta"].values)
        m_old = _compute_metrics(new_df["actual"].values,
                                 old_df[old_col].reindex(new_df.index).values)
        m_naive = _compute_metrics(new_df["actual"].values, new_df["pred_naive"].values)
        logger.info("  %s 变化率:  MAE=%.2f  RMSE=%.2f  sMAPE=%.2f%%  wMAPE=%.2f%%",
                    name, m_new["MAE"], m_new["RMSE"], m_new["sMAPE(%)"], m_new["wMAPE(%)"])
        logger.info("  %s 绝对值:  MAE=%.2f  RMSE=%.2f  sMAPE=%.2f%%  wMAPE=%.2f%%",
                    name, m_old["MAE"], m_old["RMSE"], m_old["sMAPE(%)"], m_old["wMAPE(%)"])
        logger.info("  %s Naive:   MAE=%.2f", name, m_naive["MAE"])


def _plot_comparison(new_df, old_df, name, old_col):
    """单模型缩放对比 — 选高误差日做详细对比。"""
    new_df["date"] = new_df.index.date
    daily_mae = new_df.groupby("date").apply(
        lambda g: np.mean(np.abs(g["actual"] - g["pred_delta"]))
    )
    worst4 = daily_mae.nlargest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, d in enumerate(worst4):
        ax = axes[i]
        day_new = new_df[new_df["date"] == d]
        hours = day_new.index.hour

        ax.plot(hours, day_new["actual"], "o-", color="black", linewidth=2, markersize=4,
                label="实际", zorder=3)
        ax.plot(hours, day_new["pred_delta"], "s-", color="#E91E63", linewidth=1.5, markersize=3,
                label="变化率", zorder=2)

        old_day = old_df[old_col].reindex(day_new.index)
        ax.plot(hours, old_day, "^--", color="#90CAF9", linewidth=1.2, markersize=3,
                label="绝对值", zorder=1, alpha=0.8)

        ax.fill_between(hours, day_new["actual"], day_new["pred_delta"],
                        alpha=0.15, color="#E91E63")

        mae_new = np.mean(np.abs(day_new["actual"] - day_new["pred_delta"]))
        mae_old = np.mean(np.abs(day_new["actual"].values - old_day.values))
        ax.set_title(f"{name} {str(d)[5:]}\n变化率 MAE={mae_new:.1f} | 绝对值 MAE={mae_old:.1f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("小时")
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"{name} — 高误差日对比（变化率 vs 绝对值）", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"delta_{name.lower()}_worst_days.png", dpi=120, bbox_inches="tight")
    logger.info("Saved delta_%s_worst_days.png", name.lower())
    plt.close()


def _plot_worst_week(new_df, old_df, name, old_col):
    """缩放到 03-01 ~ 03-07（含最差日期）。"""
    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2026-03-07 23:00:00")

    w_new = new_df.loc[start:end]
    w_old = old_df[old_col].reindex(w_new.index)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(w_new.index, w_new["actual"], "o-", color="black", markersize=2,
            linewidth=1.5, label="实际", zorder=3)
    ax.plot(w_new.index, w_new["pred_delta"], "s-", color="#E91E63", markersize=2,
            linewidth=1.5, label="变化率模型", zorder=2)
    ax.plot(w_new.index, w_old, "^--", color="#90CAF9", markersize=2,
            linewidth=1.2, label="绝对值模型", alpha=0.8, zorder=1)
    ax.fill_between(w_new.index, w_new["actual"], w_new["pred_delta"],
                    alpha=0.1, color="#E91E63")

    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name} — 最差周缩放 (03-01 ~ 03-07)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"delta_{name.lower()}_worst_week.png", dpi=120, bbox_inches="tight")
    logger.info("Saved delta_%s_worst_week.png", name.lower())
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_delta()
