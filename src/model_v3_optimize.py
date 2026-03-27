"""
四项增量优化实验 — V3

Exp1: RT spread 目标 (rt_price - da_d0) vs 原始目标
Exp2: DA/RT 分位数回归 P10/P50/P90
Exp3: Ramp 特征 + 状态 Flag 注入 (ablation)
Exp4: 近期样本加权

所有结果输出到 output/v3_optimize/
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
    PEAK_HOURS,
    VALLEY_HOURS,
    TEST_START,
    TRAIN_END,
    _compute_metrics,
    _load_dataset,
)

logger = logging.getLogger(__name__)
V3_DIR = OUTPUT_DIR / "v3_optimize"
V3_DIR.mkdir(exist_ok=True)


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


def _compute_shape_corr(actual: np.ndarray, pred: np.ndarray,
                        index: pd.DatetimeIndex) -> float:
    dates = index.date
    corrs = []
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() == 24:
            a, p = actual[mask], pred[mask]
            if np.std(a) > 1e-6 and np.std(p) > 1e-6:
                c = np.corrcoef(a, p)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
    return np.mean(corrs) if corrs else np.nan


def _train_model(X_tr, y_tr, X_va, y_va, params, weights=None):
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=weights)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
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


# ======================================================================
# Exp1: RT Spread 目标
# ======================================================================

def _add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RT-DA spread history features to an RT dataset."""
    out = df.copy()
    if "rt_clearing_price_lag24h" in df.columns and "da_clearing_price_lag24h" in df.columns:
        spread_lag24 = df["rt_clearing_price_lag24h"] - df["da_clearing_price_lag24h"]
        out["rt_da_spread_lag24h"] = spread_lag24
        out["rt_da_spread_roll24h_mean"] = spread_lag24.rolling(24, min_periods=12).mean()
        out["rt_da_spread_roll24h_std"] = spread_lag24.rolling(24, min_periods=12).std()
    if "rt_price_lag48h" in df.columns and "da_price_lag48h" in df.columns:
        out["rt_da_spread_lag48h"] = df["rt_price_lag48h"] - df["da_price_lag48h"]
    return out


def run_exp1_rt_spread():
    logger.info("=" * 60)
    logger.info("EXP1: RT Spread Target")
    logger.info("=" * 60)

    df = _load_dataset("rt")
    params = _load_tuned_params("rt")
    target_col = "target_rt_clearing_price"
    feature_cols = [c for c in df.columns if c != target_col]

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()

    # --- RT-A: original target ---
    logger.info("--- RT-A: Original target ---")
    model_a = _train_model(
        train_df[feature_cols], train_df[target_col],
        test_df[feature_cols], test_df[target_col], params)
    pred_a = model_a.predict(test_df[feature_cols])

    # --- RT-B: spread target ---
    logger.info("--- RT-B: Spread target (rt - da_d0) ---")
    da_d0_col = "da_clearing_price_d0"

    df_b = _add_spread_features(df)
    df_b["target_spread"] = df_b[target_col] - df_b[da_d0_col]
    feat_cols_b = [c for c in df_b.columns
                   if c not in (target_col, "target_spread")]

    train_b = df_b.loc[:TRAIN_END].copy()
    test_b = df_b.loc[TEST_START:].copy()

    model_b = _train_model(
        train_b[feat_cols_b], train_b["target_spread"],
        test_b[feat_cols_b], test_b["target_spread"], params)
    spread_pred = model_b.predict(test_b[feat_cols_b])
    pred_b = test_b[da_d0_col].values + spread_pred

    actual = test_df[target_col].values
    m_a = _compute_metrics(actual, pred_a)
    m_b = _compute_metrics(actual, pred_b)
    sr_a = _compute_shape_corr(actual, pred_a, test_df.index)
    sr_b = _compute_shape_corr(actual, pred_b, test_df.index)

    comparison = pd.DataFrame([
        {"method": "RT-A_original", **m_a, "shape_r": round(sr_a, 4)},
        {"method": "RT-B_spread", **m_b, "shape_r": round(sr_b, 4)},
    ])
    comparison.to_csv(V3_DIR / "rt_spread_comparison.csv", index=False)
    logger.info("RT-A: MAE=%.2f RMSE=%.2f shape_r=%.3f", m_a["MAE"], m_a["RMSE"], sr_a)
    logger.info("RT-B: MAE=%.2f RMSE=%.2f shape_r=%.3f", m_b["MAE"], m_b["RMSE"], sr_b)

    result = pd.DataFrame({
        "actual": actual,
        "pred_rt_a": pred_a,
        "pred_rt_b_spread": pred_b,
        "spread_pred": spread_pred,
        "da_d0": test_b[da_d0_col].values,
    }, index=test_df.index)
    result.to_csv(V3_DIR / "rt_spread_result.csv")

    # plots
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(result.index, result["actual"], color="black", linewidth=1.0, label="实际")
    ax.plot(result.index, result["pred_rt_a"], color="#90CAF9", linewidth=0.8,
            alpha=0.7, label=f"RT-A 原始 (MAE={m_a['MAE']:.1f})")
    ax.plot(result.index, result["pred_rt_b_spread"], color="#E91E63", linewidth=0.8,
            alpha=0.85, label=f"RT-B Spread (MAE={m_b['MAE']:.1f})")
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title("RT Spread 目标实验 - 全时段对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(V3_DIR / "rt_spread_full.png", dpi=120, bbox_inches="tight")
    plt.close()

    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2026-03-07 23:00:00")
    week = result.loc[start:end]
    if len(week) > 0:
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(week.index, week["actual"], "o-", color="black", markersize=2,
                linewidth=1.2, label="实际")
        ax.plot(week.index, week["pred_rt_a"], color="#90CAF9", linewidth=1.0,
                alpha=0.7, label="RT-A 原始")
        ax.plot(week.index, week["pred_rt_b_spread"], color="#E91E63", linewidth=1.0,
                alpha=0.85, label="RT-B Spread")
        ax.set_ylabel("电价 (元/MWh)")
        ax.set_title("RT Spread 目标 - 缩放周 03-01~03-07", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))
        plt.tight_layout()
        plt.savefig(V3_DIR / "rt_spread_zoom.png", dpi=120, bbox_inches="tight")
        plt.close()

    return comparison


# ======================================================================
# Exp2: 分位数回归
# ======================================================================

def _run_quantile_for_task(name: str, target_col: str):
    logger.info("--- Quantile regression: %s ---", name.upper())
    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()

    results = {"actual": test_df[target_col].values}
    for alpha, label in [(0.1, "P10"), (0.5, "P50"), (0.9, "P90")]:
        q_params = params.copy()
        q_params["objective"] = "quantile"
        q_params["alpha"] = alpha
        q_params["metric"] = "quantile"
        model = _train_model(
            train_df[feature_cols], train_df[target_col],
            test_df[feature_cols], test_df[target_col], q_params)
        results[label] = model.predict(test_df[feature_cols])
        logger.info("  %s trained (iter=%d)", label, model.best_iteration)

    result_df = pd.DataFrame(results, index=test_df.index)
    result_df["interval_width"] = result_df["P90"] - result_df["P10"]
    result_df.to_csv(V3_DIR / f"{name}_quantile_result.csv")

    coverage = np.mean(
        (result_df["actual"] >= result_df["P10"]) &
        (result_df["actual"] <= result_df["P90"])
    )
    avg_width = result_df["interval_width"].mean()
    p50_mae = np.mean(np.abs(result_df["actual"] - result_df["P50"]))
    logger.info("  Coverage (P10-P90): %.1f%%  Avg width: %.1f  P50 MAE: %.2f",
                coverage * 100, avg_width, p50_mae)

    # plot
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.fill_between(result_df.index, result_df["P10"], result_df["P90"],
                     alpha=0.2, color="#E91E63", label="P10-P90 区间")
    ax.plot(result_df.index, result_df["actual"], color="black", linewidth=1.0,
            label="实际")
    ax.plot(result_df.index, result_df["P50"], color="#E91E63", linewidth=0.8,
            alpha=0.85, label=f"P50 (MAE={p50_mae:.1f})")
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} 分位数回归 - P10/P50/P90 (覆盖率={coverage*100:.0f}%)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(V3_DIR / f"{name}_quantile_band.png", dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "task": name, "coverage_pct": round(coverage * 100, 1),
        "avg_interval_width": round(avg_width, 1),
        "P50_MAE": round(p50_mae, 2),
    }


def run_exp2_quantile():
    logger.info("=" * 60)
    logger.info("EXP2: Quantile Regression")
    logger.info("=" * 60)
    rows = []
    for name, target_col in [
        ("da", "target_da_clearing_price"),
        ("rt", "target_rt_clearing_price"),
    ]:
        rows.append(_run_quantile_for_task(name, target_col))
    return pd.DataFrame(rows)


# ======================================================================
# Exp3: Ramp + Flag 特征 ablation
# ======================================================================

def _add_ramp_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add 8 ramp features. Returns augmented df and list of new column names."""
    out = df.copy()
    new_cols = []

    if "load_forecast" in df.columns:
        out["load_ramp_1h"] = df["load_forecast"].diff(1)
        out["load_ramp_24h"] = df["load_forecast"] - df["load_forecast"].shift(24)
        new_cols += ["load_ramp_1h", "load_ramp_24h"]

    if "renewable_fcst_total_am" in df.columns:
        out["renewable_ramp_1h"] = df["renewable_fcst_total_am"].diff(1)
        out["renewable_ramp_24h"] = df["renewable_fcst_total_am"] - df["renewable_fcst_total_am"].shift(24)
        new_cols += ["renewable_ramp_1h", "renewable_ramp_24h"]

    if "net_load_forecast" in df.columns:
        out["net_load_ramp_1h"] = df["net_load_forecast"].diff(1)
        out["net_load_ramp_24h"] = df["net_load_forecast"] - df["net_load_forecast"].shift(24)
        new_cols += ["net_load_ramp_1h", "net_load_ramp_24h"]

    if "tie_line_fcst_am" in df.columns:
        out["tie_line_ramp_1h"] = df["tie_line_fcst_am"].diff(1)
        new_cols.append("tie_line_ramp_1h")

    if "supply_demand_gap" in df.columns:
        out["supply_gap_ramp_1h"] = df["supply_demand_gap"].diff(1)
        new_cols.append("supply_gap_ramp_1h")

    return out, new_cols


def _add_flag_features(df: pd.DataFrame, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add 5 state flag features. Thresholds computed from train_df."""
    out = df.copy()
    new_cols = []

    out["peak_flag"] = df.index.hour.isin(PEAK_HOURS).astype(int)
    out["valley_flag"] = df.index.hour.isin(VALLEY_HOURS).astype(int)
    new_cols += ["peak_flag", "valley_flag"]

    if "renewable_ratio" in df.columns and "renewable_ratio" in train_df.columns:
        threshold = train_df["renewable_ratio"].quantile(0.75)
        out["high_renewable_flag"] = (df["renewable_ratio"] > threshold).astype(int)
        new_cols.append("high_renewable_flag")

    if "da_price_roll24h_mean" in df.columns and "da_price_roll24h_mean" in train_df.columns:
        threshold = train_df["da_price_roll24h_mean"].quantile(0.10)
        out["low_price_risk_flag"] = (df["da_price_roll24h_mean"] < threshold).astype(int)
        new_cols.append("low_price_risk_flag")

    if "supply_demand_gap" in df.columns and "supply_demand_gap" in train_df.columns:
        threshold = train_df["supply_demand_gap"].abs().quantile(0.90)
        out["extreme_gap_flag"] = (df["supply_demand_gap"].abs() > threshold).astype(int)
        new_cols.append("extreme_gap_flag")

    return out, new_cols


def run_exp3_features():
    logger.info("=" * 60)
    logger.info("EXP3: Ramp + Flag Feature Ablation")
    logger.info("=" * 60)

    rows = []
    for name, target_col in [
        ("da", "target_da_clearing_price"),
        ("rt", "target_rt_clearing_price"),
    ]:
        logger.info("--- %s ---", name.upper())
        df = _load_dataset(name)
        params = _load_tuned_params(name)
        base_feat_cols = [c for c in df.columns if c != target_col]

        train_df = df.loc[:TRAIN_END].copy()
        test_df = df.loc[TEST_START:].copy()

        # Group A: baseline
        logger.info("  [A] Baseline (%d features)", len(base_feat_cols))
        model_a = _train_model(
            train_df[base_feat_cols], train_df[target_col],
            test_df[base_feat_cols], test_df[target_col], params)
        pred_a = model_a.predict(test_df[base_feat_cols])
        m_a = _compute_metrics(test_df[target_col].values, pred_a)
        sr_a = _compute_shape_corr(test_df[target_col].values, pred_a, test_df.index)
        rows.append({"task": name, "group": "A_baseline",
                      "n_features": len(base_feat_cols), **m_a, "shape_r": round(sr_a, 4)})

        # Group B: baseline + ramp
        df_ramp, ramp_cols = _add_ramp_features(df)
        train_r = df_ramp.loc[:TRAIN_END].copy()
        test_r = df_ramp.loc[TEST_START:].copy()
        feat_b = base_feat_cols + ramp_cols
        logger.info("  [B] Baseline + %d ramp (%d features)", len(ramp_cols), len(feat_b))
        model_b = _train_model(
            train_r[feat_b], train_r[target_col],
            test_r[feat_b], test_r[target_col], params)
        pred_b = model_b.predict(test_r[feat_b])
        m_b = _compute_metrics(test_df[target_col].values, pred_b)
        sr_b = _compute_shape_corr(test_df[target_col].values, pred_b, test_df.index)
        rows.append({"task": name, "group": "B_ramp",
                      "n_features": len(feat_b), **m_b, "shape_r": round(sr_b, 4)})

        # Group C: baseline + ramp + flags
        df_rf, flag_cols = _add_flag_features(df_ramp, train_r)
        train_rf = df_rf.loc[:TRAIN_END].copy()
        test_rf = df_rf.loc[TEST_START:].copy()
        feat_c = feat_b + flag_cols
        logger.info("  [C] Baseline + ramp + %d flags (%d features)", len(flag_cols), len(feat_c))
        model_c = _train_model(
            train_rf[feat_c], train_rf[target_col],
            test_rf[feat_c], test_rf[target_col], params)
        pred_c = model_c.predict(test_rf[feat_c])
        m_c = _compute_metrics(test_df[target_col].values, pred_c)
        sr_c = _compute_shape_corr(test_df[target_col].values, pred_c, test_df.index)
        rows.append({"task": name, "group": "C_ramp_flag",
                      "n_features": len(feat_c), **m_c, "shape_r": round(sr_c, 4)})

    ablation = pd.DataFrame(rows)
    ablation.to_csv(V3_DIR / "feature_ablation.csv", index=False)
    logger.info("\nAblation results:\n%s", ablation.to_string(index=False))
    return ablation


# ======================================================================
# Exp4: 近期样本加权
# ======================================================================

def run_exp4_sample_weight():
    logger.info("=" * 60)
    logger.info("EXP4: Recent Sample Weighting")
    logger.info("=" * 60)

    rows = []
    for name, target_col in [
        ("da", "target_da_clearing_price"),
        ("rt", "target_rt_clearing_price"),
    ]:
        logger.info("--- %s ---", name.upper())
        df = _load_dataset(name)
        params = _load_tuned_params(name)
        feature_cols = [c for c in df.columns if c != target_col]

        train_df = df.loc[:TRAIN_END].copy()
        test_df = df.loc[TEST_START:].copy()
        actual = test_df[target_col].values

        # baseline (no weights)
        logger.info("  [no weight]")
        model_base = _train_model(
            train_df[feature_cols], train_df[target_col],
            test_df[feature_cols], test_df[target_col], params)
        pred_base = model_base.predict(test_df[feature_cols])
        m_base = _compute_metrics(actual, pred_base)
        sr_base = _compute_shape_corr(actual, pred_base, test_df.index)
        rows.append({"task": name, "weighting": "none", **m_base, "shape_r": round(sr_base, 4)})

        # weighted
        train_end_date = pd.Timestamp(TRAIN_END).date()
        days_to_end = np.array([(train_end_date - d).days for d in train_df.index.date])
        weights = np.where(days_to_end <= 30, 1.4,
                           np.where(days_to_end <= 60, 1.2, 1.0))
        logger.info("  [weighted] w=1.4/1.2/1.0 by recency")
        model_w = _train_model(
            train_df[feature_cols], train_df[target_col],
            test_df[feature_cols], test_df[target_col], params,
            weights=weights)
        pred_w = model_w.predict(test_df[feature_cols])
        m_w = _compute_metrics(actual, pred_w)
        sr_w = _compute_shape_corr(actual, pred_w, test_df.index)
        rows.append({"task": name, "weighting": "recency_1.4_1.2_1.0",
                      **m_w, "shape_r": round(sr_w, 4)})

    weight_df = pd.DataFrame(rows)
    weight_df.to_csv(V3_DIR / "sample_weight_comparison.csv", index=False)
    logger.info("\nWeight comparison:\n%s", weight_df.to_string(index=False))
    return weight_df


# ======================================================================
# 汇总
# ======================================================================

def run_all():
    _setup_cn_font()

    logger.info("=" * 60)
    logger.info("V3 OPTIMIZATION EXPERIMENTS")
    logger.info("=" * 60)

    exp1 = run_exp1_rt_spread()
    exp2 = run_exp2_quantile()
    exp3 = run_exp3_features()
    exp4 = run_exp4_sample_weight()

    summary_parts = []

    for _, row in exp1.iterrows():
        summary_parts.append({
            "experiment": "Exp1_RT_Spread", "task": "rt",
            "variant": row["method"], "MAE": row["MAE"], "RMSE": row["RMSE"],
            "shape_r": row.get("shape_r", ""),
            "note": "",
        })

    for _, row in exp2.iterrows():
        summary_parts.append({
            "experiment": "Exp2_Quantile", "task": row["task"],
            "variant": "P10/P50/P90",
            "MAE": row["P50_MAE"], "RMSE": "",
            "shape_r": "",
            "note": f"coverage={row['coverage_pct']}%, width={row['avg_interval_width']}",
        })

    for _, row in exp3.iterrows():
        summary_parts.append({
            "experiment": "Exp3_Features", "task": row["task"],
            "variant": row["group"],
            "MAE": row["MAE"], "RMSE": row["RMSE"],
            "shape_r": row.get("shape_r", ""),
            "note": f"n_feat={row['n_features']}",
        })

    for _, row in exp4.iterrows():
        summary_parts.append({
            "experiment": "Exp4_Weight", "task": row["task"],
            "variant": row["weighting"],
            "MAE": row["MAE"], "RMSE": row["RMSE"],
            "shape_r": row.get("shape_r", ""),
            "note": "",
        })

    summary = pd.DataFrame(summary_parts)
    summary.to_csv(V3_DIR / "summary.csv", index=False)
    logger.info("\n=== V3 Summary ===\n%s", summary.to_string(index=False))

    logger.info("=" * 60)
    logger.info("ALL V3 EXPERIMENTS COMPLETE")
    logger.info("Results in output/v3_optimize/")
    logger.info("=" * 60)

    return {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "summary": summary}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
