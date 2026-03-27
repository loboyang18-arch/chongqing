"""
V11 特征增强验证 — 跑 LGB DA baseline 并输出形状指标对比报告。

1. 训练增强后的 LGB DA 模型
2. 计算全量形状指标 + 逐日诊断
3. 输出 summary / daily_shape_diagnostics / v11_vs_baseline 到 output/v11_feature_enhanced/
"""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .shape_metrics import compute_shape_report, _split_daily

logger = logging.getLogger(__name__)

TRAIN_END = "2026-02-08 23:00:00"
TEST_START = "2026-02-09 00:00:00"

LGB_PARAMS_DA = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}
LGB_PARAMS_RT = {
    "objective": "huber",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100

OLD_BASELINE = {
    "da": {
        "MAE": 32.02,
        "RMSE": 55.56,
        "profile_corr": 0.3050,
        "peak_hour_err": 6.93,
        "valley_hour_err": 5.47,
        "amplitude_err": 62.37,
        "direction_acc": 0.5342,
    },
}

OUT_DIR = OUTPUT_DIR / "v11_feature_enhanced"


def _daily_diagnostics(actual: np.ndarray, pred: np.ndarray,
                       index: pd.DatetimeIndex) -> pd.DataFrame:
    """逐日计算 corr / peak/valley hour / amplitude 等诊断。"""
    rows = []
    dates = index.date
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() != 24:
            continue
        a = actual[mask]
        p = pred[mask]
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


def _run_lgb_model(name: str, target_col: str, params: dict):
    """训练 LGB 模型并返回测试结果。"""
    path = OUTPUT_DIR / f"feature_{name}.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    logger.info("Loaded %s: %d rows × %d cols", path.name, len(df), len(df.columns))

    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:]
    feature_cols = [c for c in df.columns if c != target_col]

    X_tr, y_tr = train[feature_cols], train[target_col]
    X_te, y_te = test[feature_cols], test[target_col]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)
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

    y_pred = model.predict(X_te)

    results = pd.DataFrame({
        "actual": y_te.values,
        "pred_lgb": y_pred,
    }, index=test.index)
    results.index.name = "ts"

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    return results, importance, model


def run_v11_validation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V11 Feature Enhanced Validation — DA")
    logger.info("=" * 60)

    da_results, da_imp, da_model = _run_lgb_model(
        "da", "target_da_clearing_price", LGB_PARAMS_DA,
    )

    actual = da_results["actual"].values
    pred = da_results["pred_lgb"].values
    idx = da_results.index

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    shape = compute_shape_report(actual, pred, idx, include_v7=True)

    diag = _daily_diagnostics(actual, pred, idx)
    neg_ratio = float(diag["neg_corr_flag"].mean())

    summary = {
        "model": "LGB_DA_v11",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "neg_corr_day_ratio": round(neg_ratio, 4),
    }
    summary.update(shape)

    logger.info("-" * 40)
    logger.info("V11 DA Results:")
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    da_results.to_csv(OUT_DIR / "da_result.csv")
    diag.to_csv(OUT_DIR / "daily_shape_diagnostics.csv", index=False)

    da_imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    logger.info("Top-15 features (gain):")
    for _, row in da_imp.head(15).iterrows():
        logger.info("  %-55s %.1f", row["feature"], row["importance"])

    old = OLD_BASELINE.get("da", {})
    if old:
        compare_rows = []
        for metric in ["MAE", "RMSE", "profile_corr", "peak_hour_err",
                        "valley_hour_err", "amplitude_err", "direction_acc"]:
            old_val = old.get(metric, np.nan)
            new_val = summary.get(metric, np.nan)
            delta = new_val - old_val if np.isfinite(old_val) and np.isfinite(new_val) else np.nan
            pct = delta / abs(old_val) * 100 if np.isfinite(delta) and abs(old_val) > 1e-9 else np.nan
            compare_rows.append({
                "metric": metric,
                "old_baseline": old_val,
                "v11_enhanced": new_val,
                "delta": round(delta, 4) if np.isfinite(delta) else np.nan,
                "change_pct": round(pct, 1) if np.isfinite(pct) else np.nan,
            })
        compare_df = pd.DataFrame(compare_rows)
        compare_df.to_csv(OUT_DIR / "v11_vs_baseline.csv", index=False)
        logger.info("\n=== V11 vs Old Baseline ===")
        for _, row in compare_df.iterrows():
            logger.info(
                "  %-20s old=%-10s new=%-10s Δ=%-10s (%s%%)",
                row["metric"], row["old_baseline"], row["v11_enhanced"],
                row["delta"], row["change_pct"],
            )

    logger.info("=" * 60)
    logger.info("V11 Feature Enhanced Validation — RT")
    logger.info("=" * 60)

    rt_results, rt_imp, rt_model = _run_lgb_model(
        "rt", "target_rt_clearing_price", LGB_PARAMS_RT,
    )

    rt_actual = rt_results["actual"].values
    rt_pred = rt_results["pred_lgb"].values
    rt_idx = rt_results.index

    rt_mae = float(np.mean(np.abs(rt_actual - rt_pred)))
    rt_rmse = float(np.sqrt(np.mean((rt_actual - rt_pred) ** 2)))
    rt_shape = compute_shape_report(rt_actual, rt_pred, rt_idx, include_v7=True)

    rt_diag = _daily_diagnostics(rt_actual, rt_pred, rt_idx)
    rt_neg_ratio = float(rt_diag["neg_corr_flag"].mean())

    rt_summary = {
        "model": "LGB_RT_v11",
        "MAE": round(rt_mae, 2),
        "RMSE": round(rt_rmse, 2),
        "neg_corr_day_ratio": round(rt_neg_ratio, 4),
    }
    rt_summary.update(rt_shape)

    logger.info("-" * 40)
    logger.info("V11 RT Results:")
    for k, v in rt_summary.items():
        logger.info("  %-30s %s", k, v)

    rt_summary_df = pd.DataFrame([rt_summary])
    rt_summary_df.to_csv(OUT_DIR / "summary_rt.csv", index=False)
    rt_results.to_csv(OUT_DIR / "rt_result.csv")
    rt_diag.to_csv(OUT_DIR / "daily_shape_diagnostics_rt.csv", index=False)
    rt_imp.to_csv(OUT_DIR / "feature_importance_rt.csv", index=False)

    all_summary = pd.concat([summary_df, rt_summary_df], ignore_index=True)
    all_summary.to_csv(OUT_DIR / "summary_all.csv", index=False)

    logger.info("=" * 60)
    logger.info("V11 validation complete. Output dir: %s", OUT_DIR)
    logger.info("=" * 60)

    return {"da": summary, "rt": rt_summary}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v11_validation()
