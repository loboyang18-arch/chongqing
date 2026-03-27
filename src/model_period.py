"""
分时段建模 — 按 peak/valley/flat 分别训练 LightGBM，并与单模型对比。

使用 Optuna 调参后的最优参数作为基础参数。
对比方案：
  A) 单模型 + hour 特征（当前 baseline）
  B) 3 个分时段独立模型（peak/valley/flat 各自训练）

产出：
  - output/period_model_result.csv — 分时段模型预测结果
  - output/period_model_metrics.csv — 分时段 vs 单模型对比
"""

import json
import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, PARAMS_DIR
from .model_baseline import (
    EARLY_STOPPING_ROUNDS,
    FLAT_HOURS,
    NUM_BOOST_ROUND,
    PEAK_HOURS,
    TEST_START,
    TRAIN_END,
    VALLEY_HOURS,
    _compute_metrics,
    _evaluate_model,
    _load_dataset,
    _period_label,
    _time_split,
)

logger = logging.getLogger(__name__)

PERIOD_MAP = {
    "peak": PEAK_HOURS,
    "valley": VALLEY_HOURS,
    "flat": FLAT_HOURS,
}


def _load_tuned_params(name: str) -> Dict:
    path = PARAMS_DIR / f"tuning_{name}_best_params.json"
    with open(path) as f:
        return json.load(f)


def _train_period_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Dict,
    period_name: str,
    hours: set,
) -> Tuple[np.ndarray, np.ndarray, lgb.Booster]:
    """Train on subset of hours, predict on same subset."""
    train_mask = train_df.index.hour.isin(hours)
    test_mask = test_df.index.hour.isin(hours)

    X_tr = train_df.loc[train_mask, feature_cols]
    y_tr = train_df.loc[train_mask, target_col]
    X_te = test_df.loc[test_mask, feature_cols]
    y_te = test_df.loc[test_mask, target_col]

    logger.info(
        "  %s: train=%d test=%d", period_name, len(X_tr), len(X_te),
    )

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    y_pred = model.predict(X_te)
    logger.info("  %s: best_iter=%d, MAE=%.2f", period_name, model.best_iteration,
                np.mean(np.abs(y_te.values - y_pred)))
    return y_te.values, y_pred, model


def run_period_model(
    name: str,
    target_col: str,
    naive_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("PERIOD MODEL: %s", name.upper())
    logger.info("=" * 60)

    params = _load_tuned_params(name)
    df = _load_dataset(name)
    train_df, test_df = _time_split(df)
    feature_cols = [c for c in df.columns if c != target_col]

    all_actuals = []
    all_preds = []
    all_indices = []

    for period_name, hours in PERIOD_MAP.items():
        y_actual, y_pred, _ = _train_period_model(
            train_df, test_df, feature_cols, target_col, params, period_name, hours,
        )
        test_mask = test_df.index.hour.isin(hours)
        all_actuals.append(y_actual)
        all_preds.append(y_pred)
        all_indices.append(test_df.index[test_mask])

    idx = np.concatenate([i.values for i in all_indices])
    actual = np.concatenate(all_actuals)
    pred = np.concatenate(all_preds)

    sort_order = np.argsort(idx)
    idx = idx[sort_order]
    actual = actual[sort_order]
    pred = pred[sort_order]

    naive_vals = test_df.loc[pd.DatetimeIndex(idx), naive_col].values

    results = pd.DataFrame(
        {"actual": actual, "pred_period": pred, "pred_naive": naive_vals},
        index=pd.DatetimeIndex(idx, name="ts"),
    )
    results.to_csv(OUTPUT_DIR / f"period_{name}_result.csv")

    metrics_records = []
    metrics_records.extend(_evaluate_model(results, name, "lgb_period", "actual", "pred_period"))
    metrics_records.extend(_evaluate_model(results, name, "naive", "actual", "pred_naive"))
    metrics_df = pd.DataFrame(metrics_records)

    return results, metrics_df


def run_all_period_models() -> Dict:
    da_res, da_met = run_period_model("da", "target_da_clearing_price", "da_clearing_price_lag24h")
    rt_res, rt_met = run_period_model("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h")

    all_metrics = pd.concat([da_met, rt_met], ignore_index=True)
    col_order = [
        "model", "method", "group_type", "group_value",
        "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "wMAPE(%)", "MAPE_filtered(%)", "count",
    ]
    all_metrics = all_metrics[col_order]
    all_metrics.to_csv(OUTPUT_DIR / "period_model_metrics.csv", index=False)

    logger.info("=" * 60)
    logger.info("PERIOD MODEL SUMMARY")
    logger.info("=" * 60)
    overall = all_metrics[all_metrics["group_type"] == "overall"]
    for _, row in overall.iterrows():
        logger.info(
            "  %s %-12s | MAE=%.2f  sMAPE=%.1f%%  wMAPE=%.1f%%",
            row["model"].upper(), row["method"],
            row["MAE"], row["sMAPE(%)"], row["wMAPE(%)"],
        )

    return {"da": {"results": da_res, "metrics": da_met}, "rt": {"results": rt_res, "metrics": rt_met}}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_period_models()
