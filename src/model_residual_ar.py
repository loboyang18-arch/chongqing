"""
残差 AR 修正 — 在 LightGBM 预测基础上叠加 AR(p) 残差修正。

流程：
  1. 用 Optuna 调参后的 LGB 模型预测
  2. 在训练集上拟合 AR(p) 模型（p 通过 AIC 选择，max_lag=48）
  3. 用 AR 模型预测测试集残差，叠加到 LGB 预测上
  4. 对比 LGB-only vs LGB+AR 的测试集表现

产出：
  - output/residual_ar_da_result.csv
  - output/residual_ar_rt_result.csv
  - output/residual_ar_metrics.csv
"""

import json
import logging
import warnings
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from .config import OUTPUT_DIR, PARAMS_DIR
from .model_baseline import (
    EARLY_STOPPING_ROUNDS,
    NUM_BOOST_ROUND,
    TEST_START,
    TRAIN_END,
    _compute_metrics,
    _evaluate_model,
    _load_dataset,
    _time_split,
)

logger = logging.getLogger(__name__)

MAX_AR_LAG = 48


def _load_tuned_params(name: str) -> Dict:
    path = PARAMS_DIR / f"tuning_{name}_best_params.json"
    with open(path) as f:
        return json.load(f)


def _fit_ar_on_residuals(
    residuals: pd.Series,
    max_lag: int = MAX_AR_LAG,
) -> Tuple[AutoReg, int]:
    """Fit AR(p) on residuals, select p by AIC."""
    best_aic = np.inf
    best_p = 1
    best_model = None

    for p in [1, 2, 3, 6, 12, 24, 48]:
        if p > max_lag:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AutoReg(residuals.values, lags=p, old_names=False).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_p = p
                best_model = model
        except Exception:
            continue

    logger.info("  AR(%d) selected (AIC=%.2f)", best_p, best_aic)
    return best_model, best_p


def run_residual_ar(
    name: str,
    target_col: str,
    naive_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("RESIDUAL AR: %s", name.upper())
    logger.info("=" * 60)

    params = _load_tuned_params(name)
    df = _load_dataset(name)
    train_df, test_df = _time_split(df)
    feature_cols = [c for c in df.columns if c != target_col]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # ── Train LGB ────────────────────────────────
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    lgb_model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    logger.info("LGB best iteration: %d", lgb_model.best_iteration)

    # ── Compute train residuals ──────────────────
    y_pred_train = lgb_model.predict(X_train)
    train_residuals = pd.Series(y_train.values - y_pred_train, index=train_df.index)

    # ── Fit AR on train residuals ────────────────
    ar_model, ar_p = _fit_ar_on_residuals(train_residuals)

    # ── Predict test ─────────────────────────────
    y_pred_lgb_test = lgb_model.predict(X_test)

    # AR correction: use rolling prediction
    full_residuals = pd.Series(
        np.concatenate([train_residuals.values, np.zeros(len(test_df))]),
        index=pd.DatetimeIndex(np.concatenate([train_df.index.values, test_df.index.values])),
    )

    ar_corrections = np.zeros(len(test_df))
    for i in range(len(test_df)):
        train_end = len(train_df) + i
        history = full_residuals.values[:train_end]
        if len(history) < ar_p + 1:
            ar_corrections[i] = 0
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                refit = AutoReg(history, lags=ar_p, old_names=False).fit()
                pred = refit.predict(start=len(history), end=len(history))
                ar_corrections[i] = pred.values[0] if len(pred) > 0 else 0
        except Exception:
            ar_corrections[i] = 0

        actual_residual = y_test.values[i] - y_pred_lgb_test[i]
        full_residuals.values[train_end] = actual_residual

    y_pred_combined = y_pred_lgb_test + ar_corrections

    naive_vals = test_df[naive_col].values

    results = pd.DataFrame(
        {
            "actual": y_test.values,
            "pred_lgb": y_pred_lgb_test,
            "pred_lgb_ar": y_pred_combined,
            "pred_naive": naive_vals,
            "ar_correction": ar_corrections,
        },
        index=test_df.index,
    )
    results.index.name = "ts"
    results.to_csv(OUTPUT_DIR / f"residual_ar_{name}_result.csv")

    lgb_m = _compute_metrics(results["actual"].values, results["pred_lgb"].values)
    ar_m = _compute_metrics(results["actual"].values, results["pred_lgb_ar"].values)
    logger.info("  LGB-only:  MAE=%.2f  RMSE=%.2f  sMAPE=%.2f%%", lgb_m["MAE"], lgb_m["RMSE"], lgb_m["sMAPE(%)"])
    logger.info("  LGB+AR(%d): MAE=%.2f  RMSE=%.2f  sMAPE=%.2f%%", ar_p, ar_m["MAE"], ar_m["RMSE"], ar_m["sMAPE(%)"])

    metrics_records = []
    metrics_records.extend(_evaluate_model(results, name, "lgb_tuned", "actual", "pred_lgb"))
    metrics_records.extend(_evaluate_model(results, name, "lgb_ar", "actual", "pred_lgb_ar"))
    metrics_records.extend(_evaluate_model(results, name, "naive", "actual", "pred_naive"))
    metrics_df = pd.DataFrame(metrics_records)

    return results, metrics_df


def run_all_residual_ar() -> Dict:
    da_res, da_met = run_residual_ar("da", "target_da_clearing_price", "da_clearing_price_lag24h")
    rt_res, rt_met = run_residual_ar("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h")

    all_metrics = pd.concat([da_met, rt_met], ignore_index=True)
    col_order = [
        "model", "method", "group_type", "group_value",
        "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "wMAPE(%)", "MAPE_filtered(%)", "count",
    ]
    all_metrics = all_metrics[col_order]
    all_metrics.to_csv(OUTPUT_DIR / "residual_ar_metrics.csv", index=False)

    logger.info("=" * 60)
    logger.info("RESIDUAL AR SUMMARY")
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
    run_all_residual_ar()
