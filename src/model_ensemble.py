"""
多模型集成 — LightGBM + XGBoost + CatBoost Stacking。

第一层：3 个基模型（LGB/XGB/CatBoost）各自预测
第二层：Ridge 回归融合 + 简单加权平均

产出：
  - output/ensemble_da_result.csv
  - output/ensemble_rt_result.csv
  - output/ensemble_metrics.csv
"""

import json
import logging
from typing import Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

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


def _load_tuned_params(name: str) -> Dict:
    path = PARAMS_DIR / f"tuning_{name}_best_params.json"
    with open(path) as f:
        return json.load(f)


def _train_lgb(X_tr, y_tr, X_va, y_va, params):
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    logger.info("  LGB best_iter=%d", model.best_iteration)
    return model


def _train_xgb(X_tr, y_tr, X_va, y_va, is_rt):
    params = {
        "objective": "reg:pseudohubererror" if is_rt else "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    logger.info("  XGB best_iter=%d", model.best_iteration)
    return model


def _train_catboost(X_tr, y_tr, X_va, y_va, is_rt):
    model = cb.CatBoostRegressor(
        loss_function="Huber:delta=1.0" if is_rt else "MAE",
        iterations=NUM_BOOST_ROUND,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    logger.info("  CatBoost best_iter=%d", model.get_best_iteration())
    return model


def _get_cv_oof_preds(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    is_rt: bool,
    lgb_params: Dict,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions for stacking (level 1)."""
    n = len(train_df)
    fold_size = n // (n_splits + 1)

    oof_lgb = np.full(n, np.nan)
    oof_xgb = np.full(n, np.nan)
    oof_cb = np.full(n, np.nan)

    for i in range(n_splits):
        t_end = fold_size * (i + 1)
        v_start = t_end
        v_end = min(t_end + fold_size, n)
        if v_end <= v_start:
            continue

        X_tr = train_df.iloc[:t_end][feature_cols]
        y_tr = train_df.iloc[:t_end][target_col]
        X_va = train_df.iloc[v_start:v_end][feature_cols]
        y_va = train_df.iloc[v_start:v_end][target_col]

        lgb_m = _train_lgb(X_tr, y_tr, X_va, y_va, lgb_params)
        xgb_m = _train_xgb(X_tr, y_tr, X_va, y_va, is_rt)
        cb_m = _train_catboost(X_tr, y_tr, X_va, y_va, is_rt)

        oof_lgb[v_start:v_end] = lgb_m.predict(X_va)
        oof_xgb[v_start:v_end] = xgb_m.predict(xgb.DMatrix(X_va))
        oof_cb[v_start:v_end] = cb_m.predict(X_va)

        logger.info("  Fold %d done (train=%d val=%d)", i + 1, t_end, v_end - v_start)

    return oof_lgb, oof_xgb, oof_cb


def run_ensemble(
    name: str,
    target_col: str,
    naive_col: str,
    is_rt: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("ENSEMBLE: %s", name.upper())
    logger.info("=" * 60)

    lgb_params = _load_tuned_params(name)
    df = _load_dataset(name)
    train_df, test_df = _time_split(df)
    feature_cols = [c for c in df.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # ── OOF predictions for stacking meta-learner ─
    logger.info("Generating out-of-fold predictions...")
    oof_lgb, oof_xgb, oof_cb = _get_cv_oof_preds(
        train_df, feature_cols, target_col, is_rt, lgb_params,
    )

    # ── Train final base models on full training set
    logger.info("Training final base models on full training set...")
    lgb_model = _train_lgb(X_train, y_train, X_test, y_test, lgb_params)
    xgb_model = _train_xgb(X_train, y_train, X_test, y_test, is_rt)
    cb_model = _train_catboost(X_train, y_train, X_test, y_test, is_rt)

    test_lgb = lgb_model.predict(X_test)
    test_xgb = xgb_model.predict(xgb.DMatrix(X_test))
    test_cb = cb_model.predict(X_test)

    # ── Stacking: Ridge via numpy (avoid sklearn version issues) ──
    valid_mask = np.isfinite(oof_lgb) & np.isfinite(oof_xgb) & np.isfinite(oof_cb)
    meta_X_train = np.column_stack([
        oof_lgb[valid_mask], oof_xgb[valid_mask], oof_cb[valid_mask],
        np.ones(valid_mask.sum()),
    ])
    meta_y_train = y_train.values[valid_mask]

    alpha = 1.0
    XtX = meta_X_train.T @ meta_X_train + alpha * np.eye(meta_X_train.shape[1])
    Xty = meta_X_train.T @ meta_y_train
    ridge_coef = np.linalg.solve(XtX, Xty)
    logger.info("  Ridge weights: LGB=%.3f XGB=%.3f CB=%.3f intercept=%.3f",
                ridge_coef[0], ridge_coef[1], ridge_coef[2], ridge_coef[3])

    meta_X_test = np.column_stack([test_lgb, test_xgb, test_cb, np.ones(len(test_lgb))])
    test_stacking = meta_X_test @ ridge_coef

    # ── Simple weighted average (1/MAE weights) ──
    lgb_mae = np.mean(np.abs(meta_y_train - oof_lgb[valid_mask]))
    xgb_mae = np.mean(np.abs(meta_y_train - oof_xgb[valid_mask]))
    cb_mae = np.mean(np.abs(meta_y_train - oof_cb[valid_mask]))
    w = np.array([1 / lgb_mae, 1 / xgb_mae, 1 / cb_mae])
    w = w / w.sum()
    test_wavg = test_lgb * w[0] + test_xgb * w[1] + test_cb * w[2]
    logger.info("  Weighted avg: LGB=%.3f XGB=%.3f CB=%.3f", w[0], w[1], w[2])

    naive_vals = test_df[naive_col].values

    results = pd.DataFrame(
        {
            "actual": y_test.values,
            "pred_lgb": test_lgb,
            "pred_xgb": test_xgb,
            "pred_cb": test_cb,
            "pred_stacking": test_stacking,
            "pred_wavg": test_wavg,
            "pred_naive": naive_vals,
        },
        index=test_df.index,
    )
    results.index.name = "ts"
    results.to_csv(OUTPUT_DIR / f"ensemble_{name}_result.csv")

    metrics_records = []
    for method, col in [
        ("lgb", "pred_lgb"), ("xgb", "pred_xgb"), ("catboost", "pred_cb"),
        ("stacking", "pred_stacking"), ("wavg", "pred_wavg"), ("naive", "pred_naive"),
    ]:
        metrics_records.extend(_evaluate_model(results, name, method, "actual", col))
    metrics_df = pd.DataFrame(metrics_records)

    return results, metrics_df


def run_all_ensembles() -> Dict:
    da_res, da_met = run_ensemble("da", "target_da_clearing_price", "da_clearing_price_lag24h", is_rt=False)
    rt_res, rt_met = run_ensemble("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h", is_rt=True)

    all_metrics = pd.concat([da_met, rt_met], ignore_index=True)
    col_order = [
        "model", "method", "group_type", "group_value",
        "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "wMAPE(%)", "MAPE_filtered(%)", "count",
    ]
    all_metrics = all_metrics[col_order]
    all_metrics.to_csv(OUTPUT_DIR / "ensemble_metrics.csv", index=False)

    logger.info("=" * 60)
    logger.info("ENSEMBLE SUMMARY")
    logger.info("=" * 60)
    overall = all_metrics[all_metrics["group_type"] == "overall"]
    for _, row in overall.iterrows():
        logger.info(
            "  %s %-10s | MAE=%.2f  RMSE=%.2f  sMAPE=%.1f%%  wMAPE=%.1f%%",
            row["model"].upper(), row["method"],
            row["MAE"], row["RMSE"], row["sMAPE(%)"], row["wMAPE(%)"],
        )

    return {"da": {"results": da_res, "metrics": da_met}, "rt": {"results": rt_res, "metrics": rt_met}}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_ensembles()
