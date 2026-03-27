"""
Optuna 超参数搜索 — 对 DA/RT 分别调优 LightGBM 超参数。

使用 TimeSeriesSplit 5-fold 作为内部验证，目标为 MAE 最小化。
搜索完成后，用最优参数在完整训练集上训练，并在测试集上评估。

产出：
  - params/tuning_da_best_params.json
  - params/tuning_rt_best_params.json
  - output/tuning_metrics.csv
"""

import json
import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

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

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 60
CV_FOLDS = 5


def _objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    is_rt: bool,
) -> float:
    params = {
        "objective": "huber" if is_rt else "regression",
        "metric": "mae",
        "verbosity": -1,
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
    }

    n = len(train_df)
    fold_size = n // (CV_FOLDS + 1)
    fold_maes: List[float] = []

    for i in range(CV_FOLDS):
        t_end = fold_size * (i + 1)
        v_start = t_end
        v_end = min(t_end + fold_size, n)
        if v_end <= v_start:
            continue

        X_tr = train_df.iloc[:t_end][feature_cols]
        y_tr = train_df.iloc[:t_end][target_col]
        X_va = train_df.iloc[v_start:v_end][feature_cols]
        y_va = train_df.iloc[v_start:v_end][target_col]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

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
        y_pred = model.predict(X_va)
        fold_maes.append(np.mean(np.abs(y_va.values - y_pred)))

    return float(np.mean(fold_maes))


def tune_model(
    name: str,
    target_col: str,
    naive_col: str,
    is_rt: bool,
    n_trials: int = N_TRIALS,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("TUNING: %s | target: %s | %d trials", name.upper(), target_col, n_trials)
    logger.info("=" * 60)

    df = _load_dataset(name)
    train_df, test_df = _time_split(df)
    feature_cols = [c for c in df.columns if c != target_col]

    study = optuna.create_study(direction="minimize", study_name=f"tune_{name}")
    study.optimize(
        lambda trial: _objective(trial, train_df, feature_cols, target_col, is_rt),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best["objective"] = "huber" if is_rt else "regression"
    best["metric"] = "mae"
    best["verbosity"] = -1
    best["seed"] = 42

    logger.info("Best CV MAE: %.4f", study.best_value)
    logger.info("Best params: %s", json.dumps(best, indent=2))

    params_path = PARAMS_DIR / f"tuning_{name}_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Saved %s", params_path.name)

    # ── Final training with best params ──────────
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    model = lgb.train(
        best, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    logger.info("Final model best iteration: %d", model.best_iteration)
    y_pred = model.predict(X_test)

    y_pred_naive = test_df[naive_col].values

    results = pd.DataFrame(
        {"actual": y_test.values, "pred_lgb": y_pred, "pred_naive": y_pred_naive},
        index=test_df.index,
    )
    results.index.name = "ts"
    results.to_csv(OUTPUT_DIR / f"tuning_{name}_result.csv")

    metrics_records = []
    metrics_records.extend(_evaluate_model(results, name, "lgb_tuned", "actual", "pred_lgb"))
    metrics_records.extend(_evaluate_model(results, name, "naive", "actual", "pred_naive"))
    metrics_df = pd.DataFrame(metrics_records)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    logger.info("Top-10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info("  %-50s %.1f", row["feature"], row["importance"])

    return best, results, metrics_df


def run_tuning() -> Dict:
    da_params, da_results, da_metrics = tune_model(
        "da", "target_da_clearing_price", "da_clearing_price_lag24h", is_rt=False,
    )
    rt_params, rt_results, rt_metrics = tune_model(
        "rt", "target_rt_clearing_price", "rt_clearing_price_lag24h", is_rt=True,
    )

    all_metrics = pd.concat([da_metrics, rt_metrics], ignore_index=True)
    col_order = [
        "model", "method", "group_type", "group_value",
        "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "wMAPE(%)", "MAPE_filtered(%)", "count",
    ]
    all_metrics = all_metrics[col_order]
    all_metrics.to_csv(OUTPUT_DIR / "tuning_metrics.csv", index=False)

    logger.info("=" * 60)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 60)
    overall = all_metrics[all_metrics["group_type"] == "overall"]
    for _, row in overall.iterrows():
        logger.info(
            "  %s %-10s | MAE=%.2f  RMSE=%.2f  sMAPE=%.1f%%  wMAPE=%.1f%%",
            row["model"].upper(), row["method"],
            row["MAE"], row["RMSE"], row["sMAPE(%)"], row["wMAPE(%)"],
        )

    return {"da_params": da_params, "rt_params": rt_params}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_tuning()
