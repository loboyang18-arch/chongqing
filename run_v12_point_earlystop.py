"""V12 Point LGB 默认版：早停训练（日前出清价）。"""
import json
import logging
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PARAMS_DIR
from price_forecast_eval import quick_shape_report
from price_forecast_eval.viz import run_standard_visualization
from src.feature_engineering import TARGET_DA_COL
from src.model_baseline import (
    EARLY_STOPPING_ROUNDS,
    NUM_BOOST_ROUND,
    TEST_END,
    TEST_START,
    TRAIN_END,
    _load_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_EVERY = int(os.environ.get("V12_POINT_LOG_EVERY", "200"))
_sub = os.environ.get("V12_EXPERIMENT_SUBDIR", "").strip()
if _sub:
    OUT_DIR = OUTPUT_DIR / _sub
else:
    OUT_DIR = OUTPUT_DIR / "v12_point_default"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_tuned_params(name: str) -> dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


def main() -> None:
    name = "da"
    target_col = TARGET_DA_COL
    logger.info("=" * 60)
    logger.info(
        "V12 Point LGB — early stopping (max_round=%d, es_round=%d)",
        NUM_BOOST_ROUND,
        EARLY_STOPPING_ROUNDS,
    )
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]
    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:TEST_END].copy()

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
    ]
    if LOG_EVERY > 0:
        callbacks.append(lgb.log_evaluation(LOG_EVERY))

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    best_iter = int(getattr(model, "best_iteration", 0) or 0)
    logger.info("Best iteration: %d", best_iter)

    pred_train = model.predict(train_df[feature_cols], num_iteration=best_iter)
    pred_test = model.predict(test_df[feature_cols], num_iteration=best_iter)

    rows = []
    for split_name, actual, pred, idx in [
        ("train", train_df[target_col].values, pred_train, train_df.index),
        ("test", test_df[target_col].values, pred_test, test_df.index),
    ]:
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        shape = quick_shape_report(actual, pred, idx, include_extended=False)
        logger.info("── %s ──", split_name.upper())
        logger.info("  MAE: %.4f  RMSE: %.4f", mae, rmse)
        row = {"split": split_name, "mae": mae, "rmse": rmse, **shape}
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.insert(0, "best_iteration", best_iter)
    summary.to_csv(OUT_DIR / "metrics_train_test.csv", index=False)

    pd.DataFrame({"actual": train_df[target_col], "pred": pred_train}, index=train_df.index).to_csv(
        OUT_DIR / "pred_train.csv"
    )
    pd.DataFrame({"actual": test_df[target_col], "pred": pred_test}, index=test_df.index).to_csv(
        OUT_DIR / "pred_test.csv"
    )
    pd.DataFrame({"actual": test_df[target_col], "predicted": pred_test}, index=test_df.index).to_csv(
        OUT_DIR / "da_result.csv"
    )

    run_standard_visualization(
        OUT_DIR / "da_result.csv",
        out_dir=PLOTS_DIR,
        label="V12",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )
    model.save_model(str(OUT_DIR / "point_lgb_earlystop.txt"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("run_v12_point_earlystop failed")
        sys.exit(1)
