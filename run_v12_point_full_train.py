"""V12 Point LGB：无早停、满轮训练，打印训练集/测试集拟合（与 V12 [A] 同数据与超参）。"""
import json
import logging
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import OUTPUT_DIR, PARAMS_DIR
from price_forecast_eval import quick_shape_report
from price_forecast_eval.viz import run_standard_visualization
from src.model_baseline import (
    NUM_BOOST_ROUND,
    TEST_END,
    TEST_START,
    TRAIN_END,
    _load_dataset,
)

# 默认与 model_baseline.NUM_BOOST_ROUND 一致；可环境变量覆盖，例如 V12_POINT_ROUNDS=10000
NUM_ROUNDS = int(os.environ.get("V12_POINT_ROUNDS", str(NUM_BOOST_ROUND)))
LOG_EVERY = int(os.environ.get("V12_POINT_LOG_EVERY", "500"))

# 按轮数分子目录；可选 V12_EXPERIMENT_SUBDIR=experiments/<exp_id> 供 run_experiment 统一产物
_v12_sub = os.environ.get("V12_EXPERIMENT_SUBDIR", "").strip()
if _v12_sub:
    OUT_DIR = OUTPUT_DIR / _v12_sub / f"r{NUM_ROUNDS}"
else:
    OUT_DIR = OUTPUT_DIR / "v12_point_full_train" / f"r{NUM_ROUNDS}"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_tuned_params(name: str) -> dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


def main() -> None:
    name = "da"
    from src.feature_engineering import TARGET_DA_COL
    target_col = TARGET_DA_COL
    logger.info("=" * 60)
    logger.info("V12 Point LGB — full train (no early stopping)")
    logger.info("  num_boost_round=%d  log_every=%d", NUM_ROUNDS, LOG_EVERY)
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:TEST_END].copy()

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtrain)

    callbacks = [lgb.log_evaluation(LOG_EVERY)] if LOG_EVERY > 0 else []

    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    try:
        logger.info("Finished training. num_trees=%d", lgb_model.num_trees())
    except Exception:
        logger.info("Finished training.")

    pred_train = lgb_model.predict(train_df[feature_cols])
    pred_test = lgb_model.predict(test_df[feature_cols])

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
        for k, v in shape.items():
            logger.info("  %-18s %s", k, v)
        row = {"split": split_name, "mae": mae, "rmse": rmse, **shape}
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.insert(0, "num_boost_round", NUM_ROUNDS)
    summary.to_csv(OUT_DIR / "metrics_train_test.csv", index=False)
    logger.info("Saved: %s", OUT_DIR / "metrics_train_test.csv")

    pd.DataFrame(
        {"actual": train_df[target_col], "pred": pred_train},
        index=train_df.index,
    ).to_csv(OUT_DIR / "pred_train.csv")
    pd.DataFrame(
        {"actual": test_df[target_col], "pred": pred_test},
        index=test_df.index,
    ).to_csv(OUT_DIR / "pred_test.csv")
    pd.DataFrame(
        {"actual": test_df[target_col], "predicted": pred_test},
        index=test_df.index,
    ).to_csv(OUT_DIR / "da_result.csv")
    logger.info("Saved: pred_train.csv, pred_test.csv -> %s", OUT_DIR)
    logger.info("Saved: da_result.csv -> %s", OUT_DIR)

    run_standard_visualization(
        OUT_DIR / "da_result.csv",
        out_dir=PLOTS_DIR,
        label="V12",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    # 可选：保存模型
    lgb_model.save_model(str(OUT_DIR / "point_lgb_full.txt"))
    logger.info("Saved model: %s", OUT_DIR / "point_lgb_full.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("run_v12_point_full_train failed")
        sys.exit(1)
