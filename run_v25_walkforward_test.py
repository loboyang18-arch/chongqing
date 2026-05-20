#!/usr/bin/env python3
"""V25 dual02 配置：在官方测试窗 (TEST_START~TEST_END) 上按周走步回测。

每周重训：训练集 = 该周第一日之前的所有有效日历日（可含原 val 段，不含当周及未来）。
不将当周标签用于训练/早停（val_days=[]，固定 epoch）。

输出：
  output/<V25_WF_OUT_DIR>/da_result.csv
  output/<V25_WF_OUT_DIR>/walkforward_summary.json
  output/<V25_WF_OUT_DIR>/fold_metrics.csv
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from price_forecast_eval import evaluate_predictions_csv, quick_shape_report, write_metrics_json
from src.config import OUTPUT_DIR
from src.experiment.splits import TEST_END, TEST_START
from src.model_v18_conv2d import (
    LOOKBACK_DAYS,
    MAX_EPOCHS,
    _build_daily_arrays,
    compute_norm,
    predict_days,
    train_model,
)
from src.model_v24_da import (
    _patch_v18_for_v24_direct,
    _restore_v18,
    _snapshot_v18,
    load_sql_feature_matrix,
)
from src.model_v25_resconv import DualHeadResConv2dPriceNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WF_OUT = os.environ.get("V25_WF_OUT_DIR", "v25_dual02_walkforward_test").strip()
WEEK_DAYS = int(os.environ.get("V25_WF_WEEK_DAYS", "7"))


def _chunk_dates(dates: List, week_days: int) -> List[List]:
    chunks: List[List] = []
    for i in range(0, len(dates), week_days):
        ch = dates[i : i + week_days]
        if ch:
            chunks.append(ch)
    return chunks


def run_v25_walkforward_test(out_dir: Path | None = None) -> Dict:
    out_dir = Path(out_dir or OUTPUT_DIR / WF_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(os.environ.get("V18_EPOCHS", str(MAX_EPOCHS)))
    model_cls = DualHeadResConv2dPriceNet

    logger.info("=" * 60)
    logger.info(
        "V25 walk-forward | dual02 | test %s ~ %s | week=%dd | epochs=%d",
        TEST_START.date(), TEST_END.date(), WEEK_DAYS, epochs,
    )

    snap = _snapshot_v18()
    try:
        df = load_sql_feature_matrix()
        _patch_v18_for_v24_direct()
        valid_dates, day_lag0, day_lag1, day_lag2, day_targets, _, day_delta_targets = (
            _build_daily_arrays(df)
        )
        valid_dates = sorted(valid_dates)

        ts_first = TEST_START.date()
        ts_last = TEST_END.date()
        test_dates = [d for d in valid_dates if ts_first <= d <= ts_last]
        chunks = _chunk_dates(test_dates, WEEK_DAYS)
        logger.info("Test window: %d days → %d folds", len(test_dates), len(chunks))

        all_rows: List[Dict] = []
        fold_records: List[Dict] = []

        for fold, test_days in enumerate(chunks):
            train_days = [d for d in valid_dates if d < test_days[0]]
            if len(train_days) < LOOKBACK_DAYS + 3:
                logger.warning("Fold %d: skip (train_days=%d)", fold, len(train_days))
                continue

            fold_dir = out_dir / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            logger.info("-" * 60)
            logger.info(
                "Fold %d | train %d d (%s..%s) | test %d d (%s..%s)",
                fold, len(train_days), train_days[0], train_days[-1],
                len(test_days), test_days[0], test_days[-1],
            )

            norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
            tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
            y_mean = float(tgt_stack.mean())
            y_std = float(tgt_stack.std()) + 1e-8

            model, _ = train_model(
                train_days=train_days,
                val_days=[],
                day_lag0=day_lag0,
                day_lag1=day_lag1,
                day_lag2=day_lag2,
                day_targets=day_targets,
                day_delta_targets=day_delta_targets,
                norm_mean=norm_mean,
                norm_std=norm_std,
                y_mean=y_mean,
                y_std=y_std,
                epochs=epochs,
                out_dir=fold_dir,
                model_cls=model_cls,
            )

            p24, a24, dates = predict_days(
                model, test_days,
                day_lag0, day_lag1, day_lag2, day_targets,
                norm_mean, norm_std, y_mean, y_std,
            )
            if not dates:
                logger.warning("Fold %d: no predictions", fold)
                continue

            flat_act, flat_pred, idx = [], [], []
            for i, d in enumerate(dates):
                for h in range(24):
                    ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
                    ap, pp = float(a24[i, h]), float(p24[i, h])
                    flat_act.append(ap)
                    flat_pred.append(pp)
                    idx.append(ts)
                    all_rows.append({"ts": ts, "actual": ap, "predicted": pp, "fold": fold})

            fa, fp = np.array(flat_act), np.array(flat_pred)
            mae_f = float(np.mean(np.abs(fa - fp)))
            rmse_f = float(np.sqrt(np.mean((fa - fp) ** 2)))
            shape_f = quick_shape_report(fa, fp, pd.DatetimeIndex(idx))
            fold_records.append({
                "fold": fold,
                "test_week_start": str(test_days[0]),
                "test_week_end": str(test_days[-1]),
                "n_train_days": len(train_days),
                "n_test_days": len(dates),
                "mae": mae_f,
                "rmse": rmse_f,
                "profile_corr": float(shape_f.get("profile_corr", float("nan"))),
                "direction_acc": float(shape_f.get("direction_acc", float("nan"))),
            })
            logger.info(
                "Fold %d MAE=%.2f RMSE=%.2f profile_corr=%.4f",
                fold, mae_f, rmse_f, shape_f.get("profile_corr", float("nan")),
            )

        if not all_rows:
            raise RuntimeError("walk-forward produced no rows")

        result = pd.DataFrame(all_rows).set_index("ts").sort_index()
        result_path = out_dir / "da_result.csv"
        result[["actual", "predicted"]].to_csv(result_path)

        af, pf = result["actual"].values, result["predicted"].values
        pooled_shape = quick_shape_report(af, pf, result.index)
        pooled_mae = float(np.mean(np.abs(af - pf)))
        pooled_rmse = float(np.sqrt(np.mean((af - pf) ** 2)))

        fold_df = pd.DataFrame(fold_records)
        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

        fold_maes = fold_df["mae"].tolist()
        fold_corrs = fold_df["profile_corr"].tolist()
        summary = {
            "protocol": "walk_forward_weekly_retrain",
            "model": "V25 DualHeadResConv2dPriceNet (dual02)",
            "config": {
                "V25_DUAL": True,
                "V18_DELTA_LAMBDA": os.environ.get("V18_DELTA_LAMBDA", "0.2"),
                "V18_EPOCHS": epochs,
                "V18_CTX_BEFORE": os.environ.get("V18_CTX_BEFORE", "5"),
                "V18_CTX_AFTER": os.environ.get("V18_CTX_AFTER", "1"),
            },
            "test_start": str(TEST_START),
            "test_end": str(TEST_END),
            "week_days": WEEK_DAYS,
            "n_folds": len(fold_records),
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
            "pooled_profile_corr": float(pooled_shape.get("profile_corr", float("nan"))),
            "pooled_direction_acc": float(pooled_shape.get("direction_acc", float("nan"))),
            "fold_mae_mean": float(np.mean(fold_maes)),
            "fold_mae_std": float(np.std(fold_maes, ddof=0)),
            "fold_corr_mean": float(np.mean(fold_corrs)),
            "fold_corr_std": float(np.std(fold_corrs, ddof=0)),
            "folds": fold_records,
            "holdout_reference": {
                "model": "v25_resconv_dual02",
                "protocol": "single_train_leq_val_end_predict_test",
                "test_mae": 86.7,
                "test_profile_corr": 0.24,
            },
        }
        with (out_dir / "walkforward_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        ev = evaluate_predictions_csv(
            result_path, actual_col="actual", pred_col="predicted", task_type="da",
        )
        write_metrics_json(ev, out_dir / "metrics_eval_standard_da_result.json")

        logger.info("=" * 60)
        logger.info("WALK-FORWARD POOLED (%d folds, %d rows)", len(fold_records), len(result))
        logger.info(
            "  MAE:  %.2f  (fold mean±std: %.2f ± %.2f)",
            pooled_mae, summary["fold_mae_mean"], summary["fold_mae_std"],
        )
        logger.info(
            "  profile_corr: %.4f  (fold mean±std: %.4f ± %.4f)",
            summary["pooled_profile_corr"], summary["fold_corr_mean"], summary["fold_corr_std"],
        )
        logger.info("Saved %s", out_dir)
        logger.info("=" * 60)
        return summary
    finally:
        _restore_v18(snap)


if __name__ == "__main__":
    run_v25_walkforward_test()
    root = (OUTPUT_DIR / WF_OUT).resolve()
    subprocess.run(
        [
            sys.executable,
            "run_evaluate_all_models.py",
            "--output-root", str(root),
            "--summary", str(root / "evaluation_summary_appendix_v1.csv"),
            "--task", "da",
            "--no-baseline",
        ],
        cwd=str(Path(__file__).resolve().parent),
        check=False,
    )
