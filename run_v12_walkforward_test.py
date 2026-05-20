#!/usr/bin/env python3
"""V12 Shape-Opt (variant A) 在官方测试窗上按周走步回测。

每周重训：训练截止 = 该周第一日 00:00 之前（含扩展训练池，可含原 val 段）。
与 V25 walk-forward 协议对齐，便于对照。

输出：output/<V12_WF_OUT_DIR>/
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

import src.model_baseline as mb
import src.model_v12_shape_opt as m12
from price_forecast_eval import evaluate_predictions_csv, quick_shape_report, write_metrics_json
from src.config import OUTPUT_DIR
from src.experiment.splits import HOURLY_TEST_END, HOURLY_TEST_START
from src.model_v12_shape_opt import run_v12_variant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WF_OUT = os.environ.get("V12_WF_OUT_DIR", "v12_shape_opt_walkforward_test").strip()
WEEK_DAYS = int(os.environ.get("V12_WF_WEEK_DAYS", "7"))
VARIANT = os.environ.get("V12_WF_VARIANT", "A").strip() or "A"


def _test_week_chunks(test_start: pd.Timestamp, test_end: pd.Timestamp, week_days: int) -> List[tuple]:
    """返回 [(week_start, week_end_hourly), ...] 按日历日切周。"""
    days = pd.date_range(test_start.normalize(), test_end.normalize(), freq="D")
    chunks: List[tuple] = []
    for i in range(0, len(days), week_days):
        wdays = days[i : i + week_days]
        if len(wdays) == 0:
            continue
        ws = wdays[0]
        we = wdays[-1] + pd.Timedelta(hours=23)
        if we > test_end:
            we = test_end
        chunks.append((ws, we))
    return chunks


def run_v12_walkforward_test(out_dir: Path | None = None) -> Dict:
    out_dir = Path(out_dir or OUTPUT_DIR / WF_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_start = pd.Timestamp(HOURLY_TEST_START)
    test_end = pd.Timestamp(HOURLY_TEST_END)
    chunks = _test_week_chunks(test_start, test_end, WEEK_DAYS)

    logger.info("=" * 60)
    logger.info(
        "V12 walk-forward | variant=%s | test %s ~ %s | %d folds",
        VARIANT, test_start, test_end, len(chunks),
    )

    all_parts: List[pd.DataFrame] = []
    fold_records: List[Dict] = []

    for fold, (week_start, week_end) in enumerate(chunks):
        train_end = (week_start - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        ts_start = week_start.strftime("%Y-%m-%d %H:%M:%S")
        ts_end = week_end.strftime("%Y-%m-%d %H:%M:%S")

        for mod in (mb, m12):
            mod.TRAIN_END = train_end
            mod.TEST_START = ts_start
            mod.TEST_END = ts_end

        fold_dir = out_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 60)
        logger.info(
            "Fold %d | train_end=%s | test %s ~ %s",
            fold, train_end, ts_start, ts_end,
        )

        try:
            from src.model_baseline import _load_dataset
            probe = _load_dataset("da")
            n_test_rows = len(probe.loc[ts_start:ts_end])
            if n_test_rows == 0:
                logger.warning("Fold %d: skip (no rows in feature_da for %s~%s)", fold, ts_start, ts_end)
                fold_records.append({
                    "fold": fold,
                    "test_week_start": str(week_start.date()),
                    "test_week_end": str(week_end.date()),
                    "train_end": train_end,
                    "status": "skipped",
                    "error": "no_feature_rows",
                })
                continue

            summary, result = run_v12_variant(VARIANT)
        except Exception as e:
            logger.exception("Fold %d failed: %s", fold, e)
            fold_records.append({
                "fold": fold,
                "test_week_start": str(week_start.date()),
                "test_week_end": str(week_end.date()),
                "status": "failed",
                "error": str(e),
            })
            continue

        part = result.rename(columns={"pred": "predicted"}).copy()
        part["fold"] = fold
        all_parts.append(part)

        act = part["actual"].values
        pred = part["predicted"].values
        mae_f = float(np.mean(np.abs(act - pred)))
        rmse_f = float(np.sqrt(np.mean((act - pred) ** 2)))
        shape_f = quick_shape_report(act, pred, part.index)

        fold_records.append({
            "fold": fold,
            "test_week_start": str(week_start.date()),
            "test_week_end": str(week_end.date()),
            "train_end": train_end,
            "status": "ok",
            "mae": mae_f,
            "rmse": rmse_f,
            "profile_corr": float(shape_f.get("profile_corr", float("nan"))),
            "direction_acc": float(shape_f.get("direction_acc", float("nan"))),
            "summary_mae": summary.get("MAE"),
        })
        with (fold_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        part[["actual", "predicted"]].to_csv(fold_dir / "da_result.csv")

        logger.info(
            "Fold %d MAE=%.2f profile_corr=%.4f",
            fold, mae_f, shape_f.get("profile_corr", float("nan")),
        )

    if not all_parts:
        raise RuntimeError("V12 walk-forward produced no predictions")

    pooled = pd.concat(all_parts).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep="first")]
    result_path = out_dir / "da_result.csv"
    pooled[["actual", "predicted"]].to_csv(result_path)

    af, pf = pooled["actual"].values, pooled["predicted"].values
    pooled_shape = quick_shape_report(af, pf, pooled.index)
    pooled_mae = float(np.mean(np.abs(af - pf)))
    pooled_rmse = float(np.sqrt(np.mean((af - pf) ** 2)))

    ok_folds = [r for r in fold_records if r.get("status") == "ok"]
    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    fold_maes = [r["mae"] for r in ok_folds]
    fold_corrs = [r["profile_corr"] for r in ok_folds]

    summary_out = {
        "protocol": "walk_forward_weekly_retrain",
        "model": f"V12 Shape-Opt variant {VARIANT}",
        "target": "target_da_clearing_price",
        "test_start": str(test_start),
        "test_end": str(test_end),
        "week_days": WEEK_DAYS,
        "n_folds": len(ok_folds),
        "n_failed_folds": len(fold_records) - len(ok_folds),
        "pooled_mae": pooled_mae,
        "pooled_rmse": pooled_rmse,
        "pooled_profile_corr": float(pooled_shape.get("profile_corr", float("nan"))),
        "pooled_direction_acc": float(pooled_shape.get("direction_acc", float("nan"))),
        "fold_mae_mean": float(np.mean(fold_maes)) if fold_maes else None,
        "fold_mae_std": float(np.std(fold_maes, ddof=0)) if fold_maes else None,
        "fold_corr_mean": float(np.mean(fold_corrs)) if fold_corrs else None,
        "fold_corr_std": float(np.std(fold_corrs, ddof=0)) if fold_corrs else None,
        "folds": fold_records,
        "holdout_reference": {
            "model": "v12_shape_opt",
            "protocol": "single_train_leq_train_end_predict_test",
            "note": "见 output/v12_shape_opt/metrics_eval_standard_da_result.json",
        },
    }
    with (out_dir / "walkforward_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    ev = evaluate_predictions_csv(
        result_path, actual_col="actual", pred_col="predicted", task_type="da",
    )
    write_metrics_json(ev, out_dir / "metrics_eval_standard_da_result.json")

    logger.info("=" * 60)
    logger.info("V12 WALK-FORWARD POOLED (%d folds, %d rows)", len(ok_folds), len(pooled))
    logger.info("  MAE: %.2f | profile_corr: %.4f", pooled_mae, summary_out["pooled_profile_corr"])
    logger.info("Saved %s", out_dir)
    logger.info("=" * 60)
    return summary_out


if __name__ == "__main__":
    run_v12_walkforward_test()
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
