#!/usr/bin/env python3
"""重庆日前市场出清电价 — Moirai 零样本预测 + 标准评估与朴素基线对比。

示例（power 环境）：
  HF_HUB_OFFLINE=1 python run_moirai_da_clearing.py --freq 1h
  HF_HUB_OFFLINE=1 python run_moirai_da_clearing.py --freq 15min
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from price_forecast_eval import evaluate_predictions_csv, write_metrics_json
from src.config import OUTPUT_DIR
from src.model_moirai_da_clearing import DEFAULT_COVARIATES, FREQ_PRESETS, TARGET_COL, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HOURLY_DWS = OUTPUT_DIR / "dws_hourly_features.csv"
CONV2D_BASELINE_CANDIDATES = (
    OUTPUT_DIR / "v25_resconv" / "test_predictions.csv",
    OUTPUT_DIR / "v24_da" / "test_predictions.csv",
    OUTPUT_DIR / "v18_conv2d" / "test_predictions.csv",
)


def _default_out_dir(freq: str, use_cov: bool) -> Path:
    suffix = "cov" if use_cov else "uni"
    sub = os.environ.get("MOIRAI_EXPERIMENT_SUBDIR", "").strip()
    if sub:
        return OUTPUT_DIR / sub
    return OUTPUT_DIR / f"moirai_da_clearing_{freq}_{suffix}"


def _compare_baseline(
    pred_df: pd.DataFrame,
    baseline_path: Path,
    label: str,
) -> Optional[Dict[str, Any]]:
    if not baseline_path.is_file():
        return None
    base = pd.read_csv(baseline_path, parse_dates=["ts"], index_col="ts")
    pred_col = "predicted" if "predicted" in base.columns else "pred"
    if pred_col not in base.columns:
        return None
    merged = pred_df.join(base[[pred_col]].rename(columns={pred_col: "baseline_pred"}), how="inner")
    merged = merged.dropna(subset=["actual", "predicted", "baseline_pred"])
    if merged.empty:
        return None
    mae_m = float(np.mean(np.abs(merged["actual"] - merged["predicted"])))
    mae_b = float(np.mean(np.abs(merged["actual"] - merged["baseline_pred"])))
    return {
        "baseline": label,
        "n_points": int(len(merged)),
        "moirai_mae": round(mae_m, 4),
        "baseline_mae": round(mae_b, 4),
        "mae_delta_moirai_minus_baseline": round(mae_m - mae_b, 4),
    }


def _evaluate_naive_lag24(pred_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if not HOURLY_DWS.is_file():
        return None
    hourly = pd.read_csv(HOURLY_DWS, parse_dates=["ts"], index_col="ts")
    if TARGET_COL not in hourly.columns:
        return None
    actual = hourly[TARGET_COL].astype(float)
    naive_pred = actual.shift(24)
    common_idx = pred_df.index.intersection(naive_pred.dropna().index)
    if common_idx.empty:
        return None
    y = actual.loc[common_idx].values
    p = naive_pred.loc[common_idx].values
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return None
    mae_b = float(np.mean(np.abs(y[m] - p[m])))
    rmse_b = float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    return {
        "baseline": "naive_lag24h",
        "n_points": int(m.sum()),
        "baseline_mae": round(mae_b, 4),
        "baseline_rmse": round(rmse_b, 4),
    }


def _appendix_row(ev: dict, model_key: str, pred_col: str) -> dict:
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    co = ev.get("composite") or {}
    return {
        "model_key": model_key,
        "pred_col": pred_col,
        "mae": pm.get("mae"),
        "rmse": pm.get("rmse"),
        "valid_point_count": pm.get("valid_point_count"),
        "profile_corr": sm.get("profile_corr"),
        "neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
        "neg_corr_day_count": sm.get("neg_corr_day_count"),
        "amplitude_err": sm.get("amplitude_err"),
        "direction_acc": sm.get("direction_acc"),
        "normalized_profile_mae": sm.get("normalized_profile_mae"),
        "peak_hour_error": sm.get("peak_hour_error"),
        "valley_hour_error": sm.get("valley_hour_error"),
        "valid_shape_days": sm.get("valid_shape_days"),
        "turning_point_match_rate": sm.get("turning_point_match_rate"),
        "block_rank_acc": sm.get("block_rank_acc"),
        "composite_score": co.get("composite_score") if co else None,
        "mae_norm": co.get("mae_norm") if co else None,
        "corr_loss_norm": co.get("corr_loss_norm") if co else None,
        "neg_corr_norm": co.get("neg_corr_norm") if co else None,
        "amp_err_norm": co.get("amp_err_norm") if co else None,
        "dir_loss_norm": co.get("dir_loss_norm") if co else None,
    }


def _merge_shape_into_metrics_csv(out_dir: Path, shape: dict) -> None:
    """将 profile_corr 等形状指标并入 metrics.csv（与 V25 评估口径对齐）。"""
    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.is_file():
        return
    base = pd.read_csv(metrics_path, index_col=0).iloc[:, 0]
    for k in (
        "profile_corr", "neg_corr_day_ratio", "amplitude_err", "direction_acc",
        "normalized_profile_mae", "peak_hour_error", "valley_hour_error",
        "valid_shape_days", "turning_point_match_rate", "block_rank_acc",
    ):
        if k in shape and shape[k] is not None:
            base[k] = shape[k]
    base.to_csv(metrics_path)


def post_evaluate(out_dir: Path) -> Dict[str, Any]:
    da_result = out_dir / "da_result.csv"
    hourly_pred = out_dir / "test_predictions_hourly.csv"
    pred_path = da_result if da_result.is_file() else hourly_pred
    if not pred_path.is_file():
        raise FileNotFoundError(f"缺少预测结果: {pred_path}")

    if pred_path.name == "da_result.csv":
        ev = evaluate_predictions_csv(
            pred_path,
            actual_col="actual",
            pred_col="predicted",
            task_type="da",
        )
    else:
        tmp = pd.read_csv(pred_path, parse_dates=["ts"])
        if "ts" not in tmp.columns:
            tmp = tmp.rename(columns={tmp.columns[0]: "ts"})
        tmp = tmp.rename(columns={"pred": "predicted"})
        tmp_path = out_dir / "_eval_tmp.csv"
        tmp.to_csv(tmp_path, index=False)
        ev = evaluate_predictions_csv(
            tmp_path,
            actual_col="actual",
            pred_col="predicted",
            task_type="da",
        )
    write_metrics_json(ev, out_dir / "metrics.json")

    pred_col_name = "predicted"
    shape = ev.get("shape_metrics") or {}
    _merge_shape_into_metrics_csv(out_dir, shape)

    appendix = _appendix_row(ev, "da_result.csv", pred_col_name)
    pd.DataFrame([appendix]).to_csv(
        out_dir / "evaluation_summary_appendix_v1.csv", index=False,
    )

    pred_df = pd.read_csv(pred_path, parse_dates=["ts"], index_col="ts")
    if "predicted" not in pred_df.columns and "pred" in pred_df.columns:
        pred_df = pred_df.rename(columns={"pred": "predicted"})
    pred_df = pred_df.dropna(subset=["actual", "predicted"])

    comparisons = []
    for path in CONV2D_BASELINE_CANDIDATES:
        cmp = _compare_baseline(pred_df, path, path.parent.name)
        if cmp:
            comparisons.append(cmp)
            break
    naive = _evaluate_naive_lag24(pred_df)
    if naive:
        mae_m = float(np.mean(np.abs(pred_df["actual"] - pred_df["predicted"])))
        naive["moirai_mae"] = round(mae_m, 4)
        naive["mae_delta_moirai_minus_baseline"] = round(
            mae_m - float(naive["baseline_mae"]), 4,
        )
        comparisons.append(naive)

    summary = {
        "target": TARGET_COL,
        "point_metrics": ev.get("point_metrics"),
        "shape_metrics": shape,
        "baseline_comparison": comparisons,
    }
    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    cmp_df = pd.DataFrame(comparisons) if comparisons else pd.DataFrame()
    if not cmp_df.empty:
        cmp_df.to_csv(out_dir / "baseline_comparison.csv", index=False)

    logger.info(
        "标准评估: MAE=%.2f RMSE=%.2f profile_corr=%.4f direction_acc=%.4f → %s",
        float((ev.get("point_metrics") or {}).get("mae", float("nan"))),
        float((ev.get("point_metrics") or {}).get("rmse", float("nan"))),
        float(shape.get("profile_corr", float("nan"))),
        float(shape.get("direction_acc", float("nan"))),
        out_dir / "metrics.json",
    )
    if comparisons:
        logger.info("基线对比:\n%s", cmp_df.to_string(index=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Salesforce/moirai-1.1-R-small")
    p.add_argument("--freq", default="1h", choices=list(FREQ_PRESETS))
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--prediction-length", type=int, default=None)
    p.add_argument("--use-covariates", action="store_true")
    p.add_argument("--covariates", default=None)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-eval", action="store_true")
    args = p.parse_args()

    cov_tuple = DEFAULT_COVARIATES
    if args.covariates:
        cov_tuple = tuple(c.strip() for c in args.covariates.split(",") if c.strip())

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args.freq, args.use_covariates)
    run(
        model_id=args.model,
        test_start=args.test_start,
        test_end=args.test_end,
        use_covariates=args.use_covariates,
        covariates=cov_tuple,
        freq=args.freq,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        out_dir=out_dir,
        batch_size=args.batch_size,
    )

    skip_eval = args.no_eval or os.environ.get("MOIRAI_NO_EVAL", "").strip() in ("1", "true", "yes")
    if not skip_eval:
        post_evaluate(out_dir)


if __name__ == "__main__":
    main()
