#!/usr/bin/env python3
"""
批量附录标准评估：扫描 output 下预测 CSV，写出每文件 JSON + 汇总 CSV。

默认仅扫描 output/（不含 _archive）；可用 --include-archive 纳入归档实验。
同目录多个 CSV 时写入 metrics_eval_standard_<stem>.json，避免互相覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR
from price_forecast_eval import (
    evaluate_model_predictions,
    from_result_columns,
    load_baseline_from_naive_summary_csv,
)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            return str(obj)
    return obj


def _infer_pred_col(df: pd.DataFrame) -> str | None:
    for c in ("predicted", "pred", "pred_lgb", "pred_v12"):
        if c in df.columns:
            return c
    return None


def _collect_csvs(root: Path, include_archive: bool) -> list[Path]:
    out: list[Path] = []
    for pattern in (
        "**/da_result*.csv",
        "**/rt_result*.csv",
        "**/spread_result*.csv",
        "**/pred_test.csv",
    ):
        for p in root.glob(pattern):
            if not include_archive and "_archive" in p.parts:
                continue
            out.append(p)
    return sorted(set(out), key=lambda x: str(x))


def _eval_one(
    csv_path: Path,
    baseline: dict | None,
    task: str,
    include_extended: bool,
) -> tuple[dict, Path, str]:
    df = pd.read_csv(csv_path, parse_dates=["ts"])
    if "ts" not in df.columns:
        raise ValueError("需要 ts 列")
    df = df.set_index("ts")
    pred_col = _infer_pred_col(df)
    if pred_col is None:
        raise ValueError("无法推断预测列（predicted/pred/...）")
    if "actual" not in df.columns:
        raise ValueError("需要 actual 列")

    ef = from_result_columns(df, actual_col="actual", pred_col=pred_col, ts_index=True)
    ev = evaluate_model_predictions(
        ef,
        baseline_metrics=baseline,
        task_type=task,
        include_extended=include_extended,
    )
    out_json = csv_path.parent / f"metrics_eval_standard_{csv_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(ev), f, ensure_ascii=False, indent=2)
    return ev, out_json, pred_col


def _flatten(ev: dict, rel_key: str, pred_col: str) -> dict:
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    co = ev.get("composite") or {}
    return {
        "model_key": rel_key,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUTPUT_DIR, help="默认项目 output/")
    ap.add_argument("--include-archive", action="store_true")
    ap.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_DIR / "evaluation_summary_appendix_v1.csv",
    )
    ap.add_argument("--task", default="da", choices=("da", "rt"))
    ap.add_argument("--no-extended", action="store_true")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=OUTPUT_DIR / "naive_settlement_baselines" / "summary.csv",
    )
    ap.add_argument("--baseline-task", default="da")
    ap.add_argument("--baseline-variant", default="lag24h")
    ap.add_argument("--no-baseline", action="store_true", help="不算 composite")
    args = ap.parse_args()

    root = args.output_root.resolve()
    baseline = None
    if not args.no_baseline and args.baseline.is_file():
        baseline = load_baseline_from_naive_summary_csv(
            args.baseline, args.baseline_task, args.baseline_variant
        )

    csvs = _collect_csvs(root, args.include_archive)
    rows = []
    errors: list[tuple[str, str]] = []

    for p in csvs:
        rel = str(p.relative_to(root))
        try:
            ev, jpath, pred_col = _eval_one(
                p,
                baseline,
                args.task,
                include_extended=not args.no_extended,
            )
            rows.append(_flatten(ev, rel, pred_col))
            print("OK", rel, "->", jpath.name)
        except Exception as e:
            errors.append((rel, str(e)))
            print("FAIL", rel, e)

    if rows:
        pd.DataFrame(rows).to_csv(args.summary, index=False)
        print("Wrote", args.summary, "rows=", len(rows))
    if errors:
        err_path = args.summary.with_name(args.summary.stem + "_errors.txt")
        err_path.write_text("\n".join(f"{a}\t{b}" for a, b in errors), encoding="utf-8")
        print("Errors logged to", err_path)


if __name__ == "__main__":
    main()
