#!/usr/bin/env python3
"""
实验驱动入口：读 experiments/*.yaml → 设置环境变量 → 子进程跑训练 → 标准评估落盘。

产物目录：
  output/experiments/<experiment_id>/   训练产物 + metrics.json + plots/
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from src.config import BASE_DIR, OUTPUT_DIR
from price_forecast_eval import evaluate_predictions_csv, write_metrics_json


def _subst(text: str, experiment_id: str) -> str:
    return text.replace("{{experiment_id}}", experiment_id)


def _find_predictions(artifact_root: Path, filename: str) -> Optional[Path]:
    direct = artifact_root / filename
    if direct.is_file():
        return direct
    for sub in sorted(artifact_root.iterdir()):
        if sub.is_dir():
            p = sub / filename
            if p.is_file():
                return p
    return None


def _safe(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return None
    return v


def _extract_metrics_row(experiment_id: str, ev: dict[str, Any]) -> dict[str, Any]:
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    co = ev.get("composite") or {}
    return {
        "experiment_id": experiment_id,
        "mae": _safe(pm.get("mae")),
        "rmse": _safe(pm.get("rmse")),
        "valid_point_count": _safe(pm.get("valid_point_count")),
        "profile_corr": _safe(sm.get("profile_corr")),
        "neg_corr_day_ratio": _safe(sm.get("neg_corr_day_ratio")),
        "neg_corr_day_count": _safe(sm.get("neg_corr_day_count")),
        "amplitude_err": _safe(sm.get("amplitude_err")),
        "direction_acc": _safe(sm.get("direction_acc")),
        "normalized_profile_mae": _safe(sm.get("normalized_profile_mae")),
        "peak_hour_error": _safe(sm.get("peak_hour_error")),
        "valley_hour_error": _safe(sm.get("valley_hour_error")),
        "valid_shape_days": _safe(sm.get("valid_shape_days")),
        "turning_point_match_rate": _safe(sm.get("turning_point_match_rate")),
        "block_rank_acc": _safe(sm.get("block_rank_acc")),
        "composite_score": _safe(co.get("composite_score")) if co else None,
        "mae_norm": _safe(co.get("mae_norm")) if co else None,
        "corr_loss_norm": _safe(co.get("corr_loss_norm")) if co else None,
        "neg_corr_norm": _safe(co.get("neg_corr_norm")) if co else None,
        "amp_err_norm": _safe(co.get("amp_err_norm")) if co else None,
        "dir_loss_norm": _safe(co.get("dir_loss_norm")) if co else None,
    }


def _write_metrics_tables(row: dict[str, Any], out_dir: Path) -> None:
    headers = list(row.keys())
    csv_path = out_dir / "metrics_table.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="YAML 驱动实验（产物 output/experiments/<id>/）")
    ap.add_argument("--config", type=Path, required=True, help="experiments/*.yaml")
    ap.add_argument("--skip-train", action="store_true", help="仅做 post_evaluate（训练已跑完）")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    exp_id: str = raw["experiment_id"]
    if not exp_id or exp_id == "exp_template":
        raise SystemExit("请在 YAML 中设置有效的 experiment_id（勿使用模板默认值）")

    art_root = OUTPUT_DIR / "experiments" / exp_id
    art_root.mkdir(parents=True, exist_ok=True)

    runner = raw.get("runner") or {}
    cmd = runner.get("command")
    cwd = BASE_DIR / (runner.get("cwd") or ".")
    env_extra = runner.get("artifact_env") or {}

    child_env = os.environ.copy()
    for k, v in env_extra.items():
        child_env[k] = _subst(str(v), exp_id)

    if not args.skip_train:
        if not cmd:
            raise SystemExit("YAML 缺少 runner.command")
        print("RUN", cmd, "cwd=", cwd)
        r = subprocess.run(cmd, cwd=str(cwd), env=child_env)
        if r.returncode != 0:
            sys.exit(r.returncode)

    pe = raw.get("post_evaluate") or {}
    if not pe.get("enabled", False):
        print("Done (no post_evaluate). Artifacts:", art_root)
        return

    pred_name = pe.get("predictions_csv", "da_result.csv")
    pred_path = _find_predictions(art_root, pred_name)
    if pred_path is None:
        raise SystemExit(f"未找到预测文件 {pred_name} 于 {art_root}")

    seg_cols = pe.get("segment_cols") or []
    if pe.get("with_scenario_tags") and not seg_cols:
        seg_cols = ["tag_weekend", "tag_vol_class", "tag_extreme", "tag_holiday"]

    bl = pe.get("baseline_csv")
    bl_path = (BASE_DIR / bl).resolve() if bl else None

    ev = evaluate_predictions_csv(
        pred_path,
        actual_col=pe.get("actual_col", "actual"),
        pred_col=pe.get("pred_col", "predicted"),
        task_type=pe.get("task_type", "da"),
        include_extended=not pe.get("no_extended", False),
        baseline_path=bl_path,
        baseline_task=pe.get("baseline_task", "da"),
        baseline_variant=pe.get("baseline_variant", "lag24h"),
        auto_baseline=pe.get("auto_baseline"),
        with_scenario_tags=bool(pe.get("with_scenario_tags", False)),
        segment_cols=seg_cols if seg_cols else None,
    )
    write_metrics_json(ev, art_root / "metrics.json")
    print("Wrote", art_root / "metrics.json")
    table_row = _extract_metrics_row(exp_id, ev)
    _write_metrics_tables(table_row, art_root)
    print("Wrote", art_root / "metrics_table.csv")


if __name__ == "__main__":
    main()
