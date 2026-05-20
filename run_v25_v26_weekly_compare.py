#!/usr/bin/env python3
"""V25 vs V26：官方测试窗按周统计日前/实时指标并对比。

默认输入：
  V25: output/v25_deploy_5p0_lam02/da_result_v25.csv, rt_result_v25.csv
  V26: output/v26_multitask_dual_5p0/da_result.csv, rt_result.csv

输出：
  output/v25_v26_weekly_compare/weekly_metrics.csv
  output/v25_v26_weekly_compare/weekly_compare_pivot.csv
  output/v25_v26_weekly_compare/summary.md

示例：
  python run_v25_v26_weekly_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from price_forecast_eval import evaluate_predictions_csv
from src.config import OUTPUT_DIR
from src.experiment.splits import TEST_END, TEST_START

OUT_DIR = OUTPUT_DIR / "v25_v26_weekly_compare"
WEEK_DAYS = 7

MODELS = [
    ("V25", "da", OUTPUT_DIR / "v25_deploy_5p0_lam02" / "da_result_v25.csv"),
    ("V25", "rt", OUTPUT_DIR / "v25_deploy_5p0_lam02" / "rt_result_v25.csv"),
    ("V26", "da", OUTPUT_DIR / "v26_multitask_dual_5p0" / "da_result.csv"),
    ("V26", "rt", OUTPUT_DIR / "v26_multitask_dual_5p0" / "rt_result.csv"),
]


def _week_chunks(test_start: pd.Timestamp, test_end: pd.Timestamp) -> List[Dict[str, Any]]:
    days = pd.date_range(test_start.normalize(), test_end.normalize(), freq="D")
    chunks = []
    for i in range(0, len(days), WEEK_DAYS):
        wdays = days[i : i + WEEK_DAYS]
        if len(wdays) == 0:
            continue
        ws, we = wdays[0], wdays[-1]
        chunks.append({
            "week_idx": len(chunks) + 1,
            "week_start": ws.date().isoformat(),
            "week_end": we.date().isoformat(),
            "n_calendar_days": len(wdays),
            "ts_start": ws,
            "ts_end": we + pd.Timedelta(hours=23),
        })
    return chunks


def _load_result(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    df = df.sort_index()
    for c in ("actual", "predicted"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["actual", "predicted"])


def _eval_slice(df: pd.DataFrame, task: str) -> Dict[str, Any]:
    if df.empty:
        return {"n_rows": 0, "mae": None, "rmse": None, "profile_corr": None,
                "direction_acc": None, "neg_corr_day_ratio": None}
    tmp = df.reset_index()
    tmp.to_csv("/tmp/_weekly_eval_slice.csv", index=False)
    ev = evaluate_predictions_csv(
        Path("/tmp/_weekly_eval_slice.csv"),
        actual_col="actual",
        pred_col="predicted",
        task_type=task,
    )
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    val = ev.get("validation") or {}
    return {
        "n_rows": val.get("valid_point_count", len(df)),
        "n_shape_days": val.get("valid_shape_days", 0),
        "mae": pm.get("mae"),
        "rmse": pm.get("rmse"),
        "profile_corr": sm.get("profile_corr"),
        "direction_acc": sm.get("direction_acc"),
        "neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _week_chunks(TEST_START, TEST_END)

    rows: List[Dict[str, Any]] = []
    for model, task, path in MODELS:
        if not path.is_file():
            raise FileNotFoundError(path)
        full = _load_result(path)
        for w in chunks:
            sub = full.loc[w["ts_start"]:w["ts_end"]]
            m = _eval_slice(sub, task)
            rows.append({
                "model": model,
                "task": task,
                "week_idx": w["week_idx"],
                "week_start": w["week_start"],
                "week_end": w["week_end"],
                "n_calendar_days": w["n_calendar_days"],
                **m,
            })

    df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "weekly_metrics.csv"
    df.to_csv(metrics_path, index=False, float_format="%.4f")

    # 透视：每周 V25 vs V26 的 MAE / corr 及差值
    pivot_rows = []
    for task in ("da", "rt"):
        for widx in sorted(df["week_idx"].unique()):
            sub = df[(df["task"] == task) & (df["week_idx"] == widx)]
            if len(sub) != 2:
                continue
            v25 = sub[sub["model"] == "V25"].iloc[0]
            v26 = sub[sub["model"] == "V26"].iloc[0]
            pivot_rows.append({
                "task": "日前" if task == "da" else "实时",
                "week_idx": int(widx),
                "week": f"{v25['week_start']}~{v25['week_end']}",
                "V25_mae": v25["mae"],
                "V26_mae": v26["mae"],
                "mae_diff_v26_minus_v25": float(v26["mae"]) - float(v25["mae"]),
                "V25_corr": v25["profile_corr"],
                "V26_corr": v26["profile_corr"],
                "corr_diff": float(v26["profile_corr"]) - float(v25["profile_corr"]),
                "V25_neg_corr_ratio": v25["neg_corr_day_ratio"],
                "V26_neg_corr_ratio": v26["neg_corr_day_ratio"],
                "winner_mae": "V26" if v26["mae"] < v25["mae"] else "V25",
            })
    pivot = pd.DataFrame(pivot_rows)
    pivot_path = OUT_DIR / "weekly_compare_pivot.csv"
    pivot.to_csv(pivot_path, index=False, float_format="%.4f")

    # 全窗汇总
    full_summary = []
    for model, task, path in MODELS:
        full = _load_result(path)
        m = _eval_slice(full, task)
        full_summary.append({"model": model, "task": task, "scope": "full_test", **m})

    summary_md = OUT_DIR / "summary.md"
    lines = [
        "# V25 vs V26 测试窗按周对比",
        "",
        f"测试窗：{TEST_START.date()} ~ {TEST_END.date()}（按 {WEEK_DAYS} 日历日切周）",
        "",
        "## 全窗汇总",
        "",
        "| 模型 | 任务 | MAE | profile_corr | direction_acc | neg_corr_day_ratio |",
        "|------|------|-----|--------------|---------------|------------------|",
    ]
    for r in full_summary:
        tname = "日前" if r["task"] == "da" else "实时"
        lines.append(
            f"| {r['model']} | {tname} | {r['mae']:.2f} | {r['profile_corr']:.4f} | "
            f"{r['direction_acc']:.4f} | {r['neg_corr_day_ratio']:.4f} |"
        )

    for task_code, task_name in (("da", "日前"), ("rt", "实时")):
        lines.extend([
            "",
            f"## {task_name} — 按周 MAE / profile_corr",
            "",
            "| 周 | V25 MAE | V26 MAE | ΔMAE | V25 corr | V26 corr | Δcorr | MAE优 |",
            "|----|---------|---------|------|----------|----------|-------|------|",
        ])
        pt = pivot[pivot["task"] == task_name]
        for _, r in pt.iterrows():
            lines.append(
                f"| {r['week']} | {r['V25_mae']:.2f} | {r['V26_mae']:.2f} | "
                f"{r['mae_diff_v26_minus_v25']:+.2f} | {r['V25_corr']:.4f} | {r['V26_corr']:.4f} | "
                f"{r['corr_diff']:+.4f} | {r['winner_mae']} |"
            )
        v25_wins = int((pt["winner_mae"] == "V25").sum())
        v26_wins = int((pt["winner_mae"] == "V26").sum())
        lines.extend([
            "",
            f"- MAE 更优周数：V25 **{v25_wins}** 周，V26 **{v26_wins}** 周（共 {len(pt)} 周）",
        ])

    lines.extend([
        "",
        "## 文件",
        "",
        f"- `{metrics_path.relative_to(OUTPUT_DIR.parent)}`",
        f"- `{pivot_path.relative_to(OUTPUT_DIR.parent)}`",
    ])
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    with (OUT_DIR / "full_test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {pivot_path}")
    print(f"Wrote {summary_md}")
    print("\n--- 日前按周 MAE ---")
    print(pivot[pivot["task"] == "日前"][["week", "V25_mae", "V26_mae", "mae_diff_v26_minus_v25", "winner_mae"]].to_string(index=False))
    print("\n--- 实时按周 MAE ---")
    print(pivot[pivot["task"] == "实时"][["week", "V25_mae", "V26_mae", "mae_diff_v26_minus_v25", "winner_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
