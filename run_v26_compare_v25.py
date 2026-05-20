#!/usr/bin/env python3
"""V26 vs V25 对比表（日前 / 实时分别）。"""
import json
from pathlib import Path

from price_forecast_eval import evaluate_predictions_csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"


def _load_metrics(csv_path: Path, task: str) -> dict:
    ev = evaluate_predictions_csv(csv_path, task_type=task)
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    return {
        "mae": pm.get("mae"),
        "rmse": pm.get("rmse"),
        "profile_corr": sm.get("profile_corr"),
        "direction_acc": sm.get("direction_acc"),
        "neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
    }


def main():
    v25_da = OUT / "v25_deploy_5p0_lam02" / "da_result_v25.csv"
    v25_rt = OUT / "v25_deploy_5p0_lam02" / "rt_result_v25.csv"
    v26_da = OUT / "v26_multitask_5p0" / "da_result.csv"
    v26_rt = OUT / "v26_multitask_5p0" / "rt_result.csv"

    rows = []
    for name, path, task in [
        ("V25 日前", v25_da, "da"),
        ("V26 日前", v26_da, "da"),
        ("V25 实时", v25_rt, "rt"),
        ("V26 实时", v26_rt, "rt"),
    ]:
        m = _load_metrics(path, task)
        rows.append({"model": name, **m})

    cmp_path = OUT / "v26_multitask_5p0" / "compare_v25_v26.csv"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model", "mae", "rmse", "profile_corr", "direction_acc", "neg_corr_day_ratio"]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    cmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    v26_meta = OUT / "v26_multitask_5p0" / "v26_meta.json"
    dir_acc = None
    if v26_meta.is_file():
        dir_acc = json.loads(v26_meta.read_text()).get("test_dir_acc")

    print("\n=== 日前 (DA) ===")
    for r in rows[:2]:
        print(f"{r['model']}: MAE={r['mae']:.2f} corr={r['profile_corr']:.4f} dir_acc={r['direction_acc']:.4f}")
    print("\n=== 实时 (RT) ===")
    for r in rows[2:]:
        print(f"{r['model']}: MAE={r['mae']:.2f} corr={r['profile_corr']:.4f} dir_acc={r['direction_acc']:.4f}")
    if dir_acc is not None:
        print(f"\nV26 价差涨跌平准确率: {dir_acc:.3f}")
    print(f"\nWrote {cmp_path}")


if __name__ == "__main__":
    main()
