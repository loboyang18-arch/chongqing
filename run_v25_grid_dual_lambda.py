#!/usr/bin/env python3
"""
V25 双头任务：对损失系数 V18_DELTA_LAMBDA 做网格搜索并汇总指标。

双头损失（见 model_v18_conv2d.train_model）:
  L = L1(price_pred, price_tgt) + λ * L1(delta_pred, delta_tgt)

默认 λ ∈ {0.1, 0.2, …, 0.9}，每个 λ 单独输出目录并跑标准评估，最后写一张对比表。

示例:
  python run_v25_grid_dual_lambda.py
  V18_EPOCHS=30 python run_v25_grid_dual_lambda.py --parent-dir v25_dual_grid_quick
  python run_v25_grid_dual_lambda.py --aggregate-only --parent-dir v25_dual_grid_quick
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("v25_grid")


def _lambda_tag(lam: float) -> str:
    s = f"{lam:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _grid_lambdas(lo: float, hi: float, step: float) -> list[float]:
    out: list[float] = []
    x = lo
    while x <= hi + 1e-9:
        out.append(round(x, 6))
        x = round(x + step, 6)
    return out


def _run_one(lam: float, parent_rel: str) -> Path:
    """子进程跑 run_v25_resconv.py；返回该次实验 output 子目录（绝对路径）。"""
    tag = _lambda_tag(lam)
    out_rel = f"{parent_rel.strip('/')}/lambda_{tag}"
    env = os.environ.copy()
    env["V25_DUAL"] = "1"
    env["V18_DELTA_LAMBDA"] = str(lam)
    env["V25_OUT_DIR"] = out_rel
    cmd = [sys.executable, str(ROOT / "run_v25_resconv.py")]
    LOG.info("=== λ=%s → V25_OUT_DIR=output/%s ===", lam, out_rel)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"run_v25_resconv failed for lambda={lam} exit={r.returncode}")
    return OUTPUT_DIR / out_rel


def _load_metrics_row(out_dir: Path, lam: float) -> dict:
    jpath = out_dir / "metrics_eval_standard_da_result.json"
    if not jpath.is_file():
        raise FileNotFoundError(f"missing {jpath}")
    with open(jpath, encoding="utf-8") as f:
        ev = json.load(f)
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    co = ev.get("composite") or {}
    return {
        "v18_delta_lambda": lam,
        "out_dir": str(out_dir.relative_to(OUTPUT_DIR)),
        "mae": pm.get("mae"),
        "rmse": pm.get("rmse"),
        "profile_corr": sm.get("profile_corr"),
        "direction_acc": sm.get("direction_acc"),
        "amplitude_err": sm.get("amplitude_err"),
        "normalized_profile_mae": sm.get("normalized_profile_mae"),
        "composite_score": (co or {}).get("composite_score"),
        "mae_norm": (co or {}).get("mae_norm"),
    }


def _aggregate(parent_rel: str, lambdas: list[float]) -> Path:
    rows = []
    for lam in lambdas:
        tag = _lambda_tag(lam)
        out_dir = OUTPUT_DIR / parent_rel.strip("/") / f"lambda_{tag}"
        try:
            rows.append(_load_metrics_row(out_dir, lam))
        except Exception as e:
            LOG.warning("skip aggregate λ=%s: %s", lam, e)
    if not rows:
        raise SystemExit("no rows to aggregate")
    df = pd.DataFrame(rows)
    df = df.sort_values("v18_delta_lambda").reset_index(drop=True)
    out_csv = OUTPUT_DIR / parent_rel.strip("/") / "v25_dual_lambda_grid_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOG.info("Wrote %s (%d rows)", out_csv, len(df))
    # 终端友好打印
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="V25 dual-head V18_DELTA_LAMBDA grid search")
    ap.add_argument("--min", dest="lo", type=float, default=0.1, help="最小 λ（默认 0.1）")
    ap.add_argument("--max", dest="hi", type=float, default=0.9, help="最大 λ（默认 0.9）")
    ap.add_argument("--step", type=float, default=0.1, help="步长（默认 0.1）")
    ap.add_argument(
        "--parent-dir",
        type=str,
        default="v25_dual_lambda_grid",
        help="output 下父目录名（默认 v25_dual_lambda_grid）",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="不训练，只根据已有子目录汇总 v25_dual_lambda_grid_summary.csv",
    )
    args = ap.parse_args()

    lambdas = _grid_lambdas(args.lo, args.hi, args.step)
    LOG.info("Grid V18_DELTA_LAMBDA: %s (n=%d)", lambdas, len(lambdas))

    if not args.aggregate_only:
        parent = args.parent_dir.strip().strip("/")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "parent_dir": parent,
            "lambdas": lambdas,
            "v25_dual": True,
            "note": "loss = L1(price) + lambda * L1(delta); env V18_DELTA_LAMBDA",
        }
        meta_path = OUTPUT_DIR / parent / "v25_dual_lambda_grid_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        for lam in lambdas:
            _run_one(lam, parent)

    _aggregate(args.parent_dir.strip().strip("/"), lambdas)


if __name__ == "__main__":
    main()
