#!/usr/bin/env python3
"""V26 多任务：日前 + 实时，各带 V25 式 price+delta 双头（无 L_dir）。

默认 5+0、δ=0.2，与 V25 部署一致；Lag2 保留 realtime_clearing_*。

示例：
  python run_v26_multitask.py
  V26_OUT_DIR=v26_multitask_dual_5p0 python run_v26_multitask.py
"""
import logging
import os
import subprocess
import sys

os.environ.setdefault("V18_CTX_BEFORE", "5")
os.environ.setdefault("V18_CTX_AFTER", "0")
os.environ.setdefault("V26_OUT_DIR", "v26_multitask_dual_5p0")
os.environ.setdefault("V18_DELTA_LAMBDA", "0.2")

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_sub = os.environ.get("V26_OUT_DIR", "v26_multitask_5p0").strip()
out_dir = OUTPUT_DIR / out_sub

if __name__ == "__main__":
    from src.model_v26_multitask import run_v26

    run_v26(out_dir=out_dir)

    for task in ("da", "rt"):
        root = out_dir.resolve()
        summary = root / f"evaluation_summary_{task}_v1.csv"
        cmd = [
            sys.executable,
            "run_evaluate_all_models.py",
            "--output-root", str(root),
            "--summary", str(summary),
            "--task", task,
            "--no-baseline",
        ]
        logging.info("Eval %s: %s", task, " ".join(cmd))
        subprocess.run(cmd, check=False)
