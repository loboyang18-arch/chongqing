#!/usr/bin/env python3
"""V25：10 层 ResConv2D + 残差连接 → 日前出清价预测

数据：V24 sql_data 特征（默认无天气、无 PCA）
模型：ResConv2dPriceNet / DualHeadResConv2dPriceNet（10 层 Conv + 3 ResBlock）
图例：默认 V25-ResConv-DA-{CTX}-λ*（环境变量 V18_VIZ_LABEL 可覆盖）

示例：
  V18_EPOCHS=100 python run_v25_resconv.py
  V25_OUT_DIR=v25_test V18_MERGE_VAL=1 python run_v25_resconv.py
"""
import logging
import os
import subprocess
import sys

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_sub = os.environ.get("V25_OUT_DIR", "v25_resconv").strip() or "v25_resconv"
out_dir = OUTPUT_DIR / out_sub

if __name__ == "__main__":
    from src.model_v25_resconv import run_v25
    run_v25(out_dir=out_dir)

    no_eval = os.environ.get("V25_NO_EVAL", "").strip() in ("1", "true", "yes")
    if not no_eval:
        root = out_dir.resolve()
        summary = root / "evaluation_summary_appendix_v1.csv"
        cmd = [
            sys.executable,
            "run_evaluate_all_models.py",
            "--output-root", str(root),
            "--summary", str(summary),
            "--task", "da",
            "--no-baseline",
        ]
        logging.info("Running standard eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
