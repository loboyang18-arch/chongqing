#!/usr/bin/env python3
"""V24：基于 sql_data 的 30 列市场特征 + 天气 → V18 Conv2D 日前出清价预测。

数据来源：sql_data/chongqing_market_join.csv
特征 Lag 定义见 src/model_v24_da.py

示例：
  V18_EPOCHS=100 python run_v24_da.py
  V24_OUT_DIR=v24_da_exp1 V18_EPOCHS=200 python run_v24_da.py
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

out_sub = os.environ.get("V24_OUT_DIR", "v24_da").strip() or "v24_da"
out_dir = OUTPUT_DIR / out_sub

if __name__ == "__main__":
    from src.model_v24_da import run_v24
    run_v24(out_dir=out_dir)

    # 标准评估
    no_eval = os.environ.get("V24_NO_EVAL", "").strip() in ("1", "true", "yes")
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
