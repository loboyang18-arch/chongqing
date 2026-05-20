#!/usr/bin/env python3
"""V25 ResConv — 预测实时出清电价 realtime_clearing_price（sql_data 特征）。

与日前 V25 dual02 对齐：
  - 同一 train/val/test 切分（src/experiment/splits.py）
  - 同一网络结构（DualHeadResConv2dPriceNet，默认开启）
  - Lag2 默认保留 realtime_clearing_*（与 DA 同为 34 通道）；旧行为：V24_STRIP_RT_LAG2=1

须在 import V24/V25 模块前设置 V24_TARGET_COL。

示例：
  python run_v25_resconv_rt.py

  V18_EPOCHS=100 V25_OUT_DIR=v25_resconv_rt_dual02 python run_v25_resconv_rt.py

标准评估（脚本内自动，--task rt）：
  python run_evaluate_all_models.py --output-root output/v25_resconv_rt_dual02 --task rt --no-baseline
"""
import logging
import os
import subprocess
import sys

# 须在 import v24/v25 前设置（TARGET_COL / LAG2 在模块加载时解析）
os.environ["V24_TARGET_COL"] = "realtime_clearing_price"
os.environ.setdefault("V25_DUAL", "1")
os.environ.setdefault("V18_DELTA_LAMBDA", "0.2")
os.environ.setdefault("V25_OUT_DIR", "v25_resconv_rt_dual02")

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_sub = os.environ.get("V25_OUT_DIR", "v25_resconv_rt_dual02").strip() or "v25_resconv_rt_dual02"
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
            "--task", "rt",
            "--no-baseline",
        ]
        logging.info("Running standard eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
