#!/usr/bin/env python3
"""V18 Conv2D — 预测实时出清电价 rt_clearing_price，产物目录 output/v18_conv2d_rt。

必须在导入 model_v18_conv2d 之前设置 V18_TARGET_COL，否则仍为默认 da_clearing_price。

环境变量与 run_v18_conv2d.py 相同（V18_EPOCHS、V18_BS 等）；输出目录默认 V18_OUT_DIR=v18_conv2d_rt。

标准评估示例：
  python run_evaluate_all_models.py --output-root output/v18_conv2d_rt --task rt --no-baseline
"""
import logging
import os

# 须在 import v18 模块前设置（TARGET_COL 在模块加载时读取）
os.environ["V18_TARGET_COL"] = "rt_clearing_price"
os.environ.setdefault("V18_OUT_DIR", "v18_conv2d_rt")

from src.config import OUTPUT_DIR
from src.model_v18_conv2d import run_v18

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_dir_name = os.environ.get("V18_OUT_DIR", "v18_conv2d_rt").strip() or "v18_conv2d_rt"
out_dir = OUTPUT_DIR / out_dir_name
out_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    run_v18(out_dir=out_dir)
