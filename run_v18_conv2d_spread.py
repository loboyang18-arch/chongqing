#!/usr/bin/env python3
"""V18 Conv2D — 预测小时级 (实时−日前) 价差，产物目录 output/v18_conv2d_spread。

目标定义：当日 15min 的 rt_clearing_price、da_clearing_price 各自按小时 4 点均值后相减，
与 V18 单序列小时标签口径一致。

必须在导入 model_v18_conv2d 之前设置 V18_TARGET_COL。

环境变量同 run_v18_conv2d.py；默认 V18_OUT_DIR=v18_conv2d_spread。
主结果文件：spread_result.csv

评估示例（点指标，无 baseline）：
  python run_evaluate_all_models.py --output-root output/v18_conv2d_spread --task da --no-baseline
"""
import logging
import os

os.environ["V18_TARGET_COL"] = "spread_rt_minus_da"
os.environ.setdefault("V18_OUT_DIR", "v18_conv2d_spread")

from src.config import OUTPUT_DIR
from src.model_v18_conv2d import run_v18

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_dir_name = os.environ.get("V18_OUT_DIR", "v18_conv2d_spread").strip() or "v18_conv2d_spread"
out_dir = OUTPUT_DIR / out_dir_name
out_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    run_v18(out_dir=out_dir)
