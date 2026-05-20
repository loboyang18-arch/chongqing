#!/usr/bin/env python3
"""V18 末 28 天按周滚动评估（每周之前全部为训练集，每周重训）。

环境变量（节选，与 model_v18_conv2d 一致）：
  V18_ROLLING_OUT_DIR   输出子目录 (默认 v18_conv2d_rolling28)
  V18_ROLLING_TOTAL_DAYS  末段天数 (默认 28)
  V18_ROLLING_WEEK_DAYS   每折测试天数 (默认 7)
  V18_ROLLING_EPOCHS    每折训练轮数；不设则与 V18_EPOCHS 相同
  V18_EPOCHS / V18_BS / …  同 run_v18_conv2d.py

无缓冲输出示例：
  PYTHONUNBUFFERED=1 python -u run_v18_rolling28.py
"""
import logging
import os

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.model_v18_conv2d import run_v18_rolling_last28

if __name__ == "__main__":
    sub = os.environ.get("V18_ROLLING_OUT_DIR", "v18_conv2d_rolling28").strip()
    run_v18_rolling_last28(OUTPUT_DIR / sub)
