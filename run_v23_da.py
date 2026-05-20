#!/usr/bin/env python3
"""V23：对 V18 全部输入特征（Lag0+Lag1+Lag2）联合 PCA → V18 日前出清；默认 output/v23_da。

与是否启用天气无关；特征列来自当前 V18 的 LAG* 定义。详见 src/model_v23_da.py。

示例：
  V23_PCA_COMPONENTS=32 V18_EPOCHS=100 python run_v23_da.py
"""
import logging
import os

from src.config import OUTPUT_DIR
from src.model_v23_da import run_v23

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_sub = os.environ.get("V23_OUT_DIR", "v23_da").strip() or "v23_da"
out_dir = OUTPUT_DIR / out_sub
no_eval = os.environ.get("V23_NO_EVAL", "").strip() in ("1", "true", "yes")

if __name__ == "__main__":
    run_v23(out_dir=out_dir, run_eval=not no_eval)
