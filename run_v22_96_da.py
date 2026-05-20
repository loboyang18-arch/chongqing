#!/usr/bin/env python3
"""V22：96 点 15min 日前出清价；默认 output/v22_96_da。环境变量见 model_v22_96_da。

终端里要「边训边看」：本脚本启用行缓冲 + 每条日志 flush；仍可用
  PYTHONUNBUFFERED=1 python run_v22_96_da.py
  或  python -u run_v22_96_da.py
"""
import os

from src.config import OUTPUT_DIR
from src.model_v22_96_da import configure_realtime_console_logging, run_v22

configure_realtime_console_logging()

out_sub = os.environ.get("V22_OUT_DIR", "v22_96_da").strip()
out_dir = OUTPUT_DIR / out_sub if out_sub else OUTPUT_DIR / "v22_96_da"
no_eval = os.environ.get("V22_NO_EVAL", "").strip() in ("1", "true", "yes")

if __name__ == "__main__":
    run_v22(out_dir=out_dir, run_eval=not no_eval)
