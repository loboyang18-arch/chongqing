#!/usr/bin/env python3
"""V25 早停实验：5+0 dual λ=0.2，每 epoch 终端打印 train/val MAE 与 corr。

默认与部署包一致；权重为验证集 MAE 最优（非末 epoch）。

示例：
  conda activate power
  cd /root/workspace/chongqing_prj
  python -u run_v25_earlystop.py

  V25_VAL_WEEKS=2 python -u run_v25_earlystop.py   # 验证 14 天（训练少 7 天）

  V25_EPOCHS=80 V25_PATIENCE=20 V25_OUT_DIR=v25_earlystop_da python -u run_v25_earlystop.py
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

# 验证集周数：2 = 14 日历日（从训练末尾划出，测试窗不变）
_val_weeks = float(os.environ.get("V25_VAL_WEEKS", "1"))
if _val_weeks == 2:
    os.environ.setdefault("SPLIT_TRAIN_END", "2026-02-16 23:45:00")
    os.environ.setdefault("SPLIT_VAL_END", "2026-03-02 23:45:00")
    os.environ.setdefault("V25_OUT_DIR", "v25_earlystop_val14d_5p0_lam02_da")

# V25 训练超参（在 import 模块前设置）
os.environ.setdefault("V25_DUAL", "1")
os.environ.setdefault("V25_CTX_BEFORE", "5")
os.environ.setdefault("V25_CTX_AFTER", "0")
os.environ.setdefault("V25_DELTA_LAMBDA", "0.2")
os.environ.setdefault("V25_EPOCHS", "100")
os.environ.setdefault("V25_PATIENCE", "15")
os.environ.setdefault("V25_MIN_EPOCHS", "20")
os.environ.setdefault("V25_DROPOUT", "0.15")
os.environ.setdefault("V25_LR", "1e-3")
os.environ.setdefault("V25_WD", "1e-4")
os.environ.setdefault("V25_OUT_DIR", "v25_earlystop_5p0_lam02_da")

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

out_sub = os.environ.get("V25_OUT_DIR", "v25_earlystop_5p0_lam02_da").strip()
out_dir = OUTPUT_DIR / out_sub
out_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    from src.model_v25_train import run_v25_early_stop

    meta = run_v25_early_stop(out_dir=out_dir)

    if os.environ.get("V25_SKIP_EVAL", "").strip() not in ("1", "true", "yes"):
        cmd = [
            sys.executable,
            "run_evaluate_all_models.py",
            "--output-root", str(out_dir.resolve()),
            "--task", "da",
            "--no-baseline",
        ]
        logging.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)

    print(f"\n完成。输出目录: {out_dir.resolve()}\n", flush=True)
