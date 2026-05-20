#!/usr/bin/env python3
"""V25 固定 100 epoch、末轮权重（不早停）。默认 14 天验证 + 5+0 dual。

  python -u run_v25_lastepoch.py
  V25_DROPOUT=0.2 V25_OUT_DIR=v25_lastepoch_drop02 python -u run_v25_lastepoch.py
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

os.environ["V25_EARLY_STOP"] = "0"
os.environ.setdefault("V25_EPOCHS", "100")
os.environ.setdefault("V25_DUAL", "1")
os.environ.setdefault("V25_CTX_BEFORE", "5")
os.environ.setdefault("V25_CTX_AFTER", "0")
os.environ.setdefault("V25_DELTA_LAMBDA", "0.2")
os.environ.setdefault("V25_DROPOUT", "0.15")
os.environ.setdefault("V25_LR", "1e-3")
os.environ.setdefault("V25_WD", "1e-4")

_val_weeks = float(os.environ.get("V25_VAL_WEEKS", "2"))
if _val_weeks == 2:
    os.environ.setdefault("SPLIT_TRAIN_END", "2026-02-16 23:45:00")
    os.environ.setdefault("SPLIT_VAL_END", "2026-03-02 23:45:00")
    os.environ.setdefault("V25_OUT_DIR", "v25_lastepoch_val14d_5p0_lam02_da")

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

out_sub = os.environ.get("V25_OUT_DIR", "v25_lastepoch_val14d_5p0_lam02_da").strip()
out_dir = OUTPUT_DIR / out_sub
out_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    from src.model_v25_train import run_v25_early_stop

    print("[V25] mode=last_epoch  early_stop=off  epochs=100\n", flush=True)
    meta = run_v25_early_stop(out_dir=out_dir)

    if os.environ.get("V25_SKIP_EVAL", "").strip() not in ("1", "true", "yes"):
        subprocess.run(
            [
                sys.executable,
                "run_evaluate_all_models.py",
                "--output-root", str(out_dir.resolve()),
                "--task", "da",
                "--no-baseline",
            ],
            check=False,
        )

    print(f"\n完成。输出: {out_dir.resolve()}\n", flush=True)
