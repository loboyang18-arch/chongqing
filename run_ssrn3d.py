#!/usr/bin/env python3
"""SSRN3D：光谱-空间 3D 残差卷积 → 日前出清价（默认 5+0 双头）

借鉴 SSRN 思想；上下文固定为 h 及之前（无 after）；默认 DualHead + λ=0.2。

示例：
  conda activate power
  cd /root/workspace/chongqing_prj
  python run_ssrn3d.py

  SSRN3D_DUAL=0 python run_ssrn3d.py   # 单头对照
"""
import os
import sys

# 固定 5+0，禁止 after（须在 import src 之前）
os.environ.setdefault("V18_CTX_BEFORE", "5")
os.environ["V18_CTX_AFTER"] = "0"
os.environ.setdefault("V25_CTX_BEFORE", os.environ["V18_CTX_BEFORE"])
os.environ["V25_CTX_AFTER"] = "0"
os.environ.setdefault("SSRN3D_CTX_BEFORE", os.environ["V18_CTX_BEFORE"])
os.environ["SSRN3D_CTX_AFTER"] = "0"

os.environ.setdefault("SSRN3D_DUAL", "1")
os.environ.setdefault("V18_DELTA_LAMBDA", os.environ.get("SSRN3D_DELTA_LAMBDA", "0.2"))
os.environ.setdefault("V18_DROPOUT", "0.2")

import logging
import subprocess

from src.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

out_sub = os.environ.get("SSRN3D_OUT_DIR", "ssrn3d_da_5p0").strip() or "ssrn3d_da_5p0"
out_dir = OUTPUT_DIR / out_sub

if __name__ == "__main__":
    from src.model_ssrn3d import run_ssrn3d

    run_ssrn3d(out_dir=out_dir)

    no_eval = os.environ.get("SSRN3D_NO_EVAL", "").strip() in ("1", "true", "yes")
    if not no_eval:
        summary = out_dir.resolve() / "evaluation_summary_appendix_v1.csv"
        cmd = [
            sys.executable,
            "run_evaluate_all_models.py",
            "--output-root", str(out_dir.resolve()),
            "--summary", str(summary),
            "--task", "da",
            "--no-baseline",
        ]
        logging.info("Running standard eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
