#!/usr/bin/env python3
"""V25 时段三模型（8+8+8）：与单模相同的逐小时单点预测，训练按小时切分，预测拼接。

  python -u run_v25_segment_pipeline.py train
  python -u run_v25_segment_pipeline.py stitch
  python -u run_v25_segment_pipeline.py all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["V25_EARLY_STOP"] = "0"
os.environ["V25_EPOCHS"] = "100"
os.environ["V25_DUAL"] = "1"
os.environ["V25_CTX_BEFORE"] = "5"
os.environ["V25_CTX_AFTER"] = "0"
os.environ["V25_DELTA_LAMBDA"] = "0.2"
os.environ["V25_DROPOUT"] = "0.44"
os.environ["V25_MERGE_VAL"] = "1"
os.environ["V25_SEG_ROOT"] = "v25_segment_8x3"

from src.model_v25_segment import SEGMENTS_888, SegmentSpec
from src.model_v25_segment_train import V25_SEG_ROOT, run_v25_segment_stitch, run_v25_segment_train


def cmd_train() -> None:
    shared = V25_SEG_ROOT / "_shared_norm"
    shared.mkdir(parents=True, exist_ok=True)

    # 先算一份共享归一化（与单模一致：全训练日、全 24 小时）
    if not (shared / "v25_norm.npz").is_file():
        from src.experiment.splits import VAL_END
        from src.model_v18_conv2d import _build_daily_arrays, compute_norm
        from src.model_v24_da import (
            _patch_v18_for_v24_direct,
            _restore_v18,
            _snapshot_v18,
            load_sql_feature_matrix,
        )
        from src.model_v25_train import save_v25_norm
        import numpy as np

        snap = _snapshot_v18()
        try:
            df = load_sql_feature_matrix()
            _patch_v18_for_v24_direct()
            valid_dates, day_lag0, day_lag1, day_lag2, day_targets, _, _ = _build_daily_arrays(df)
            train_days = [d for d in valid_dates if d <= VAL_END.date()]
            norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
            tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
            y_mean = float(tgt_stack.mean())
            y_std = float(tgt_stack.std()) + 1e-8
            save_v25_norm(shared, norm_mean, norm_std, y_mean, y_std)
        finally:
            _restore_v18(snap)

    for h0, h1, name in SEGMENTS_888:
        seg = SegmentSpec(h0, h1, name)
        print(f"\n{'#' * 72}\n# {name}  hours {h0}–{h1 - 1}\n{'#' * 72}")
        run_v25_segment_train(seg, V25_SEG_ROOT / name, shared_norm_dir=shared)


def cmd_stitch() -> None:
    run_v25_segment_stitch(V25_SEG_ROOT)
    subprocess.run(
        [
            sys.executable,
            "run_evaluate_all_models.py",
            "--output-root",
            str((V25_SEG_ROOT / "stitched").resolve()),
            "--task",
            "da",
            "--no-baseline",
        ],
        check=False,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="V25 时段三模型 8+8+8（逐小时单点）")
    ap.add_argument("cmd", choices=["train", "stitch", "all"])
    args = ap.parse_args()

    V25_SEG_ROOT.mkdir(parents=True, exist_ok=True)
    with (V25_SEG_ROOT / "segments.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "hourly_single_point",
                "layout": "8+8+8",
                "segments": [
                    {"name": nm, "hour_start": h0, "hour_end": h1}
                    for h0, h1, nm in SEGMENTS_888
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if args.cmd in ("train", "all"):
        cmd_train()
    if args.cmd in ("stitch", "all"):
        cmd_stitch()


if __name__ == "__main__":
    main()
