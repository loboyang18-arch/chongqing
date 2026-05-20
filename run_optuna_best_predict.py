#!/usr/bin/env python3
"""加载 Optuna 搜索最优 trial 的模型权重，直接预测 + 评估（不重新训练）。"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optuna_best_predict")

os.environ.setdefault("V24_USE_WEATHER", "0")
os.environ.setdefault("V24_PCA_COMPONENTS", "0")

SEARCH_DIR = Path("output/v24_optuna_test_search")
OUT_DIR = Path("output/v24_da_optuna_best_v2")


def main():
    best_path = SEARCH_DIR / "best_params.json"
    with open(best_path, "r") as f:
        best = json.load(f)
    params = best["params"]
    trial_num = best["trial_number"]
    ckpt_path = SEARCH_DIR / f"trial_{trial_num:04d}" / "seed0.pt"

    logger.info("Best trial #%d  test_mae=%.2f", trial_num, best["test_mae"])
    for k, v in params.items():
        logger.info("  %-14s = %s", k, v)
    logger.info("Checkpoint: %s", ckpt_path)

    from src.experiment.splits import TRAIN_END, VAL_END, TEST_START, TEST_END
    from src.model_v24_da import (
        _load_raw_df, _patch_v18_for_v24_direct, _snapshot_v18, _restore_v18,
    )
    import src.model_v18_conv2d as m18

    snap = _snapshot_v18()
    _patch_v18_for_v24_direct()

    df = _load_raw_df()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets, *_ = m18._build_daily_arrays(df)

    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
    train_days = [d for d in valid_dates if d <= val_last]
    test_days = [d for d in valid_dates if ts_first <= d <= ts_last]

    norm_mean, norm_std = m18.compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    logger.info("Train days: %d, Test days: %d", len(train_days), len(test_days))

    _cb = params["ctx_before"]
    _ca = params["ctx_after"]
    _h_slots = (_cb + 1 + _ca) * m18.SLOTS_PER_HOUR
    _dropout = params["dropout"]

    orig_cb, orig_ca, orig_hs = m18.CONTEXT_BEFORE, m18.CONTEXT_AFTER, m18.H_SLOTS
    m18.CONTEXT_BEFORE = _cb
    m18.CONTEXT_AFTER = _ca
    m18.H_SLOTS = _h_slots

    model = m18.Conv2dPriceNet(
        c_in=m18.C_TOTAL, h_slots=_h_slots, dropout=_dropout,
    ).to(m18.DEVICE)
    state = torch.load(ckpt_path, map_location=m18.DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s (ctx=%d+%d, h_slots=%d)",
                ckpt_path.name, _cb, _ca, _h_slots)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p24, a24, dates = m18.predict_days(
        model, test_days,
        day_lag0, day_lag1, day_lag2, day_targets,
        norm_mean, norm_std, y_mean, y_std,
    )

    rows = []
    for i, d in enumerate(dates):
        for h in range(24):
            rows.append({
                "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                "actual": a24[i, h],
                "predicted": p24[i, h],
            })
    result = pd.DataFrame(rows).set_index("ts").sort_index()
    result_path = OUT_DIR / "da_result.csv"
    result.to_csv(result_path)
    logger.info("Saved: %s (%d rows, %d days)", result_path.name, len(result), len(dates))

    from price_forecast_eval.viz import run_standard_visualization
    run_standard_visualization(
        result_path,
        out_dir=OUT_DIR / "plots",
        label="V24-Optuna-Best",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    af = result["actual"].values
    pf = result["predicted"].values
    mae = float(np.mean(np.abs(af - pf)))
    rmse = float(np.sqrt(np.mean((af - pf) ** 2)))

    from src.model_v18_conv2d import quick_shape_report
    shape = quick_shape_report(af, pf, result.index)

    logger.info("=" * 60)
    logger.info("RESULTS (from trial #%d checkpoint)", trial_num)
    logger.info("  MAE:  %.2f", mae)
    logger.info("  RMSE: %.2f", rmse)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    m18.CONTEXT_BEFORE, m18.CONTEXT_AFTER, m18.H_SLOTS = orig_cb, orig_ca, orig_hs
    _restore_v18(snap)

    meta = {
        "source": f"optuna trial #{trial_num} checkpoint",
        "checkpoint": str(ckpt_path),
        "params": params,
        "search_test_mae": best["test_mae"],
        "predict_test_mae": mae,
    }
    with open(OUT_DIR / "v24_optuna_best_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    cmd = [
        sys.executable, "run_evaluate_all_models.py",
        "--output-root", str(OUT_DIR.resolve()),
        "--summary", str((OUT_DIR / "evaluation_summary_appendix_v1.csv").resolve()),
        "--task", "da", "--no-baseline",
    ]
    logger.info("Running standard eval ...")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
