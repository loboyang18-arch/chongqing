#!/usr/bin/env python3
"""V24 超参数搜索 — Optuna 贝叶斯优化

搜索空间：LR, WD, Dropout, CTX_BEFORE, CTX_AFTER, Epochs
目标：最小化验证集 MAE（元/MWh）
模型配置：V24 无天气、无 PCA

用法：
  python run_optuna_v24.py                    # 默认 50 trials
  python run_optuna_v24.py --n-trials 100     # 自定义 trial 数
  python run_optuna_v24.py --n-trials 30 --timeout 3600  # 限时 1h
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optuna_v24")

os.environ.setdefault("V24_USE_WEATHER", "0")
os.environ.setdefault("V24_PCA_COMPONENTS", "0")


def main():
    parser = argparse.ArgumentParser(description="V24 Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--out-dir", type=str, default="output/v24_optuna",
                        help="Output directory for results")
    args = parser.parse_args()

    import optuna
    from optuna.pruners import MedianPruner

    from src.config import OUTPUT_DIR
    from src.experiment.splits import TRAIN_END, VAL_END, TEST_START, TEST_END
    from src.model_v24_da import (
        _load_raw_df, _patch_v18_for_v24_direct, _snapshot_v18, _restore_v18,
        LAG0_COLS, LAG1_COLS, LAG2_COLS, TARGET_COL,
    )
    import src.model_v18_conv2d as m18

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据只加载一次 ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading data (once) ...")
    snap = _snapshot_v18()
    _patch_v18_for_v24_direct()

    df = _load_raw_df()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets, *_ = m18._build_daily_arrays(df)

    tr_last = TRAIN_END.date()
    val_last = VAL_END.date()
    ts_first = TEST_START.date()
    ts_last = TEST_END.date()
    train_days = [d for d in valid_dates if d <= val_last]
    eval_days = [d for d in valid_dates if ts_first <= d <= ts_last]

    norm_mean, norm_std = m18.compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    logger.info("Data ready: train=%d days (incl. val), eval(test)=%d days",
                len(train_days), len(eval_days))
    logger.info("=" * 60)

    # ── Optuna objective ─────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1, 0.2, 0.3])
        ctx_before = trial.suggest_categorical("ctx_before", [2, 3, 5, 7])
        ctx_after = trial.suggest_categorical("ctx_after", [0, 1, 2])
        epochs = trial.suggest_categorical("epochs", [50, 75, 100, 150])

        logger.info(
            "Trial %d: lr=%.1e wd=%.1e dropout=%.2f ctx=%d+%d epochs=%d",
            trial.number, lr, wd, dropout, ctx_before, ctx_after, epochs,
        )

        def epoch_callback(ep: int, val_mae: float):
            trial.report(val_mae, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_dir = out_dir / f"trial_{trial.number:04d}"

        t0 = time.time()
        try:
            _, test_mae = m18.train_model(
                train_days=train_days,
                val_days=eval_days,
                day_lag0=day_lag0, day_lag1=day_lag1,
                day_lag2=day_lag2, day_targets=day_targets,
                norm_mean=norm_mean, norm_std=norm_std,
                y_mean=y_mean, y_std=y_std,
                epochs=epochs, out_dir=trial_dir,
                lr=lr, weight_decay=wd,
                dropout=dropout,
                ctx_before=ctx_before, ctx_after=ctx_after,
                epoch_callback=epoch_callback,
            )
        except optuna.TrialPruned:
            elapsed = time.time() - t0
            logger.info("Trial %d PRUNED after %.1fs", trial.number, elapsed)
            raise

        elapsed = time.time() - t0
        logger.info(
            "Trial %d done: test_mae=%.2f (%.1fs)",
            trial.number, test_mae, elapsed,
        )
        return test_mae

    # ── 运行搜索 ─────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        study_name="v24_hp_search",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30),
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    _restore_v18(snap)

    # ── 输出结果 ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Search complete: %d trials", len(study.trials))

    best = study.best_trial
    logger.info("Best trial #%d  test_mae=%.2f", best.number, best.value)
    for k, v in best.params.items():
        logger.info("  %-14s = %s", k, v)

    # CSV: 所有 trial 结果
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "test_mae": t.value, "state": t.state.name}
        row.update(t.params)
        rows.append(row)
    results_df = pd.DataFrame(rows).sort_values("test_mae", na_position="last")
    csv_path = out_dir / "optuna_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved all trial results → %s", csv_path)

    # JSON: 最优参数
    best_info = {
        "trial_number": best.number,
        "test_mae": best.value,
        "params": best.params,
    }
    json_path = out_dir / "best_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    logger.info("Saved best params → %s", json_path)

    # Top-10 排名
    logger.info("\n" + "=" * 60)
    logger.info("Top-10 trials:")
    logger.info(
        "%-6s %-10s %-10s %-10s %-8s %-6s %-6s %-6s",
        "Trial", "test_mae", "lr", "wd", "dropout", "ctx_b", "ctx_a", "epochs",
    )
    for _, row in results_df.head(10).iterrows():
        logger.info(
            "%-6d %-10.2f %-10.1e %-10.1e %-8.2f %-6s %-6s %-6s",
            row["trial"],
            row["test_mae"] if pd.notna(row["test_mae"]) else float("inf"),
            row.get("lr", float("nan")),
            row.get("wd", float("nan")),
            row.get("dropout", float("nan")),
            row.get("ctx_before", "?"),
            row.get("ctx_after", "?"),
            row.get("epochs", "?"),
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
