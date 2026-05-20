#!/usr/bin/env python3
"""V25 末轮权重 + 100 epoch：dropout 扩大网格（默认 14 天验证，不早停）。

  python -u run_v25_dropout_lastepoch_grid.py

  V25_DROPOUT_GRID=0,0.1,0.2,0.3,0.4,0.5 python -u run_v25_dropout_lastepoch_grid.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("AUTONOMOUS_PYTHON", "/root/miniconda3/envs/power/bin/python")
OUT_ROOT = ROOT / "output" / os.environ.get(
    "V25_DROPOUT_GRID_DIR", "v25_dropout_lastepoch_grid_val14d"
)

# 扩大搜索：0～0.5 步长 0.02，并补充 0.55/0.6
DEFAULT_GRID = ",".join(
    [f"{x:.2f}" for x in [i * 0.02 for i in range(26)]]  # 0.00 .. 0.50
    + ["0.55", "0.60"]
)


def _parse_grid() -> list[float]:
    raw = os.environ.get("V25_DROPOUT_GRID", DEFAULT_GRID)
    return sorted(set(float(x.strip()) for x in raw.split(",") if x.strip()))


def _drop_tag(d: float) -> str:
    s = f"{d:.3f}".rstrip("0").rstrip(".")
    return "drop" + s.replace(".", "p")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid = _parse_grid()
    n = len(grid)
    print(
        f"[V25] 末轮权重 100ep  dropout 网格 n={n} (val14d, no early_stop)\n",
        flush=True,
    )

    rows = []
    for i, d in enumerate(grid, 1):
        tag = _drop_tag(d)
        out_rel = f"v25_dropout_lastepoch_grid_val14d/{tag}"
        out_dir = ROOT / "output" / out_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train_console.log"

        env = os.environ.copy()
        env.update({
            "V25_EARLY_STOP": "0",
            "V25_EPOCHS": "100",
            "V25_VAL_WEEKS": "2",
            "V25_DROPOUT": str(d),
            "V25_OUT_DIR": out_rel,
            "V25_DUAL": "1",
            "V25_CTX_BEFORE": "5",
            "V25_CTX_AFTER": "0",
            "V25_DELTA_LAMBDA": "0.2",
            "SPLIT_TRAIN_END": "2026-02-16 23:45:00",
            "SPLIT_VAL_END": "2026-03-02 23:45:00",
        })

        print(f"[{i}/{n}] dropout={d}  →  {out_dir.name}", flush=True)

        with log_path.open("w", encoding="utf-8") as logf:
            subprocess.run(
                [PYTHON, "-u", str(ROOT / "run_v25_lastepoch.py")],
                cwd=str(ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        hist_path = out_dir / "v25_train_history.json"
        meta_path = out_dir / "v25_meta.json"
        row = {"dropout": d, "out_dir": out_rel}
        if hist_path.is_file():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            row["weight_policy"] = hist.get("weight_policy")
            row["final_epoch"] = hist.get("final_epoch")
            row["best_val_mae"] = hist.get("best_val_mae")
            last = hist.get("history", [])[-1] if hist.get("history") else {}
            row["ep100_train_mae"] = last.get("train_mae")
            row["ep100_train_corr"] = last.get("train_corr")
            row["ep100_val_mae"] = last.get("val_mae")
            row["ep100_val_corr"] = last.get("val_corr")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["test_mae"] = meta.get("test_mae")
            row["test_profile_corr"] = meta.get("test_profile_corr")

        rows.append(row)
        print(
            f"    ep100 train_mae={row.get('ep100_train_mae')} corr={row.get('ep100_train_corr')} "
            f"| test_mae={row.get('test_mae')} test_corr={row.get('test_profile_corr')}",
            flush=True,
        )

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("dropout")
    csv_path = OUT_ROOT / "dropout_lastepoch_grid_summary.csv"
    df.to_csv(csv_path, index=False)

    best_mae = df.loc[df["test_mae"].idxmin()] if df["test_mae"].notna().any() else None
    best_corr = df.loc[df["test_profile_corr"].idxmax()] if df["test_profile_corr"].notna().any() else None
    print(f"\n汇总 → {csv_path}\n", flush=True)
    print(df[["dropout", "ep100_train_mae", "ep100_val_mae", "test_mae", "test_profile_corr"]].to_string(index=False), flush=True)
    if best_mae is not None:
        print(f"\n测试 MAE 最低: dropout={best_mae['dropout']}  MAE={best_mae['test_mae']:.2f}", flush=True)
    if best_corr is not None:
        print(f"测试 corr 最高: dropout={best_corr['dropout']}  corr={best_corr['test_profile_corr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
