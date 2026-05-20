#!/usr/bin/env python3
"""V25 早停 + 14 天验证：dropout 网格扫描。

  V25_VAL_WEEKS=2 python -u run_v25_dropout_grid.py
  V25_DROPOUT_GRID=0,0.1,0.2,0.3 python -u run_v25_dropout_grid.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("AUTONOMOUS_PYTHON", "/root/miniconda3/envs/power/bin/python")
OUT_ROOT = ROOT / "output" / os.environ.get("V25_DROPOUT_GRID_DIR", "v25_dropout_grid_val14d")


def _parse_grid() -> list[float]:
    raw = os.environ.get("V25_DROPOUT_GRID", "0,0.05,0.1,0.15,0.2,0.25,0.3")
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _drop_tag(d: float) -> str:
    return f"drop{str(d).replace('.', 'p')}"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid = _parse_grid()
    rows = []

    print(f"[V25] dropout 网格 (val14d): {grid}\n", flush=True)

    for d in grid:
        tag = _drop_tag(d)
        out_rel = f"v25_dropout_grid_val14d/{tag}"
        env = os.environ.copy()
        env.update({
            "V25_VAL_WEEKS": "2",
            "V25_DROPOUT": str(d),
            "V25_OUT_DIR": out_rel,
            "V25_DUAL": "1",
            "V25_CTX_BEFORE": "5",
            "V25_CTX_AFTER": "0",
            "V25_DELTA_LAMBDA": "0.2",
            "V25_EPOCHS": os.environ.get("V25_EPOCHS", "100"),
            "V25_PATIENCE": os.environ.get("V25_PATIENCE", "15"),
            "V25_MIN_EPOCHS": os.environ.get("V25_MIN_EPOCHS", "20"),
        })

        out_dir = ROOT / "output" / out_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train_console.log"

        print("=" * 72, flush=True)
        print(f"[V25] dropout={d}  →  {out_dir}", flush=True)
        print("=" * 72, flush=True)

        with log_path.open("w", encoding="utf-8") as logf:
            r = subprocess.run(
                [PYTHON, "-u", str(ROOT / "run_v25_earlystop.py")],
                cwd=str(ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        if r.returncode != 0:
            print(f"  WARN: 子进程 exit={r.returncode} (若训练完成可忽略 tee 类错误)", flush=True)

        hist_path = out_dir / "v25_train_history.json"
        meta_path = out_dir / "v25_meta.json"
        metrics_path = out_dir / "metrics_eval_standard_da_result.json"

        row = {"dropout": d, "out_dir": out_rel, "status": "ok" if hist_path.is_file() else "fail"}
        if hist_path.is_file():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            row["best_epoch"] = hist.get("best_epoch")
            row["best_val_mae"] = hist.get("best_val_mae")
            row["final_epoch"] = hist.get("final_epoch")
            last = hist.get("history", [])[-1] if hist.get("history") else {}
            row["final_train_mae"] = last.get("train_mae")
            row["final_train_corr"] = last.get("train_corr")
            row["final_val_mae"] = last.get("val_mae")
            row["final_val_corr"] = last.get("val_corr")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["test_mae"] = meta.get("test_mae")
            row["test_profile_corr"] = meta.get("test_profile_corr")
        if metrics_path.is_file():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            row["test_mae_eval"] = m.get("point_metrics", {}).get("mae")
            row["test_corr_eval"] = m.get("shape_metrics", {}).get("profile_corr")

        rows.append(row)
        print(
            f"  best_ep={row.get('best_epoch')} val_mae={row.get('best_val_mae')} "
            f"test_mae={row.get('test_mae')} test_corr={row.get('test_profile_corr')}",
            flush=True,
        )

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("dropout")
    csv_path = OUT_ROOT / "dropout_grid_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[V25] 汇总 → {csv_path}\n", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
