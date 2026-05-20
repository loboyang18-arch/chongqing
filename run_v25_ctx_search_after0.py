#!/usr/bin/env python3
"""V25 dual02 方案1：V18_CTX_AFTER=0，在 BEFORE 上网格搜索（验证集选模）。

固定：V25_DUAL=1, V18_DELTA_LAMBDA=0.2, 与 dual02 一致。
每个 trial 仅写 val_result.csv，按 val MAE 排序；结束后对最优配置重训并评估测试集。

示例：
  python run_v25_ctx_search_after0.py
  python run_v25_ctx_search_after0.py --before 3 4 5 6 7 8 --epochs 100
  python run_v25_ctx_search_after0.py --skip-test   # 只搜 val
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from price_forecast_eval import evaluate_predictions_csv, write_metrics_json
from src.config import OUTPUT_DIR

PYTHON = os.environ.get("AUTONOMOUS_PYTHON", "/root/miniconda3/envs/power/bin/python")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SEARCH_DIR = OUTPUT_DIR / "v25_ctx_after0_search" / RUN_ID


def _rank_score(mae: float | None, corr: float | None) -> float | None:
    if mae is None or corr is None:
        return None
    return round(-float(mae) + 50.0 * float(corr), 4)


def _eval_val(val_csv: Path, metrics_out: Path) -> dict[str, Any]:
    ev = evaluate_predictions_csv(
        val_csv, actual_col="actual", pred_col="predicted", task_type="da",
    )
    write_metrics_json(ev, metrics_out)
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    mae, corr = pm.get("mae"), sm.get("profile_corr")
    return {
        "val_mae": mae,
        "val_rmse": pm.get("rmse"),
        "val_profile_corr": corr,
        "val_direction_acc": sm.get("direction_acc"),
        "rank_score": _rank_score(mae, corr),
    }


def _run_trial(before: int, epochs: int, log: logging.Logger) -> dict[str, Any]:
    tid = f"cb{before}_ca0_e{epochs}"
    out_rel = f"v25_ctx_after0_search/{RUN_ID}/trials/{tid}"
    out_dir = OUTPUT_DIR / out_rel
    env = os.environ.copy()
    env.update({
        "V25_DUAL": "1",
        "V18_DELTA_LAMBDA": "0.2",
        "V18_CTX_BEFORE": str(before),
        "V18_CTX_AFTER": "0",
        "V18_EPOCHS": str(epochs),
        "V18_SAVE_VAL_PRED": "1",
        "V18_SKIP_TEST_PRED": "1",
        "V25_NO_EVAL": "1",
        "V25_OUT_DIR": out_rel,
    })
    log.info("RUN %s | BEFORE=%d AFTER=0 epochs=%d", tid, before, epochs)
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, str(ROOT / "run_v25_resconv.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    dur = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trial_stdout.log").write_text(
        (proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or ""),
        encoding="utf-8",
    )
    row: dict[str, Any] = {
        "trial_id": tid,
        "V18_CTX_BEFORE": before,
        "V18_CTX_AFTER": 0,
        "h_slots": (before + 1) * 4,
        "epochs": epochs,
        "duration_sec": dur,
        "out_dir": str(out_dir),
    }
    if proc.returncode != 0:
        row["status"] = "failed"
        row["error"] = f"exit={proc.returncode}"
        return row
    val_csv = out_dir / "val_result.csv"
    if not val_csv.is_file():
        row["status"] = "failed"
        row["error"] = "missing val_result.csv"
        return row
    try:
        m = _eval_val(val_csv, out_dir / "metrics_val.json")
        row.update(m)
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)
    return row


def _write_leaderboard(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("val_mae") is not None]
    ok.sort(key=lambda r: (float(r["val_mae"]), -float(r.get("val_profile_corr") or 0)))
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    failed = [r for r in rows if r.get("status") != "ok"]
    ordered = ok + failed
    headers = [
        "rank", "trial_id", "status", "V18_CTX_BEFORE", "V18_CTX_AFTER", "h_slots",
        "epochs", "val_mae", "val_profile_corr", "val_direction_acc", "rank_score",
        "duration_sec", "out_dir", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow(r)
    return ok


def _run_best_test(best: dict[str, Any], log: logging.Logger) -> None:
    before = int(best["V18_CTX_BEFORE"])
    epochs = int(best["epochs"])
    out_rel = "v25_resconv_dual02_after0_best"
    env = os.environ.copy()
    env.update({
        "V25_DUAL": "1",
        "V18_DELTA_LAMBDA": "0.2",
        "V18_CTX_BEFORE": str(before),
        "V18_CTX_AFTER": "0",
        "V18_EPOCHS": str(epochs),
        "V25_OUT_DIR": out_rel,
    })
    log.info("Retrain best on full pipeline → %s", out_rel)
    subprocess.run([PYTHON, str(ROOT / "run_v25_resconv.py")], cwd=str(ROOT), env=env, check=True)
    meta = {
        "selected_from": str(SEARCH_DIR),
        "best_trial": best.get("trial_id"),
        "V18_CTX_BEFORE": before,
        "V18_CTX_AFTER": 0,
        "h_slots": (before + 1) * 4,
        "val_mae": best.get("val_mae"),
        "val_profile_corr": best.get("val_profile_corr"),
    }
    out_dir = OUTPUT_DIR / out_rel
    with (out_dir / "ctx_search_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="V25 dual02: AFTER=0, search BEFORE on val")
    ap.add_argument(
        "--before", type=int, nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9],
        help="V18_CTX_BEFORE candidates (default 3..9)",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-test", action="store_true", help="Do not retrain best on test")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("v25_ctx_after0")
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Search dir: %s", SEARCH_DIR)
    log.info("AFTER=0 fixed | BEFORE=%s | epochs=%d", args.before, args.epochs)

    rows: list[dict[str, Any]] = []
    for b in args.before:
        rows.append(_run_trial(b, args.epochs, log))

    lb_path = SEARCH_DIR / "leaderboard.csv"
    ok = _write_leaderboard(rows, lb_path)
    summary = {
        "run_id": RUN_ID,
        "policy": "AFTER=0, dual02 lambda=0.2, val MAE primary",
        "epochs": args.epochs,
        "before_grid": args.before,
        "n_ok": len(ok),
        "best": ok[0] if ok else None,
    }
    with (SEARCH_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Leaderboard: %s", lb_path)
    if ok:
        b = ok[0]
        log.info(
            "BEST val: BEFORE=%s H_SLOTS=%s MAE=%.4f corr=%.4f",
            b["V18_CTX_BEFORE"], b["h_slots"], b["val_mae"], b["val_profile_corr"],
        )
        if not args.skip_test:
            _run_best_test(b, log)
    else:
        log.error("No successful trials")
        sys.exit(1)


if __name__ == "__main__":
    main()
