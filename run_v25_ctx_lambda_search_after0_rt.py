#!/usr/bin/env python3
"""V25 实时出清价：AFTER=0 固定，BEFORE × λ 网格（验证集选模）。

须在 import V24 前设置 V24_TARGET_COL=realtime_clearing_price。
Lag2 自动剔除 realtime_clearing_* 防标签泄漏。

示例：
  python run_v25_ctx_lambda_search_after0_rt.py
  python run_v25_ctx_lambda_search_after0_rt.py --skip-test
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 须在 import v24/v25 前
os.environ.setdefault("V24_TARGET_COL", "realtime_clearing_price")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from price_forecast_eval import evaluate_predictions_csv, write_metrics_json
from src.config import OUTPUT_DIR

PYTHON = os.environ.get("AUTONOMOUS_PYTHON", "/root/miniconda3/envs/power/bin/python")


def _rank_score(mae: float | None, corr: float | None) -> float | None:
    if mae is None or corr is None:
        return None
    return round(-float(mae) + 50.0 * float(corr), 4)


def _eval_val(val_csv: Path, metrics_out: Path) -> dict[str, Any]:
    ev = evaluate_predictions_csv(
        val_csv, actual_col="actual", pred_col="predicted", task_type="rt",
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
        "val_neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
        "rank_score": _rank_score(mae, corr),
    }


def _run_trial(
    before: int,
    lam: float,
    epochs: int,
    run_id: str,
    log: logging.Logger,
) -> dict[str, Any]:
    lam_s = f"{lam:g}".replace(".", "p")
    tid = f"cb{before}_lam{lam_s}_ca0"
    out_rel = f"v25_ctx_lambda_after0_rt_search/{run_id}/trials/{tid}"
    out_dir = OUTPUT_DIR / out_rel
    env = os.environ.copy()
    env["V24_TARGET_COL"] = "realtime_clearing_price"
    env.update({
        "V25_DUAL": "1",
        "V18_DELTA_LAMBDA": str(lam),
        "V18_CTX_BEFORE": str(before),
        "V18_CTX_AFTER": "0",
        "V18_EPOCHS": str(epochs),
        "V18_SAVE_VAL_PRED": "1",
        "V18_SKIP_TEST_PRED": "1",
        "V25_NO_EVAL": "1",
        "V25_OUT_DIR": out_rel,
    })
    log.info("RUN RT %s | BEFORE=%d λ=%s AFTER=0", tid, before, lam)
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
        "target": "realtime_clearing_price",
        "V18_CTX_BEFORE": before,
        "V18_CTX_AFTER": 0,
        "V18_DELTA_LAMBDA": lam,
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


def _write_leaderboard(
    rows: list[dict[str, Any]],
    path: Path,
    sort_key,
) -> list[dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("val_mae") is not None]
    ok.sort(key=sort_key)
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    failed = [r for r in rows if r.get("status") != "ok"]
    ordered = ok + failed
    headers = [
        "rank", "trial_id", "status", "target", "V18_CTX_BEFORE", "V18_CTX_AFTER",
        "V18_DELTA_LAMBDA", "h_slots", "epochs",
        "val_mae", "val_rmse", "val_profile_corr", "val_direction_acc",
        "val_neg_corr_day_ratio", "rank_score", "duration_sec", "out_dir", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow(r)
    return ok


def _run_best_test(best: dict[str, Any], search_dir: Path, log: logging.Logger) -> Path:
    before = int(best["V18_CTX_BEFORE"])
    lam = float(best["V18_DELTA_LAMBDA"])
    epochs = int(best["epochs"])
    out_rel = "v25_resconv_rt_after0_ctx_lambda_best"
    env = os.environ.copy()
    env["V24_TARGET_COL"] = "realtime_clearing_price"
    env.update({
        "V25_DUAL": "1",
        "V18_DELTA_LAMBDA": str(lam),
        "V18_CTX_BEFORE": str(before),
        "V18_CTX_AFTER": "0",
        "V18_EPOCHS": str(epochs),
        "V25_OUT_DIR": out_rel,
    })
    log.info("Retrain RT best → %s (BEFORE=%d λ=%s)", out_rel, before, lam)
    proc = subprocess.run(
        [PYTHON, str(ROOT / "run_v25_resconv_rt.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("Test retrain failed:\n%s", proc.stderr)
        raise SystemExit(proc.returncode)
    out_dir = OUTPUT_DIR / out_rel
    meta = {
        "selected_from": str(search_dir),
        "target": "realtime_clearing_price",
        "best_trial": best.get("trial_id"),
        "V18_CTX_BEFORE": before,
        "V18_CTX_AFTER": 0,
        "V18_DELTA_LAMBDA": lam,
        "h_slots": (before + 1) * 4,
        "val_mae": best.get("val_mae"),
        "val_profile_corr": best.get("val_profile_corr"),
        "rank_score": best.get("rank_score"),
    }
    with (out_dir / "ctx_lambda_search_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_dir


def _load_test_metrics(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "metrics_eval_standard_rt_result.json"
    with p.open(encoding="utf-8") as f:
        ev = json.load(f)
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    return {
        "test_mae": pm.get("mae"),
        "test_rmse": pm.get("rmse"),
        "test_profile_corr": sm.get("profile_corr"),
        "test_direction_acc": sm.get("direction_acc"),
        "test_neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V25 RT: AFTER=0, search BEFORE×λ on val")
    ap.add_argument("--before", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    ap.add_argument(
        "--lambda", dest="lambdas", type=float, nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = OUTPUT_DIR / "v25_ctx_lambda_after0_rt_search" / run_id
    search_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("v25_ctx_lambda_rt")
    combos = list(itertools.product(args.before, args.lambdas))
    log.info("RT search dir: %s", search_dir)
    log.info("AFTER=0 | %d trials | epochs=%d", len(combos), args.epochs)

    rows: list[dict[str, Any]] = []
    for b, lam in combos:
        rows.append(_run_trial(b, lam, args.epochs, run_id, log))

    ok_mae = _write_leaderboard(
        rows, search_dir / "leaderboard_by_val_mae.csv",
        sort_key=lambda r: (float(r["val_mae"]), -float(r.get("val_profile_corr") or 0)),
    )
    ok_score = _write_leaderboard(
        rows, search_dir / "leaderboard_by_rank_score.csv",
        sort_key=lambda r: (-float(r.get("rank_score") or -1e9), float(r["val_mae"])),
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "target": "realtime_clearing_price",
        "policy": "AFTER=0, val MAE primary for test retrain",
        "epochs": args.epochs,
        "before_grid": args.before,
        "lambda_grid": args.lambdas,
        "n_trials": len(combos),
        "n_ok": len(ok_mae),
        "best_by_val_mae": ok_mae[0] if ok_mae else None,
        "best_by_rank_score": ok_score[0] if ok_score else None,
    }

    if not args.skip_test and ok_mae:
        test_dir = _run_best_test(ok_mae[0], search_dir, log)
        summary["test_metrics_best_val_mae"] = _load_test_metrics(test_dir)

    with (search_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("leaderboard: %s", search_dir / "leaderboard_by_val_mae.csv")
    if ok_mae:
        b = ok_mae[0]
        log.info(
            "BEST val MAE: BEFORE=%s λ=%s MAE=%.4f corr=%.4f",
            b["V18_CTX_BEFORE"], b["V18_DELTA_LAMBDA"], b["val_mae"], b["val_profile_corr"],
        )
    if ok_score:
        s = ok_score[0]
        log.info(
            "BEST rank_score: BEFORE=%s λ=%s MAE=%.4f corr=%.4f score=%s",
            s["V18_CTX_BEFORE"], s["V18_DELTA_LAMBDA"],
            s["val_mae"], s["val_profile_corr"], s.get("rank_score"),
        )
    if not ok_mae:
        log.error("No successful trials")
        sys.exit(1)


if __name__ == "__main__":
    main()
