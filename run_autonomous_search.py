#!/usr/bin/env python3
"""
自主算法搜索编排器（验证集选模，不碰测试集）。

- 仅写 val_result.csv 并在验证窗评估排序
- V18_SKIP_TEST_PRED=1 跳过测试预测与标准 test 评估
- 落盘 output/autonomous_search/<run_id>/：status.json、leaderboard.csv、search.log

示例（10 小时）:
  python run_autonomous_search.py --hours 10
  python run_autonomous_search.py --status-only   # 查看进展
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
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from price_forecast_eval import evaluate_predictions_csv, write_metrics_json
from src.config import OUTPUT_DIR

SEARCH_PARENT = OUTPUT_DIR / "autonomous_search"
PYTHON = os.environ.get("AUTONOMOUS_PYTHON", "/root/miniconda3/envs/power/bin/python")

METRIC_COLS = [
    "trial_id",
    "status",
    "family",
    "val_mae",
    "val_rmse",
    "val_profile_corr",
    "val_direction_acc",
    "val_amplitude_err",
    "val_neg_corr_day_ratio",
    "rank_score",
    "duration_sec",
    "out_dir",
    "config_json",
]


@dataclass
class TrialConfig:
    family: str
    trial_id: str
    out_rel: str
    env: dict[str, str] = field(default_factory=dict)
    epochs: int = 50

    def label(self) -> str:
        parts = [self.family]
        for k in sorted(self.env):
            if k.startswith("V18_") or k.startswith("V25_"):
                parts.append(f"{k}={self.env[k]}")
        return " ".join(parts)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rank_score(mae: Optional[float], profile_corr: Optional[float]) -> Optional[float]:
    if mae is None or profile_corr is None:
        return None
    return round(-float(mae) + 50.0 * float(profile_corr), 4)


def _extract_val_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open(encoding="utf-8") as f:
        ev = json.load(f)
    pm = ev.get("point_metrics") or {}
    sm = ev.get("shape_metrics") or {}
    mae = pm.get("mae")
    corr = sm.get("profile_corr")
    return {
        "val_mae": mae,
        "val_rmse": pm.get("rmse"),
        "val_profile_corr": corr,
        "val_direction_acc": sm.get("direction_acc"),
        "val_amplitude_err": sm.get("amplitude_err"),
        "val_neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
        "rank_score": _rank_score(mae, corr),
    }


def _eval_val_csv(val_csv: Path, metrics_out: Path) -> dict[str, Any]:
    ev = evaluate_predictions_csv(
        val_csv,
        actual_col="actual",
        pred_col="predicted",
        task_type="da",
    )
    write_metrics_json(ev, metrics_out)
    row = _extract_val_metrics(metrics_out)
    row["metrics_path"] = str(metrics_out)
    return row


def _build_trial_queue() -> list[TrialConfig]:
    """生成搜索队列：先广搜（50 epoch），穿插单头/上下文/正则。"""
    trials: list[TrialConfig] = []
    seq = 0

    def add(family: str, env: dict[str, str], epochs: int = 50) -> None:
        nonlocal seq
        seq += 1
        tid = f"t{seq:04d}_{family}"
        out_rel = f"autonomous_search/trials/{tid}"
        base = {
            "V18_SAVE_VAL_PRED": "1",
            "V18_SKIP_TEST_PRED": "1",
            "V25_NO_EVAL": "1",
            "V18_EPOCHS": str(epochs),
            "V25_OUT_DIR": out_rel,
        }
        base.update(env)
        trials.append(TrialConfig(family=family, trial_id=tid, out_rel=out_rel, env=base, epochs=epochs))

    # 1) 双头 λ 细网格（当前最强方向）
    for lam in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
        add("v25_dual_lambda", {"V25_DUAL": "1", "V18_DELTA_LAMBDA": str(lam)})

    # 2) λ=0.2 周围 lr / wd / dropout
    for lr, wd, drop in itertools.product(
        ["5e-4", "1e-3", "2e-3"],
        ["1e-5", "1e-4", "1e-3"],
        ["0.05", "0.1", "0.15"],
    ):
        add(
            "v25_dual_hparam",
            {
                "V25_DUAL": "1",
                "V18_DELTA_LAMBDA": "0.2",
                "V18_LR": lr,
                "V18_WD": wd,
                "V18_DROPOUT": drop,
            },
        )

    # 3) 上下文窗口
    for cb, ca in [(4, 1), (5, 1), (6, 1), (7, 1), (5, 2), (6, 2), (7, 2)]:
        add(
            "v25_dual_ctx",
            {
                "V25_DUAL": "1",
                "V18_DELTA_LAMBDA": "0.2",
                "V18_CTX_BEFORE": str(cb),
                "V18_CTX_AFTER": str(ca),
            },
        )

    # 4) 单头对照
    for lr in ["5e-4", "1e-3", "2e-3"]:
        add("v25_single", {"V25_DUAL": "0", "V18_LR": lr})

    # 5) 训练过采样 / 残差 MC（轻量尝试）
    for osample in ["2", "3"]:
        add(
            "v25_dual_oversample",
            {"V25_DUAL": "1", "V18_DELTA_LAMBDA": "0.2", "V18_TRAIN_OVERSAMPLE": osample},
        )
    add("v25_dual_resid_mc", {"V25_DUAL": "1", "V18_DELTA_LAMBDA": "0.2", "V18_RESIDUAL_MC": "1"})

    # 6) 高 epoch 精调（队列后部，若时间够会跑到）
    for lam in [0.15, 0.2, 0.25, 0.3]:
        add("v25_dual_refine100", {"V25_DUAL": "1", "V18_DELTA_LAMBDA": str(lam)}, epochs=100)

    return trials


def _run_v25_trial(trial: TrialConfig, log: logging.Logger) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(trial.env)
    cmd = [PYTHON, str(ROOT / "run_v25_resconv.py")]
    log.info("RUN %s | %s", trial.trial_id, trial.label())
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    dur = round(time.time() - t0, 1)
    out_dir = OUTPUT_DIR / trial.out_rel
    log_path = out_dir / "trial_stdout.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or ""), encoding="utf-8")

    if proc.returncode != 0:
        return {
            "status": "failed",
            "duration_sec": dur,
            "error": f"exit={proc.returncode}",
            "log_path": str(log_path),
        }

    val_csv = out_dir / "val_result.csv"
    if not val_csv.is_file():
        return {
            "status": "failed",
            "duration_sec": dur,
            "error": "missing val_result.csv",
            "log_path": str(log_path),
        }

    metrics_out = out_dir / "metrics_val.json"
    try:
        m = _eval_val_csv(val_csv, metrics_out)
    except Exception as e:
        return {
            "status": "failed",
            "duration_sec": dur,
            "error": f"val_eval: {e}",
            "log_path": str(log_path),
        }

    m["status"] = "ok"
    m["duration_sec"] = dur
    m["log_path"] = str(log_path)
    return m


def _write_leaderboard(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("val_mae") is not None]
    ok_rows.sort(
        key=lambda r: (
            float(r["val_mae"]),
            -float(r.get("val_profile_corr") or 0.0),
        ),
    )
    for i, r in enumerate(ok_rows, 1):
        r["rank"] = i

    all_rows = ok_rows + [r for r in rows if r.get("status") != "ok"]
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["rank"] + METRIC_COLS
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)


def _write_status(
    run_dir: Path,
    *,
    started_at: str,
    deadline_at: str,
    rows: list[dict[str, Any]],
    current: Optional[str],
    n_pending: int,
) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") == "failed"]
    best = None
    if ok:
        best = min(
            ok,
            key=lambda r: (
                float(r["val_mae"]),
                -float(r.get("val_profile_corr") or 0.0),
            ),
        )
    status = {
        "run_id": run_dir.name,
        "started_at": started_at,
        "deadline_at": deadline_at,
        "updated_at": _now_iso(),
        "policy": "validation_only_no_test",
        "n_completed_ok": len(ok),
        "n_failed": len(failed),
        "n_total_recorded": len(rows),
        "n_pending": n_pending,
        "current_trial": current,
        "best_val": best,
        "top5_val": sorted(
            ok,
            key=lambda r: (float(r["val_mae"]), -float(r.get("val_profile_corr") or 0.0)),
        )[:5],
        "reference_test_baseline": {
            "model": "v25_resconv_dual02",
            "test_mae": 86.7,
            "test_profile_corr": 0.24,
            "note": "测试集仅作参照，搜索过程不评估测试集",
        },
    }
    (run_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _latest_run_dir() -> Optional[Path]:
    if not SEARCH_PARENT.is_dir():
        return None
    runs = sorted([p for p in SEARCH_PARENT.iterdir() if p.is_dir()], key=lambda p: p.name)
    return runs[-1] if runs else None


def _print_status(run_dir: Path) -> None:
    status_path = run_dir / "status.json"
    lb_path = run_dir / "leaderboard.csv"
    if not status_path.is_file():
        print(f"无 status.json: {status_path}")
        return
    st = json.loads(status_path.read_text(encoding="utf-8"))
    print("=" * 72)
    print(f"运行 ID: {st.get('run_id')}")
    print(f"策略: {st.get('policy')}")
    print(f"开始: {st.get('started_at')}")
    print(f"截止: {st.get('deadline_at')}")
    print(f"更新: {st.get('updated_at')}")
    print(f"进度: ok={st.get('n_completed_ok')} failed={st.get('n_failed')} pending={st.get('n_pending')}")
    if st.get("current_trial"):
        print(f"当前: {st['current_trial']}")
    best = st.get("best_val")
    if best:
        print("-" * 72)
        print(
            f"当前最优(验证集): {best.get('trial_id')} | "
            f"MAE={best.get('val_mae'):.2f} | "
            f"profile_corr={best.get('val_profile_corr'):.4f} | "
            f"rank_score={best.get('rank_score')}"
        )
    ref = st.get("reference_test_baseline") or {}
    print(
        f"参照(测试集 baseline): {ref.get('model')} "
        f"MAE={ref.get('test_mae')} profile_corr={ref.get('test_profile_corr')}"
    )
    print("-" * 72)
    print("验证集 Top5:")
    for i, r in enumerate(st.get("top5_val") or [], 1):
        print(
            f"  {i}. {r.get('trial_id')} mae={r.get('val_mae'):.2f} "
            f"corr={r.get('val_profile_corr'):.4f} score={r.get('rank_score')} "
            f"({r.get('family')})"
        )
    if lb_path.is_file():
        print(f"\n完整榜单: {lb_path}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="自主搜索（验证集选模，不碰测试集）")
    ap.add_argument("--hours", type=float, default=10.0, help="最长运行小时数")
    ap.add_argument("--run-id", type=str, default="", help="指定 run 子目录名")
    ap.add_argument("--status-only", action="store_true", help="仅打印最新/指定 run 进展")
    ap.add_argument("--run-dir", type=Path, default=None, help="与 --status-only 联用")
    args = ap.parse_args()

    if args.status_only:
        run_dir = args.run_dir or _latest_run_dir()
        if run_dir is None:
            print("尚无 autonomous_search 运行记录")
            sys.exit(1)
        _print_status(run_dir)
        return

    run_id = args.run_id.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = SEARCH_PARENT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("autonomous_search")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "search.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(sh)

    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=args.hours)
    started_s = started.isoformat(timespec="seconds")
    deadline_s = deadline.isoformat(timespec="seconds")

    queue = _build_trial_queue()
    (run_dir / "queue.json").write_text(
        json.dumps([asdict(t) for t in queue], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Run %s | trials=%d | deadline=%s", run_id, len(queue), deadline_s)

    rows: list[dict[str, Any]] = []
    for i, trial in enumerate(queue):
        if datetime.now().astimezone() >= deadline:
            log.info("到达时间上限，停止队列")
            break

        _write_status(
            run_dir,
            started_at=started_s,
            deadline_at=deadline_s,
            rows=rows,
            current=trial.trial_id,
            n_pending=len(queue) - i,
        )

        result = _run_v25_trial(trial, log)
        row = {
            "trial_id": trial.trial_id,
            "family": trial.family,
            "out_dir": trial.out_rel,
            "config_json": json.dumps(trial.env, ensure_ascii=False, sort_keys=True),
            **result,
        }
        rows.append(row)
        _write_leaderboard(rows, run_dir / "leaderboard.csv")
        _write_status(
            run_dir,
            started_at=started_s,
            deadline_at=deadline_s,
            rows=rows,
            current=None,
            n_pending=len(queue) - i - 1,
        )

        if result.get("status") == "ok":
            log.info(
                "DONE %s val_mae=%.2f corr=%.4f score=%s (%.0fs)",
                trial.trial_id,
                result.get("val_mae", float("nan")),
                result.get("val_profile_corr", float("nan")),
                result.get("rank_score"),
                result.get("duration_sec", 0),
            )
        else:
            log.warning("FAIL %s: %s", trial.trial_id, result.get("error"))

    _write_status(
        run_dir,
        started_at=started_s,
        deadline_at=deadline_s,
        rows=rows,
        current=None,
        n_pending=0,
    )
    log.info("搜索结束 | ok=%d failed=%d | 见 %s", 
             sum(1 for r in rows if r.get("status") == "ok"),
             sum(1 for r in rows if r.get("status") == "failed"),
             run_dir)


if __name__ == "__main__":
    main()
