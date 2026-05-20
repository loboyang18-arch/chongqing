#!/usr/bin/env python3
"""V25 部署包：5+0 + λ=0.2，日前 + 实时出清价，统一输出目录。

交付物（仅此目录下）：
  da_result_v25.csv, rt_result_v25.csv
  plots_da/, plots_rt/
  seed0_da.pt, seed0_rt.pt
  da_v25_meta.json, rt_v25_meta.json
  train_da.log, train_rt.log
  ctx_deploy_meta.json

训练在系统临时目录完成，结束后只保留上述文件（无 _work_*）。

示例：
  python run_v25_deploy_bundle.py
  python run_v25_deploy_bundle.py --replot-only
  V25_DEPLOY_DIR=my_deploy python run_v25_deploy_bundle.py
  V25_DEPLOY_KEEP_TMP=1   # 调试：保留临时训练目录不删
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR

PYTHON = os.environ.get("AUTONOMOUS_PYTHON", "/root/miniconda3/envs/power/bin/python")

CTX_BEFORE = int(os.environ.get("V18_CTX_BEFORE", "5"))
CTX_AFTER = int(os.environ.get("V18_CTX_AFTER", "0"))
DELTA_LAMBDA = os.environ.get("V18_DELTA_LAMBDA", "0.2")
EPOCHS = int(os.environ.get("V18_EPOCHS", "100"))
DEPLOY_SUB = os.environ.get("V25_DEPLOY_DIR", "v25_deploy_5p0_lam02").strip() or "v25_deploy_5p0_lam02"
KEEP_TMP = os.environ.get("V25_DEPLOY_KEEP_TMP", "").strip().lower() in ("1", "true", "yes")


def _viz_label(task: str) -> str:
    t = "RT" if task == "rt" else "DA"
    return f"V25-ResConv-{t}-{CTX_BEFORE}+{CTX_AFTER}-λ{DELTA_LAMBDA}"


def _remove_legacy_work_dirs(deploy_dir: Path, log: logging.Logger) -> None:
    """删除旧版 bundle 留下的 _work_da / _work_rt。"""
    for name in ("_work_da", "_work_rt"):
        p = deploy_dir / name
        if p.is_dir():
            shutil.rmtree(p)
            log.info("Removed legacy dir %s/", name)
    for name in ("work_da_v25_meta.json", "work_rt_v25_meta.json"):
        p = deploy_dir / name
        if p.is_file():
            p.unlink()
            log.info("Removed legacy file %s", name)


def _copy_plots(work_dir: Path, deploy_dir: Path, plots_name: str, log: logging.Logger) -> None:
    src = work_dir / "plots"
    if not src.is_dir():
        log.warning("No plots dir: %s", src)
        return
    dst = deploy_dir / plots_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log.info("Copied plots → %s/", plots_name)


def _run_task(
    task: str,
    work_dir: Path,
    deploy_dir: Path,
    log: logging.Logger,
) -> Path:
    """在 work_dir 内训练；返回 result.csv 路径。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        work_rel = str(work_dir.relative_to(OUTPUT_DIR))
    except ValueError as e:
        raise ValueError(
            f"Training dir must be under OUTPUT_DIR={OUTPUT_DIR}, got {work_dir}"
        ) from e

    env = os.environ.copy()
    env.update({
        "V25_DUAL": "1",
        "V18_DELTA_LAMBDA": DELTA_LAMBDA,
        "V18_CTX_BEFORE": str(CTX_BEFORE),
        "V18_CTX_AFTER": str(CTX_AFTER),
        "V18_EPOCHS": str(EPOCHS),
        "V25_OUT_DIR": work_rel,
        "V25_NO_EVAL": "1",
        "V18_VIZ_LABEL": _viz_label(task),
    })
    if task == "rt":
        env["V24_TARGET_COL"] = "realtime_clearing_price"
    else:
        env.pop("V24_TARGET_COL", None)

    log.info("Train+predict %s (tmp %s)", task.upper(), work_dir)
    proc = subprocess.run(
        [PYTHON, str(ROOT / "run_v25_resconv.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    (deploy_dir / f"train_{task}.log").write_text(
        (proc.stdout or "") + "\n---stderr---\n" + (proc.stderr or ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        log.error("%s failed exit=%s (log: train_%s.log)", task, proc.returncode, task)
        if KEEP_TMP:
            log.info("V25_DEPLOY_KEEP_TMP=1 → kept %s", work_dir)
        raise SystemExit(proc.returncode)

    result_name = "rt_result.csv" if task == "rt" else "da_result.csv"
    src = work_dir / result_name
    if not src.is_file():
        raise FileNotFoundError(f"Missing {src}")
    return src


def _finalize_task(
    task: str,
    work_dir: Path,
    deploy_dir: Path,
    result_src: Path,
    log: logging.Logger,
) -> None:
    """把单次训练产物写入 deploy 根目录。"""
    csv_dst = deploy_dir / ("rt_result_v25.csv" if task == "rt" else "da_result_v25.csv")
    shutil.copy2(result_src, csv_dst)
    log.info("Wrote %s (%d bytes)", csv_dst.name, csv_dst.stat().st_size)

    plots_name = "plots_rt" if task == "rt" else "plots_da"
    _copy_plots(work_dir, deploy_dir, plots_name, log)

    pt = work_dir / "seed0.pt"
    if pt.is_file():
        shutil.copy2(pt, deploy_dir / f"seed0_{task}.pt")

    meta = work_dir / "v25_meta.json"
    if meta.is_file():
        shutil.copy2(meta, deploy_dir / f"{task}_v25_meta.json")


def replot_only(deploy_dir: Path, log: logging.Logger) -> None:
    """仅根据已有 *_v25.csv 重画 plots_da / plots_rt（不重训）。"""
    from price_forecast_eval.viz import run_standard_visualization

    for task, csv_name, plots_name in (
        ("da", "da_result_v25.csv", "plots_da"),
        ("rt", "rt_result_v25.csv", "plots_rt"),
    ):
        csv_path = deploy_dir / csv_name
        if not csv_path.is_file():
            log.error("Missing %s", csv_path)
            raise SystemExit(1)
        plots_dir = deploy_dir / plots_name
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True)
        label = _viz_label(task)
        run_standard_visualization(
            csv_path,
            out_dir=plots_dir,
            label=label,
            actual_col="actual",
            pred_col="predicted",
            mode="appendix",
            weekly=True,
        )
        log.info("Replotted %s → %s/ (label=%s)", csv_name, plots_name, label)


def main() -> None:
    log = logging.getLogger("v25_deploy")

    deploy_dir = OUTPUT_DIR / DEPLOY_SUB
    deploy_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_work_dirs(deploy_dir, log)

    log.info("Deploy dir: %s", deploy_dir)
    log.info("Config: BEFORE=%d AFTER=%d λ=%s epochs=%d",
             CTX_BEFORE, CTX_AFTER, DELTA_LAMBDA, EPOCHS)

    for task in ("da", "rt"):
        if KEEP_TMP:
            work_dir = deploy_dir / f".tmp_train_{task}"
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True)
            result_src = _run_task(task, work_dir, deploy_dir, log)
            _finalize_task(task, work_dir, deploy_dir, result_src, log)
            log.info("V25_DEPLOY_KEEP_TMP=1 → kept %s", work_dir)
        else:
            with tempfile.TemporaryDirectory(
                prefix=f"v25_{task}_",
                dir=OUTPUT_DIR,
            ) as tmp:
                work_dir = Path(tmp)
                result_src = _run_task(task, work_dir, deploy_dir, log)
                _finalize_task(task, work_dir, deploy_dir, result_src, log)

    meta = {
        "model": "DualHeadResConv2dPriceNet",
        "V18_CTX_BEFORE": CTX_BEFORE,
        "V18_CTX_AFTER": CTX_AFTER,
        "V18_DELTA_LAMBDA": float(DELTA_LAMBDA),
        "V18_EPOCHS": EPOCHS,
        "h_slots": (CTX_BEFORE + 1 + CTX_AFTER) * 4,
        "da_result": "da_result_v25.csv",
        "rt_result": "rt_result_v25.csv",
        "da_viz_label": _viz_label("da"),
        "rt_viz_label": _viz_label("rt"),
        "plots_da": "plots_da/",
        "plots_rt": "plots_rt/",
        "format": "ts,actual,predicted (hourly, official test window)",
    }
    with (deploy_dir / "ctx_deploy_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Done. %s", deploy_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="V25 5+0 部署包")
    ap.add_argument(
        "--replot-only",
        action="store_true",
        help="仅重画 plots_da/plots_rt（需已有 da/rt_result_v25.csv）",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("v25_deploy")
    deploy_dir = OUTPUT_DIR / DEPLOY_SUB
    if args.replot_only:
        replot_only(deploy_dir, log)
    else:
        main()
