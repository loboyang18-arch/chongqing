#!/usr/bin/env python3
"""V25 场景双专家：打标 → 训练 normal/low → 硬路由预测。

默认：merge val、dropout=0.44、100 epoch 末轮、5+0 dual。

  python -u run_v25_scene_pipeline.py label
  python -u run_v25_scene_pipeline.py train
  python -u run_v25_scene_pipeline.py route
  python -u run_v25_scene_pipeline.py all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _apply_train_env() -> None:
    """须在 import model_v25_train 之前调用（该模块 import 时会快照部分 env）。"""
    os.environ["V25_EARLY_STOP"] = "0"
    os.environ["V25_EPOCHS"] = "100"
    os.environ["V25_DUAL"] = "1"
    os.environ["V25_CTX_BEFORE"] = "5"
    os.environ["V25_CTX_AFTER"] = "0"
    os.environ["V25_DELTA_LAMBDA"] = "0.2"
    os.environ["V25_DROPOUT"] = "0.44"
    os.environ["V25_MERGE_VAL"] = "1"


_apply_train_env()

from price_forecast_eval import quick_shape_report

from src.config import OUTPUT_DIR
from src.experiment.splits import TEST_END, TEST_START
from src.model_v25_resconv import DualHeadResConv2dPriceNet
from src.model_v25_train import (
    V25_CTX_AFTER,
    V25_CTX_BEFORE,
    V25_DUAL,
    load_v25_norm,
    run_v25_early_stop,
)
from src.v25_scene import (
    SCENE_TO_EXPERT,
    build_daily_stats,
    default_all_label_days,
    default_fit_days,
    fit_thresholds,
    label_all_days,
    load_label_artifacts,
    save_label_artifacts,
    test_dates_with_expert,
    train_dates_for_expert,
)

SCENE_ROOT = OUTPUT_DIR / os.environ.get("V25_SCENE_ROOT", "v25_scene_2way").strip()
EXPERTS = ("normal", "low")


def cmd_label(merge_val: bool) -> None:
    daily = build_daily_stats()
    fit_days = default_fit_days(merge_val=merge_val)
    all_days = default_all_label_days()
    th = fit_thresholds(fit_days, daily)
    table = label_all_days(all_days, daily, th)
    save_label_artifacts(SCENE_ROOT, table, th, merge_val)

    table["date_obj"] = pd.to_datetime(table["date"]).dt.date
    fit_set = set(fit_days)
    test_mask = (
        (table["date_obj"] >= TEST_START.date())
        & (table["date_obj"] <= TEST_END.date())
    )
    print(f"[scene] thresholds: p_low={th.p_low:.2f} p_high={th.p_high:.2f} p_vol={th.p_vol:.2f}")
    print(f"[scene] wrote {SCENE_ROOT / 'day_scene.csv'}")
    for split_name, mask in [
        ("fit/train", table["date_obj"].isin(fit_set)),
        ("test", test_mask),
    ]:
        sub = table.loc[mask]
        if len(sub) == 0:
            continue
        print(f"\n{split_name} ({len(sub)} days) expert:")
        print(sub.groupby("expert").size().to_string())
        print("scene:")
        print(sub.groupby("scene").size().to_string())


def cmd_train(merge_val: bool) -> None:
    day_table, _ = load_label_artifacts(SCENE_ROOT)
    for expert in EXPERTS:
        dates = train_dates_for_expert(day_table, expert, merge_val=merge_val)
        out_dir = SCENE_ROOT / f"expert_{expert}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n[V25] train expert={expert}  days={len(dates)}\n{'='*60}", flush=True)
        if len(dates) < 8:
            print(f"WARN: expert {expert} only {len(dates)} train days", flush=True)
        os.environ["V18_VIZ_LABEL"] = f"V25-Scene-{expert}"
        run_v25_early_stop(
            out_dir=out_dir,
            restrict_train_dates=dates,
            skip_test_predict=True,
            skip_plots=True,
        )


def cmd_route() -> None:
    from src.model_v18_conv2d import (
        DEVICE,
        _build_daily_arrays,
        predict_days,
    )
    from src.model_v24_da import (
        _patch_v18_for_v24_direct,
        _restore_v18,
        _snapshot_v18,
        load_sql_feature_matrix,
    )

    day_table, th = load_label_artifacts(SCENE_ROOT)
    test_df = test_dates_with_expert(day_table)

    snap = _snapshot_v18()
    experts: dict = {}
    try:
        import src.model_v18_conv2d as m18
        m18.CONTEXT_BEFORE = V25_CTX_BEFORE
        m18.CONTEXT_AFTER = V25_CTX_AFTER
        m18.H_SLOTS = (V25_CTX_BEFORE + 1 + V25_CTX_AFTER) * 4
        df = load_sql_feature_matrix()
        _patch_v18_for_v24_direct()
        h_slots = m18.H_SLOTS
        drop = float(os.environ.get("V25_DROPOUT", "0.44"))
        for expert in EXPERTS:
            edir = SCENE_ROOT / f"expert_{expert}"
            norm_mean, norm_std, y_mean, y_std = load_v25_norm(edir)
            model = DualHeadResConv2dPriceNet(c_in=m18.C_TOTAL, h_slots=h_slots, dropout=drop)
            model.load_state_dict(torch.load(edir / "seed0.pt", map_location=DEVICE, weights_only=True))
            model.to(DEVICE).eval()
            experts[expert] = (model, norm_mean, norm_std, y_mean, y_std)

        valid_dates, day_lag0, day_lag1, day_lag2, day_targets, _, _ = _build_daily_arrays(df)

        rows = []
        route_counts = {"normal": 0, "low": 0}
        for _, r in test_df.iterrows():
            d = pd.to_datetime(r["date"]).date()
            expert = str(r["expert"])
            if expert not in experts:
                expert = "normal"
            route_counts[expert] = route_counts.get(expert, 0) + 1
            model, norm_mean, norm_std, y_mean, y_std = experts[expert]
            p24, a24, dates = predict_days(
                model, [d],
                day_lag0, day_lag1, day_lag2, day_targets,
                norm_mean, norm_std, y_mean, y_std,
            )
            if not dates:
                continue
            for h in range(24):
                rows.append({
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": float(a24[0, h]),
                    "predicted": float(p24[0, h]),
                    "scene": r["scene"],
                    "expert": expert,
                })

        result = pd.DataFrame(rows).set_index("ts").sort_index()
        out_dir = SCENE_ROOT / "routed"
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "da_result.csv"
        result.to_csv(result_path)

        af = result["actual"].values
        pf = result["predicted"].values
        mae = float(np.mean(np.abs(af - pf)))
        shape = quick_shape_report(af, pf, result.index)
        corr = float(shape.get("profile_corr", float("nan")))

        summary = {
            "test_mae": mae,
            "test_profile_corr": corr,
            "route_days": route_counts,
            "thresholds": th.to_dict(),
            "scene_to_expert": SCENE_TO_EXPERT,
        }
        with (out_dir / "route_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n[V25 scene route] test MAE={mae:.2f}  profile_corr={corr:.4f}")
        print(f"route days: {route_counts}")
        print(f"→ {result_path}")

        subprocess.run(
            [
                sys.executable,
                "run_evaluate_all_models.py",
                "--output-root", str(out_dir.resolve()),
                "--task", "da",
                "--no-baseline",
            ],
            check=False,
        )
    finally:
        _restore_v18(snap)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="V25 场景双专家 pipeline")
    ap.add_argument("cmd", choices=["label", "train", "route", "all"])
    ap.add_argument("--no-merge-val", action="store_true")
    args = ap.parse_args()
    merge_val = not args.no_merge_val

    if args.cmd in ("label", "all"):
        cmd_label(merge_val)
    if args.cmd in ("train", "all"):
        cmd_train(merge_val)
    if args.cmd in ("route", "all"):
        cmd_route()


if __name__ == "__main__":
    main()
