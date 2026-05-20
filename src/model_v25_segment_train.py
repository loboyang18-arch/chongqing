"""三时段 V25：逐小时单点，训练集按小时切分，预测拼接。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from price_forecast_eval import quick_shape_report

from .config import OUTPUT_DIR
from .experiment.splits import TEST_END, TEST_START, TRAIN_END, VAL_END
from .model_v18_conv2d import (
    DEVICE,
    V18_DELTA_TARGET,
    _build_daily_arrays,
    compute_norm,
    predict_days,
)
from .model_v24_da import (
    V24_PCA_COMPONENTS,
    _patch_v18_for_v24_direct,
    _patch_v18_for_v24_pca,
    _restore_v18,
    _snapshot_v18,
    load_sql_feature_matrix,
    load_sql_feature_matrix_pca,
)
from .model_v25_resconv import DualHeadResConv2dPriceNet, ResConv2dPriceNet, default_v25_viz_label
from .model_v25_train import (
    V25_DUAL,
    load_v25_norm,
    save_v25_norm,
    train_v25_model,
    _v25_dropout,
)
from .model_v25_segment import SEGMENTS_888, SegmentSpec, stitch_hourly_predictions

V25_SEG_ROOT = OUTPUT_DIR / os.environ.get("V25_SEG_ROOT", "v25_segment_8x3").strip()


def run_v25_segment_train(
    segment: SegmentSpec,
    out_dir: Path,
    *,
    shared_norm_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """训练一个时段模型（结构与 V25 单模相同，仅 hour 过滤）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_dual = V25_DUAL
    cls = DualHeadResConv2dPriceNet if use_dual else ResConv2dPriceNet
    viz = f"{default_v25_viz_label(use_dual)}-{segment.name}"
    os.environ["V18_VIZ_LABEL"] = viz

    print("\n" + "=" * 72, flush=True)
    print(
        f"[V25-seg] 逐小时单点  {segment.name}  hours [{segment.hour_start},{segment.hour_end})  "
        f"out={out_dir}",
        flush=True,
    )
    print("=" * 72, flush=True)

    snap = _snapshot_v18()
    try:
        import src.model_v18_conv2d as m18
        from .model_v25_train import V25_CTX_AFTER, V25_CTX_BEFORE

        m18.CONTEXT_BEFORE = V25_CTX_BEFORE
        m18.CONTEXT_AFTER = V25_CTX_AFTER
        m18.H_SLOTS = (V25_CTX_BEFORE + 1 + V25_CTX_AFTER) * 4

        if V24_PCA_COMPONENTS > 0:
            df, pca_names, _ = load_sql_feature_matrix_pca(V24_PCA_COMPONENTS)
            _patch_v18_for_v24_pca(pca_names)
        else:
            df = load_sql_feature_matrix()
            _patch_v18_for_v24_direct()

        valid_dates, day_lag0, day_lag1, day_lag2, day_targets, _, day_delta_targets = (
            _build_daily_arrays(df)
        )

        merge_val = os.environ.get("V25_MERGE_VAL", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if merge_val:
            train_days = [d for d in valid_dates if d <= VAL_END.date()]
            val_days = []
        else:
            train_days = [d for d in valid_dates if d <= TRAIN_END.date()]
            val_days = [
                d for d in valid_dates
                if TRAIN_END.date() < d <= VAL_END.date()
            ]

        if shared_norm_dir and (shared_norm_dir / "v25_norm.npz").is_file():
            norm_mean, norm_std, y_mean, y_std = load_v25_norm(shared_norm_dir)
        else:
            norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
            tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
            y_mean = float(tgt_stack.mean())
            y_std = float(tgt_stack.std()) + 1e-8
        save_v25_norm(out_dir, norm_mean, norm_std, y_mean, y_std)

        train_v25_model(
            train_days=train_days,
            val_days=val_days,
            day_lag0=day_lag0,
            day_lag1=day_lag1,
            day_lag2=day_lag2,
            day_targets=day_targets,
            norm_mean=norm_mean,
            norm_std=norm_std,
            y_mean=y_mean,
            y_std=y_std,
            out_dir=out_dir,
            model_cls=cls,
            day_delta_targets=day_delta_targets if not V18_DELTA_TARGET else None,
            hour_start=segment.hour_start,
            hour_end=segment.hour_end,
        )

        meta = {
            "model": cls.__name__,
            "mode": "hourly_single_point",
            "segment": {
                "name": segment.name,
                "hour_start": segment.hour_start,
                "hour_end": segment.hour_end,
            },
            "viz_label": viz,
        }
        with (out_dir / "v25_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta
    finally:
        _restore_v18(snap)


def run_v25_segment_stitch(root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(root or V25_SEG_ROOT)
    snap = _snapshot_v18()
    try:
        import src.model_v18_conv2d as m18
        from .model_v25_train import V25_CTX_AFTER, V25_CTX_BEFORE

        m18.CONTEXT_BEFORE = V25_CTX_BEFORE
        m18.CONTEXT_AFTER = V25_CTX_AFTER
        m18.H_SLOTS = (V25_CTX_BEFORE + 1 + V25_CTX_AFTER) * 4

        df = load_sql_feature_matrix()
        _patch_v18_for_v24_direct()
        valid_dates, day_lag0, day_lag1, day_lag2, day_targets, _, _ = _build_daily_arrays(df)
        test_days = [
            d for d in valid_dates
            if TEST_START.date() <= d <= TEST_END.date()
        ]

        use_dual = V25_DUAL
        cls = DualHeadResConv2dPriceNet if use_dual else ResConv2dPriceNet
        h_slots = m18.H_SLOTS
        drop = _v25_dropout()

        seg_preds = []
        segs = SegmentSpec.all_segments()
        common_dates = None

        for seg in segs:
            edir = root / seg.name
            norm_mean, norm_std, y_mean, y_std = load_v25_norm(edir)
            model = cls(c_in=m18.C_TOTAL, h_slots=h_slots, dropout=drop)
            model.load_state_dict(
                torch.load(edir / "seed0.pt", map_location=DEVICE, weights_only=True)
            )
            model.to(DEVICE).eval()

            p24_part, _, dates = predict_days(
                model,
                test_days,
                day_lag0,
                day_lag1,
                day_lag2,
                day_targets,
                norm_mean,
                norm_std,
                y_mean,
                y_std,
                hour_start=seg.hour_start,
                hour_end=seg.hour_end,
            )
            # 只取该段列（其余为 nan）
            pred_seg = p24_part[:, seg.hour_start : seg.hour_end]
            if common_dates is None:
                common_dates = dates
            elif dates != common_dates:
                raise ValueError(f"日期不一致: {seg.name}")
            seg_preds.append(pred_seg)

        p24 = stitch_hourly_predictions(seg_preds, segs)
        a24 = np.stack([day_targets[d] for d in common_dates], axis=0)

        rows = []
        for i, d in enumerate(common_dates):
            for h in range(24):
                rows.append(
                    {
                        "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                        "actual": float(a24[i, h]),
                        "predicted": float(p24[i, h]),
                    }
                )
        result = pd.DataFrame(rows).set_index("ts").sort_index()
        out_dir = root / "stitched"
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
            "mode": "hourly_single_point",
            "layout": "8+8+8",
            "segments": [
                {"name": s.name, "hour_start": s.hour_start, "hour_end": s.hour_end}
                for s in segs
            ],
        }
        with (out_dir / "stitch_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n[V25-seg stitch] test MAE={mae:.2f}  profile_corr={corr:.4f}")
        print(f"→ {result_path}")
        return summary
    finally:
        _restore_v18(snap)
