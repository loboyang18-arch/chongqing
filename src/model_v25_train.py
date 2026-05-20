"""
V25 训练循环：早停、验证集最优权重、每 epoch 打印 train/val MAE 与 profile_corr。

环境变量（V25 命名）：
  V25_EPOCHS, V25_BS, V25_LR, V25_WD, V25_DROPOUT
  V25_CTX_BEFORE, V25_CTX_AFTER, V25_DELTA_LAMBDA
  V25_PATIENCE          验证 MAE 无改善则停止（默认 15）
  V25_EARLY_STOP        1=早停+val最优权重；0=跑满 epoch 且用末轮权重（默认 1）
  V25_WARMUP_EPOCHS     默认 10
  V25_MIN_EPOCHS        至少训练轮数再允许早停（默认 20）
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from price_forecast_eval import quick_shape_report

from .model_v18_conv2d import (
    DEVICE,
    HourlyConv2dDataset,
    TRAIN_END,
    VAL_END,
    TEST_START,
    TEST_END,
    V18_DELTA_TARGET,
    _build_daily_arrays,
    _plot_train_last_week,
    _seed,
    _v18_result_csv_path,
    _v18_viz_label,
    compute_norm,
    predict_days,
)
from .model_v24_da import (
    TARGET_COL,
    V24_PCA_COMPONENTS,
    V24_USE_WEATHER,
    _patch_v18_for_v24_direct,
    _patch_v18_for_v24_pca,
    _restore_v18,
    _snapshot_v18,
    load_sql_feature_matrix,
    load_sql_feature_matrix_pca,
)
from .model_v25_resconv import (
    DualHeadResConv2dPriceNet,
    ResConv2dPriceNet,
    V25_DUAL,
    default_v25_viz_label,
)
from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

V25_DIR = OUTPUT_DIR / os.environ.get("V25_OUT_DIR", "v25_resconv").strip()

V25_EPOCHS = int(os.environ.get("V25_EPOCHS", "100"))
V25_BS = int(os.environ.get("V25_BS", "64"))
V25_LR = float(os.environ.get("V25_LR", "1e-3"))
V25_WD = float(os.environ.get("V25_WD", "1e-4"))
V25_DROPOUT = float(os.environ.get("V25_DROPOUT", "0.15"))


def _v25_dropout() -> float:
    """训练时读取 dropout（避免 import 早于入口脚本 setenv）。"""
    return float(os.environ.get("V25_DROPOUT", str(V25_DROPOUT)))
V25_CTX_BEFORE = int(os.environ.get("V25_CTX_BEFORE", "5"))
V25_CTX_AFTER = int(os.environ.get("V25_CTX_AFTER", "0"))
V25_DELTA_LAMBDA = float(os.environ.get("V25_DELTA_LAMBDA", "0.2"))
V25_PATIENCE = int(os.environ.get("V25_PATIENCE", "15"))
V25_WARMUP_EPOCHS = int(os.environ.get("V25_WARMUP_EPOCHS", "10"))
V25_MIN_EPOCHS = int(os.environ.get("V25_MIN_EPOCHS", "20"))
SLOTS_PER_HOUR = 4


def save_v25_norm(
    out_dir: Path,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    y_mean: float,
    y_std: float,
) -> None:
    np.savez_compressed(
        out_dir / "v25_norm.npz",
        norm_mean=norm_mean,
        norm_std=norm_std,
        y_mean=np.float32(y_mean),
        y_std=np.float32(y_std),
    )


def load_v25_norm(out_dir: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
    z = np.load(out_dir / "v25_norm.npz")
    return (
        z["norm_mean"],
        z["norm_std"],
        float(z["y_mean"]),
        float(z["y_std"]),
    )


def _v25_early_stop_enabled() -> bool:
    return os.environ.get("V25_EARLY_STOP", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _eval_v25_metrics(
    model: nn.Module,
    dataset: HourlyConv2dDataset,
    y_mean: float,
    y_std: float,
    batch_size: int = 512,
) -> Tuple[float, float]:
    """返回 (MAE, profile_corr)。"""
    if len(dataset) == 0:
        return float("nan"), float("nan")

    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=False,
    )
    model.eval()
    _dual = getattr(model, "_dual_head", False)
    ps, ts = [], []
    with torch.no_grad():
        for grid, tgt, _dtgt in loader:
            out = model(grid.to(DEVICE))
            pred = out[0] if _dual else out
            ps.append(pred.cpu().numpy())
            ts.append(tgt.numpy())

    p = np.concatenate(ps) * y_std + y_mean
    t = np.concatenate(ts) * y_std + y_mean
    mae = float(np.mean(np.abs(p - t)))

    index = pd.DatetimeIndex(
        [pd.Timestamp(d) + pd.Timedelta(hours=int(h))
         for d, h in dataset.meta[: len(p)]]
    )
    shape = quick_shape_report(t, p, index)
    corr = float(shape.get("profile_corr", float("nan")))
    return mae, corr


def train_v25_model(
    train_days: List,
    val_days: List,
    day_lag0: Dict,
    day_lag1: Dict,
    day_lag2: Dict,
    day_targets: Dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    y_mean: float,
    y_std: float,
    out_dir: Path,
    model_cls: Type[nn.Module],
    day_delta_targets: Optional[Dict] = None,
    epochs: Optional[int] = None,
    hour_start: int = 0,
    hour_end: int = 24,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """V25 训练：每 epoch 打印指标；val MAE 最优存 seed0.pt；早停。"""
    epochs = epochs or V25_EPOCHS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)

    h_slots = (V25_CTX_BEFORE + 1 + V25_CTX_AFTER) * SLOTS_PER_HOUR

    delta_y_mean, delta_y_std = 0.0, 1.0
    if day_delta_targets:
        dvals = []
        for d in train_days:
            if d not in day_delta_targets:
                continue
            dvals.append(day_delta_targets[d][hour_start:hour_end])
        if dvals:
            dall = np.concatenate(dvals)
            delta_y_mean = float(np.mean(dall))
            delta_y_std = max(float(np.std(dall)), 1e-6)

    ds_kw = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        ctx_before=V25_CTX_BEFORE, ctx_after=V25_CTX_AFTER,
        train_oversample=1,
        day_delta_targets=day_delta_targets,
        delta_y_mean=delta_y_mean, delta_y_std=delta_y_std,
        hour_start=hour_start,
        hour_end=hour_end,
    )
    train_ds = HourlyConv2dDataset(sample_dates=train_days, **ds_kw)
    val_ds = (
        HourlyConv2dDataset(sample_dates=val_days, **ds_kw) if val_days else None
    )

    train_loader = DataLoader(train_ds, V25_BS, shuffle=True, drop_last=True)
    val_loader = (
        DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)
        if val_ds and len(val_ds) > 0 else None
    )

    import src.model_v18_conv2d as m18
    dropout = _v25_dropout()
    model = model_cls(c_in=m18.C_TOTAL, h_slots=h_slots, dropout=dropout).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[V25] {model_cls.__name__}  params={n_params:,}  "
        f"C_in={m18.C_TOTAL}  h_slots={h_slots}  dropout={dropout}",
        flush=True,
    )
    n_h = hour_end - hour_start
    print(
        f"[V25] train={len(train_ds)} samples ({len(train_days)}d×{n_h}h)  "
        f"val={len(val_ds) if val_ds else 0} samples ({len(val_days)}d)  "
        f"hours=[{hour_start},{hour_end})",
        flush=True,
    )
    early_stop = _v25_early_stop_enabled()
    print(
        f"[V25] epochs={epochs}  early_stop={'on' if early_stop else 'off'}  "
        f"weights={'val_best' if early_stop else 'last_epoch'}  "
        f"patience={V25_PATIENCE}  min_epochs={V25_MIN_EPOCHS}  "
        f"lr={V25_LR}  wd={V25_WD}  λ_delta={V25_DELTA_LAMBDA}",
        flush=True,
    )
    print(
        f"{'ep':>4}  {'train_mae':>10}  {'train_corr':>11}  "
        f"{'val_mae':>10}  {'val_corr':>11}  {'lr':>9}  note",
        flush=True,
    )
    print("-" * 72, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=V25_LR, weight_decay=V25_WD)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=V25_WARMUP_EPOCHS,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - V25_WARMUP_EPOCHS, 1), eta_min=1e-6,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched],
        milestones=[V25_WARMUP_EPOCHS],
    )

    _is_dual = getattr(model, "_dual_head", False)
    best_ckpt = out_dir / "seed0_best.pt"
    best_val_mae = float("inf")
    best_epoch = -1
    patience_left = V25_PATIENCE
    history: List[Dict[str, Any]] = []
    stopped_early = False

    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for grid, tgt, dtgt in train_loader:
            grid, tgt, dtgt = grid.to(DEVICE), tgt.to(DEVICE), dtgt.to(DEVICE)
            opt.zero_grad()
            out = model(grid)
            if _is_dual:
                price_pred, delta_pred = out
                loss = (
                    F.l1_loss(price_pred, tgt)
                    + V25_DELTA_LAMBDA * F.l1_loss(delta_pred, dtgt)
                )
            else:
                loss = F.l1_loss(out, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        train_mae, train_corr = _eval_v25_metrics(model, train_ds, y_mean, y_std, V25_BS)
        if val_loader is not None and len(val_ds) > 0:
            val_mae, val_corr = _eval_v25_metrics(model, val_ds, y_mean, y_std)
        else:
            val_mae, val_corr = float("nan"), float("nan")

        lr_now = opt.param_groups[0]["lr"]
        note = ""
        if val_loader is not None and val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = ep
            if early_stop:
                torch.save(model.state_dict(), best_ckpt)
            patience_left = V25_PATIENCE
            note = "★ best"
        elif early_stop and val_loader is not None and ep + 1 >= V25_MIN_EPOCHS:
            patience_left -= 1
            if patience_left <= 0:
                note = "early_stop"
                stopped_early = True

        row = {
            "epoch": ep + 1,
            "train_mae": train_mae,
            "train_corr": train_corr,
            "val_mae": val_mae,
            "val_corr": val_corr,
            "lr": lr_now,
            "loss": ep_loss / max(nb, 1),
            "note": note.strip(),
        }
        history.append(row)

        print(
            f"{ep + 1:4d}  {train_mae:10.2f}  {train_corr:11.4f}  "
            f"{val_mae:10.2f}  {val_corr:11.4f}  {lr_now:9.2e}  {note}",
            flush=True,
        )

        if early_stop and stopped_early:
            print(
                f"\n[V25] 早停于 epoch {ep + 1}，验证集最优 epoch {best_epoch + 1}  "
                f"(val_mae={best_val_mae:.2f})",
                flush=True,
            )
            break

    if early_stop and val_loader is not None and best_epoch >= 0 and best_ckpt.is_file():
        model.load_state_dict(
            torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
        )
        print(
            f"[V25] 已加载验证集最优权重 (epoch {best_epoch + 1}, val_mae={best_val_mae:.2f})",
            flush=True,
        )
    else:
        print(
            f"[V25] 使用最后一轮权重 (epoch {epochs}, early_stop={'on' if early_stop else 'off'})",
            flush=True,
        )

    seed_path = out_dir / "seed0.pt"
    torch.save(model.state_dict(), seed_path)

    hist_path = out_dir / "v25_train_history.json"
    summary = {
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "best_val_mae": best_val_mae if best_epoch >= 0 else None,
        "stopped_early": stopped_early,
        "weight_policy": "val_best" if early_stop else "last_epoch",
        "final_epoch": history[-1]["epoch"] if history else 0,
        "config": {
            "V25_EPOCHS": epochs,
            "V25_EARLY_STOP": early_stop,
            "V25_PATIENCE": V25_PATIENCE,
            "V25_MIN_EPOCHS": V25_MIN_EPOCHS,
            "V25_DROPOUT": V25_DROPOUT,
            "V25_LR": V25_LR,
            "V25_WD": V25_WD,
            "V25_CTX_BEFORE": V25_CTX_BEFORE,
            "V25_CTX_AFTER": V25_CTX_AFTER,
            "V25_DELTA_LAMBDA": V25_DELTA_LAMBDA,
        },
        "history": history,
    }
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", hist_path.name)

    return model, summary


def run_v25_early_stop(
    out_dir: Optional[Path] = None,
    model_cls: Optional[Type[nn.Module]] = None,
    *,
    restrict_train_dates: Optional[Set] = None,
    skip_test_predict: bool = False,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """V25 全流程：sql 特征 → 早停训练 → 测试预测与作图。"""
    out_dir = Path(out_dir or V25_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_dual = V25_DUAL
    cls = model_cls or (DualHeadResConv2dPriceNet if use_dual else ResConv2dPriceNet)

    viz = os.environ.get("V25_VIZ_LABEL", "").strip() or default_v25_viz_label(use_dual)
    os.environ["V18_VIZ_LABEL"] = viz  # 复用作图标签

    print("\n" + "=" * 72, flush=True)
    print(f"[V25] 早停训练  model={cls.__name__}  target={TARGET_COL}", flush=True)
    print(f"[V25] ctx={V25_CTX_BEFORE}+{V25_CTX_AFTER}  out={out_dir}", flush=True)
    print("=" * 72, flush=True)

    snap = _snapshot_v18()
    try:
        # 同步上下文到 v18 模块（预测/构图用）
        import src.model_v18_conv2d as m18
        m18.CONTEXT_BEFORE = V25_CTX_BEFORE
        m18.CONTEXT_AFTER = V25_CTX_AFTER
        m18.H_SLOTS = (V25_CTX_BEFORE + 1 + V25_CTX_AFTER) * SLOTS_PER_HOUR

        if V24_PCA_COMPONENTS > 0:
            df, pca_names, _ = load_sql_feature_matrix_pca(V24_PCA_COMPONENTS)
            _patch_v18_for_v24_pca(pca_names)
        else:
            df = load_sql_feature_matrix()
            _patch_v18_for_v24_direct()

        valid_dates, day_lag0, day_lag1, day_lag2, day_targets, day_anchors, day_delta_targets = (
            _build_daily_arrays(df)
        )

        tr_last = TRAIN_END.date()
        val_last = VAL_END.date()
        ts_first = TEST_START.date()
        ts_last = TEST_END.date()
        merge_val = os.environ.get("V25_MERGE_VAL", os.environ.get("V18_MERGE_VAL", "0"))
        merge_val = str(merge_val).strip().lower() in ("1", "true", "yes", "on")
        if merge_val:
            train_days = [d for d in valid_dates if d <= val_last]
            val_days = []
            print("[V25] V25_MERGE_VAL=1 → 验证集并入训练，无独立 val", flush=True)
        else:
            train_days = [d for d in valid_dates if d <= tr_last]
            val_days = [d for d in valid_dates if tr_last < d <= val_last]
        test_days = [d for d in valid_dates if ts_first <= d <= ts_last]

        if restrict_train_dates is not None:
            allow = set(restrict_train_dates)
            train_days = [d for d in train_days if d in allow]
            print(
                f"[V25] restrict_train_dates → {len(train_days)} days",
                flush=True,
            )
            if len(train_days) < 5:
                raise ValueError(f"Too few train days after scene filter: {len(train_days)}")

        print(
            f"[V25] 切分: train {len(train_days)}d ({train_days[0]}~{train_days[-1]}) | "
            f"val {len(val_days)}d ({val_days[0] if val_days else '—'}~{val_days[-1] if val_days else '—'}) | "
            f"test {len(test_days)}d ({test_days[0]}~{test_days[-1]})",
            flush=True,
        )
        print(
            f"[V25] splits: TRAIN_END={TRAIN_END.date()} VAL_END={VAL_END.date()} "
            f"TEST={TEST_START.date()}~{TEST_END.date()}",
            flush=True,
        )

        norm_mean, norm_std = compute_norm(day_lag0, day_lag1, day_lag2, train_days)
        tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
        y_mean = float(tgt_stack.mean())
        y_std = float(tgt_stack.std()) + 1e-8
        save_v25_norm(out_dir, norm_mean, norm_std, y_mean, y_std)

        model, train_summary = train_v25_model(
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
        )

        if not skip_plots:
            train_last7 = train_days[-7:] if len(train_days) >= 7 else train_days
            p24_tr, a24_tr, dates_tr = predict_days(
                model, train_last7,
                day_lag0, day_lag1, day_lag2, day_targets,
                norm_mean, norm_std, y_mean, y_std,
            )
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            if len(dates_tr) > 0:
                _plot_train_last_week(p24_tr, a24_tr, dates_tr, plots_dir)

        test_mae, test_corr = float("nan"), float("nan")
        result_path = out_dir / "da_result.csv"
        if not skip_test_predict:
            p24, a24, dates = predict_days(
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
            result_path = _v18_result_csv_path(out_dir)
            result.to_csv(result_path)

            if not skip_plots:
                from price_forecast_eval.viz import run_standard_visualization
                run_standard_visualization(
                    result_path,
                    out_dir=out_dir / "plots",
                    label=viz,
                    actual_col="actual",
                    pred_col="predicted",
                    mode="appendix",
                    weekly=True,
                )

            af = result["actual"].values
            pf = result["predicted"].values
            test_mae = float(np.mean(np.abs(af - pf)))
            shape = quick_shape_report(af, pf, result.index)
            test_corr = float(shape.get("profile_corr", float("nan")))

            print("\n" + "=" * 72, flush=True)
            print(f"[V25] 测试集  MAE={test_mae:.2f}  profile_corr={test_corr:.4f}", flush=True)
            print(f"[V25] 结果 → {result_path.name}", flush=True)
            print("=" * 72 + "\n", flush=True)

        meta = {
            "model": cls.__name__,
            "target": TARGET_COL,
            "viz_label": viz,
            "train_summary": {
                "best_epoch": train_summary.get("best_epoch"),
                "best_val_mae": train_summary.get("best_val_mae"),
                "stopped_early": train_summary.get("stopped_early"),
            },
            "test_mae": test_mae,
            "test_profile_corr": test_corr,
            "V25_CTX_BEFORE": V25_CTX_BEFORE,
            "V25_CTX_AFTER": V25_CTX_AFTER,
            "V25_DELTA_LAMBDA": V25_DELTA_LAMBDA,
            "V25_DROPOUT": V25_DROPOUT,
            "V25_MERGE_VAL": merge_val,
            "weight_policy": train_summary.get("weight_policy"),
        }
        with (out_dir / "v25_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return meta
    finally:
        _restore_v18(snap)
