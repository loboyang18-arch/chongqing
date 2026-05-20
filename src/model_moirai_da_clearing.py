"""
重庆日前市场出清电价 — Moirai-1.1-R 零样本分位数预测

借鉴 neimeng_prj/src/model_v12_moirai.py：
  - univariate：仅历史 da_clearing_price
  - covariate-aware：D 日已知的负荷/新能源/申报均价等作 future covariate

数据：
  - 1h：output/dws_hourly_features.csv
  - 15min：build_feature_matrix()（可选缓存 output/dws_15min_moirai_<target>.parquet）
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 本地已有 HF 缓存时优先离线，避免镜像超时
if Path.home().joinpath(".cache/huggingface/hub/models--Salesforce--moirai-1.1-R-small").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .config import OUTPUT_DIR
from .experiment.splits import (
    HOURLY_TEST_END,
    HOURLY_TEST_START,
    TEST_END,
    TEST_START,
)

logger = logging.getLogger(__name__)

TARGET_COL = os.environ.get("CQ_MOIRAI_TARGET", "da_clearing_price")
HOURLY_DWS_DEFAULT = OUTPUT_DIR / "dws_hourly_features.csv"


def _cache_15min_path() -> Path:
    safe = TARGET_COL.replace("/", "_")
    return OUTPUT_DIR / f"dws_15min_moirai_{safe}.parquet"

DEFAULT_COVARIATES = (
    "load_forecast",
    "renewable_fcst",
    "avg_bid_price",
)
COVARIATE_FALLBACKS = {
    "renewable_fcst": ("renewable_fcst_wind_pm", "renewable_fcst_total_pm"),
    "avg_bid_price": ("reliability_da_price", "da_reliability_clearing_price"),
}
DEFAULT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)

FREQ_PRESETS = {
    "1h": {"gluonts_freq": "h", "context_steps": 720, "pred_steps": 24, "sph": 1},
    "15min": {"gluonts_freq": "15min", "context_steps": 720, "pred_steps": 96, "sph": 4},
}


def _resolve_covariates(available: set[str], requested: Tuple[str, ...]) -> List[str]:
    """在宽表中解析协变量列，缺失时尝试 fallback。"""
    resolved: List[str] = []
    for col in requested:
        if col in available:
            resolved.append(col)
            continue
        for alt in COVARIATE_FALLBACKS.get(col, ()):
            if alt in available:
                logger.info("协变量 %s 缺失，使用 %s", col, alt)
                resolved.append(alt)
                break
        else:
            logger.warning("协变量 %s 及 fallback 均缺失，跳过", col)
    return resolved


def _mean4_to_hourly(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        if a.shape[0] != 96:
            raise ValueError(f"mean4 需要 96 slots，got {a.shape}")
        return a.reshape(24, 4).mean(axis=1).astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 96:
        return a.reshape(a.shape[0], 24, 4).mean(axis=2).astype(np.float32)
    if a.ndim == 3 and a.shape[1] == 96:
        return a.reshape(a.shape[0], 24, 4, a.shape[2]).mean(axis=2).astype(np.float32)
    raise ValueError(f"mean4 不支持的 shape: {a.shape}")


def _effective_test_bounds(
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
    freq: str,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """按标签可用性截断测试窗（标签缺失时收窄至最后有效时刻）。"""
    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)
    labeled = df["target"].dropna()
    if labeled.empty:
        raise RuntimeError(f"target={TARGET_COL} 无有效标签")
    last_label = labeled.index.max()
    eff_end = min(ts_end, last_label)
    if eff_end < ts_start:
        raise RuntimeError(
            f"有效测试窗为空: test_start={ts_start}, last_label={last_label}"
        )
    if eff_end < ts_end:
        logger.warning(
            "标签 %s 止于 %s，测试窗从 %s 截断为 %s（原计划 %s）",
            TARGET_COL, last_label, ts_end, eff_end, ts_end,
        )
    return ts_start, eff_end


def _load_hourly_frame(
    covariates: List[str],
    dws_csv: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    path = Path(dws_csv) if dws_csv else Path(
        os.environ.get("CQ_DWS_CSV", str(HOURLY_DWS_DEFAULT))
    )
    if not path.is_file():
        raise FileNotFoundError(f"缺少小时 DWS: {path}")
    raw = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    if TARGET_COL not in raw.columns:
        raise KeyError(f"{path} 中无目标列 {TARGET_COL}")

    cov_resolved = _resolve_covariates(set(raw.columns), tuple(covariates))
    cols = [TARGET_COL, *cov_resolved]
    df = raw[cols].astype(float).rename(columns={TARGET_COL: "target"})
    df = df.dropna(subset=["target"])
    logger.info(
        "1h: %d rows from %s | covs=%s | target range %s ~ %s",
        len(df), path.name, cov_resolved,
        df.index.min(), df.index.max(),
    )
    return df, cov_resolved


def _build_15min_frame(covariates: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """从 build_feature_matrix 构建 15min 宽表，可选 parquet 缓存。"""
    cache_path = _cache_15min_path()
    use_cache = os.environ.get("CQ_MOIRAI_NO_CACHE", "").strip() not in ("1", "true", "yes")
    if use_cache and cache_path.is_file():
        raw = pd.read_parquet(cache_path)
        logger.info("15min: 从缓存加载 %s (%d rows)", cache_path.name, len(raw))
    else:
        from .model_v16_nhits import build_feature_matrix

        logger.info("15min: 构建 feature matrix …")
        raw = build_feature_matrix()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw.to_parquet(cache_path)
            logger.info("15min: 已缓存 %s", cache_path)
        except Exception as exc:
            logger.warning("15min parquet 缓存失败（可安装 pyarrow）: %s", exc)

    if TARGET_COL not in raw.columns:
        raise KeyError(f"15min 矩阵中无目标列 {TARGET_COL}")

    cov_resolved = _resolve_covariates(set(raw.columns), tuple(covariates))
    cols = [TARGET_COL, *cov_resolved]
    df = raw[cols].astype(float).rename(columns={TARGET_COL: "target"})
    df = df.dropna(subset=["target"])
    logger.info(
        "15min: %d rows | covs=%s | target range %s ~ %s",
        len(df), cov_resolved, df.index.min(), df.index.max(),
    )
    return df, cov_resolved


def _load_series_with_covariates(
    covariates: List[str],
    freq: str = "1h",
    dws_csv: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if freq not in FREQ_PRESETS:
        raise ValueError(f"未知 freq={freq}，可选 {list(FREQ_PRESETS)}")
    if freq == "1h":
        return _load_hourly_frame(covariates, dws_csv=dws_csv)
    return _build_15min_frame(covariates)


def _build_test_inputs(
    df: pd.DataFrame,
    test_days: List,
    context_steps: int,
    pred_steps: int,
    covariates: List[str],
    freq: str = "1h",
):
    step_min = 60 if freq == "1h" else 15
    samples = []
    for d in test_days:
        d0 = pd.Timestamp(d)
        if freq == "1h":
            ctx_start = d0 - pd.Timedelta(hours=context_steps)
            ctx_end = d0 - pd.Timedelta(hours=1)
            fut_start = d0
            fut_end = d0 + pd.Timedelta(hours=23)
        else:
            ctx_start = d0 - pd.Timedelta(minutes=step_min * context_steps)
            ctx_end = d0 - pd.Timedelta(minutes=step_min)
            fut_start = d0
            fut_end = d0 + pd.Timedelta(hours=23, minutes=45)

        history = df.loc[ctx_start:ctx_end]
        future = df.loc[fut_start:fut_end]
        min_ctx = context_steps - (24 if freq == "1h" else 96)
        if len(history) < min_ctx or len(future) != pred_steps:
            continue
        if np.isnan(future["target"].values).any():
            continue

        target_hist = history["target"].values.astype(np.float32)
        actual_fut = future["target"].values.astype(np.float32)
        cov_full = None
        if covariates:
            cov_full = pd.concat(
                [history[covariates], future[covariates]], axis=0,
            ).values.astype(np.float32).T

        samples.append({
            "date": d,
            "start": history.index[0],
            "target_hist": target_hist,
            "cov_full": cov_full,
            "actual_fut": actual_fut,
        })
    logger.info(
        "有效 test 天数: %d / %d (freq=%s context_steps=%d pred_steps=%d)",
        len(samples), len(test_days), freq, context_steps, pred_steps,
    )
    return samples


def _make_gluonts_dataset(samples: List[dict], use_covariates: bool, gluonts_freq: str):
    from gluonts.dataset.common import ListDataset

    items = []
    for s in samples:
        item = {
            "target": s["target_hist"],
            "start": pd.Period(s["start"], freq=gluonts_freq),
        }
        if use_covariates:
            item["feat_dynamic_real"] = s["cov_full"]
        items.append(item)
    return ListDataset(items, freq=gluonts_freq, one_dim_target=True)


def run(
    model_id: str = "Salesforce/moirai-1.1-R-small",
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    use_covariates: bool = False,
    covariates: tuple = DEFAULT_COVARIATES,
    freq: str = "1h",
    context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    patch_size: int = 32,
    num_samples: int = 100,
    quantile_levels: tuple = DEFAULT_QUANTILES,
    out_dir: Optional[Path] = None,
    batch_size: int = 8,
) -> dict:
    preset = FREQ_PRESETS[freq]
    ctx_steps = context_length if context_length is not None else preset["context_steps"]
    pred_steps = prediction_length if prediction_length is not None else preset["pred_steps"]
    gluonts_freq = preset["gluonts_freq"]

    if test_start is None:
        test_start = HOURLY_TEST_START if freq == "1h" else str(TEST_START)
    if test_end is None:
        test_end = HOURLY_TEST_END if freq == "1h" else str(TEST_END)

    base_covs = list(covariates) if use_covariates else []
    use_cov = use_covariates
    if out_dir is None:
        suffix = "cov" if use_cov else "uni"
        sub = f"moirai_da_clearing_{freq}_{suffix}"
        out_dir = OUTPUT_DIR / sub
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Moirai da_clearing: %s", model_id)
    logger.info("  target=%s | test=%s ~ %s | freq=%s", TARGET_COL, test_start, test_end, freq)
    logger.info("  context=%d | pred=%d | covariates=%s",
                ctx_steps, pred_steps, base_covs if use_cov else "(univariate)")
    logger.info("=" * 60)

    dws_csv = os.environ.get("CQ_DWS_CSV")
    df, cov_list = _load_series_with_covariates(base_covs, freq=freq, dws_csv=dws_csv)
    eff_start, eff_end = _effective_test_bounds(df, test_start, test_end, freq)

    test_dt = eff_start.date()
    test_end_dt = eff_end.date()
    all_days = sorted(set(df.index.normalize().date.tolist()))
    test_days = [d for d in all_days if test_dt <= d <= test_end_dt]

    samples = _build_test_inputs(
        df, test_days,
        context_steps=ctx_steps,
        pred_steps=pred_steps,
        covariates=cov_list if use_cov else [],
        freq=freq,
    )
    if not samples:
        raise RuntimeError("无有效 test 样本")
    actual_fut = np.stack([s["actual_fut"] for s in samples], axis=0)

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    logger.info("加载 %s …", model_id)
    module = MoiraiModule.from_pretrained(model_id)
    n_params = sum(p.numel() for p in module.parameters())
    logger.info("  params: %d", n_params)

    feat_dim = len(cov_list) if use_cov else 0
    model = MoiraiForecast(
        module=module,
        prediction_length=pred_steps,
        context_length=ctx_steps,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=feat_dim,
        past_feat_dynamic_real_dim=0,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    ds = _make_gluonts_dataset(samples, use_covariates=use_cov, gluonts_freq=gluonts_freq)
    predictor = model.create_predictor(batch_size=batch_size)

    logger.info("开始推理 %d 天 …", len(samples))
    forecasts = list(predictor.predict(ds))
    logger.info("推理完成: %d forecasts", len(forecasts))

    quantiles_native = np.zeros(
        (len(samples), pred_steps, len(quantile_levels)), dtype=np.float32,
    )
    for i, fc in enumerate(forecasts):
        for q_i, q in enumerate(quantile_levels):
            quantiles_native[i, :, q_i] = fc.quantile(q)
    quantiles_native = np.sort(quantiles_native, axis=-1)

    p50_idx = len(quantile_levels) // 2
    if freq == "15min":
        actual_arr = _mean4_to_hourly(actual_fut)
        quantiles_arr = _mean4_to_hourly(quantiles_native)
        eval_note = "hourly_mean4_from_96x15min"
        native_steps = pred_steps
    else:
        actual_arr = actual_fut
        quantiles_arr = quantiles_native
        eval_note = "native_hourly"
        native_steps = pred_steps

    rows_long, rows_p50, rows_native = [], [], []
    for i, s in enumerate(samples):
        d = s["date"]
        for h in range(24):
            row = {
                "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                "actual": float(actual_arr[i, h]),
            }
            for q_i, q in enumerate(quantile_levels):
                row[f"p{int(q * 100):02d}"] = float(quantiles_arr[i, h, q_i])
            rows_long.append(row)
            rows_p50.append({
                "ts": row["ts"],
                "actual": row["actual"],
                "pred": float(quantiles_arr[i, h, p50_idx]),
            })
        if freq == "15min":
            for slot in range(native_steps):
                ts = pd.Timestamp(d) + pd.Timedelta(minutes=15 * slot)
                row_n = {"ts": ts, "actual": float(actual_fut[i, slot])}
                for q_i, q in enumerate(quantile_levels):
                    row_n[f"p{int(q * 100):02d}"] = float(
                        quantiles_native[i, slot, q_i],
                    )
                rows_native.append(row_n)

    pd.DataFrame(rows_long).sort_values("ts").reset_index(drop=True).to_csv(
        out_dir / "test_predictions_quantile.csv", index=False,
    )
    pd.DataFrame(rows_p50).set_index("ts").sort_index().to_csv(
        out_dir / "test_predictions_hourly.csv",
    )
    da_result = pd.DataFrame(
        {"actual": [r["actual"] for r in rows_p50],
         "predicted": [r["pred"] for r in rows_p50]},
        index=pd.DatetimeIndex([r["ts"] for r in rows_p50]),
    )
    da_result.index.name = "ts"
    da_result.reset_index().to_csv(out_dir / "da_result.csv", index=False)
    if freq == "15min":
        pd.DataFrame(rows_native).sort_values("ts").reset_index(drop=True).to_csv(
            out_dir / "test_predictions_15min.csv", index=False,
        )
    np.save(out_dir / "quantile_levels.npy", np.array(quantile_levels))

    flat_a = actual_arr.reshape(-1)
    flat_q = quantiles_arr.reshape(-1, len(quantile_levels))
    mask = ~np.isnan(flat_a)
    fa, fq = flat_a[mask], flat_q[mask]
    p50 = fq[:, p50_idx]
    mae = float(np.mean(np.abs(p50 - fa)))
    rmse = float(np.sqrt(np.mean((p50 - fa) ** 2)))
    bias = float(np.mean(p50 - fa))
    cov80 = float(np.mean((fa >= fq[:, 0]) & (fa <= fq[:, -1])))
    width = float(np.mean(fq[:, -1] - fq[:, 0]))

    metrics = {
        "model_id": model_id,
        "target": TARGET_COL,
        "use_covariates": use_cov,
        "freq": freq,
        "eval_granularity": eval_note,
        "covariates": cov_list if use_cov else [],
        "context_steps": ctx_steps,
        "prediction_steps_native": pred_steps,
        "patch_size": patch_size,
        "num_samples": num_samples,
        "n_test_days": len(samples),
        "test_start": str(eff_start),
        "test_end": str(eff_end),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "coverage_80": round(cov80, 3),
        "interval_width": round(width, 1),
    }
    pd.Series(metrics).to_csv(out_dir / "metrics.csv")

    logger.info("=" * 60)
    logger.info("Moirai da_clearing (%s) [%s]", "cov" if use_cov else "uni", eval_note)
    logger.info("  MAE (P50):       %.2f", mae)
    logger.info("  RMSE (P50):      %.2f", rmse)
    logger.info("  Bias (P50):      %.2f", bias)
    logger.info("  Coverage 80%%:    %.3f", cov80)
    logger.info("  Interval Width:  %.1f", width)
    logger.info("  保存: %s", out_dir)
    logger.info("=" * 60)
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="重庆日前出清价 Moirai 零样本预测")
    p.add_argument("--model", default="Salesforce/moirai-1.1-R-small")
    p.add_argument("--freq", default="1h", choices=list(FREQ_PRESETS))
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--prediction-length", type=int, default=None)
    p.add_argument("--use-covariates", action="store_true")
    p.add_argument("--covariates", default=None,
                   help="逗号分隔协变量列名；需配合 --use-covariates")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    cov_tuple = DEFAULT_COVARIATES
    if args.covariates:
        cov_tuple = tuple(c.strip() for c in args.covariates.split(",") if c.strip())
    out = Path(args.out_dir) if args.out_dir else None
    run(
        model_id=args.model,
        test_start=args.test_start,
        test_end=args.test_end,
        use_covariates=args.use_covariates,
        covariates=cov_tuple,
        freq=args.freq,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        out_dir=out,
        batch_size=args.batch_size,
    )
