"""
官方结算价 naive 基线 — 与 V16d 同一 y 定义与测试时间窗。

- 日前：y = settlement_da_price，ŷ = 昨日同一小时（shift 24）。
- 实时：y = settlement_rt_price
    - lag24：ŷ = 昨日同一小时实时结算
    - from_da：ŷ = 同一时刻 settlement_da_price（用日前结算充当实时预测）

数据：output/dws_hourly_features.csv（小时整点索引）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from price_forecast_eval import (
    compute_shape_metrics,
    quick_shape_report as compute_shape_report,
    to_eval_frame,
)

from .config import OUTPUT_DIR
from .experiment.splits import HOURLY_TEST_END, TEST_START

logger = logging.getLogger(__name__)

HOURLY_PATH = OUTPUT_DIR / "dws_hourly_features.csv"
DEFAULT_OUT_DIR = OUTPUT_DIR / "naive_settlement_baselines"

COL_DA = "settlement_da_price"
COL_RT = "settlement_rt_price"


def _test_end_hourly() -> pd.Timestamp:
    return pd.Timestamp(HOURLY_TEST_END)


def load_hourly_settlements(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    p = path or HOURLY_PATH
    if not p.is_file():
        raise FileNotFoundError(f"缺少小时特征表: {p}")
    df = pd.read_csv(p, parse_dates=["ts"], index_col="ts")
    df = df.sort_index()
    for c in (COL_DA, COL_RT):
        if c not in df.columns:
            raise KeyError(f"{p} 中无列 {c}")
    return df


def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def _scalar_metrics(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, int]:
    m = _finite_mask(y, p)
    if not m.any():
        return float("nan"), float("nan"), 0
    yt, pt = y[m], p[m]
    err = yt - pt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse, int(m.sum())


def evaluate_one(
    actual: pd.Series,
    pred: pd.Series,
    index: pd.DatetimeIndex,
) -> Dict[str, Any]:
    y = actual.values.astype(float)
    pr = pred.values.astype(float)
    m = _finite_mask(y, pr)
    empty_shape = {
        "profile_corr": float("nan"),
        "norm_profile_mae": float("nan"),
        "peak_hour_err": float("nan"),
        "valley_hour_err": float("nan"),
        "amplitude_err": float("nan"),
        "direction_acc": float("nan"),
        "neg_corr_day_ratio": float("nan"),
    }
    if not m.any():
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "n_samples": 0,
            **empty_shape,
        }
    mae, rmse, n = _scalar_metrics(y, pr)
    shape = compute_shape_report(y[m], pr[m], index[m], include_extended=False)
    out = dict(shape)
    out["mae"] = round(mae, 4)
    out["rmse"] = round(rmse, 4)
    out["n_samples"] = n
    try:
        ef = to_eval_frame(index[m], y[m], pr[m])
        sm = compute_shape_metrics(ef, include_extended=False)
        out["neg_corr_day_ratio"] = round(float(sm["neg_corr_day_ratio"]), 6)
    except Exception:
        out["neg_corr_day_ratio"] = float("nan")
    return out


def run_all(
    test_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
    hourly_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在 [test_start, test_end] 上评估三条 naive，返回 (summary_df, detail_df)。
    """
    t0 = test_start if test_start is not None else pd.Timestamp(TEST_START)
    t1 = test_end if test_end is not None else _test_end_hourly()

    df = load_hourly_settlements(hourly_path)
    t1_eff = min(t1, df.index.max())
    win = df.loc[t0:t1_eff].copy()
    if win.empty:
        raise ValueError(f"测试窗 {t0} ~ {t1_eff} 无数据")

    da_full = df[COL_DA].astype(float)
    rt_full = df[COL_RT].astype(float)
    da_lag_full = da_full.shift(24)
    rt_lag_full = rt_full.shift(24)

    da = win[COL_DA].astype(float)
    rt = win[COL_RT].astype(float)
    da_lag = da_lag_full.reindex(win.index)
    rt_lag = rt_lag_full.reindex(win.index)

    idx = win.index
    rows: List[Dict[str, Any]] = []

    ev_da = evaluate_one(da, da_lag, idx)
    rows.append({"task": "da", "variant": "lag24h", **ev_da})

    ev_rt1 = evaluate_one(rt, rt_lag, idx)
    rows.append({"task": "rt", "variant": "lag24h", **ev_rt1})

    ev_rt2 = evaluate_one(rt, da, idx)
    rows.append({"task": "rt", "variant": "pred_from_da_settlement", **ev_rt2})

    summary = pd.DataFrame(rows)
    col_order = [
        "task",
        "variant",
        "n_samples",
        "mae",
        "rmse",
        "profile_corr",
        "neg_corr_day_ratio",
        "norm_profile_mae",
        "peak_hour_err",
        "valley_hour_err",
        "amplitude_err",
        "direction_acc",
    ]
    summary = summary[[c for c in col_order if c in summary.columns]]

    detail = pd.DataFrame(
        {
            COL_DA: da,
            "da_naive_lag24h": da_lag,
            COL_RT: rt,
            "rt_naive_lag24h": rt_lag,
            "rt_naive_pred_from_da": da,
        },
        index=idx,
    )
    detail.index.name = "ts"

    od = out_dir or DEFAULT_OUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    summary.to_csv(od / "summary.csv", index=False)
    detail.to_csv(od / "test_window_predictions.csv")
    logger.info("写入 %s", od / "summary.csv")
    logger.info("写入 %s", od / "test_window_predictions.csv")

    return summary, detail


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    s, _ = run_all()
    print(s.to_string(index=False))
