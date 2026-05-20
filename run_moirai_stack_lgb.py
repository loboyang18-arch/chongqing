#!/usr/bin/env python3
"""Moirai 分位数 + DWS lag0 表格特征 → LightGBM stacking（日前出清价，不含 V25）。

训练：至 HOURLY_TRAIN_END；早停验证：HOURLY_VAL；测试：HOURLY_TEST（3/3～4/13）。

特征：
  - DWS lag0（负荷/新能源/联络线/检修等）
  - 日历：hour, dayofweek, is_peak, is_valley
  - da_clearing_price lag24
  - Moirai：p10/p30/p50/p70/p90, moirai_spread_80

示例（power）：
  HF_HUB_OFFLINE=1 python run_moirai_stack_lgb.py --skip-moirai-regen
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from price_forecast_eval import evaluate_predictions_csv, quick_shape_report, write_metrics_json
from src.config import OUTPUT_DIR
from src.experiment.splits import (
    HOURLY_TEST_END,
    HOURLY_TEST_START,
    HOURLY_TRAIN_END,
    HOURLY_VAL_END,
)
from src.feature_engineering import LAG0_DIRECT
from src.model_baseline import EARLY_STOPPING_ROUNDS, LGB_PARAMS_DA, NUM_BOOST_ROUND

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET = "da_clearing_price"
DWS_PATH = OUTPUT_DIR / "dws_hourly_features.csv"
DEFAULT_MOIRAI_CSV = OUTPUT_DIR / "moirai_da_clearing_1h_uni" / "test_predictions_quantile.csv"
DEFAULT_OUT = OUTPUT_DIR / "moirai_stack_lgb"

MOIRAI_COLS = ("p10", "p30", "p50", "p70", "p90")
PEAK_HOURS = set(range(8, 12)) | set(range(17, 21))
VALLEY_HOURS = set(range(0, 8)) | {23}


def _calendar_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    h = idx.hour
    return pd.DataFrame(
        {
            "hour": h,
            "dayofweek": idx.dayofweek,
            "is_peak": h.isin(PEAK_HOURS).astype(np.int8),
            "is_valley": h.isin(VALLEY_HOURS).astype(np.int8),
        },
        index=idx,
    )


def _load_dws_base() -> pd.DataFrame:
    raw = pd.read_csv(DWS_PATH, parse_dates=["ts"], index_col="ts").sort_index()
    lag0_cols = [c for c in LAG0_DIRECT if c in raw.columns]
    df = pd.concat(
        [raw[[TARGET]].rename(columns={TARGET: "y"}), raw[lag0_cols].astype(float)],
        axis=1,
    )
    df["lag24"] = df["y"].shift(24)
    return pd.concat([df, _calendar_feats(df.index)], axis=1)


def _load_moirai_quantiles(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path, parse_dates=["ts"])
    if "ts" not in q.columns:
        q = q.rename(columns={q.columns[0]: "ts"})
    q = q.set_index("ts").sort_index()
    out = q[list(MOIRAI_COLS)].astype(float)
    out["moirai_spread_80"] = out["p90"] - out["p10"]
    return out


def _ensure_moirai_coverage(
    moirai_path: Path,
    regen: bool,
    regen_start: str,
    regen_end: str,
) -> pd.DataFrame:
    need_start = pd.Timestamp(HOURLY_TRAIN_END) - pd.Timedelta(hours=23)
    need_end = pd.Timestamp(HOURLY_TEST_END)

    if moirai_path.is_file() and not regen:
        q = _load_moirai_quantiles(moirai_path)
        if q.index.min() <= need_start and q.index.max() >= need_end:
            logger.info("Moirai 分位数已覆盖 train+test: %s", moirai_path)
            return q
        logger.warning("Moirai 覆盖不足，将重新生成")

    logger.info("生成 Moirai 分位数 %s ~ %s …", regen_start, regen_end)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from src.model_moirai_da_clearing import run as moirai_run

    out_dir = moirai_path.parent
    moirai_run(
        test_start=regen_start,
        test_end=regen_end,
        use_covariates=False,
        out_dir=out_dir,
    )
    return _load_moirai_quantiles(out_dir / "test_predictions_quantile.csv")


def build_stack_frame(moirai_q: pd.DataFrame) -> pd.DataFrame:
    df = _load_dws_base().join(moirai_q, how="left")
    return df.dropna(subset=["y", "lag24", "p50"])


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[:HOURLY_TRAIN_END].copy()
    val = df.loc[pd.Timestamp(HOURLY_TRAIN_END) + pd.Timedelta(hours=1): HOURLY_VAL_END].copy()
    test = df.loc[HOURLY_TEST_START:HOURLY_TEST_END].copy()
    return train, val, test


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "y" and df[c].dtype != "O"]


def _metrics_block(y: np.ndarray, p: np.ndarray, idx: pd.DatetimeIndex) -> Dict:
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    yt, pt = y[m], p[m]
    return {
        "mae": float(np.mean(np.abs(yt - pt))),
        "rmse": float(np.sqrt(np.mean((yt - pt) ** 2))),
        "n": int(m.sum()),
        **quick_shape_report(yt, pt, idx[m], include_extended=False),
    }


def train_and_evaluate(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = _split(df)
    feats = _feature_cols(df)
    logger.info("特征数 %d", len(feats))

    dtrain = lgb.Dataset(train[feats], label=train["y"])
    dval = lgb.Dataset(val[feats], label=val["y"], reference=dtrain)
    model = lgb.train(
        LGB_PARAMS_DA,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    best_iter = int(getattr(model, "best_iteration", 0) or NUM_BOOST_ROUND)
    pred_test = model.predict(test[feats], num_iteration=best_iter)

    pd.DataFrame(
        {"ts": test.index, "actual": test["y"].values, "predicted": pred_test},
    ).to_csv(out_dir / "da_result.csv", index=False)

    ev = evaluate_predictions_csv(
        out_dir / "da_result.csv",
        actual_col="actual",
        pred_col="predicted",
        task_type="da",
    )
    write_metrics_json(ev, out_dir / "metrics.json")
    sm, pm = ev.get("shape_metrics") or {}, ev.get("point_metrics") or {}
    logger.info(
        "Stack LGB test: MAE=%.2f profile_corr=%.4f",
        pm.get("mae", float("nan")),
        sm.get("profile_corr", float("nan")),
    )

    y, idx = test["y"].values, test.index
    cmp_df = pd.DataFrame([
        {"model": "stack_lgb", **_metrics_block(y, pred_test, idx)},
        {"model": "moirai_p50", **_metrics_block(y, test["p50"].values, idx)},
        {"model": "naive_lag24", **_metrics_block(y, test["lag24"].values, idx)},
    ])
    cmp_df.to_csv(out_dir / "comparison_test.csv", index=False)
    with (out_dir / "stack_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"best_iteration": best_iter, "feature_cols": feats,
             "n_train": len(train), "n_val": len(val), "n_test": len(test)},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("对比表:\n%s", cmp_df.to_string(index=False))
    return cmp_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moirai-csv", type=Path, default=DEFAULT_MOIRAI_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-moirai-regen", action="store_true")
    ap.add_argument("--moirai-regen-start", default="2025-12-01")
    ap.add_argument("--moirai-regen-end", default=HOURLY_TEST_END)
    args = ap.parse_args()

    moirai_q = _ensure_moirai_coverage(
        args.moirai_csv,
        regen=not args.skip_moirai_regen,
        regen_start=args.moirai_regen_start,
        regen_end=args.moirai_regen_end,
    )
    df = build_stack_frame(moirai_q)
    logger.info("Stack 帧: %d 行, %s ~ %s", len(df), df.index.min(), df.index.max())
    train_and_evaluate(df, args.out_dir)


if __name__ == "__main__":
    main()
