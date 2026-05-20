#!/usr/bin/env python3
"""方案 B：Moirai encoder 表征 + DWS lag0 表格特征 → LightGBM（不含 V25）。

1. 导出 Moirai backbone 隐层表征（future token 上采样 + context mean 池化）
2. 与 DWS lag0、日历、lag24 拼接训练 LGB

示例（power）：
  HF_HUB_OFFLINE=1 python run_moirai_encoder_stack_lgb.py
  HF_HUB_OFFLINE=1 python run_moirai_encoder_stack_lgb.py --skip-embed-regen
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
from src.moirai_encoder_features import export_hourly_encoder_embeddings, load_encoder_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET = "da_clearing_price"
DWS_PATH = OUTPUT_DIR / "dws_hourly_features.csv"
DEFAULT_EMBED_CSV = OUTPUT_DIR / "moirai_encoder_1h_uni" / "hourly_encoder_features.csv"
DEFAULT_MOIRAI_P50_CSV = OUTPUT_DIR / "moirai_da_clearing_1h_uni" / "test_predictions_quantile.csv"
DEFAULT_OUT = OUTPUT_DIR / "moirai_encoder_stack_lgb"

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


def _ensure_encoder_coverage(
    embed_path: Path,
    regen: bool,
    regen_start: str,
    regen_end: str,
    patch_size: int,
) -> pd.DataFrame:
    need_start = pd.Timestamp(HOURLY_TRAIN_END) - pd.Timedelta(hours=23)
    need_end = pd.Timestamp(HOURLY_TEST_END)

    if embed_path.is_file() and not regen:
        enc = load_encoder_frame(embed_path)
        if enc.index.min() <= need_start and enc.index.max() >= need_end:
            logger.info("Encoder 特征已覆盖 train+test: %s", embed_path)
            return enc
        logger.warning("Encoder CSV 覆盖不足，将重新导出")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    export_hourly_encoder_embeddings(
        regen_start,
        regen_end,
        patch_size=patch_size,
        out_csv=embed_path,
    )
    return load_encoder_frame(embed_path)


def _load_moirai_p50(path: Path) -> pd.Series:
    if not path.is_file():
        return pd.Series(dtype=float, name="moirai_p50")
    q = pd.read_csv(path, parse_dates=["ts"]).set_index("ts").sort_index()
    return q["p50"].rename("moirai_p50") if "p50" in q.columns else pd.Series(dtype=float)


def build_stack_frame(enc: pd.DataFrame) -> pd.DataFrame:
    df = _load_dws_base().join(enc, how="left")
    return df.dropna(subset=["y", "lag24"] + [c for c in enc.columns if c.startswith("enc_fut_")])


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


def train_and_evaluate(df: pd.DataFrame, out_dir: Path, moirai_p50: pd.Series) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = _split(df)
    feats = _feature_cols(df)
    logger.info("特征数 %d（含 encoder %d 维）", len(feats), sum(c.startswith("enc_") for c in feats))

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
        "Encoder Stack LGB test: MAE=%.2f profile_corr=%.4f",
        pm.get("mae", float("nan")),
        sm.get("profile_corr", float("nan")),
    )

    y, idx = test["y"].values, test.index
    p50 = moirai_p50.reindex(test.index).values if not moirai_p50.empty else np.full(len(test), np.nan)
    cmp_df = pd.DataFrame([
        {"model": "encoder_stack_lgb", **_metrics_block(y, pred_test, idx)},
        {"model": "moirai_p50", **_metrics_block(y, p50, idx)},
        {"model": "naive_lag24", **_metrics_block(y, test["lag24"].values, idx)},
    ])
    cmp_df.to_csv(out_dir / "comparison_test.csv", index=False)
    with (out_dir / "stack_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme": "B_encoder_lgb",
                "best_iteration": best_iter,
                "n_features": len(feats),
                "n_encoder_dims": sum(c.startswith("enc_") for c in feats),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("对比表:\n%s", cmp_df.to_string(index=False))
    return cmp_df


def main() -> None:
    ap = argparse.ArgumentParser(description="方案 B: Moirai encoder + LGB stacking")
    ap.add_argument("--embed-csv", type=Path, default=DEFAULT_EMBED_CSV)
    ap.add_argument("--moirai-p50-csv", type=Path, default=DEFAULT_MOIRAI_P50_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-embed-regen", action="store_true")
    ap.add_argument("--embed-regen-start", default="2025-12-01")
    ap.add_argument("--embed-regen-end", default=HOURLY_TEST_END)
    ap.add_argument("--patch-size", type=int, default=8, help="Moirai patch_size（默认 8，future 3 token→24h）")
    args = ap.parse_args()

    enc = _ensure_encoder_coverage(
        args.embed_csv,
        regen=not args.skip_embed_regen,
        regen_start=args.embed_regen_start,
        regen_end=args.embed_regen_end,
        patch_size=args.patch_size,
    )
    df = build_stack_frame(enc)
    logger.info("Stack 帧: %d 行, %s ~ %s", len(df), df.index.min(), df.index.max())
    moirai_p50 = _load_moirai_p50(args.moirai_p50_csv)
    train_and_evaluate(df, args.out_dir, moirai_p50)


if __name__ == "__main__":
    main()
