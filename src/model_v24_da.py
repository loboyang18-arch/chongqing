"""
V24 — 基于 sql_data/chongqing_market_join.csv 的 30 列市场特征 + 天气 + 时间编码，
使用 V18 Conv2D 架构预测日前出清价（market_clearing_price → 小时均值）。

特征来源与 Lag 级别（以决策日 D 为基准）：
  Lag0（D 日可得）：8 列 *_pred_v1 [+ 38 天气]
  Lag1（D-1 可得）：7 列 *_pred_v2 + 6 列日前/可靠性出清
  Lag2（D-2 可得）：6 列 *_actual + 3 列实时出清

两种模式：
  非 PCA（默认）：df 不做 shift，V18 通过 dates0/dates1/dates2 天然实现 lag 对齐。
  PCA 模式：先对 LAG1 shift+1d、LAG2 shift+2d 构建 lag 对齐矩阵 → PCA → 结果作为 Lag0。

环境变量：
  V24_USE_WEATHER    是否加入天气列（默认 0=关）
  V24_PCA_COMPONENTS PCA 主成分数（默认 0=不做 PCA）
  V24_PCA_RANDOM_STATE PCA 随机种子（默认 42）
  V24_OUT_DIR        输出子目录（默认 v24_da）
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BASE_DIR, OUTPUT_DIR
from .experiment.splits import (
    EFFECTIVE_START, EFFECTIVE_END, TRAIN_END, VAL_END,
    TEST_START, TEST_END,
)
from .model_v18_conv2d import run_v18

logger = logging.getLogger(__name__)

V24_DIR = OUTPUT_DIR / os.environ.get("V24_OUT_DIR", "v24_da").strip()
SQL_DATA_PATH = BASE_DIR / "sql_data" / "chongqing_market_join.csv"
SPD = 96  # 15-min slots per day

_DA_TARGET = "market_clearing_price"
_RT_TARGET = "realtime_clearing_price"
_DEFAULT_TARGET = _DA_TARGET
V24_TARGET_COL = os.environ.get("V24_TARGET_COL", _DEFAULT_TARGET).strip() or _DEFAULT_TARGET
if V24_TARGET_COL == "rt_clearing_price":
    V24_TARGET_COL = _RT_TARGET
TARGET_COL = V24_TARGET_COL
_RT_TARGET_NAMES = frozenset({_RT_TARGET, "rt_clearing_price"})

# ── Lag0: D 日当天可得（v1 预测 + 天气） ────────────────────────
LAG0_MARKET_COLS = [
    "total_gen_pred_v1",
    "total_load_pred_v1",
    "renewable_pred_v1",
    "solar_pred_v1",
    "wind_pred_v1",
    "hydro_pred_v1",
    "trans_pred_v1",
    "nonmarket_gen_pred_v1",
]

LAG0_WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth",
    "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "shortwave_radiation_instant",
    "direct_radiation", "direct_radiation_instant",
    "direct_normal_irradiance", "direct_normal_irradiance_instant",
    "diffuse_radiation", "diffuse_radiation_instant",
    "terrestrial_radiation", "terrestrial_radiation_instant",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
]

V24_USE_WEATHER = os.environ.get("V24_USE_WEATHER", "0").strip().lower() not in (
    "0", "false", "no", "off", "",
)
LAG0_COLS = LAG0_MARKET_COLS + LAG0_WEATHER_COLS if V24_USE_WEATHER else list(LAG0_MARKET_COLS)

# ── Lag1: D-1 可得（v2 修正预测 + 日前/可靠性出清） ──────────
LAG1_COLS = [
    "total_gen_pred_v2",
    "renewable_pred_v2",
    "solar_pred_v2",
    "wind_pred_v2",
    "hydro_pred_v2",
    "trans_pred_v2",
    "nonmarket_gen_pred_v2",
    "market_clearing_price",
    "market_clearing_power",
    "market_unit_count",
    "reliability_clearing_price",
    "reliability_clearing_power",
    "reliability_unit_count",
]

# ── Lag2: D-2 可得（实际值 + 实时出清） ──────────────────────
LAG2_COLS_BASE = [
    "total_gen_actual",
    "total_load_actual",
    "renewable_actual",
    "hydro_actual",
    "trans_actual",
    "nonmarket_gen_actual",
    "realtime_clearing_price",
    "realtime_clearing_energy",
    "realtime_unit_count",
]

_RT_LEAK_COLS = frozenset({
    "realtime_clearing_price",
    "realtime_clearing_energy",
    "realtime_unit_count",
})

# 默认保留 Lag2 实时出清（历史 D-2 语义）；仅当 V24_STRIP_RT_LAG2=1 时剔除（旧 V25 RT 单任务行为）
V24_STRIP_RT_LAG2 = os.environ.get("V24_STRIP_RT_LAG2", "0").strip().lower() in (
    "1", "true", "yes", "on",
)


def lag2_cols_for_target(target: str = TARGET_COL) -> List[str]:
    """Lag2 列列表。RT 目标默认与 DA 相同保留 realtime_clearing_*。"""
    if target in _RT_TARGET_NAMES and V24_STRIP_RT_LAG2:
        return [c for c in LAG2_COLS_BASE if c not in _RT_LEAK_COLS]
    return list(LAG2_COLS_BASE)


LAG2_COLS = lag2_cols_for_target()
ALL_FEAT_COLS = LAG0_COLS + LAG1_COLS + LAG2_COLS

# PCA 降维（环境变量控制；0 = 不做 PCA）
V24_PCA_COMPONENTS = int(os.environ.get("V24_PCA_COMPONENTS", "0"))
V24_PCA_RANDOM_STATE = int(os.environ.get("V24_PCA_RANDOM_STATE", "42"))


# ── 数据加载（不做 shift，V18 dates0/1/2 天然处理 lag） ──────

def _load_raw_df() -> pd.DataFrame:
    """从 sql_data CSV 加载原始数据，对齐到 15min 网格，不做任何 shift。"""
    logger.info("Loading sql_data: %s", SQL_DATA_PATH.name)
    raw = pd.read_csv(SQL_DATA_PATH, parse_dates=["datetime"])
    raw = raw.set_index("datetime").sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    idx = pd.date_range(EFFECTIVE_START, EFFECTIVE_END, freq="15min")
    df = pd.DataFrame(index=idx)
    df.index.name = "ts"

    all_needed = list(dict.fromkeys(ALL_FEAT_COLS + [TARGET_COL]))
    for col in all_needed:
        if col in raw.columns:
            df[col] = raw[col].reindex(idx)

    n_valid = df.dropna(how="all").shape[0]
    logger.info("Raw SQL matrix: %d rows × %d cols, %d non-empty rows",
                len(df), len(df.columns), n_valid)
    return df


def load_sql_feature_matrix() -> pd.DataFrame:
    """非 PCA 模式：返回原始 df，V18 的 dates0/dates1/dates2 实现 lag 对齐。"""
    return _load_raw_df()


# ── PCA 模式：先 lag 对齐再 PCA ──────────────────────────────

def load_sql_feature_matrix_pca(
    n_components: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    1. 加载原始数据
    2. 构建 lag 对齐矩阵：LAG0 不偏移，LAG1 shift+1d，LAG2 shift+2d
       → 每行 = 该时刻决策时能看到的全部信息
    3. 在训练窗内拟合 StandardScaler + PCA
    4. 返回 (df with PCA cols + 原始目标列, pca_col_names, meta)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = _load_raw_df()

    # 构建 lag 对齐矩阵（仅用于 PCA，不改原 df 的列）
    parts: List[np.ndarray] = []
    aligned_names: List[str] = []

    for col in LAG0_COLS:
        if col in df.columns:
            parts.append(df[col].values.astype(np.float64))
            aligned_names.append(col)

    for col in LAG1_COLS:
        if col in df.columns:
            parts.append(df[col].shift(SPD).values.astype(np.float64))
            aligned_names.append(f"{col}_lag1")

    for col in LAG2_COLS:
        if col in df.columns:
            parts.append(df[col].shift(2 * SPD).values.astype(np.float64))
            aligned_names.append(f"{col}_lag2")

    X = np.column_stack(parts)
    logger.info("Lag-aligned matrix for PCA: %d rows × %d cols", X.shape[0], X.shape[1])

    # PCA（仅训练窗拟合）
    train_m = np.asarray(df.index <= TRAIN_END, dtype=bool)
    n_train = int(train_m.sum())

    col_mean = np.nanmean(X[train_m], axis=0)
    X_filled = np.where(np.isfinite(X), X, col_mean)

    scaler = StandardScaler()
    scaler.fit(X_filled[train_m])
    Z = scaler.transform(X_filled)

    k = max(1, min(int(n_components), Z.shape[1], n_train - 1))
    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(Z[train_m])
    P = pca.transform(Z)

    pca_names = [f"feat_pca_{i}" for i in range(k)]
    pca_df = pd.DataFrame(P, index=df.index, columns=pca_names)

    # 输出 df = PCA 列 + 原始目标列（不含原始特征列）
    out = pd.DataFrame(index=df.index)
    out.index.name = "ts"
    out = out.join(pca_df, how="left")
    if TARGET_COL in df.columns:
        out[TARGET_COL] = df[TARGET_COL]

    meta = {
        "n_components": k,
        "requested_components": int(n_components),
        "aligned_feature_names": aligned_names,
        "n_aligned_features": len(aligned_names),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_explained_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
    }
    return out, pca_names, meta


# ── V18 通道 Patch ─────────────────────────────────────────────

def _snapshot_v18() -> Dict[str, Any]:
    import src.model_v18_conv2d as m18
    return {
        "LAG0_COLS": list(m18.LAG0_COLS),
        "LAG1_COLS": list(m18.LAG1_COLS),
        "LAG2_COLS": list(m18.LAG2_COLS),
        "C_LAG0": int(m18.C_LAG0),
        "C_LAG1": int(m18.C_LAG1),
        "C_LAG2": int(m18.C_LAG2),
        "C_TOTAL": int(m18.C_TOTAL),
        "TARGET_COL": str(m18.TARGET_COL),
    }


def _restore_v18(snap: Dict[str, Any]) -> None:
    import src.model_v18_conv2d as m18
    m18.LAG0_COLS = list(snap["LAG0_COLS"])
    m18.LAG1_COLS = list(snap["LAG1_COLS"])
    m18.LAG2_COLS = list(snap["LAG2_COLS"])
    m18.C_LAG0 = int(snap["C_LAG0"])
    m18.C_LAG1 = int(snap["C_LAG1"])
    m18.C_LAG2 = int(snap["C_LAG2"])
    m18.C_TOTAL = int(snap["C_TOTAL"])
    m18.TARGET_COL = str(snap["TARGET_COL"])


def _patch_v18_for_v24_direct() -> None:
    """非 PCA 模式：V18 的 dates0/1/2 天然提供 lag 对齐。"""
    import src.model_v18_conv2d as m18
    m18.LAG0_COLS = list(LAG0_COLS)
    m18.LAG1_COLS = list(LAG1_COLS)
    lag2 = lag2_cols_for_target(TARGET_COL)
    m18.LAG2_COLS = list(lag2)
    m18.C_LAG0 = len(LAG0_COLS) + 4
    m18.C_LAG1 = len(LAG1_COLS)
    m18.C_LAG2 = len(lag2)
    m18.C_TOTAL = m18.C_LAG0 + m18.C_LAG1 + m18.C_LAG2
    m18.TARGET_COL = TARGET_COL


def _patch_v18_for_v24_pca(pca_names: List[str]) -> None:
    """PCA 模式：lag 对齐已在 PCA 前完成，全部主成分放 Lag0。"""
    import src.model_v18_conv2d as m18
    k = len(pca_names)
    m18.LAG0_COLS = list(pca_names)
    m18.LAG1_COLS = []
    m18.LAG2_COLS = []
    m18.C_LAG0 = k + 4
    m18.C_LAG1 = 0
    m18.C_LAG2 = 0
    m18.C_TOTAL = m18.C_LAG0
    m18.TARGET_COL = TARGET_COL


# ── 主入口 ────────────────────────────────────────────────────

def run_v24(out_dir: Optional[Path] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir or V24_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V24 — sql_data 特征 → V18 Conv2D, da_clearing_price")
    logger.info("  Lag0: %d cols (v1 pred%s)", len(LAG0_COLS),
                " + weather" if V24_USE_WEATHER else "")
    logger.info("  Lag1: %d cols (v2 pred + da/reliability clearing)", len(LAG1_COLS))
    logger.info("  Lag2: %d cols (actual + rt clearing)", len(LAG2_COLS))
    logger.info("  Target: %s", TARGET_COL)

    use_pca = V24_PCA_COMPONENTS > 0
    n_comp = V24_PCA_COMPONENTS
    rs = V24_PCA_RANDOM_STATE

    if use_pca:
        logger.info("  PCA: %d components, random_state=%d", n_comp, rs)
    else:
        logger.info("  PCA: off")

    snap = _snapshot_v18()
    meta: Dict[str, Any] = {}
    try:
        if use_pca:
            df, pca_names, pca_meta = load_sql_feature_matrix_pca(n_comp, rs)
            _patch_v18_for_v24_pca(pca_names)
            logger.info(
                "  PCA k=%d | cumulative explained var=%.4f",
                pca_meta["n_components"], pca_meta["cumulative_explained_variance"],
            )
            meta["pca"] = pca_meta
        else:
            df = load_sql_feature_matrix()
            _patch_v18_for_v24_direct()

        meta.update({
            "lag0_cols": LAG0_COLS,
            "lag1_cols": LAG1_COLS,
            "lag2_cols": LAG2_COLS,
            "n_lag0": len(LAG0_COLS),
            "n_lag1": len(LAG1_COLS),
            "n_lag2": len(LAG2_COLS),
            "use_pca": use_pca,
            "pca_components": n_comp if use_pca else 0,
            "use_weather": V24_USE_WEATHER,
            "target": TARGET_COL,
            "data_source": str(SQL_DATA_PATH.name),
        })
        meta_path = out_dir / "v24_feature_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s", meta_path.name)

        run_v18(out_dir=out_dir, feature_df=df)
    finally:
        _restore_v18(snap)
        logger.info("Restored V18 channel definitions")

    return meta


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v24()
