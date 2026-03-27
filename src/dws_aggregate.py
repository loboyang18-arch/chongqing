"""
DWS 聚合 — 将 DWD 层多粒度数据聚合到小时级并拼接为特征宽表。

核心函数:
  resample_to_hourly  : 通用重采样 (15min/5min → 1h)
  build_hourly_system_ts : 系统时序 → 小时宽表
  build_hourly_clearing  : 出清结果 → 小时宽表
  build_hourly_nodal     : 节点电价(系统级) → 小时宽表
  build_hourly_section   : 断面约束(关键断面) → 小时宽表
  merge_hourly_features  : 拼接所有小时宽表 → dws_hourly_features
"""
import logging

import pandas as pd

from .config import KEY_SECTIONS, SYSTEM_NODAL_PRICE_FILTER

logger = logging.getLogger(__name__)


def resample_to_hourly(
    df: pd.DataFrame, ts_col: str, value_col: str, method: str = "mean"
) -> pd.Series:
    """对单值列按小时重采样。"""
    s = df.set_index(ts_col)[value_col]
    if method == "mean":
        return s.resample("1h").mean()
    elif method == "sum":
        return s.resample("1h").sum(min_count=1)
    else:
        return s.resample("1h").mean()


SUB_HOUR_STATS_METRICS = {
    "actual_load",
    "renewable_gen",
    "tie_line_power",
}

SUB_HOUR_STATS_CLEARING = {
    "da_clearing_price",
    "rt_clearing_price",
}


def _sub_hour_stats(s_hourly_groups) -> dict:
    """对已按小时桶分组的 Series 计算 std / range / max_ramp。"""
    std = s_hourly_groups.std()
    range_ = s_hourly_groups.max() - s_hourly_groups.min()
    max_ramp = s_hourly_groups.apply(lambda x: x.diff().abs().max() if len(x) > 1 else 0.0)
    return {"std": std, "range": range_, "max_ramp": max_ramp}


# ── 系统时序 → 小时宽表 ─────────────────────────────────
def build_hourly_system_ts(dwd_ts: pd.DataFrame) -> pd.DataFrame:
    """
    输入: (ts, metric, version, value, is_anomaly)
    输出: 以 ts 为索引的宽表，每个 metric+version 一列。
    """
    normal = dwd_ts[dwd_ts["is_anomaly"] == 0].copy()

    pivot_parts = []
    for (metric, version), grp in normal.groupby(["metric", "version"]):
        col_name = f"{metric}_{version}" if version else metric
        hourly = resample_to_hourly(grp, "ts", "value", "mean")
        hourly.name = col_name
        pivot_parts.append(hourly)

        if metric in SUB_HOUR_STATS_METRICS and not version:
            s = grp.set_index("ts")["value"]
            groups = s.resample("1h")
            for suffix, vals in _sub_hour_stats(groups).items():
                vals.name = f"{metric}_{suffix}"
                pivot_parts.append(vals)

    if not pivot_parts:
        return pd.DataFrame()

    result = pd.concat(pivot_parts, axis=1)
    result.index.name = "ts"
    logger.info("Hourly system_ts: %d hours × %d cols", len(result), len(result.columns))
    return result


# ── 出清结果 → 小时宽表 ─────────────────────────────────
def build_hourly_clearing(dwd_clearing: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for market, grp in dwd_clearing.groupby("market"):
        grp = grp.set_index("ts")
        value_cols = [c for c in grp.columns if c not in ("market",)]
        for vc in value_cols:
            s = pd.to_numeric(grp[vc], errors="coerce")
            if s.notna().sum() == 0:
                continue
            hourly = s.resample("1h").mean()
            col_name = f"{market}_clearing_{vc}"
            hourly.name = col_name
            parts.append(hourly)

            if col_name in SUB_HOUR_STATS_CLEARING:
                groups = s.dropna().resample("1h")
                for suffix, vals in _sub_hour_stats(groups).items():
                    vals.name = f"{col_name}_{suffix}"
                    parts.append(vals)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, axis=1)
    result.index.name = "ts"
    logger.info("Hourly clearing: %d hours × %d cols", len(result), len(result.columns))
    return result


# ── 统一结算点电价 → 小时宽表（已是小时粒度） ─────────────
def build_hourly_settlement(dwd_settlement: pd.DataFrame) -> pd.DataFrame:
    df = dwd_settlement.set_index("ts")
    value_cols = [c for c in df.columns if c != "ts"]
    result = df[value_cols].resample("1h").mean()
    result.index.name = "ts"
    logger.info("Hourly settlement: %d hours × %d cols", len(result), len(result.columns))
    return result


# ── 节点电价(系统级) → 小时宽表 ──────────────────────────
def build_hourly_nodal(dwd_nodal_da: pd.DataFrame, dwd_nodal_rt: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for label, df in [("da", dwd_nodal_da), ("rt", dwd_nodal_rt)]:
        if df.empty:
            continue
        mask = (
            (df["data_type"] == SYSTEM_NODAL_PRICE_FILTER["data_type"]) &
            (df["node_name"] == SYSTEM_NODAL_PRICE_FILTER["node_name"])
        )
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        hourly = resample_to_hourly(sub, "ts", "price", "mean")
        hourly.name = f"nodal_{label}_energy_price"
        parts.append(hourly)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, axis=1)
    result.index.name = "ts"
    logger.info("Hourly nodal: %d hours × %d cols", len(result), len(result.columns))
    return result


# ── 断面约束 → 小时宽表 (关键断面利用率) ─────────────────
def build_hourly_section(dwd_section: pd.DataFrame) -> pd.DataFrame:
    if dwd_section.empty:
        return pd.DataFrame()

    parts = []
    for section_name in KEY_SECTIONS:
        actual = dwd_section[
            (dwd_section["device_name"] == section_name) &
            (dwd_section["device_type"] == "实际潮流")
        ]
        limit = dwd_section[
            (dwd_section["device_name"] == section_name) &
            (dwd_section["device_type"] == "限额")
        ]

        if actual.empty or limit.empty:
            continue

        h_actual = resample_to_hourly(actual, "ts", "value", "mean")
        h_limit = resample_to_hourly(limit, "ts", "value", "mean")

        ratio = h_actual / h_limit.replace(0, float("nan"))
        safe_name = section_name.replace(" ", "_")[:30]
        ratio.name = f"section_ratio_{safe_name}"
        parts.append(ratio)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, axis=1)
    result.index.name = "ts"
    logger.info("Hourly section: %d hours × %d cols", len(result), len(result.columns))
    return result


# ── 日级指标 → 展开到小时 ───────────────────────────────
def expand_daily_to_hourly(
    daily_price: pd.DataFrame,
    maintenance: pd.DataFrame,
    ts_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """将日级指标按日展开到每个小时。"""
    hourly_dt = ts_index.normalize()

    result = pd.DataFrame(index=ts_index)
    result.index.name = "ts"

    if not daily_price.empty:
        daily_price = daily_price.copy()
        daily_price["dt_norm"] = pd.to_datetime(daily_price["dt"]).dt.normalize()
        for col in ["da_avg_clearing_price", "rt_avg_clearing_price", "avg_bid_price"]:
            if col in daily_price.columns:
                mapping = daily_price.set_index("dt_norm")[col]
                result[col] = hourly_dt.map(mapping)

    if not maintenance.empty:
        maintenance = maintenance.copy()
        maintenance["dt_norm"] = pd.to_datetime(maintenance["dt"]).dt.normalize()
        for col in ["maintenance_gen_count", "maintenance_grid_count"]:
            if col in maintenance.columns:
                mapping = maintenance.set_index("dt_norm")[col]
                result[col] = hourly_dt.map(mapping).fillna(0).astype(int)

    logger.info("Daily→hourly expansion: %d hours × %d cols", len(result), len(result.columns))
    return result


# ── 拼接所有小时宽表 ───────────────────────────────────
def merge_hourly_features(
    hourly_system: pd.DataFrame,
    hourly_clearing: pd.DataFrame,
    hourly_settlement: pd.DataFrame,
    hourly_nodal: pd.DataFrame,
    hourly_section: pd.DataFrame,
    hourly_daily: pd.DataFrame,
) -> pd.DataFrame:
    """外连接所有小时宽表。"""
    dfs = [
        hourly_system,
        hourly_clearing,
        hourly_settlement,
        hourly_nodal,
        hourly_section,
        hourly_daily,
    ]
    dfs = [d for d in dfs if not d.empty]

    if not dfs:
        return pd.DataFrame()

    result = dfs[0]
    for d in dfs[1:]:
        result = result.join(d, how="outer")

    result = result.sort_index()
    result.index.name = "ts"
    logger.info("Merged hourly features: %d hours × %d cols", len(result), len(result.columns))
    return result
