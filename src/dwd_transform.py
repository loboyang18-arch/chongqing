"""
DWD 清洗与标准化 — 对 ODS 加载结果做以下处理：

1. drop_null_value_rows : 删除 value 全为 NaN 的占位行
2. flag_anomalies       : 异常值标记（非市场机组 < -1000, 出清电价 = 0）
3. build_dwd_system_ts  : 合并所有15min系统时序为统一表
4. build_dwd_clearing   : 合并出清结果
5. build_dwd_settlement : 清洗统一结算点电价（去掉纯日期占位行）
6. build_dwd_section    : 断面约束清洗
7. build_dwd_nodal_price : 节点电价（从 chunked melt 结果清洗）
8. build_dwd_maintenance : 检修计划汇总为日级计数
9. build_dwd_daily_price : 日均价 pivot 为宽表
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_null_value_rows(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    mask = df[value_cols].notna().any(axis=1)
    dropped = (~mask).sum()
    if dropped:
        logger.info("Dropped %d all-null rows", dropped)
    return df.loc[mask].copy()



def flag_anomalies(df: pd.DataFrame, metric_col: str, value_col: str) -> pd.DataFrame:
    """添加 is_anomaly 列。"""
    df = df.copy()
    df["is_anomaly"] = 0
    non_market_mask = df[metric_col].str.contains("non_market", na=False)
    df.loc[non_market_mask & (df[value_col] < -1000), "is_anomaly"] = 1
    return df


# ── DWD 构建函数 ────────────────────────────────────────

def build_dwd_system_ts(
    single_dfs: list[pd.DataFrame],
    dual_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """合并 Format A 单值 + 双值为统一的 (ts, metric, version, value) 表。"""
    parts = single_dfs + dual_dfs
    df = pd.concat(parts, ignore_index=True)
    df = drop_null_value_rows(df, ["value"])
    df = flag_anomalies(df, "metric", "value")
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info("DWD system_ts: %d rows, metrics=%s",
                len(df), sorted(df["metric"].unique()))
    return df


def build_dwd_clearing(clearing_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(clearing_dfs, ignore_index=True)
    value_cols = [c for c in df.columns if c not in ("ts", "market")]
    df = drop_null_value_rows(df, value_cols)
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info("DWD clearing: %d rows", len(df))
    return df


def build_dwd_settlement(df: pd.DataFrame) -> pd.DataFrame:
    """统一结算点电价：去掉无小时信息的占位行。"""
    value_cols = [c for c in df.columns if c != "ts"]
    df = drop_null_value_rows(df, value_cols)
    has_time = df["ts"].dt.hour.ne(0) | df["ts"].dt.minute.ne(0)
    first_valid = df[value_cols].notna().any(axis=1)
    df = df[has_time | first_valid].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info("DWD settlement: %d rows", len(df))
    return df



def build_dwd_section(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_null_value_rows(df, ["value"])
    df = df.rename(columns={"设备名称": "device_name", "设备类型": "device_type"})
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info("DWD section_constraint: %d rows, devices=%d",
                len(df), df["device_name"].nunique())
    return df


def build_dwd_nodal_price(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df = drop_null_value_rows(df, ["price"])
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info("DWD nodal_price: %d rows", len(df))
    return df


def build_dwd_maintenance_daily(df: pd.DataFrame) -> pd.DataFrame:
    """检修计划 → 按日统计发电/电网设备检修数。"""
    df = df.dropna(subset=["dt"])
    gen_mask = df["equipment_type"] == "发电设备"
    grid_mask = df["equipment_type"] == "电网设备"

    gen_daily = df.loc[gen_mask].groupby("dt").size().rename("maintenance_gen_count")
    grid_daily = df.loc[grid_mask].groupby("dt").size().rename("maintenance_grid_count")
    out = pd.DataFrame({"maintenance_gen_count": gen_daily,
                         "maintenance_grid_count": grid_daily}).fillna(0).astype(int)
    out.index.name = "dt"
    out = out.reset_index()
    out["dt"] = pd.to_datetime(out["dt"])
    logger.info("DWD maintenance_daily: %d days", len(out))
    return out


def build_dwd_daily_price(df: pd.DataFrame) -> pd.DataFrame:
    """日均价 → pivot 为 (dt, da_avg_clearing_price, rt_avg_clearing_price, avg_bid_price)。"""
    df = df.dropna(subset=["dt", "price"])
    pivot = df.pivot_table(index="dt", columns="metric", values="price", aggfunc="first")
    pivot = pivot.reset_index()
    pivot["dt"] = pd.to_datetime(pivot["dt"])
    for c in ["da_avg_clearing_price", "rt_avg_clearing_price", "avg_bid_price"]:
        if c not in pivot.columns:
            pivot[c] = float("nan")
    pivot.columns.name = None
    logger.info("DWD daily_price: %d days", len(pivot))
    return pivot
