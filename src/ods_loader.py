"""
ODS 加载器 — 从 source_data/ 读取原始文件，标准化列名和日期，返回 DataFrame。

四种格式：
  load_format_a_single   : 长表 datetime + 单值列
  load_format_a_dual     : 长表 datetime + 上午/下午双值列 → melt
  load_format_a_clearing : 长表 datetime + 出清三元组
  load_format_a_settlement : 统一结算点电价（混合粒度）
  load_format_b          : 枢纽长表 (entity, ts, value)
  load_format_c_chunked  : 宽表 V-columns → melt（分块生成器）
  load_format_d_daily    : 日均价事务表
  load_format_d_maintenance : 检修计划
"""
import logging
from datetime import timedelta
from typing import Generator

import pandas as pd

from .config import SOURCE_DIR

logger = logging.getLogger(__name__)


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# ── Format A: 单值长表 ──────────────────────────────────
def load_format_a_single(filename: str, meta: dict) -> pd.DataFrame:
    path = SOURCE_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ts"] = _parse_datetime(df[meta["date_col"]])
    records = []
    for orig_col, metric_name in meta["value_cols"].items():
        tmp = df[["ts"]].copy()
        tmp["metric"] = metric_name
        tmp["version"] = ""
        tmp["value"] = pd.to_numeric(df[orig_col], errors="coerce")
        records.append(tmp)
    out = pd.concat(records, ignore_index=True)
    out = out.dropna(subset=["ts"])
    logger.info("Loaded %s: %d rows → %d records", filename, len(df), len(out))
    return out


# ── Format A: 上午/下午双值 → melt ─────────────────────
def load_format_a_dual(filename: str, meta: dict) -> pd.DataFrame:
    path = SOURCE_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ts"] = _parse_datetime(df[meta["date_col"]])
    records = []
    for orig_col, (metric_name, version) in meta["value_cols"].items():
        tmp = df[["ts"]].copy()
        tmp["metric"] = metric_name
        tmp["version"] = version
        tmp["value"] = pd.to_numeric(df[orig_col], errors="coerce")
        records.append(tmp)
    out = pd.concat(records, ignore_index=True)
    out = out.dropna(subset=["ts"])
    logger.info("Loaded %s: %d rows → %d records", filename, len(df), len(out))
    return out


# ── Format A: 出清结果 ─────────────────────────────────
def load_format_a_clearing(filename: str, meta: dict) -> pd.DataFrame:
    path = SOURCE_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ts"] = _parse_datetime(df[meta["date_col"]])
    df["market"] = meta["market"]
    rename = {}
    for orig_col, std_name in meta["value_cols"].items():
        rename[orig_col] = std_name
    df = df.rename(columns=rename)
    keep = ["ts", "market"] + list(meta["value_cols"].values())
    out = df[keep].copy()
    for c in meta["value_cols"].values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["ts"])
    logger.info("Loaded clearing %s: %d rows", filename, len(out))
    return out


# ── Format A: 统一结算点电价 ─────────────────────────────
def load_format_a_settlement(filename: str, meta: dict) -> pd.DataFrame:
    path = SOURCE_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ts"] = _parse_datetime(df[meta["date_col"]])
    rename = {}
    for orig_col, std_name in meta["value_cols"].items():
        rename[orig_col] = std_name
    df = df.rename(columns=rename)
    keep = ["ts"] + list(meta["value_cols"].values())
    out = df[keep].copy()
    for c in meta["value_cols"].values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["ts"])
    logger.info("Loaded settlement %s: %d rows", filename, len(out))
    return out


# ── Format B: 枢纽长表 ─────────────────────────────────
def load_format_b(filename: str, meta: dict) -> pd.DataFrame:
    path = SOURCE_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ts"] = _parse_datetime(df[meta["date_col"]])
    df["value"] = pd.to_numeric(df[meta["value_col"]], errors="coerce")
    keep = ["ts", "value"] + meta["entity_cols"]
    out = df[keep].dropna(subset=["ts"]).copy()
    logger.info("Loaded pivot %s: %d rows", filename, len(out))
    return out


# ── Format C: 宽表 melt（分块生成器） ──────────────────
def _v_col_to_time(v_col: str, suffix: str, interval_min: int) -> timedelta:
    """V0015 → timedelta(minutes=0), 即往前推 interval_min 分钟。"""
    stripped = v_col.replace("V", "").replace(suffix, "")
    hh = int(stripped[:2])
    mm = int(stripped[2:4])
    total_min = hh * 60 + mm - interval_min
    if total_min < 0:
        total_min = 0
    return timedelta(minutes=total_min)


def load_format_c_chunked(
    filename: str, meta: dict
) -> Generator[pd.DataFrame, None, None]:
    """逐块读取宽表并 melt 为长表。每个 chunk 产出一个 DataFrame。"""
    path = SOURCE_DIR / filename
    skip = meta["skip_placeholder_rows"]
    chunksize = meta["chunksize"]
    suffix = meta["v_suffix"]
    interval_min = meta["interval_min"]

    v_cols = None
    time_offsets = None

    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        skiprows=range(1, skip + 1),
        chunksize=chunksize,
        low_memory=False,
    ):
        if v_cols is None:
            v_cols = [c for c in chunk.columns if c.startswith("V") and c != "V"]
            time_offsets = {
                vc: _v_col_to_time(vc, suffix, interval_min) for vc in v_cols
            }

        dt = pd.to_datetime(chunk[meta["date_col"]].astype(str), errors="coerce")
        valid_mask = dt.notna()
        chunk = chunk.loc[valid_mask].copy()
        dt = dt.loc[valid_mask]
        if chunk.empty:
            continue

        base_dates = dt.dt.normalize()

        melted_parts = []
        for vc in v_cols:
            vals = pd.to_numeric(chunk[vc], errors="coerce")
            not_null = vals.notna()
            if not not_null.any():
                continue
            part = chunk.loc[not_null, meta["meta_cols"]].copy()
            part["dt"] = dt.loc[not_null].dt.date
            part["ts"] = base_dates.loc[not_null] + time_offsets[vc]
            part["price"] = vals.loc[not_null].values
            melted_parts.append(part)

        if melted_parts:
            result = pd.concat(melted_parts, ignore_index=True)
            result = result.rename(columns={
                "节点类型": "node_type",
                "数据类型": "data_type",
                "节点名称": "node_name",
            })
            yield result

    logger.info("Finished chunked melt for %s", filename)


# ── Format D: 日均价 ───────────────────────────────────
def load_format_d_daily_prices() -> pd.DataFrame:
    """合并3个日均价文件为一张表。"""
    from .config import FORMAT_D

    records = []
    for filename in ["日前平均出清电价.csv", "实时平均出清电价.csv", "平均申报电价.csv"]:
        meta = FORMAT_D[filename]
        path = SOURCE_DIR / filename
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["dt"] = _parse_datetime(df[meta["date_col"]]).dt.date
        df["metric"] = meta["metric"]
        df["price"] = pd.to_numeric(df[meta["value_col"]], errors="coerce")
        records.append(df[["dt", "metric", "price"]].dropna(subset=["dt"]))

    out = pd.concat(records, ignore_index=True)
    logger.info("Loaded daily prices: %d rows", len(out))
    return out


# ── Format D: 检修计划 ──────────────────────────────────
def load_format_d_maintenance() -> pd.DataFrame:
    from .config import FORMAT_D

    meta = FORMAT_D["发输变电检修计划.csv"]
    path = SOURCE_DIR / "发输变电检修计划.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["dt"] = _parse_datetime(df["日期"]).dt.date
    df = df.rename(columns=meta["rename"])
    keep = ["dt", "version", "equipment_name", "equipment_type",
            "voltage_level", "planned_start", "planned_end"]
    out = df[keep].dropna(subset=["dt"]).copy()
    out["planned_start"] = _parse_datetime(out["planned_start"])
    out["planned_end"] = _parse_datetime(out["planned_end"])
    logger.info("Loaded maintenance plan: %d rows", len(out))
    return out
