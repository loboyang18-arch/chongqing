"""
主流程编排 — ODS → DWD → DWS，输出 parquet / csv / quality_report。

用法:
    python -m src.pipeline            # 完整流程(含大文件节点电价 melt)
    python -m src.pipeline --skip-nodal  # 跳过节点电价(加速调试)
"""
import argparse
import logging
import time

import pandas as pd

from .config import (
    FORMAT_A_CLEARING,
    FORMAT_A_DUAL,
    FORMAT_A_SETTLEMENT,
    FORMAT_A_SINGLE,
    FORMAT_B,
    FORMAT_C,
    OUTPUT_DIR,
    SYSTEM_NODAL_PRICE_FILTER,
)
from .dwd_transform import (
    build_dwd_clearing,
    build_dwd_daily_price,
    build_dwd_maintenance_daily,
    build_dwd_nodal_price,
    build_dwd_section,
    build_dwd_settlement,
    build_dwd_system_ts,
)
from .dws_aggregate import (
    build_hourly_clearing,
    build_hourly_nodal,
    build_hourly_section,
    build_hourly_settlement,
    build_hourly_system_ts,
    expand_daily_to_hourly,
    merge_hourly_features,
)
from .ods_loader import (
    load_format_a_clearing,
    load_format_a_dual,
    load_format_a_settlement,
    load_format_a_single,
    load_format_b,
    load_format_c_chunked,
    load_format_d_daily_prices,
    load_format_d_maintenance,
)
from .quality_report import generate_quality_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(skip_nodal: bool = False):
    t0 = time.time()

    # ═══════════════════════════════════════════════════════
    # 第1步: ODS 加载
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: ODS — Loading source files")
    logger.info("=" * 60)

    # Format A — 单值长表
    single_dfs = []
    for fname, meta in FORMAT_A_SINGLE.items():
        single_dfs.append(load_format_a_single(fname, meta))

    # Format A — 双值长表
    dual_dfs = []
    for fname, meta in FORMAT_A_DUAL.items():
        dual_dfs.append(load_format_a_dual(fname, meta))

    # Format A — 出清结果
    clearing_dfs = []
    for fname, meta in FORMAT_A_CLEARING.items():
        clearing_dfs.append(load_format_a_clearing(fname, meta))

    # Format A — 统一结算点电价
    fname_stl = "统一结算点电价.csv"
    ods_settlement = load_format_a_settlement(fname_stl, FORMAT_A_SETTLEMENT[fname_stl])

    # Format B — 断面约束
    fname_sec = "实际运行输电断面约束情况.csv"
    ods_section = load_format_b(fname_sec, FORMAT_B[fname_sec])

    # Format C — 节点电价（边加载边过滤，只保留系统级节点，避免内存溢出）
    nodal_da_chunks: list[pd.DataFrame] = []
    nodal_rt_chunks: list[pd.DataFrame] = []
    _nf = SYSTEM_NODAL_PRICE_FILTER

    if not skip_nodal:
        logger.info("Loading day-ahead nodal prices (141MB, chunked melt + filter)...")
        fname_nda = "日前市场交易出清节点电价.csv"
        for chunk_df in load_format_c_chunked(fname_nda, FORMAT_C[fname_nda]):
            filtered = chunk_df[
                (chunk_df["data_type"] == _nf["data_type"]) &
                (chunk_df["node_name"] == _nf["node_name"])
            ]
            if not filtered.empty:
                nodal_da_chunks.append(filtered)
        logger.info("Day-ahead nodal (filtered): %d chunks, %d rows",
                    len(nodal_da_chunks),
                    sum(len(c) for c in nodal_da_chunks))

        logger.info("Loading real-time nodal prices (1GB, chunked melt + filter)...")
        fname_nrt = "实时出清节点电价.csv"
        for chunk_df in load_format_c_chunked(fname_nrt, FORMAT_C[fname_nrt]):
            filtered = chunk_df[
                (chunk_df["data_type"] == _nf["data_type"]) &
                (chunk_df["node_name"] == _nf["node_name"])
            ]
            if not filtered.empty:
                nodal_rt_chunks.append(filtered)
        logger.info("Real-time nodal (filtered): %d chunks, %d rows",
                    len(nodal_rt_chunks),
                    sum(len(c) for c in nodal_rt_chunks))
    else:
        logger.info("SKIP: nodal prices (--skip-nodal)")

    # Format D — 日均价 + 检修
    ods_daily_price = load_format_d_daily_prices()
    ods_maintenance = load_format_d_maintenance()

    logger.info("ODS loading done in %.1fs", time.time() - t0)

    # ═══════════════════════════════════════════════════════
    # 第2步: DWD 清洗
    # ═══════════════════════════════════════════════════════
    t1 = time.time()
    logger.info("=" * 60)
    logger.info("STEP 2: DWD — Cleaning & standardizing")
    logger.info("=" * 60)

    dwd_system_ts = build_dwd_system_ts(single_dfs, dual_dfs)
    dwd_clearing = build_dwd_clearing(clearing_dfs)
    dwd_settlement = build_dwd_settlement(ods_settlement)
    dwd_section = build_dwd_section(ods_section)
    dwd_nodal_da = build_dwd_nodal_price(nodal_da_chunks)
    dwd_nodal_rt = build_dwd_nodal_price(nodal_rt_chunks)
    dwd_daily_price = build_dwd_daily_price(ods_daily_price)
    dwd_maintenance = build_dwd_maintenance_daily(ods_maintenance)

    logger.info("DWD done in %.1fs", time.time() - t1)

    # ═══════════════════════════════════════════════════════
    # 第3步: DWS 聚合
    # ═══════════════════════════════════════════════════════
    t2 = time.time()
    logger.info("=" * 60)
    logger.info("STEP 3: DWS — Aggregating to hourly features")
    logger.info("=" * 60)

    hourly_system = build_hourly_system_ts(dwd_system_ts)
    hourly_clearing = build_hourly_clearing(dwd_clearing)
    hourly_settlement = build_hourly_settlement(dwd_settlement)
    hourly_nodal = build_hourly_nodal(dwd_nodal_da, dwd_nodal_rt)
    hourly_section = build_hourly_section(dwd_section)

    # 日级展开
    ts_index = hourly_system.index if not hourly_system.empty else pd.DatetimeIndex([])
    hourly_daily = expand_daily_to_hourly(dwd_daily_price, dwd_maintenance, ts_index)

    # 拼接
    dws_features = merge_hourly_features(
        hourly_system,
        hourly_clearing,
        hourly_settlement,
        hourly_nodal,
        hourly_section,
        hourly_daily,
    )

    logger.info("DWS done in %.1fs", time.time() - t2)

    # ═══════════════════════════════════════════════════════
    # 第4步: 输出
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 4: Writing output files")
    logger.info("=" * 60)

    dws_features.to_csv(OUTPUT_DIR / "dws_hourly_features.csv")
    try:
        dws_features.to_parquet(OUTPUT_DIR / "dws_hourly_features.parquet")
        logger.info("Written: dws_hourly_features.parquet / .csv  (%d rows × %d cols)",
                    len(dws_features), len(dws_features.columns))
    except ImportError:
        logger.warning("pyarrow not installed, skipping parquet output")
        logger.info("Written: dws_hourly_features.csv  (%d rows × %d cols)",
                    len(dws_features), len(dws_features.columns))

    # 质量报告
    quality = generate_quality_report(dws_features)
    quality.to_csv(OUTPUT_DIR / "quality_report.csv", index=False)
    logger.info("Written: quality_report.csv (%d rows)", len(quality))

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("  dws_hourly_features: %d hours × %d columns", len(dws_features), len(dws_features.columns))
    logger.info("  Date range: %s ~ %s",
                dws_features.index.min() if not dws_features.empty else "N/A",
                dws_features.index.max() if not dws_features.empty else "N/A")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="重庆电力数据入库 Pipeline")
    parser.add_argument("--skip-nodal", action="store_true",
                        help="跳过节点电价 melt（节省时间用于调试）")
    args = parser.parse_args()
    run(skip_nodal=args.skip_nodal)


if __name__ == "__main__":
    main()
