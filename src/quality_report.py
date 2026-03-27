"""
质量报告 — 对 DWS 小时宽表按月、按列统计完整性和异常率。

输出 quality_report.csv，每行 = (year_month, column, expected_hours,
non_null_count, non_zero_count, null_ratio, zero_ratio)。
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: dws_hourly_features (索引为 ts 的 DatetimeIndex)
    输出: 质量报告 DataFrame
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["_ym"] = df.index.to_period("M")

    value_cols = [c for c in df.columns if c != "_ym"]
    records = []

    for ym, grp in df.groupby("_ym"):
        start = ym.start_time
        end = ym.end_time
        expected_hours = int((end - start).total_seconds() / 3600) + 1

        for col in value_cols:
            series = grp[col]
            non_null = series.notna().sum()
            non_zero = (series.fillna(0) != 0).sum()
            actual_count = len(series)
            records.append({
                "year_month": str(ym),
                "column": col,
                "expected_hours": expected_hours,
                "actual_hours": actual_count,
                "non_null_count": int(non_null),
                "non_zero_count": int(non_zero),
                "null_ratio": round(1 - non_null / max(actual_count, 1), 4),
                "zero_ratio": round(
                    1 - non_zero / max(non_null, 1), 4
                ) if non_null > 0 else None,
            })

    report = pd.DataFrame(records)
    logger.info("Quality report: %d rows for %d columns × %d months",
                len(report), len(value_cols), df["_ym"].nunique())
    return report
