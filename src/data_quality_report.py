"""
数据质量检查报告 — 逐特征检查原始数据质量并输出 CSV 报告。

检查维度：基本信息、时间覆盖、缺失分析、有效区间、数值分布、异常值。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    SOURCE_DIR, OUTPUT_DIR,
    FORMAT_A_SINGLE, FORMAT_A_DUAL, FORMAT_A_CLEARING,
    FORMAT_A_SETTLEMENT, FORMAT_B, FORMAT_D,
    KEY_SECTIONS,
)

logger = logging.getLogger(__name__)

TRAIN_PERIOD = ("2025-11-01", "2026-02-08 23:59:59")
TEST_PERIOD = ("2026-02-09", "2026-03-10 23:59:59")

REPORT_COLUMNS = [
    "feature_name", "source_file", "native_granularity",
    "target_granularity", "align_method", "role",
    "ts_min", "ts_max", "expected_rows", "actual_rows", "coverage_ratio",
    "nan_count", "nan_ratio", "nan_max_consecutive",
    "nan_max_consecutive_hours", "nan_start_periods", "nan_end_periods",
    "nan_segments",
    "valid_start", "valid_end", "valid_ratio_train", "valid_ratio_test",
    "dtype", "min", "max", "mean", "median", "std", "q01", "q99",
    "n_zero", "n_negative", "n_outlier_3sigma",
    "n_constant_runs", "max_constant_run",
]

V16_ROLES = {
    "da_clearing_price": "target",
    "rt_clearing_price": "past_observed",
    "actual_load": "past_observed",
    "total_gen": "past_observed",
    "hydro_gen": "past_observed",
    "non_market_gen": "past_observed",
    "renewable_gen": "past_observed",
    "tie_line_power": "past_observed",
    "reliability_da_price": "past_observed",
    "reliability_rt_price": "past_observed",
    "settlement_da_price": "past_observed",
    "settlement_rt_price": "past_observed",
    "da_avg_clearing_price": "future_known",
    "rt_avg_clearing_price": "future_known",
    "avg_bid_price": "past_observed",
    "load_forecast": "future_known",
    "renewable_fcst": "future_known",
    "total_gen_fcst_am": "future_known",
    "total_gen_fcst_pm": "future_known",
    "hydro_gen_fcst_am": "future_known",
    "hydro_gen_fcst_pm": "future_known",
    "non_market_gen_fcst_am": "future_known",
    "non_market_gen_fcst_pm": "future_known",
    "renewable_fcst_solar_am": "future_known",
    "renewable_fcst_solar_pm": "future_known",
    "renewable_fcst_total_am": "future_known",
    "renewable_fcst_total_pm": "future_known",
    "renewable_fcst_wind_am": "future_known",
    "renewable_fcst_wind_pm": "future_known",
    "tie_line_fcst_am": "future_known",
    "tie_line_fcst_pm": "future_known",
    "maintenance_gen_count": "future_known",
    "maintenance_grid_count": "future_known",
    "da_clearing_power": "past_observed",
    "da_clearing_unit_count": "past_observed",
    "rt_clearing_volume": "past_observed",
    "rt_clearing_unit_count": "past_observed",
    "da_reliability_clearing_price": "past_observed",
    "da_reliability_clearing_power": "past_observed",
    "da_reliability_clearing_unit_count": "past_observed",
}


def _max_consecutive_nan(s: pd.Series) -> int:
    if s.isna().sum() == 0:
        return 0
    is_nan = s.isna().astype(int)
    groups = is_nan.ne(is_nan.shift()).cumsum()
    nan_groups = groups[is_nan == 1]
    if nan_groups.empty:
        return 0
    return int(nan_groups.value_counts().max())


def _nan_segments(s: pd.Series) -> int:
    if s.isna().sum() == 0:
        return 0
    is_nan = s.isna().astype(int)
    starts = (is_nan.diff().fillna(is_nan.iloc[0]) == 1).sum()
    return int(starts)


def _leading_nans(s: pd.Series) -> int:
    count = 0
    for v in s:
        if pd.isna(v):
            count += 1
        else:
            break
    return count


def _trailing_nans(s: pd.Series) -> int:
    count = 0
    for v in reversed(s.values):
        if pd.isna(v):
            count += 1
        else:
            break
    return count


def _constant_runs(s: pd.Series, min_length: int = 8) -> Tuple[int, int]:
    vals = s.dropna().values
    if len(vals) < min_length:
        return 0, 0
    diffs = np.diff(vals)
    is_same = np.concatenate([[False], diffs == 0])
    groups = np.diff(np.where(np.concatenate([[True], ~is_same[1:], [True]]))[0])

    run_lengths = []
    current_len = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            current_len += 1
        else:
            if current_len >= min_length:
                run_lengths.append(current_len)
            current_len = 1
    if current_len >= min_length:
        run_lengths.append(current_len)

    n_runs = len(run_lengths)
    max_run = max(run_lengths) if run_lengths else 0
    return n_runs, max_run


def _compute_quality(
    s: pd.Series,
    feature_name: str,
    source_file: str,
    native_gran: int,
    target_gran: int,
    align_method: str,
    role: str,
) -> Dict:
    row = {
        "feature_name": feature_name,
        "source_file": source_file,
        "native_granularity": native_gran,
        "target_granularity": target_gran,
        "align_method": align_method,
        "role": role,
    }

    row["ts_min"] = str(s.index.min()) if len(s) > 0 else ""
    row["ts_max"] = str(s.index.max()) if len(s) > 0 else ""

    if len(s) > 0 and native_gran > 0:
        total_minutes = (s.index.max() - s.index.min()).total_seconds() / 60
        row["expected_rows"] = int(total_minutes / native_gran) + 1
    else:
        row["expected_rows"] = 0
    row["actual_rows"] = len(s)
    row["coverage_ratio"] = round(
        row["actual_rows"] / row["expected_rows"], 4
    ) if row["expected_rows"] > 0 else 0

    row["nan_count"] = int(s.isna().sum())
    row["nan_ratio"] = round(row["nan_count"] / len(s), 4) if len(s) > 0 else 0
    row["nan_max_consecutive"] = _max_consecutive_nan(s)
    row["nan_max_consecutive_hours"] = round(
        row["nan_max_consecutive"] * native_gran / 60, 1
    )
    row["nan_start_periods"] = _leading_nans(s)
    row["nan_end_periods"] = _trailing_nans(s)
    row["nan_segments"] = _nan_segments(s)

    valid = s.dropna()
    row["valid_start"] = str(valid.index.min()) if len(valid) > 0 else ""
    row["valid_end"] = str(valid.index.max()) if len(valid) > 0 else ""

    for label, (start, end) in [
        ("train", TRAIN_PERIOD), ("test", TEST_PERIOD),
    ]:
        mask = (s.index >= start) & (s.index <= end)
        sub = s.loc[mask]
        if len(sub) > 0:
            row[f"valid_ratio_{label}"] = round(
                1 - sub.isna().sum() / len(sub), 4
            )
        else:
            row[f"valid_ratio_{label}"] = 0.0

    row["dtype"] = str(s.dtype)
    if len(valid) > 0:
        vals = valid.values.astype(float)
        row["min"] = round(float(np.min(vals)), 4)
        row["max"] = round(float(np.max(vals)), 4)
        row["mean"] = round(float(np.mean(vals)), 4)
        row["median"] = round(float(np.median(vals)), 4)
        row["std"] = round(float(np.std(vals)), 4)
        row["q01"] = round(float(np.percentile(vals, 1)), 4)
        row["q99"] = round(float(np.percentile(vals, 99)), 4)
    else:
        for k in ["min", "max", "mean", "median", "std", "q01", "q99"]:
            row[k] = np.nan

    if len(valid) > 0:
        vals = valid.values.astype(float)
        row["n_zero"] = int(np.sum(vals == 0))
        row["n_negative"] = int(np.sum(vals < 0))
        mu, sigma = np.mean(vals), np.std(vals)
        if sigma > 1e-10:
            row["n_outlier_3sigma"] = int(np.sum(np.abs(vals - mu) > 3 * sigma))
        else:
            row["n_outlier_3sigma"] = 0
        n_runs, max_run = _constant_runs(valid, min_length=8)
        row["n_constant_runs"] = n_runs
        row["max_constant_run"] = max_run
    else:
        for k in ["n_zero", "n_negative", "n_outlier_3sigma",
                   "n_constant_runs", "max_constant_run"]:
            row[k] = 0

    return row


def _load_raw_ts(fname: str, date_col: str = "datetime") -> pd.DataFrame:
    path = SOURCE_DIR / fname
    if not path.exists():
        logger.warning("File not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col not in df.columns:
        logger.warning("Date column '%s' not in %s", date_col, fname)
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df


def _get_align_method(native_gran: int) -> str:
    if native_gran == 5:
        return "resample_5to15"
    elif native_gran == 15:
        return "raw"
    elif native_gran == 60:
        return "repeat_hourly"
    elif native_gran >= 1440:
        return "repeat_daily"
    return "raw"


def check_all_features() -> pd.DataFrame:
    rows: List[Dict] = []

    # ── FORMAT_A_SINGLE ──
    for fname, meta in FORMAT_A_SINGLE.items():
        df = _load_raw_ts(fname, meta["date_col"])
        if df.empty:
            continue
        gran = meta["granularity"]
        for orig_col, std_name in meta["value_cols"].items():
            if orig_col not in df.columns:
                logger.warning("Column '%s' not in %s", orig_col, fname)
                continue
            s = pd.to_numeric(df[orig_col], errors="coerce")
            s.name = std_name
            role = V16_ROLES.get(std_name, "unknown")
            row = _compute_quality(
                s, std_name, fname, gran, 15,
                _get_align_method(gran), role,
            )
            rows.append(row)
            logger.info("Checked: %-35s (%s)", std_name, fname)

    # ── FORMAT_A_DUAL (AM/PM 预测) ──
    for fname, meta in FORMAT_A_DUAL.items():
        df = _load_raw_ts(fname, meta["date_col"])
        if df.empty:
            continue
        gran = meta["granularity"]
        for orig_col, (base, period) in meta["value_cols"].items():
            if orig_col not in df.columns:
                logger.warning("Column '%s' not in %s", orig_col, fname)
                continue
            std_name = f"{base}_{period}"
            s = pd.to_numeric(df[orig_col], errors="coerce")
            s.name = std_name
            role = V16_ROLES.get(std_name, "unknown")
            row = _compute_quality(
                s, std_name, fname, gran, 15,
                _get_align_method(gran), role,
            )
            rows.append(row)
            logger.info("Checked: %-35s (%s)", std_name, fname)

    # ── FORMAT_A_CLEARING ──
    for fname, meta in FORMAT_A_CLEARING.items():
        df = _load_raw_ts(fname, meta["date_col"])
        if df.empty:
            continue
        gran = meta["granularity"]
        market = meta["market"]
        for orig_col, metric in meta["value_cols"].items():
            if orig_col not in df.columns:
                logger.warning("Column '%s' not in %s", orig_col, fname)
                continue
            std_name = f"{market}_clearing_{metric}"
            s = pd.to_numeric(df[orig_col], errors="coerce")
            s.name = std_name
            role = V16_ROLES.get(std_name, "unknown")
            row = _compute_quality(
                s, std_name, fname, gran, 15,
                _get_align_method(gran), role,
            )
            rows.append(row)
            logger.info("Checked: %-35s (%s)", std_name, fname)

    # ── FORMAT_A_SETTLEMENT ──
    for fname, meta in FORMAT_A_SETTLEMENT.items():
        df = _load_raw_ts(fname, meta["date_col"])
        if df.empty:
            continue
        gran = meta["granularity"]
        for orig_col, std_name in meta["value_cols"].items():
            if orig_col not in df.columns:
                logger.warning("Column '%s' not in %s", orig_col, fname)
                continue
            s = pd.to_numeric(df[orig_col], errors="coerce")
            s.name = std_name
            role = V16_ROLES.get(std_name, "unknown")
            row = _compute_quality(
                s, std_name, fname, gran, 15,
                _get_align_method(gran), role,
            )
            rows.append(row)
            logger.info("Checked: %-35s (%s)", std_name, fname)

    # ── FORMAT_B (断面约束) ──
    for fname, meta in FORMAT_B.items():
        path = SOURCE_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        date_col = meta["date_col"]
        if date_col not in df.columns:
            continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        gran = meta["granularity"]

        entity_cols = meta["entity_cols"]
        value_col = meta["value_col"]

        for section_name in KEY_SECTIONS:
            for device_type in ["实际潮流", "限额"]:
                mask = (
                    (df[entity_cols[0]] == section_name)
                    & (df[entity_cols[1]] == device_type)
                )
                sub = df.loc[mask].copy()
                if sub.empty:
                    continue
                sub = sub.set_index(date_col).sort_index()
                s = pd.to_numeric(sub[value_col], errors="coerce")
                safe_section = section_name[:20]
                std_name = f"section_{device_type}_{safe_section}"
                s.name = std_name
                row = _compute_quality(
                    s, std_name, fname, gran, 15,
                    "repeat_hourly", "past_observed",
                )
                rows.append(row)
                logger.info("Checked: %-35s (%s)", std_name, fname)

    # ── FORMAT_D (日级数据) ──
    for fname, meta in FORMAT_D.items():
        if not meta.get("ingest", False):
            continue
        path = SOURCE_DIR / fname
        if not path.exists():
            continue

        date_col = meta.get("date_col")
        if date_col is None:
            continue

        if fname == "发输变电检修计划.csv":
            _check_maintenance(path, date_col, rows)
            continue

        value_col = meta.get("value_col")
        std_name = meta.get("metric")
        if value_col is None or std_name is None:
            continue

        df = pd.read_csv(path)
        if date_col not in df.columns or value_col not in df.columns:
            continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        s = pd.to_numeric(df[value_col], errors="coerce")
        s.name = std_name
        role = V16_ROLES.get(std_name, "unknown")
        row = _compute_quality(
            s, std_name, fname, 1440, 15,
            "repeat_daily", role,
        )
        rows.append(row)
        logger.info("Checked: %-35s (%s)", std_name, fname)

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    return report


def _check_maintenance(path: Path, date_col: str, rows: List[Dict]):
    df = pd.read_csv(path)
    if date_col not in df.columns:
        return
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if "设备类型" not in df.columns:
        return

    for equip_type, std_name in [
        ("发电", "maintenance_gen_count"),
        ("电网", "maintenance_grid_count"),
    ]:
        sub = df[df["设备类型"].str.contains(equip_type, na=False)]
        if sub.empty:
            continue
        daily_count = sub.groupby(df.loc[sub.index, date_col].dt.date).size()
        daily_count.index = pd.to_datetime(daily_count.index)
        daily_count.name = std_name

        if daily_count.empty or pd.isna(daily_count.index.min()):
            continue
        all_dates = pd.date_range(
            daily_count.index.min(), daily_count.index.max(), freq="D"
        )
        daily_count = daily_count.reindex(all_dates, fill_value=0)

        role = V16_ROLES.get(std_name, "unknown")
        row = _compute_quality(
            daily_count, std_name, "发输变电检修计划.csv", 1440, 15,
            "repeat_daily", role,
        )
        rows.append(row)
        logger.info("Checked: %-35s (发输变电检修计划.csv)", std_name)


def run_report():
    report = check_all_features()
    out_path = OUTPUT_DIR / "data_quality_report.csv"
    report.to_csv(out_path, index=False)
    logger.info("Data quality report saved: %s (%d features)", out_path, len(report))
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_report()
