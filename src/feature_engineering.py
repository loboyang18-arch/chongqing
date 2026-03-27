"""
特征工程模块 — 从 dws_hourly_features.csv 构建 DA/RT 电价预测训练数据集。

核心原则：严格按数据可获取时间设定滞后，防止未来信息泄漏。
  - lag0：D日预测时已知（负荷/出力/新能源预测、检修计划、日历特征）
  - lag1：D-1日可用（出清价格类，shift 24h）
  - lag2：D-2日可用（实际运行值、结算价，shift 48h）

产出：
  - output/feature_da.csv — 日前电价预测训练集
  - output/feature_rt.csv — 实时电价预测训练集
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, SOURCE_DIR

logger = logging.getLogger(__name__)

# ── 峰谷时段定义（重庆电力市场） ──────────────────────────
PEAK_HOURS = set(range(8, 12)) | set(range(17, 21))
VALLEY_HOURS = set(range(0, 8)) | {23}

# ── 有效数据窗口 ─────────────────────────────────────────
EFFECTIVE_START = "2025-11-01"
EFFECTIVE_END = "2026-03-10 23:00:00"

# ── lag0: D日可用 — 预测值 + 检修（无需 shift） ──────────
LAG0_DIRECT = [
    "load_forecast",
    "total_gen_fcst_am",
    "hydro_gen_fcst_am",
    "non_market_gen_fcst_am",
    "renewable_fcst",
    "renewable_fcst_solar_am",
    "renewable_fcst_wind_am",
    "renewable_fcst_total_am",
    "tie_line_fcst_am",
    "total_gen_fcst_pm",
    "hydro_gen_fcst_pm",
    "non_market_gen_fcst_pm",
    "renewable_fcst_solar_pm",
    "renewable_fcst_total_pm",
    "renewable_fcst_wind_pm",
    "tie_line_fcst_pm",
    "maintenance_gen_count",
    "maintenance_grid_count",
]

# ── lag1: D-1日可用 — 出清/可靠性价格 + 日前均价（shift 24h）
LAG1_SHIFT24 = [
    "da_clearing_price",
    "rt_clearing_price",
    "da_reliability_clearing_price",
    "reliability_da_price",
    "reliability_rt_price",
    "da_avg_clearing_price",
    "da_clearing_power",
    "da_clearing_unit_count",
    "rt_clearing_volume",
    "rt_clearing_unit_count",
]

# ── lag2: D-2日可用 — 实际值 + 结算价 + 均价（shift 48h）──
LAG2_SHIFT48 = [
    "actual_load",
    "total_gen",
    "hydro_gen",
    "non_market_gen",
    "renewable_gen",
    "tie_line_power",
    "section_ratio_C-500-洪板断面送板桥-全接线【常】",
    "section_ratio_C-500-资铜断面送铜梁-全接线【常】",
    "settlement_da_price",
    "settlement_rt_price",
    "avg_bid_price",
    "rt_avg_clearing_price",
    "actual_load_std",
    "actual_load_range",
    "actual_load_max_ramp",
    "renewable_gen_std",
    "renewable_gen_range",
    "tie_line_power_std",
    "tie_line_power_range",
    "tie_line_power_max_ramp",
]

# ── RT 模型独有 lag0（D日日前出清结果，D-1 17:30 已公布）──
RT_EXTRA_LAG0 = [
    "da_clearing_price",
    "da_clearing_power",
]


def load_hourly_features(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = OUTPUT_DIR / "dws_hourly_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    logger.info(
        "Loaded: %d rows × %d cols (%s ~ %s)",
        len(df), len(df.columns), df.index.min(), df.index.max(),
    )
    return df


def _add_calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": idx.hour,
            "day_of_week": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "month": idx.month,
        },
        index=idx,
    )


def _build_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建 DA/RT 共用的滞后特征 + 衍生特征 + 日历特征。"""
    feat = pd.DataFrame(index=df.index)

    # ── lag0 ──────────────────────────────────────
    for col in LAG0_DIRECT:
        if col in df.columns:
            feat[col] = df[col]

    # ── lag1 (shift 24h) ─────────────────────────
    for col in LAG1_SHIFT24:
        if col in df.columns:
            feat[f"{col}_lag24h"] = df[col].shift(24)

    # ── lag2 (shift 48h) ─────────────────────────
    for col in LAG2_SHIFT48:
        if col in df.columns:
            feat[f"{col}_lag48h"] = df[col].shift(48)

    # ── 断面触限标志（基于 lag48h 断面利用率）──────
    for col in list(feat.columns):
        if col.startswith("section_ratio_") and col.endswith("_lag48h"):
            flag_name = col.replace("section_ratio_", "section_congested_").replace("_lag48h", "_flag_lag48h")
            feat[flag_name] = (feat[col] > 0.9).astype(int)

    # ── 日历特征 ──────────────────────────────────
    feat = pd.concat([feat, _add_calendar(df.index)], axis=1)

    # ── 衍生特征：供需 ───────────────────────────
    if "load_forecast" in feat.columns and "total_gen_fcst_am" in feat.columns:
        feat["supply_demand_gap"] = feat["load_forecast"] - feat["total_gen_fcst_am"]

    if "renewable_fcst_total_am" in feat.columns and "load_forecast" in feat.columns:
        feat["renewable_ratio"] = (
            feat["renewable_fcst_total_am"]
            / feat["load_forecast"].replace(0, np.nan)
        )

    if "renewable_fcst_total_am" in feat.columns and "load_forecast" in feat.columns:
        feat["net_load_forecast"] = feat["load_forecast"] - feat["renewable_fcst_total_am"]

    if "load_forecast" in feat.columns:
        feat["load_x_hour"] = feat["load_forecast"] * feat["hour"]

    # ── DA 价格多级滞后 ──────────────────────────
    if "da_clearing_price" in df.columns:
        da = df["da_clearing_price"]
        feat["da_price_lag48h"] = da.shift(48)
        feat["da_price_lag72h"] = da.shift(72)
        feat["da_price_lag96h"] = da.shift(96)
        feat["da_price_lag168h"] = da.shift(168)

    # ── RT 价格多级滞后 ──────────────────────────
    if "rt_clearing_price" in df.columns:
        rt = df["rt_clearing_price"]
        feat["rt_price_lag48h"] = rt.shift(48)
        feat["rt_price_lag168h"] = rt.shift(168)

    # ── DA 价格滚动统计（基于 lag24h 序列，无泄漏）
    if "da_clearing_price" in df.columns:
        lagged_24 = df["da_clearing_price"].shift(24)
        feat["da_price_roll24h_mean"] = lagged_24.rolling(24, min_periods=12).mean()
        feat["da_price_roll24h_std"] = lagged_24.rolling(24, min_periods=12).std()
        feat["da_price_roll24h_min"] = lagged_24.rolling(24, min_periods=12).min()
        feat["da_price_roll24h_max"] = lagged_24.rolling(24, min_periods=12).max()
        feat["da_price_roll48h_mean"] = lagged_24.rolling(48, min_periods=24).mean()
        feat["da_price_roll48h_std"] = lagged_24.rolling(48, min_periods=24).std()
        feat["da_price_roll168h_mean"] = lagged_24.rolling(168, min_periods=72).mean()
        feat["da_price_roll168h_std"] = lagged_24.rolling(168, min_periods=72).std()

    # ── RT 价格滚动统计 ──────────────────────────
    if "rt_clearing_price" in df.columns:
        rt_lagged_24 = df["rt_clearing_price"].shift(24)
        feat["rt_price_roll24h_mean"] = rt_lagged_24.rolling(24, min_periods=12).mean()
        feat["rt_price_roll24h_std"] = rt_lagged_24.rolling(24, min_periods=12).std()

    # ── 同小时历史统计（按 hour 分组后 rolling，基于 lag24h）
    if "da_clearing_price" in df.columns:
        lagged_24 = df["da_clearing_price"].shift(24)
        grouped = lagged_24.groupby(df.index.hour)
        feat["da_price_same_hour_7d_mean"] = grouped.transform(
            lambda x: x.rolling(7, min_periods=3).mean()
        )
        feat["da_price_same_hour_7d_std"] = grouped.transform(
            lambda x: x.rolling(7, min_periods=3).std()
        )

    # ── 价格变化率特征 ────────────────────────────
    if "da_clearing_price" in df.columns:
        da = df["da_clearing_price"]
        feat["da_price_diff_24h"] = da.shift(24) - da.shift(48)
        lag168 = da.shift(168)
        feat["da_price_ratio_24h_168h"] = da.shift(24) / lag168.replace(0, np.nan)

    # ── 供需缺口滞后 ─────────────────────────────
    if "supply_demand_gap" in feat.columns:
        feat["supply_demand_gap_lag24h"] = feat["supply_demand_gap"].shift(24)

    # ── Ramp 特征（8 个）— 基于 lag0 可用数据 ───────
    if "load_forecast" in feat.columns:
        feat["load_ramp_1h"] = feat["load_forecast"].diff(1)
        feat["load_ramp_24h"] = (
            feat["load_forecast"] - feat["load_forecast"].shift(24)
        )

    if "renewable_fcst_total_am" in feat.columns:
        feat["renewable_ramp_1h"] = feat["renewable_fcst_total_am"].diff(1)
        feat["renewable_ramp_24h"] = (
            feat["renewable_fcst_total_am"]
            - feat["renewable_fcst_total_am"].shift(24)
        )

    if "net_load_forecast" in feat.columns:
        feat["net_load_ramp_1h"] = feat["net_load_forecast"].diff(1)
        feat["net_load_ramp_24h"] = (
            feat["net_load_forecast"] - feat["net_load_forecast"].shift(24)
        )

    if "tie_line_fcst_am" in feat.columns:
        feat["tie_line_ramp_1h"] = feat["tie_line_fcst_am"].diff(1)

    if "supply_demand_gap" in feat.columns:
        feat["supply_gap_ramp_1h"] = feat["supply_demand_gap"].diff(1)

    # ── Flag 特征（5 个）────────────────────────────
    feat["peak_flag"] = feat.index.hour.isin(PEAK_HOURS).astype(int)
    feat["valley_flag"] = feat.index.hour.isin(VALLEY_HOURS).astype(int)

    if "renewable_ratio" in feat.columns:
        rr_expanding_q75 = (
            feat["renewable_ratio"]
            .expanding(min_periods=48)
            .quantile(0.75)
            .shift(1)
        )
        feat["high_renewable_flag"] = (
            feat["renewable_ratio"] > rr_expanding_q75
        ).astype(int)

    if "da_price_roll24h_mean" in feat.columns:
        roll_expanding_q10 = (
            feat["da_price_roll24h_mean"]
            .expanding(min_periods=48)
            .quantile(0.10)
            .shift(1)
        )
        feat["low_price_risk_flag"] = (
            feat["da_price_roll24h_mean"] < roll_expanding_q10
        ).astype(int)

    if "supply_demand_gap" in feat.columns:
        gap_abs = feat["supply_demand_gap"].abs()
        gap_expanding_q90 = (
            gap_abs.expanding(min_periods=48).quantile(0.90).shift(1)
        )
        feat["extreme_gap_flag"] = (gap_abs > gap_expanding_q90).astype(int)

    # ── 形状先验特征（3 个）— 基于昨日 DA 价格 ──────
    if "da_clearing_price" in df.columns:
        da_lag24 = df["da_clearing_price"].shift(24)
        by_date = da_lag24.groupby(da_lag24.index.date)

        da_lag24_peak = da_lag24.where(
            da_lag24.index.hour.isin(PEAK_HOURS)
        )
        feat["da_price_peak_mean_lag1d"] = (
            da_lag24_peak.groupby(da_lag24_peak.index.date).transform("mean")
        )

        da_lag24_valley = da_lag24.where(
            da_lag24.index.hour.isin(VALLEY_HOURS)
        )
        feat["da_price_valley_mean_lag1d"] = (
            da_lag24_valley.groupby(da_lag24_valley.index.date)
            .transform("mean")
        )

        feat["da_price_amplitude_lag1d"] = (
            by_date.transform("max") - by_date.transform("min")
        )

    return feat


def _add_template_shape_features(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """V7/V7b：基于历史同类日构建日内偏离模板（仅用当前行日期之前的日历日，无泄漏）。

    产出：
      - template_dev_h0..h23：7 天同类日模板（与 RT 兼容）
      - template_global_h* / template_14d_h* / template_7d_h*：三级分解
      - template_blend_h*：0.25*global + 0.35*14d + 0.40*7d
      - template_n_global / template_n_14d / template_n_7d：各层有效样本日数
      - template_amplitude 等：基于融合模板
    """
    if "da_clearing_price" not in df.columns:
        return feat

    # 避免碎片化 DataFrame 上逐列 insert 导致后续 loc 赋值静默失败
    feat = feat.copy()

    da = df["da_clearing_price"]
    dates_sorted = sorted(set(da.index.date))
    profiles: Dict = {}
    for d in dates_sorted:
        mask = da.index.date == d
        if mask.sum() != 24:
            continue
        sub = da.loc[mask].sort_index()
        dm = float(sub.mean())
        arr = sub.values.astype(float) - dm
        if len(arr) == 24:
            profiles[d] = arr

    weekend_map = {d: int(pd.Timestamp(d).dayofweek >= 5) for d in dates_sorted}

    col_lists = (
        [f"template_dev_h{h}" for h in range(24)]
        + [f"template_global_h{h}" for h in range(24)]
        + [f"template_14d_h{h}" for h in range(24)]
        + [f"template_7d_h{h}" for h in range(24)]
        + [f"template_blend_h{h}" for h in range(24)]
    )
    extra_meta = (
        "template_amplitude",
        "template_peak_hour",
        "template_valley_hour",
        "template_blend_amplitude",
        "template_blend_peak_hour",
        "template_blend_valley_hour",
        "template_n_global",
        "template_n_14d",
        "template_n_7d",
    )
    tmpl_block = pd.DataFrame(
        {c: np.nan for c in col_lists + list(extra_meta)},
        index=feat.index,
    )
    feat = pd.concat([feat, tmpl_block], axis=1)

    def _mean_profile(cand_dates):
        vecs = [profiles[p] for p in cand_dates if p in profiles]
        if not vecs:
            return np.zeros(24)
        stack = np.stack(vecs, axis=0)
        # 个别小时缺测时 np.mean 会整点变 nan；用 nanmean 并填 0
        m = np.nanmean(stack, axis=0)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    a_w, b_w, c_w = 0.25, 0.35, 0.40

    for d in dates_sorted:
        mask = feat.index.date == d
        if not mask.any():
            continue
        sw = weekend_map[d]
        past_sw = [x for x in dates_sorted if x < d and weekend_map[x] == sw]
        past_any = [x for x in dates_sorted if x < d]

        past_7 = past_sw[-7:] if past_sw else []
        past_14 = past_sw[-14:] if past_sw else []
        tmpl_7 = _mean_profile(past_7)
        if np.allclose(tmpl_7, 0) and past_any:
            past_7_fb = past_any[-7:]
            tmpl_7 = _mean_profile(past_7_fb)
            n7 = len([p for p in past_7_fb if p in profiles])
        else:
            n7 = len([p for p in past_7 if p in profiles])

        tmpl_14 = _mean_profile(past_14) if past_14 else tmpl_7
        n14 = len([p for p in past_14 if p in profiles]) if past_14 else 0

        tmpl_global = _mean_profile(past_sw) if past_sw else tmpl_7
        ng = len([p for p in past_sw if p in profiles])

        blend = a_w * tmpl_global + b_w * tmpl_14 + c_w * tmpl_7

        for h in range(24):
            feat.loc[mask, f"template_dev_h{h}"] = tmpl_7[h]
            feat.loc[mask, f"template_global_h{h}"] = tmpl_global[h]
            feat.loc[mask, f"template_14d_h{h}"] = tmpl_14[h]
            feat.loc[mask, f"template_7d_h{h}"] = tmpl_7[h]
            feat.loc[mask, f"template_blend_h{h}"] = blend[h]

        # RT 兼容：峰谷振幅仍基于 7d 模板
        feat.loc[mask, "template_amplitude"] = float(np.max(tmpl_7) - np.min(tmpl_7))
        feat.loc[mask, "template_peak_hour"] = float(np.argmax(tmpl_7))
        feat.loc[mask, "template_valley_hour"] = float(np.argmin(tmpl_7))
        feat.loc[mask, "template_blend_amplitude"] = float(np.max(blend) - np.min(blend))
        feat.loc[mask, "template_blend_peak_hour"] = float(np.argmax(blend))
        feat.loc[mask, "template_blend_valley_hour"] = float(np.argmin(blend))
        feat.loc[mask, "template_n_global"] = float(ng)
        feat.loc[mask, "template_n_14d"] = float(n14)
        feat.loc[mask, "template_n_7d"] = float(n7)

    return feat


_SUB_HOUR_RAW_15MIN = {
    "da_clearing_price": ("日前市场交易出清结果.csv", "现货日前市场出清电价"),
    "rt_clearing_price": ("实时出清结果.csv", "现货实时出清电价"),
    "actual_load": ("实际负荷.csv", "实际负荷"),
    "renewable_gen": ("新能源总出力.csv", "新能源总出力"),
    "total_gen": ("发电总出力.csv", "发电总出力"),
    "hydro_gen": ("水电（含抽蓄）总出力.csv", "水电（含抽蓄）总出力"),
}
_SUB_HOUR_RAW_5MIN = {
    "tie_line_power": ("省间联络线输电.csv", "省间联络线输电"),
}


def _load_sub_hour_raw() -> pd.DataFrame:
    """加载子小时原始数据并合并为 15 分钟宽表。"""
    parts = []
    for metric, (fname, col_name) in _SUB_HOUR_RAW_15MIN.items():
        path = SOURCE_DIR / fname
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["datetime"])
        raw = raw.rename(columns={"datetime": "ts", col_name: metric})
        raw[metric] = pd.to_numeric(raw[metric], errors="coerce")
        raw = raw[["ts", metric]].dropna(subset=["ts"]).set_index("ts").sort_index()
        parts.append(raw[[metric]])

    for metric, (fname, col_name) in _SUB_HOUR_RAW_5MIN.items():
        path = SOURCE_DIR / fname
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["datetime"])
        raw = raw.rename(columns={"datetime": "ts", col_name: metric})
        raw[metric] = pd.to_numeric(raw[metric], errors="coerce")
        raw = raw[["ts", metric]].dropna(subset=["ts"]).set_index("ts").sort_index()
        parts.append(raw[[metric]].resample("15min").mean())

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1, join="outer").sort_index()


def _add_sub_hour_shape_features(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """从 D-1 的 15 分钟原始数据中提取日内形状摘要特征 (~18 维)。

    对齐方式：每行 ts 对应预测日 D 的某小时，取 D-1 全天 96 步子小时数据。
    同一天 24 行共享相同的子小时摘要值。
    """
    feat = feat.copy()
    sub_raw = _load_sub_hour_raw()
    if sub_raw.empty:
        logger.warning("Sub-hour raw data not available, skipping shape features")
        return feat

    logger.info("Sub-hour raw loaded: %d rows × %d cols", len(sub_raw), len(sub_raw.columns))

    new_cols = [
        "sub_da_amplitude", "sub_da_morning_rise", "sub_da_midday_drop",
        "sub_da_evening_rise", "sub_da_intraday_std", "sub_da_last_quarter_slope",
        "sub_rt_amplitude", "sub_rt_intraday_std", "sub_da_rt_spread_std",
        "sub_load_morning_ramp", "sub_load_peak_valley_ratio", "sub_load_intraday_std",
        "sub_renew_intraday_ramp", "sub_renew_dawn_dusk_ratio", "sub_renew_volatility",
        "sub_tie_intraday_std", "sub_tie_max_ramp",
        "sub_netload_intraday_std",
    ]
    for c in new_cols:
        feat[c] = np.nan

    all_dates = sorted(set(feat.index.date))

    for d in all_dates:
        d_ts = pd.Timestamp(d)
        d_minus1 = d_ts - pd.Timedelta(days=1)
        sub_start = d_minus1
        sub_end = d_minus1 + pd.Timedelta(hours=23, minutes=45)
        chunk = sub_raw.loc[sub_start:sub_end]

        if len(chunk) < 48:
            continue

        day_mask = feat.index.date == d
        vals = {}

        da = chunk.get("da_clearing_price")
        if da is not None:
            da_v = da.dropna()
            if len(da_v) >= 48:
                vals["sub_da_amplitude"] = float(da_v.max() - da_v.min())
                vals["sub_da_intraday_std"] = float(da_v.std())
                h_means = da_v.resample("1h").mean()
                if len(h_means) >= 21:
                    morning = h_means.iloc[6:10].mean() if len(h_means) > 9 else np.nan
                    dawn = h_means.iloc[0:6].mean() if len(h_means) > 5 else np.nan
                    midday = h_means.iloc[11:15].mean() if len(h_means) > 14 else np.nan
                    evening = h_means.iloc[17:21].mean() if len(h_means) > 20 else np.nan
                    vals["sub_da_morning_rise"] = float(morning - dawn) if np.isfinite(morning) and np.isfinite(dawn) else np.nan
                    vals["sub_da_midday_drop"] = float(morning - midday) if np.isfinite(morning) and np.isfinite(midday) else np.nan
                    vals["sub_da_evening_rise"] = float(evening - midday) if np.isfinite(evening) and np.isfinite(midday) else np.nan
                last_4h = da_v.iloc[-16:]
                if len(last_4h) >= 8:
                    x = np.arange(len(last_4h))
                    vals["sub_da_last_quarter_slope"] = float(np.polyfit(x, last_4h.values, 1)[0])

        rt = chunk.get("rt_clearing_price")
        if rt is not None:
            rt_v = rt.dropna()
            if len(rt_v) >= 48:
                vals["sub_rt_amplitude"] = float(rt_v.max() - rt_v.min())
                vals["sub_rt_intraday_std"] = float(rt_v.std())

        if da is not None and rt is not None:
            spread = (da - rt).dropna()
            if len(spread) >= 48:
                vals["sub_da_rt_spread_std"] = float(spread.std())

        load = chunk.get("actual_load")
        if load is not None:
            load_v = load.dropna()
            if len(load_v) >= 48:
                vals["sub_load_intraday_std"] = float(load_v.std())
                h_load = load_v.resample("1h").mean()
                if len(h_load) >= 12:
                    morning_load = h_load.iloc[6:10].mean() if len(h_load) > 9 else np.nan
                    dawn_load = h_load.iloc[0:6].mean() if len(h_load) > 5 else np.nan
                    vals["sub_load_morning_ramp"] = float(morning_load - dawn_load) if np.isfinite(morning_load) and np.isfinite(dawn_load) else np.nan
                peak_load = load_v.max()
                valley_load = load_v.min()
                vals["sub_load_peak_valley_ratio"] = float(peak_load / valley_load) if valley_load > 0 else np.nan

        renew = chunk.get("renewable_gen")
        if renew is not None:
            renew_v = renew.dropna()
            if len(renew_v) >= 48:
                h_renew = renew_v.resample("1h").mean()
                vals["sub_renew_intraday_ramp"] = float(h_renew.max() - h_renew.min()) if len(h_renew) >= 12 else np.nan
                dawn_r = h_renew.iloc[0:6].mean() if len(h_renew) > 5 else np.nan
                dusk_r = h_renew.iloc[17:21].mean() if len(h_renew) > 20 else np.nan
                vals["sub_renew_dawn_dusk_ratio"] = float(dusk_r / dawn_r) if np.isfinite(dawn_r) and dawn_r > 0 else np.nan
                vals["sub_renew_volatility"] = float(renew_v.diff().abs().mean())

        tie = chunk.get("tie_line_power")
        if tie is not None:
            tie_v = tie.dropna()
            if len(tie_v) >= 48:
                vals["sub_tie_intraday_std"] = float(tie_v.std())
                vals["sub_tie_max_ramp"] = float(tie_v.diff().abs().max())

        if load is not None and renew is not None:
            nl = (chunk.get("actual_load", pd.Series(dtype=float)) - chunk.get("renewable_gen", pd.Series(dtype=float))).dropna()
            if len(nl) >= 48:
                vals["sub_netload_intraday_std"] = float(nl.std())

        for col, val in vals.items():
            feat.loc[day_mask, col] = val

    filled = {c: feat[c].notna().sum() for c in new_cols}
    logger.info("Sub-hour shape features: %d new cols, fill rates: %s",
                len(new_cols),
                {k: f"{v}/{len(feat)}" for k, v in list(filled.items())[:5]})
    return feat


def build_da_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """构建日前电价预测训练数据集。"""
    logger.info("Building DA dataset...")
    feat = _build_common_features(df)
    feat = _add_template_shape_features(df, feat)
    feat = _add_sub_hour_shape_features(df, feat)
    feat["target_da_clearing_price"] = df["da_clearing_price"]

    feat = feat.loc[EFFECTIVE_START:EFFECTIVE_END]
    feat = feat.dropna(subset=["target_da_clearing_price"])

    logger.info("DA dataset: %d rows × %d cols", len(feat), len(feat.columns))
    return feat


def build_rt_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """构建实时电价预测训练数据集。"""
    logger.info("Building RT dataset...")
    feat = _build_common_features(df)
    feat = _add_template_shape_features(df, feat)

    for col in RT_EXTRA_LAG0:
        if col in df.columns:
            feat[f"{col}_d0"] = df[col]

    # ── RT-DA spread 特征（4 个）───────────────────
    rt_lag24 = feat.get("rt_clearing_price_lag24h")
    da_lag24 = feat.get("da_clearing_price_lag24h")
    if rt_lag24 is not None and da_lag24 is not None:
        spread_lag24 = rt_lag24 - da_lag24
        feat["rt_da_spread_lag24h"] = spread_lag24
        feat["rt_da_spread_roll24h_mean"] = (
            spread_lag24.rolling(24, min_periods=12).mean()
        )
        feat["rt_da_spread_roll24h_std"] = (
            spread_lag24.rolling(24, min_periods=12).std()
        )

    rt_lag48 = feat.get("rt_price_lag48h")
    da_lag48 = feat.get("da_price_lag48h")
    if rt_lag48 is not None and da_lag48 is not None:
        feat["rt_da_spread_lag48h"] = rt_lag48 - da_lag48

    feat["target_rt_clearing_price"] = df["rt_clearing_price"]

    feat = feat.loc[EFFECTIVE_START:EFFECTIVE_END]
    feat = feat.dropna(subset=["target_rt_clearing_price"])

    logger.info("RT dataset: %d rows × %d cols", len(feat), len(feat.columns))
    return feat


def validate_no_leakage(dataset: pd.DataFrame, target_col: str) -> dict:
    """检查 target 与特征的相关性，排查潜在未来泄漏。"""
    feat_cols = [c for c in dataset.columns if c != target_col]
    corr = (
        dataset[feat_cols]
        .corrwith(dataset[target_col])
        .abs()
        .sort_values(ascending=False)
    )

    top10 = corr.head(10)
    suspicious = corr[corr > 0.95]

    if len(suspicious) > 0:
        logger.warning("POTENTIAL LEAKAGE — features with |corr| > 0.95:")
        for name, val in suspicious.items():
            logger.warning("  %s: %.4f", name, val)
    else:
        logger.info("No leakage detected (all |corr| <= 0.95) for %s", target_col)

    logger.info("Top-10 correlated with %s:", target_col)
    for name, val in top10.items():
        logger.info("  %-50s %.4f", name, val)

    return {"top10": top10.to_dict(), "suspicious": suspicious.to_dict()}


def report_coverage(dataset: pd.DataFrame, label: str) -> pd.Series:
    """报告特征覆盖率。"""
    coverage = dataset.notna().mean()
    n_good = int((coverage >= 0.95).sum())

    logger.info("[%s] Coverage: %d/%d features >= 95%%", label, n_good, len(coverage))
    low = coverage[coverage < 0.95].sort_values()
    if len(low) > 0:
        for col, val in low.items():
            logger.warning("  %s: %.1f%%", col, val * 100)

    return coverage


def run_feature_engineering(
    input_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """主入口：加载数据 → 构建 DA/RT 特征集 → 验证 → 保存。"""
    df = load_hourly_features(input_path)

    da = build_da_dataset(df)
    rt = build_rt_dataset(df)

    logger.info("=" * 60)
    logger.info("VALIDATION: leakage check & coverage report")
    logger.info("=" * 60)
    validate_no_leakage(da, "target_da_clearing_price")
    validate_no_leakage(rt, "target_rt_clearing_price")
    report_coverage(da, "DA")
    report_coverage(rt, "RT")

    da.to_csv(OUTPUT_DIR / "feature_da.csv")
    rt.to_csv(OUTPUT_DIR / "feature_rt.csv")
    logger.info(
        "Saved: feature_da.csv (%d×%d), feature_rt.csv (%d×%d)",
        len(da), len(da.columns), len(rt), len(rt.columns),
    )

    return da, rt


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_feature_engineering()
