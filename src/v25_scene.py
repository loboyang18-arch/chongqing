"""
V25 日级场景：Lag 可观测规则打标 → normal / low（可扩展 high、volatile）。

预测日 D 的标签仅使用 D-1 及更早的日前出清价（market_clearing_price 小时均值）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .experiment.splits import TRAIN_END, VAL_END, TEST_START, TEST_END
from .model_v24_da import load_sql_feature_matrix

DA_COL = "market_clearing_price"
LOOKBACK_DAYS = 7


@dataclass
class SceneThresholds:
    p_low: float
    p_high: float
    p_vol: float
    lookback_days: int = LOOKBACK_DAYS

    def to_dict(self) -> dict:
        return {
            "p_low": self.p_low,
            "p_high": self.p_high,
            "p_vol": self.p_vol,
            "lookback_days": self.lookback_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneThresholds":
        return cls(
            p_low=float(d["p_low"]),
            p_high=float(d["p_high"]),
            p_vol=float(d["p_vol"]),
            lookback_days=int(d.get("lookback_days", LOOKBACK_DAYS)),
        )


def build_daily_stats() -> pd.DataFrame:
    """日表：daily_mean, daily_amp（hourly 聚合）。"""
    df = load_sql_feature_matrix()
    if DA_COL not in df.columns:
        raise KeyError(f"Missing {DA_COL}")
    raw = df[DA_COL].dropna()
    hourly = raw.resample("1h").mean()
    daily_mean = hourly.resample("1D").mean()
    daily_amp = hourly.resample("1D").apply(lambda x: float(x.max() - x.min()) if len(x) else np.nan)
    out = pd.DataFrame({
        "daily_mean": daily_mean,
        "daily_amp": daily_amp,
    })
    out.index = pd.to_datetime(out.index).date
    return out


def _lag7_mean(series: pd.Series, day: date, lookback: int) -> float:
    d = pd.Timestamp(day)
    vals = []
    for k in range(1, lookback + 1):
        pd_d = (d - pd.Timedelta(days=k)).date()
        if pd_d in series.index:
            v = series.loc[pd_d]
            if np.isfinite(v):
                vals.append(float(v))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def fit_thresholds(
    fit_days: Sequence[date],
    daily: pd.DataFrame,
    q_low: float = 0.20,
    q_high: float = 0.80,
    q_vol: float = 0.80,
    lookback: int = LOOKBACK_DAYS,
) -> SceneThresholds:
    mu7_list, amp7_list = [], []
    for d in fit_days:
        mu7 = _lag7_mean(daily["daily_mean"], d, lookback)
        amp7 = _lag7_mean(daily["daily_amp"], d, lookback)
        if np.isfinite(mu7):
            mu7_list.append(mu7)
        if np.isfinite(amp7):
            amp7_list.append(amp7)
    if len(mu7_list) < 10:
        raise ValueError(f"Too few fit samples for thresholds: {len(mu7_list)}")
    return SceneThresholds(
        p_low=float(np.quantile(mu7_list, q_low)),
        p_high=float(np.quantile(mu7_list, q_high)),
        p_vol=float(np.quantile(amp7_list, q_vol)) if amp7_list else float("inf"),
        lookback_days=lookback,
    )


def classify_day(
    day: date,
    daily: pd.DataFrame,
    th: SceneThresholds,
) -> str:
    """
    返回场景：low | high | volatile | normal（volatile/high 暂映射到专模时见 SCENE_TO_EXPERT）。
    """
    mu7 = _lag7_mean(daily["daily_mean"], day, th.lookback_days)
    amp7 = _lag7_mean(daily["daily_amp"], day, th.lookback_days)
    if not np.isfinite(mu7):
        return "normal"
    if np.isfinite(amp7) and amp7 >= th.p_vol:
        return "volatile"
    if mu7 < th.p_low:
        return "low"
    if mu7 > th.p_high:
        return "high"
    return "normal"


# 两专家路由：仅 low 独立，其余走 normal
SCENE_TO_EXPERT = {
    "low": "low",
    "normal": "normal",
    "high": "normal",
    "volatile": "normal",
}


def label_all_days(
    days: Sequence[date],
    daily: pd.DataFrame,
    th: SceneThresholds,
) -> pd.DataFrame:
    rows = []
    for d in days:
        scene = classify_day(d, daily, th)
        expert = SCENE_TO_EXPERT.get(scene, "normal")
        rows.append({
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "mu7_lag_da": _lag7_mean(daily["daily_mean"], d, th.lookback_days),
            "amp7_lag_da": _lag7_mean(daily["daily_amp"], d, th.lookback_days),
            "scene": scene,
            "expert": expert,
        })
    return pd.DataFrame(rows)


def default_fit_days(merge_val: bool = True) -> List[date]:
    daily = build_daily_stats()
    all_days = sorted(daily.index)
    if merge_val:
        end = VAL_END.date()
    else:
        end = TRAIN_END.date()
    return [d for d in all_days if d <= end]


def default_all_label_days() -> List[date]:
    daily = build_daily_stats()
    end = TEST_END.date()
    return [d for d in sorted(daily.index) if d <= end]


def save_label_artifacts(
    out_dir: Path,
    day_table: pd.DataFrame,
    th: SceneThresholds,
    merge_val: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    day_table.to_csv(out_dir / "day_scene.csv", index=False)
    with (out_dir / "scene_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"thresholds": th.to_dict(), "merge_val_fit": merge_val, "scene_to_expert": SCENE_TO_EXPERT},
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_label_artifacts(root: Path) -> Tuple[pd.DataFrame, SceneThresholds]:
    day_table = pd.read_csv(root / "day_scene.csv")
    with (root / "scene_thresholds.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    th = SceneThresholds.from_dict(meta["thresholds"])
    return day_table, th


def train_dates_for_expert(
    day_table: pd.DataFrame,
    expert: str,
    merge_val: bool = True,
) -> Set[date]:
    df = day_table.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    end = VAL_END.date() if merge_val else TRAIN_END.date()
    sub = df.loc[(df["date"] <= end) & (df["expert"] == expert), "date"]
    return set(sub.tolist())


def test_dates_with_expert(day_table: pd.DataFrame) -> pd.DataFrame:
    df = day_table.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    mask = (df["date"] >= TEST_START.date()) & (df["date"] <= TEST_END.date())
    return df.loc[mask].copy()
