"""
Shape 评估指标模块 — 日内曲线形状指标 + 汇总函数。

基础 6 项：
  1. daily_profile_corr     — 逐日 24h Pearson 相关系数均值
  2. normalized_profile_mae — 每日 z-score 标准化后 MAE
  3. peak_hour_error        — 峰值时刻偏差
  4. valley_hour_error      — 谷值时刻偏差
  5. amplitude_error        — 振幅偏差
  6. direction_accuracy     — 相邻差分符号一致率

V7 扩展：
  7-8. turning_point_*      — 转折点匹配率与平均偏移
  9-11. block_*             — 分块 MAE / 分块振幅误差 / 块间排序一致率
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _split_daily(
    actual: np.ndarray,
    pred: np.ndarray,
    index: pd.DatetimeIndex,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """将连续序列拆成逐日 (actual_24h, pred_24h) 对，只保留完整 24h 日。"""
    dates = index.date
    pairs = []
    for d in sorted(set(dates)):
        mask = dates == d
        if mask.sum() == 24:
            pairs.append((actual[mask], pred[mask]))
    return pairs


def daily_profile_corr(actual: np.ndarray, pred: np.ndarray,
                       index: pd.DatetimeIndex) -> float:
    """逐日 24h Pearson 相关系数，取所有完整日的平均。"""
    corrs = []
    for a, p in _split_daily(actual, pred, index):
        if np.std(a) > 1e-6 and np.std(p) > 1e-6:
            c = np.corrcoef(a, p)[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan


def normalized_profile_mae(actual: np.ndarray, pred: np.ndarray,
                           index: pd.DatetimeIndex) -> float:
    """每日 24 点做 z-score 标准化后计算 MAE，反映纯形状偏差。"""
    maes = []
    for a, p in _split_daily(actual, pred, index):
        a_std, p_std = np.std(a), np.std(p)
        if a_std > 1e-6 and p_std > 1e-6:
            a_z = (a - np.mean(a)) / a_std
            p_z = (p - np.mean(p)) / p_std
            maes.append(float(np.mean(np.abs(a_z - p_z))))
    return float(np.mean(maes)) if maes else np.nan


def peak_hour_error(actual: np.ndarray, pred: np.ndarray,
                    index: pd.DatetimeIndex) -> float:
    """|argmax(pred) - argmax(actual)| 的逐日平均。"""
    errors = []
    for a, p in _split_daily(actual, pred, index):
        errors.append(abs(int(np.argmax(p)) - int(np.argmax(a))))
    return float(np.mean(errors)) if errors else np.nan


def valley_hour_error(actual: np.ndarray, pred: np.ndarray,
                      index: pd.DatetimeIndex) -> float:
    """|argmin(pred) - argmin(actual)| 的逐日平均。"""
    errors = []
    for a, p in _split_daily(actual, pred, index):
        errors.append(abs(int(np.argmin(p)) - int(np.argmin(a))))
    return float(np.mean(errors)) if errors else np.nan


def amplitude_error(actual: np.ndarray, pred: np.ndarray,
                    index: pd.DatetimeIndex) -> float:
    """振幅偏差 |(max-min)_pred - (max-min)_actual| 的逐日平均。"""
    errors = []
    for a, p in _split_daily(actual, pred, index):
        amp_a = float(np.max(a) - np.min(a))
        amp_p = float(np.max(p) - np.min(p))
        errors.append(abs(amp_p - amp_a))
    return float(np.mean(errors)) if errors else np.nan


def direction_accuracy(actual: np.ndarray, pred: np.ndarray,
                       index: pd.DatetimeIndex) -> float:
    """相邻差分符号一致率：每日 23 对差分，取符号匹配比例。"""
    accs = []
    for a, p in _split_daily(actual, pred, index):
        diff_a = np.diff(a)
        diff_p = np.diff(p)
        nonzero = np.abs(diff_a) > 1e-8
        if nonzero.sum() > 0:
            match = np.sign(diff_a[nonzero]) == np.sign(diff_p[nonzero])
            accs.append(float(np.mean(match)))
    return float(np.mean(accs)) if accs else np.nan


def _turning_points(x: np.ndarray, min_abs_diff: float = 1e-6) -> List[int]:
    """一阶差分变号位置（拐点候选），返回小时索引列表。"""
    d = np.diff(x.astype(float))
    pts: List[int] = []
    for i in range(1, len(d)):
        if abs(d[i - 1]) < min_abs_diff or abs(d[i]) < min_abs_diff:
            continue
        if d[i - 1] * d[i] < 0:
            pts.append(i)
    return pts


def turning_point_match_rate(actual: np.ndarray, pred: np.ndarray,
                             index: pd.DatetimeIndex,
                             tol_hours: int = 2) -> float:
    """对每日：实际拐点与预测拐点在 tol 小时内匹配的比例（对实际每个拐点找最近预测）。"""
    rates = []
    for a, p in _split_daily(actual, pred, index):
        ta = _turning_points(a)
        tp = _turning_points(p)
        if not ta:
            continue
        matched = 0
        for t in ta:
            if any(abs(t - u) <= tol_hours for u in tp):
                matched += 1
        rates.append(matched / len(ta))
    return float(np.mean(rates)) if rates else np.nan


def turning_point_offset_mean(actual: np.ndarray, pred: np.ndarray,
                              index: pd.DatetimeIndex,
                              tol_hours: int = 2) -> float:
    """匹配到的拐点对之间的平均绝对时刻偏差。"""
    offsets = []
    for a, p in _split_daily(actual, pred, index):
        ta = _turning_points(a)
        tp = _turning_points(p)
        if not ta or not tp:
            continue
        for t in ta:
            best = min(abs(t - u) for u in tp)
            if best <= tol_hours:
                offsets.append(float(best))
    return float(np.mean(offsets)) if offsets else np.nan


# 分块：凌晨 / 上午峰 / 午间 / 晚峰 / 夜间
BLOCK_SLICES = {
    "dawn": slice(0, 6),      # 0-5
    "morning": slice(6, 12),  # 6-11
    "noon": slice(12, 17),    # 12-16
    "evening": slice(17, 22), # 17-21
    "night": slice(22, 24),   # 22-23
}


def block_mean_mae(actual: np.ndarray, pred: np.ndarray,
                   index: pd.DatetimeIndex) -> Dict[str, float]:
    """各段块内均价 MAE 的逐日平均，再对各日取平均。"""
    out = {f"block_mae_{k}": [] for k in BLOCK_SLICES}
    for a, p in _split_daily(actual, pred, index):
        for name, sl in BLOCK_SLICES.items():
            aa = a[sl]
            pp = p[sl]
            if len(aa) == 0:
                continue
            out[f"block_mae_{name}"].append(float(np.mean(np.abs(aa - pp))))
    return {
        k: float(np.mean(v)) if v else np.nan
        for k, v in out.items()
    }


def block_amplitude_error(actual: np.ndarray, pred: np.ndarray,
                          index: pd.DatetimeIndex) -> Dict[str, float]:
    """各段块内 (max-min) 的振幅误差逐日平均。"""
    out = {f"block_amp_err_{k}": [] for k in BLOCK_SLICES}
    for a, p in _split_daily(actual, pred, index):
        for name, sl in BLOCK_SLICES.items():
            aa = a[sl]
            pp = p[sl]
            if len(aa) < 2:
                continue
            amp_a = float(np.max(aa) - np.min(aa))
            amp_p = float(np.max(pp) - np.min(pp))
            out[f"block_amp_err_{name}"].append(abs(amp_p - amp_a))
    return {
        k: float(np.mean(v)) if v else np.nan
        for k, v in out.items()
    }


def block_rank_accuracy(actual: np.ndarray, pred: np.ndarray,
                        index: pd.DatetimeIndex) -> float:
    """五段块均价排序是否与真实一致（Kendall tau 简化为排序完全一致比例）。"""
    names = list(BLOCK_SLICES.keys())
    accs = []
    for a, p in _split_daily(actual, pred, index):
        ma = [float(np.mean(a[BLOCK_SLICES[n]])) for n in names]
        mp = [float(np.mean(p[BLOCK_SLICES[n]])) for n in names]
        ra = np.argsort(np.argsort(ma))
        rp = np.argsort(np.argsort(mp))
        accs.append(float(np.mean(ra == rp)))
    return float(np.mean(accs)) if accs else np.nan


def compute_shape_report(
    actual: np.ndarray,
    pred: np.ndarray,
    index: pd.DatetimeIndex,
    include_v7: bool = False,
) -> Dict[str, float]:
    """汇总 shape 指标；include_v7 为 True 时附加转折点与分块指标（V7）。"""
    base = {
        "profile_corr": round(daily_profile_corr(actual, pred, index), 4),
        "norm_profile_mae": round(normalized_profile_mae(actual, pred, index), 4),
        "peak_hour_err": round(peak_hour_error(actual, pred, index), 2),
        "valley_hour_err": round(valley_hour_error(actual, pred, index), 2),
        "amplitude_err": round(amplitude_error(actual, pred, index), 2),
        "direction_acc": round(direction_accuracy(actual, pred, index), 4),
    }
    if not include_v7:
        return base

    tp_rate = turning_point_match_rate(actual, pred, index)
    tp_off = turning_point_offset_mean(actual, pred, index)
    base["turn_point_match_rate"] = round(tp_rate, 4) if np.isfinite(tp_rate) else np.nan
    base["turn_point_offset_mean"] = round(tp_off, 3) if np.isfinite(tp_off) else np.nan
    base["block_rank_acc"] = round(block_rank_accuracy(actual, pred, index), 4)

    for k, v in block_mean_mae(actual, pred, index).items():
        base[k] = round(v, 2) if np.isfinite(v) else np.nan
    for k, v in block_amplitude_error(actual, pred, index).items():
        base[k] = round(v, 2) if np.isfinite(v) else np.nan
    return base
