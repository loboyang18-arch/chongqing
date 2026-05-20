#!/usr/bin/env python3
"""
反向推理：统一结算点电价、出清结果（15min→小时）、日度平均出清电价之间的数值关系。

依赖 source_data 下原始 CSV，与主流程列名一致（见 src/config.py）。
运行（仓库根目录）:
  python scripts/reverse_infer_price_relations.py

输出:
  - 终端摘要
  - output/reverse_infer_price_relations.txt
  - output/reverse_infer_daily_compare.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import FORMAT_A_CLEARING, FORMAT_A_SETTLEMENT, SOURCE_DIR
from src.dwd_transform import build_dwd_clearing, build_dwd_daily_price, build_dwd_settlement
from src.dws_aggregate import build_hourly_clearing, build_hourly_settlement
from src.ods_loader import load_format_a_clearing, load_format_a_settlement, load_format_d_daily_prices


def _ols_y_on_x(y: pd.Series, x: pd.Series) -> dict:
    mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(x.to_numpy(dtype=float))
    if mask.sum() < 50:
        return {"n": int(mask.sum()), "a": np.nan, "b": np.nan, "r2": np.nan, "rmse": np.nan}
    yv = y.to_numpy(dtype=float)[mask]
    xv = x.to_numpy(dtype=float)[mask]
    a, b = np.polyfit(xv, yv, 1)
    pred = a * xv + b
    ss_res = np.sum((yv - pred) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(yv)))
    return {"n": int(mask.sum()), "a": float(a), "b": float(b), "r2": float(r2), "rmse": rmse}


def _lag_corrs(s1: pd.Series, s2: pd.Series, lags: range) -> list[tuple[int, float]]:
    """corr(s1.shift(lag), s2) 对齐后皮尔逊相关：lag>0 表示 s1 滞后（更晚）。"""
    df = pd.concat([s1, s2], axis=1).dropna(how="all")
    out = []
    for lag in lags:
        c = df.iloc[:, 0].shift(lag).corr(df.iloc[:, 1])
        out.append((lag, float(c) if np.isfinite(c) else float("nan")))
    return out


def _daily_from_clearing_15m(
    df: pd.DataFrame, price_col: str, weight_col: str | None, prefix: str
) -> pd.DataFrame:
    """按日历日聚合：等权均价、可选电量加权均价（向量化）。"""
    d = df.copy()
    d["day"] = d["ts"].dt.normalize()
    g = d.groupby("day", sort=True)
    out = pd.DataFrame({f"{prefix}_mean15m": g[price_col].mean()})
    if weight_col and weight_col in d.columns:
        d["_pw"] = d[price_col] * d[weight_col]
        num = d.groupby("day", sort=True)["_pw"].sum()
        den = d.groupby("day", sort=True)[weight_col].sum().replace(0, np.nan)
        out[f"{prefix}_wavg"] = num / den
    out.index = pd.to_datetime(out.index)
    return out


def main() -> None:
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"缺少目录: {SOURCE_DIR}")

    da = load_format_a_clearing(
        "日前市场交易出清结果.csv", FORMAT_A_CLEARING["日前市场交易出清结果.csv"]
    )
    rt = load_format_a_clearing("实时出清结果.csv", FORMAT_A_CLEARING["实时出清结果.csv"])
    dwd_c = build_dwd_clearing([da, rt])
    hourly_c = build_hourly_clearing(dwd_c)

    stl = load_format_a_settlement(
        "统一结算点电价.csv", FORMAT_A_SETTLEMENT["统一结算点电价.csv"]
    )
    dwd_s = build_dwd_settlement(stl)
    hourly_s = build_hourly_settlement(dwd_s)

    daily_wide = build_dwd_daily_price(load_format_d_daily_prices())

    h = hourly_c.join(hourly_s, how="inner")
    log("=== 数据覆盖（小时内连接 inner join） ===")
    log(f"  小时样本数: {len(h)}")
    log(f"  时间范围: {h.index.min()} ~ {h.index.max()}")

    # —— A. 结算 vs 出清（小时） ——
    log("\n=== A. 统一结算点电价 vs 小时出清电价（同小时对齐） ===")
    for side, ccol, scol in (
        ("日前", "da_clearing_price", "settlement_da_price"),
        ("实时", "rt_clearing_price", "settlement_rt_price"),
    ):
        if ccol not in h.columns or scol not in h.columns:
            log(f"  [{side}] 缺列，跳过")
            continue
        pair = h[[ccol, scol]].dropna()
        corr = pair[ccol].corr(pair[scol])
        diff = pair[scol] - pair[ccol]
        fit = _ols_y_on_x(pair[scol], pair[ccol])
        log(f"  [{side}] 有效样本 {len(pair)}, Pearson 相关 {corr:.4f}")
        log(
            f"       结算 ~ a*出清+b : a={fit['a']:.4f}, b={fit['b']:.2f}, "
            f"R²={fit['r2']:.4f}, RMSE={fit['rmse']:.2f} 元/MWh"
        )
        log(
            f"       残差(结算-出清): mean={diff.mean():.3f}, std={diff.std():.3f}, "
            f"p50={diff.median():.3f}, p95={diff.quantile(0.95):.3f}, p5={diff.quantile(0.05):.3f}"
        )

    log("\n=== A'. 滞后相关：结算 与 出清（检验领先/滞后） ===")
    for side, ccol, scol in (
        ("日前", "da_clearing_price", "settlement_da_price"),
        ("实时", "rt_clearing_price", "settlement_rt_price"),
    ):
        if ccol not in h.columns or scol not in h.columns:
            continue
        aligned = h[[ccol, scol]].dropna()
        # corr(settlement.shift(lag), clearing): lag>0 结算更早对齐到更晚的出清
        lc = _lag_corrs(aligned[scol], aligned[ccol], range(-6, 7))
        best_lag, best_c = max(lc, key=lambda t: abs(t[1]) if np.isfinite(t[1]) else -1)
        log(f"  [{side}] 滞后集合 [-6h..+6h] 内 |相关| 最大: lag={best_lag}h, corr={best_c:.4f}")
        log(f"       明细 lag,corr: " + ", ".join(f"{lg}:{c:.3f}" for lg, c in lc))

    # —— B. 日度官方平均 vs 由 15min 自建聚合 ——
    log("\n=== B. 日度「平均出清电价」文件 vs 由出清结果自建日均价 ===")
    da_d = _daily_from_clearing_15m(da, "price", "power", "da")
    rt_d = _daily_from_clearing_15m(rt, "price", "volume", "rt")

    cmp_da = daily_wide.set_index("dt")[["da_avg_clearing_price"]].join(da_d, how="inner")
    cmp_rt = daily_wide.set_index("dt")[["rt_avg_clearing_price"]].join(rt_d, how="inner")

    def _summ(name: str, official: pd.Series, cand: pd.Series) -> None:
        m = pd.concat([official, cand], axis=1).dropna()
        if len(m) < 5:
            log(f"  {name}: 样本过少 ({len(m)})")
            return
        err = m.iloc[:, 0] - m.iloc[:, 1]
        log(
            f"  {name}: n={len(m)}, corr={m.iloc[:, 0].corr(m.iloc[:, 1]):.6f}, "
            f"MAE={err.abs().mean():.4f}, max|err|={err.abs().max():.4f}"
        )
        log(
            f"         误差分位 p50={err.median():.4f}, p95={err.quantile(0.95):.4f}, "
            f"p5={err.quantile(0.05):.4f}"
        )

    _summ("日前 官方 vs 15min等权日均", cmp_da["da_avg_clearing_price"], cmp_da["da_mean15m"])
    if "da_wavg" in cmp_da.columns:
        _summ("日前 官方 vs 15min×电力加权", cmp_da["da_avg_clearing_price"], cmp_da["da_wavg"])

    _summ("实时 官方 vs 15min等权日均", cmp_rt["rt_avg_clearing_price"], cmp_rt["rt_mean15m"])
    if "rt_wavg" in cmp_rt.columns:
        _summ("实时 官方 vs 15min×电量加权", cmp_rt["rt_avg_clearing_price"], cmp_rt["rt_wavg"])

    # 由小时出清再聚日（与 DWS 一致），对比官方日均价
    hc = hourly_c.copy()
    hc["day"] = hc.index.normalize()
    da_from_hour = hc.groupby("day")["da_clearing_price"].mean()
    rt_from_hour = hc.groupby("day")["rt_clearing_price"].mean()
    cmp_da2 = daily_wide.set_index("dt")["da_avg_clearing_price"].to_frame().join(
        da_from_hour.rename("da_from_hourly_mean"), how="inner"
    )
    cmp_rt2 = daily_wide.set_index("dt")["rt_avg_clearing_price"].to_frame().join(
        rt_from_hour.rename("rt_from_hourly_mean"), how="inner"
    )
    _summ("日前 官方 vs (先小时均值再日均)", cmp_da2["da_avg_clearing_price"], cmp_da2["da_from_hourly_mean"])
    _summ("实时 官方 vs (先小时均值再日均)", cmp_rt2["rt_avg_clearing_price"], cmp_rt2["rt_from_hourly_mean"])

    # 写出逐日对比表
    out_path = ROOT / "output" / "reverse_infer_daily_compare.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = daily_wide.copy()
    full = full.merge(da_d.reset_index().rename(columns={"day": "dt"}), on="dt", how="left")
    full = full.merge(rt_d.reset_index().rename(columns={"day": "dt"}), on="dt", how="left")
    full["da_from_hourly_mean"] = full["dt"].map(da_from_hour)
    full["rt_from_hourly_mean"] = full["dt"].map(rt_from_hour)
    full.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n已写逐日对比: {out_path}")

    log("\n=== C. 结论摘要（统计推断，非监管条文） ===")
    log(
        "  1) 若 A 中 R² 极高且残差接近常数，说明结算价与出清价在样本内近似仿射关系；"
        "残差大或呈结构则说明存在额外结算项。"
    )
    log(
        "  2) 若 B 中「官方日均价」与「15min等权/加权」几乎重合，则日文件口径与自建聚合一致；"
        "否则官方日指标另有定义。"
    )
    log(
        "  3) 滞后相关峰值不在 lag=0 时，提示两序列在发布时间或对齐口径上存在系统偏移。"
    )

    report = ROOT / "output" / "reverse_infer_price_relations.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n全文已写: {report}")


if __name__ == "__main__":
    main()
