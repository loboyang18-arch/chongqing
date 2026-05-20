#!/usr/bin/env python3
"""
将 sql_data/chongqing_market_join.csv 中的 Open-Meteo 天气列导出到 source_data/，
格式与现有 Format A 单值长表一致：utf-8-sig、首列 datetime，其余列为数值特征列名
（与 src.config.FORMAT_A_WEATHER 中 value_cols 键一致）。

用法（在项目根目录）:
  python scripts/export_sql_weather_to_source_data.py

依赖：已存在 sql_data/chongqing_market_join.csv（如 scripts/fetch_chongqing_market_sql.py 拉取）。
默认按 experiment.splits 的 EFFECTIVE_START～EFFECTIVE_END 裁剪，与主特征矩阵样本窗对齐。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FORMAT_A_WEATHER, SOURCE_DIR
from src.experiment.splits import EFFECTIVE_END, EFFECTIVE_START


def main() -> None:
    join_path = ROOT / "sql_data" / "chongqing_market_join.csv"
    if not join_path.is_file():
        raise SystemExit(f"缺少文件: {join_path}（请先拉取 SQL 宽表）")

    fname = next(iter(FORMAT_A_WEATHER))
    meta = FORMAT_A_WEATHER[fname]
    date_col = meta["date_col"]
    cols = [date_col] + list(meta["value_cols"].keys())

    df = pd.read_csv(join_path, encoding="utf-8-sig")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"宽表缺少列: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    out = df[cols].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out = out.set_index(date_col).sort_index()
    out = out.loc[EFFECTIVE_START:EFFECTIVE_END]
    out = out.reset_index()

    dest = SOURCE_DIR / fname
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"Wrote {dest}  rows={len(out)}  cols={len(out.columns)}")


if __name__ == "__main__":
    main()
