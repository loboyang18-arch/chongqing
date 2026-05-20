#!/usr/bin/env python3
"""
从远端 MySQL 拉取重庆市场宽表（能源流 + 市场出清 + Open-Meteo 小时天气），
导出 CSV 与 Parquet 到项目根目录 sql_data/。

依赖: pip install pymysql sqlalchemy pandas pyarrow

环境变量（密码勿写进代码）:
  MYSQL_HOST      默认 121.43.142.213
  MYSQL_PORT      默认 13306
  MYSQL_USER      默认 viewer
  MYSQL_PASSWORD  必填（或交互输入）
  MYSQL_DATABASE  默认 testdb

  SQL_OUT_BASE    输出文件名不含扩展名，默认 chongqing_market_join
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
SQL_DATA_DIR = ROOT / "sql_data"

SQL = """
SELECT
    e.*,
    m.*,
    w.*
FROM data_energy_flow e
LEFT JOIN data_market_clearing m
    ON e.datetime = m.datetime
LEFT JOIN data_openmeteo_weather_hourly w
    ON w.latitude = 29.58
   AND w.longitude = 106.53
   AND w.date = DATE(e.datetime)
   AND w.hour = HOUR(e.datetime)
"""
# 说明：天气为整点小时一条，应对齐到该钟点内的全部 4 个 15min（如 00:00/00:15/00:30/00:45 均用 hour=0）。
# 旧写法对非整点用 HOUR+1，会把 00:15～00:45 错绑到下一小时，后续整点小时整体错位。


def main() -> None:
    host = os.environ.get("MYSQL_HOST", "121.43.142.213")
    port = int(os.environ.get("MYSQL_PORT", "13306"))
    user = os.environ.get("MYSQL_USER", "viewer")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "testdb")
    base = os.environ.get("SQL_OUT_BASE", "chongqing_market_join").strip() or "chongqing_market_join"

    if not password:
        import getpass

        password = getpass.getpass("MySQL password: ")

    pwd = quote_plus(password or "")
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}"
    engine = create_engine(url, pool_pre_ping=True)

    print(f"Connecting {host}:{port}/{database} …")
    df = pd.read_sql(SQL, engine)
    print(f"Loaded rows={len(df):,} cols={len(df.columns)}")

    df = df.loc[:, ~df.columns.duplicated()]
    if "datetime" not in df.columns:
        print("警告: 结果中无 datetime 列", file=sys.stderr)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df = df.set_index("datetime", drop=False)

    SQL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SQL_DATA_DIR / f"{base}.csv"
    pq_path = SQL_DATA_DIR / f"{base}.parquet"

    print(f"Writing CSV → {csv_path}")
    df.to_csv(csv_path, index=False)

    print(f"Writing Parquet → {pq_path}")
    try:
        df.to_parquet(pq_path, index=False, engine="pyarrow")
    except ImportError:
        print("未安装 pyarrow，跳过 parquet。请: pip install pyarrow", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Parquet 写出失败: {e}", file=sys.stderr)
        raise

    print("Done.")
    print("可选: 将天气列按 ODS 格式写入 source_data/ →")
    print("  python scripts/export_sql_weather_to_source_data.py")


if __name__ == "__main__":
    main()
