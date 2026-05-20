#!/usr/bin/env python3
"""
列出 MySQL 库中的表名（SHOW TABLES）。

依赖: pip install pymysql sqlalchemy

连接信息优先读环境变量（避免把密码写进代码）:
  MYSQL_HOST      默认 121.43.142.213
  MYSQL_PORT      默认 13306
  MYSQL_USER      默认 viewer
  MYSQL_PASSWORD  必填（或运行时输入）
  MYSQL_DATABASE  默认 testdb

示例:
  export MYSQL_PASSWORD='你的密码'
  python scripts/list_mysql_tables.py
"""
import os
import sys

try:
    from sqlalchemy import create_engine, text
    from urllib.parse import quote_plus
except ImportError:
    print("请先安装: pip install pymysql sqlalchemy", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    host = os.environ.get("MYSQL_HOST", "121.43.142.213")
    port = int(os.environ.get("MYSQL_PORT", "13306"))
    user = os.environ.get("MYSQL_USER", "viewer")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "testdb")

    if not password:
        import getpass

        password = getpass.getpass("MySQL password (viewer): ")

    pwd = quote_plus(password)
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}"
    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as conn:
        rows = conn.execute(text("SHOW TABLES")).fetchall()

    if not rows:
        print(f"数据库 `{database}` 中没有任何表。")
        return

    names = [r[0] for r in rows]

    print(f"主机 {host}:{port}  库 `{database}`  共 {len(names)} 张表:\n")
    for i, name in enumerate(names, 1):
        print(f"  {i:4d}  {name}")


if __name__ == "__main__":
    main()
