"""
时间切分单一真源 — 所有 train/val/test 边界由此 import，禁止在业务脚本中重复写死。

- 15 分钟序列（V16 / V16d / V17 / build_feature_matrix）：EFFECTIVE_*、TRAIN_END、VAL_END、
  TEST_START、TEST_END（测试窗右端，含该日 23:45）
- 小时表（feature_da.csv、model_baseline、V12）：HOURLY_TRAIN_END、HOURLY_VAL_END、
  HOURLY_TEST_START、HOURLY_TEST_END
- 特征工程 CSV 裁剪：FEATURE_ENGINEERING_*（字符串）

当前策略（与 2026-04 数据对齐）：
  训练：有效窗内至 2026-02-23（含）
  验证：2026-02-24～2026-03-02（7 天，测试开始前一周）
  测试：2026-03-03～2026-04-13（含）
  特征矩阵：EFFECTIVE_END 对齐当前 dws 末条（2026-04-16 23:45）；测试评估仍以 TEST_END 为界。
"""

import os

import pandas as pd


def _split_ts(env_key: str, default: str) -> pd.Timestamp:
    """实验可覆盖：SPLIT_TRAIN_END / SPLIT_VAL_END / SPLIT_TEST_START / SPLIT_TEST_END。"""
    v = os.environ.get(env_key, "").strip()
    return pd.Timestamp(v) if v else pd.Timestamp(default)


# ── 15 分钟网格 ─────────────────────────────────────────
EFFECTIVE_START = pd.Timestamp("2025-11-01")
# 与 output/dws_hourly_features 当前最大整点 2026-04-16 23:00 对齐的末个 15min 槽
EFFECTIVE_END = pd.Timestamp("2026-04-16 23:45:00")

# 默认：训练至 02-23，验证 7 天（02-24～03-02），测试自 03-03
TRAIN_END = _split_ts("SPLIT_TRAIN_END", "2026-02-23 23:45:00")
VAL_END = _split_ts("SPLIT_VAL_END", "2026-03-02 23:45:00")
TEST_START = _split_ts("SPLIT_TEST_START", "2026-03-03 00:00:00")
TEST_END = _split_ts("SPLIT_TEST_END", "2026-04-13 23:45:00")

# ── 小时网格（与 15min 边界同日对齐）────────────────
HOURLY_TRAIN_END = str(TRAIN_END.floor("h"))[:19] + ":00"
HOURLY_VAL_END = str(VAL_END.floor("h"))[:19] + ":00"
HOURLY_TEST_START = str(TEST_START.floor("h"))[:19] + ":00"
HOURLY_TEST_END = str(TEST_END.floor("h"))[:19] + ":00"

# ── feature_engineering 产出窗口（按小时索引）────────────────
FEATURE_ENGINEERING_START = "2025-11-01"
FEATURE_ENGINEERING_END = "2026-04-16 23:00:00"
