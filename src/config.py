"""
全局配置 — 27个源数据文件的元数据注册表。

每个条目按实际文件内容（而非报告描述）定义：
  format_type : A(长表) / B(枢纽长表) / C(宽表) / D(事务表)
  granularity : 源文件的原始时间粒度(分钟)
  date_col    : 日期/时间列名
  value_cols  : {原始列名 → 标准化 metric 名}
  agg_hourly  : 聚合到小时时使用的方法
  skip_rows   : 需跳过的占位空行数
  ingest      : 是否纳入主流程入库
"""
from pathlib import Path

# ── 路径 ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "source_data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
# LightGBM 调参结果（与训练产物分离，纳入版本库）
PARAMS_DIR = BASE_DIR / "params"
PARAMS_DIR.mkdir(exist_ok=True)

# ── Format A: 长表 (datetime + value) ──────────────────
FORMAT_A_SINGLE = {
    "实际负荷.csv": {
        "date_col": "datetime",
        "value_cols": {"实际负荷": "actual_load"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "系统负荷预测.csv": {
        "date_col": "datetime",
        "value_cols": {"系统负荷预测": "load_forecast"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "发电总出力.csv": {
        "date_col": "datetime",
        "value_cols": {"发电总出力": "total_gen"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "水电（含抽蓄）总出力.csv": {
        "date_col": "datetime",
        "value_cols": {"水电（含抽蓄）总出力": "hydro_gen"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "非市场机组总出力.csv": {
        "date_col": "datetime",
        "value_cols": {"非市场机组总出力": "non_market_gen"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "新能源总出力.csv": {
        "date_col": "datetime",
        "value_cols": {"新能源总出力": "renewable_gen"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "新能源预测.csv": {
        "date_col": "datetime",
        "value_cols": {"新能源预测": "renewable_fcst"},
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "省间联络线输电.csv": {
        "date_col": "datetime",
        "value_cols": {"省间联络线输电": "tie_line_power"},
        "granularity": 5,
        "agg_hourly": "mean",
    },
    "现货交易可靠性结果查询（用电侧）.csv": {
        "date_col": "datetime",
        "value_cols": {
            "日前加权平均节点电价(元/MWh)": "reliability_da_price",
            "实时加权平均节点电价(元/MWh)": "reliability_rt_price",
        },
        "granularity": 15,
        "agg_hourly": "mean",
    },
}

FORMAT_A_DUAL = {
    "发电总出力预测.csv": {
        "date_col": "datetime",
        "value_cols": {
            "发电总出力预测上午": ("total_gen_fcst", "am"),
            "发电总出力预测下午": ("total_gen_fcst", "pm"),
        },
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "水电（含抽蓄）总出力预测.csv": {
        "date_col": "datetime",
        "value_cols": {
            "水电（含抽蓄）总出力预测上午": ("hydro_gen_fcst", "am"),
            "水电（含抽蓄）总出力预测下午": ("hydro_gen_fcst", "pm"),
        },
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "非市场机组总出力预测.csv": {
        "date_col": "datetime",
        "value_cols": {
            "非市场机组总出力预测上午传输": ("non_market_gen_fcst", "am"),
            "非市场机组总出力预测下午传输": ("non_market_gen_fcst", "pm"),
        },
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "新能源总出力预测.csv": {
        "date_col": "datetime",
        "value_cols": {
            "光伏总出力预测上午": ("renewable_fcst_solar", "am"),
            "新能源总出力预测上午": ("renewable_fcst_total", "am"),
            "风电总出力预测上午": ("renewable_fcst_wind", "am"),
            "光伏总出力预测下午": ("renewable_fcst_solar", "pm"),
            "新能源总出力预测下午": ("renewable_fcst_total", "pm"),
            "风电总出力预测下午": ("renewable_fcst_wind", "pm"),
        },
        "granularity": 15,
        "agg_hourly": "mean",
    },
    "省间联络线输电曲线预测.csv": {
        "date_col": "datetime",
        "value_cols": {
            "省间联络线输电曲线预测上午": ("tie_line_fcst", "am"),
            "省间联络线输电曲线预测下午": ("tie_line_fcst", "pm"),
        },
        "granularity": 15,
        "agg_hourly": "mean",
    },
}

FORMAT_A_CLEARING = {
    "日前市场交易出清结果.csv": {
        "date_col": "datetime",
        "market": "da",
        "value_cols": {
            "现货日前市场出清电价": "price",
            "现货日前市场出清电力": "power",
            "现货日前市场出清机组台数": "unit_count",
        },
        "granularity": 15,
        "agg_hourly": {"price": "mean", "power": "mean", "unit_count": "mean"},
    },
    "实时出清结果.csv": {
        "date_col": "datetime",
        "market": "rt",
        "value_cols": {
            "现货实时出清电量": "volume",
            "现货实时出清电价": "price",
            "现货实时出清机组台数": "unit_count",
        },
        "granularity": 15,
        "agg_hourly": {"price": "mean", "volume": "sum", "unit_count": "mean"},
    },
    "现货日前可靠性出清电量.csv": {
        "date_col": "datetime",
        "market": "da_reliability",
        "value_cols": {
            "现货日前可靠性出清电价": "price",
            "现货日前可靠性出清电力": "power",
            "现货日前可靠性出清机组台数": "unit_count",
        },
        "granularity": 15,
        "agg_hourly": {"price": "mean", "power": "mean", "unit_count": "mean"},
    },
}

FORMAT_A_SETTLEMENT = {
    "统一结算点电价.csv": {
        "date_col": "datetime",
        "value_cols": {
            "日前统一结算点电价": "settlement_da_price",
            "实时统一结算点电价": "settlement_rt_price",
        },
        "granularity": 60,
        "agg_hourly": "mean",
    },
}

# ── Format B: 枢纽长表 (entity + timestamp + value) ─────
FORMAT_B = {
    "实际运行输电断面约束情况.csv": {
        "date_col": "时点",
        "entity_cols": ["设备名称", "设备类型"],
        "value_col": "值",
        "granularity": 60,
        "agg_hourly": "mean",
    },
}

# ── Format C: 宽表 (V-columns) ──────────────────────────
FORMAT_C = {
    "日前市场交易出清节点电价.csv": {
        "date_col": "日期",
        "meta_cols": ["节点类型", "数据类型", "节点名称"],
        "v_prefix": "V",
        "v_suffix": "",
        "granularity": 15,
        "interval_min": 15,
        "skip_placeholder_rows": 304,
        "agg_hourly": "mean",
        "chunksize": 10_000,
    },
    "实时出清节点电价.csv": {
        "date_col": "日期",
        "meta_cols": ["节点类型", "数据类型", "节点名称"],
        "v_prefix": "V",
        "v_suffix": "出清节点电价",
        "granularity": 5,
        "interval_min": 5,
        "skip_placeholder_rows": 1046,
        "agg_hourly": "mean",
        "chunksize": 5_000,
    },
}

# ── Format D: 事务/事件表 ─────────────────────────────────
FORMAT_D = {
    "发输变电检修计划.csv": {
        "date_col": "日期",
        "cols": ["日期", "数据类型", "设备名称", "设备类型", "电压等级", "计划开工时间", "计划完工时间"],
        "rename": {
            "数据类型": "version",
            "设备名称": "equipment_name",
            "设备类型": "equipment_type",
            "电压等级": "voltage_level",
            "计划开工时间": "planned_start",
            "计划完工时间": "planned_end",
        },
        "ingest": True,
    },
    "日前平均出清电价.csv": {
        "date_col": "日期",
        "value_col": "日前平均出清电价",
        "metric": "da_avg_clearing_price",
        "ingest": True,
    },
    "实时平均出清电价.csv": {
        "date_col": "日期",
        "value_col": "实时平均出清电价",
        "metric": "rt_avg_clearing_price",
        "ingest": True,
    },
    "平均申报电价.csv": {
        "date_col": "日期",
        "value_col": "平均申报电价",
        "metric": "avg_bid_price",
        "ingest": True,
    },
    "能量块滚动撮合结果查询.csv": {
        "date_col": None,
        "ingest": False,
    },
    "绿电交易申报及成交情况.csv": {
        "date_col": "日期",
        "ingest": False,
    },
    "中长期合同分解曲线.csv": {
        "date_col": "分解日期",
        "ingest": False,
    },
}

FORMAT_D_XLSX = {
    "售电公司代理电量.xlsx": {
        "date_col": "日期",
        "ingest": False,
    },
}

# ── 断面约束：关键断面列表（用于 DWS 聚合） ─────────────────
KEY_SECTIONS = [
    "C-500-洪板断面送板桥-全接线【常】",
    "C-500-资铜断面送铜梁-全接线【常】",
]

# ── 节点电价：系统级 metric（用于 DWS 聚合） ────────────────
SYSTEM_NODAL_PRICE_FILTER = {
    "data_type": "电能量价格",
    "node_name": "电能量价格",
}
