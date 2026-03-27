"""
Shape 横向评估 — 对所有已有 *_result.csv 计算 6 项 shape 指标 + MAE/RMSE。

读取 output/ 下全部结果文件，对每个 (文件, 预测列) 组合调用
shape_metrics.compute_shape_report，汇总输出到
output/shape_evaluation_summary.csv。

不修改任何模型代码，不重新训练，纯后处理。
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

SKIP_COLS = {
    "ts", "actual", "naive", "delta_actual", "delta_pred", "base_price",
    "ar_correction", "spread_pred", "da_d0", "interval_width",
}

RESULT_FILES: List[Dict] = [
    {"path": "baseline_da_result.csv",       "task": "da", "source": "baseline"},
    {"path": "baseline_rt_result.csv",       "task": "rt", "source": "baseline"},
    {"path": "tuning_da_result.csv",         "task": "da", "source": "tuning"},
    {"path": "tuning_rt_result.csv",         "task": "rt", "source": "tuning"},
    {"path": "delta_da_result.csv",          "task": "da", "source": "delta"},
    {"path": "delta_rt_result.csv",          "task": "rt", "source": "delta"},
    {"path": "sequence_da_result.csv",       "task": "da", "source": "sequence"},
    {"path": "sequence_rt_result.csv",       "task": "rt", "source": "sequence"},
    {"path": "ensemble_da_result.csv",       "task": "da", "source": "ensemble"},
    {"path": "ensemble_rt_result.csv",       "task": "rt", "source": "ensemble"},
    {"path": "residual_ar_da_result.csv",    "task": "da", "source": "residual_ar"},
    {"path": "residual_ar_rt_result.csv",    "task": "rt", "source": "residual_ar"},
    {"path": "period_da_result.csv",         "task": "da", "source": "period"},
    {"path": "period_rt_result.csv",         "task": "rt", "source": "period"},
    {"path": "viz/shape_da_result.csv",      "task": "da", "source": "shape"},
    {"path": "viz/shape_rt_result.csv",      "task": "rt", "source": "shape"},
    {"path": "viz/decompose_da_result.csv",  "task": "da", "source": "decompose"},
    {"path": "viz/decompose_rt_result.csv",  "task": "rt", "source": "decompose"},
    {"path": "viz/final_da_result.csv",      "task": "da", "source": "final"},
    {"path": "viz/final_rt_result.csv",      "task": "rt", "source": "final"},
    {"path": "viz/seqshape_da_result.csv",   "task": "da", "source": "seqshape"},
    {"path": "viz/seqshape_rt_result.csv",   "task": "rt", "source": "seqshape"},
    {"path": "viz/v2_da_result.csv",         "task": "da", "source": "v2_adaptive"},
    {"path": "viz/v2_rt_result.csv",         "task": "rt", "source": "v2_adaptive"},
    {"path": "v3_optimize/rt_spread_result.csv", "task": "rt", "source": "v3_spread"},
    {"path": "v3_optimize/da_quantile_result.csv", "task": "da", "source": "v3_quantile"},
    {"path": "v3_optimize/rt_quantile_result.csv", "task": "rt", "source": "v3_quantile"},
]


def _evaluate_file(info: Dict) -> List[Dict]:
    """对单个 result CSV 的所有预测列计算 shape 指标。"""
    fpath = OUTPUT_DIR / info["path"]
    if not fpath.exists():
        logger.warning("File not found, skipping: %s", fpath)
        return []

    df = pd.read_csv(fpath, parse_dates=["ts"], index_col="ts")

    if "actual" not in df.columns:
        logger.warning("No 'actual' column in %s, skipping", fpath)
        return []

    actual = df["actual"].values
    index = df.index

    pred_cols = [
        c for c in df.columns
        if c not in SKIP_COLS and not c.startswith("P") and c != "P50"
    ]
    if "P50" in df.columns:
        pred_cols.append("P50")

    rows = []
    for col in pred_cols:
        pred = df[col].values
        mask = np.isfinite(actual) & np.isfinite(pred)
        a_clean, p_clean = actual[mask], pred[mask]
        idx_clean = index[mask]

        if len(a_clean) < 24:
            continue

        mae = float(np.mean(np.abs(a_clean - p_clean)))
        rmse = float(np.sqrt(np.mean((a_clean - p_clean) ** 2)))
        shape = compute_shape_report(a_clean, p_clean, idx_clean)

        row = {
            "source": info["source"],
            "task": info["task"],
            "method": col,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
        }
        row.update(shape)
        rows.append(row)

    return rows


def run_shape_evaluation() -> pd.DataFrame:
    """主入口：遍历所有 result 文件，汇总 shape 评估。"""
    all_rows = []
    for info in RESULT_FILES:
        rows = _evaluate_file(info)
        if rows:
            logger.info(
                "  %s (%s): %d methods evaluated",
                info["source"], info["task"], len(rows),
            )
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    col_order = [
        "source", "task", "method", "MAE", "RMSE",
        "profile_corr", "norm_profile_mae",
        "peak_hour_err", "valley_hour_err",
        "amplitude_err", "direction_acc",
    ]
    summary = summary[[c for c in col_order if c in summary.columns]]
    summary = summary.sort_values(["task", "profile_corr"], ascending=[True, False])

    out_path = OUTPUT_DIR / "shape_evaluation_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved shape evaluation summary: %s (%d rows)", out_path, len(summary))

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    df = run_shape_evaluation()
    print("\n=== Shape Evaluation Summary ===")
    print(df.to_string(index=False))
