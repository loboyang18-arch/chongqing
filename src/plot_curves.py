"""
通用曲线绘图 — 为任意版本的 da_result.csv 绘制 typical/high/low 日叠图。

实现已迁移至 `price_forecast_eval.viz`；本模块保留兼容入口。

用法:
    python run_experiment_viz.py output/v12_shape_opt/da_result.csv --label V12-A
    python -m src.plot_curves output/v12_shape_opt/da_result.csv --label V12-A
"""

import argparse
import logging
from pathlib import Path

from price_forecast_eval.viz.plotting import load_prediction_csv
from price_forecast_eval.viz import run_standard_visualization

logger = logging.getLogger(__name__)


def plot_all(result_path: str, label: str):
    """为给定的 da_result.csv 生成三类叠图 + 周级连续曲线（输出在 CSV 同目录）。"""
    path = Path(result_path)
    _, pred_col = load_prediction_csv(path)
    logger.info("Plotting curves for %s (pred_col=%s, label=%s)", path.name, pred_col, label)
    run_standard_visualization(
        path,
        out_dir=path.parent,
        label=label,
        actual_col="actual",
        pred_col=pred_col,
        mode="legacy",
        scenarios=("typical", "high", "low"),
        weekly=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("result_csv", help="Path to da_result.csv")
    parser.add_argument("--label", default="Model", help="Label for the model in plots")
    args = parser.parse_args()
    plot_all(args.result_csv, args.label)
