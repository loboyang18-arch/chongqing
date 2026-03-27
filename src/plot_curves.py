"""
通用曲线绘图 — 为任意版本的 da_result.csv 绘制 typical/high/low 日叠图。

用法:
    python -m src.plot_curves output/v12_shape_opt/da_result.csv --label V12-A
    python -m src.plot_curves output/v11_feature_enhanced/da_result.csv --label V11
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _setup_cn_font():
    for p in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Medium.ttc"]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["mathtext.fontset"] = "cm"


def _select_days(results: pd.DataFrame, pred_col: str, category: str, n: int = 6):
    daily = results.groupby(results.index.date).agg(
        actual_mean=("actual", "mean"),
    )
    daily["mae"] = results.groupby(results.index.date).apply(
        lambda g: np.mean(np.abs(g["actual"] - g[pred_col])))

    if category == "typical":
        med = daily["mae"].median()
        daily["dist"] = (daily["mae"] - med).abs()
        return list(daily.nsmallest(n, "dist").index)
    elif category == "high":
        return list(daily.nlargest(n, "actual_mean").index)
    elif category == "low":
        return list(daily.nsmallest(n, "actual_mean").index)
    return []


def _plot_day_overlay(results: pd.DataFrame, days: list, title: str,
                      filename: str, pred_col: str, pred_label: str,
                      out_dir: Path):
    _setup_cn_font()
    n = len(days)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    hours = list(range(24))
    last_i = 0
    for i, d in enumerate(sorted(days)):
        ax = axes[i // cols][i % cols]
        day_data = results.loc[str(d)]
        if len(day_data) != 24:
            continue
        last_i = i

        ax.plot(hours, day_data["actual"].values, "k-", linewidth=2.0, label="实际", zorder=3)
        ax.plot(hours, day_data[pred_col].values, "#E91E63", linewidth=1.5, alpha=0.9, label=pred_label)

        corr = np.corrcoef(day_data["actual"].values, day_data[pred_col].values)[0, 1]
        mae = np.mean(np.abs(day_data["actual"].values - day_data[pred_col].values))
        ax.set_title(f"{d}  r={corr:.2f}  MAE={mae:.1f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for j in range(last_i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / filename)


def _plot_weekly(results: pd.DataFrame, pred_col: str, pred_label: str,
                 out_dir: Path):
    """按周绘制连续时序对比曲线（actual vs pred），每周一张图。"""
    _setup_cn_font()

    results = results.sort_index()
    all_dates = sorted(set(results.index.date))
    if not all_dates:
        return

    weeks = []
    week = [all_dates[0]]
    for d in all_dates[1:]:
        if (d - week[0]).days >= 7:
            weeks.append(week)
            week = [d]
        else:
            week.append(d)
    if week:
        weeks.append(week)

    for wi, week_dates in enumerate(weeks):
        start_d, end_d = week_dates[0], week_dates[-1]
        mask = (results.index.date >= start_d) & (results.index.date <= end_d)
        chunk = results.loc[mask]
        if len(chunk) < 24:
            continue

        fig, ax = plt.subplots(figsize=(18, 5))
        x = range(len(chunk))
        ax.plot(x, chunk["actual"].values, "k-", linewidth=1.8, label="实际", zorder=3)
        ax.plot(x, chunk[pred_col].values, "#E91E63", linewidth=1.3, alpha=0.85, label=pred_label)

        day_boundaries = []
        tick_positions = []
        tick_labels_list = []
        for d in week_dates:
            d_mask = chunk.index.date == d
            idxs = np.where(d_mask)[0]
            if len(idxs) > 0:
                day_boundaries.append(idxs[0])
                tick_positions.append(idxs[0] + 12)
                a = chunk["actual"].values[d_mask]
                p = chunk[pred_col].values[d_mask]
                if len(a) == 24 and np.std(a) > 1e-6 and np.std(p) > 1e-6:
                    r = np.corrcoef(a, p)[0, 1]
                    tick_labels_list.append(f"{d}\nr={r:.2f}")
                else:
                    tick_labels_list.append(str(d))

        for bd in day_boundaries[1:]:
            ax.axvline(bd, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=8)
        ax.set_ylim(200, 600)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{pred_label} — 第{wi+1}周 ({start_d} ~ {end_d})",
                     fontsize=13, fontweight="bold")

        plt.tight_layout()
        fname = f"da_week{wi+1}.png"
        plt.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", out_dir / fname)


def plot_all(result_path: str, label: str):
    """为给定的 da_result.csv 生成三类叠图 + 周级连续曲线。"""
    path = Path(result_path)
    out_dir = path.parent

    results = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    pred_col = [c for c in results.columns if c != "actual"][0]

    logger.info("Plotting curves for %s (pred_col=%s, label=%s)", path.name, pred_col, label)

    for cat, title_suffix in [("typical", "典型日曲线叠图"),
                               ("high", "极端高价日曲线叠图"),
                               ("low", "极端低价日曲线叠图")]:
        days = _select_days(results, pred_col, cat, 6)
        _plot_day_overlay(
            results, days,
            f"{label} — {title_suffix}",
            f"da_{cat}_days.png",
            pred_col, label, out_dir,
        )

    _plot_weekly(results, pred_col, label, out_dir)


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
