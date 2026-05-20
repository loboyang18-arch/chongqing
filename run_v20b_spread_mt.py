"""V20b Spread multi-task (regression + focal sign) runner."""
import logging
import os
from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR
from src.model_v20b_spread_mt import run_v20b

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _compare_with_baselines(v20b_dir: Path) -> None:
    """与 V20 spread、V19-RT 对比 MAE / sign accuracy。"""
    rows = []

    def _load_spread(path: Path):
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        a, p = df["actual"].values, df["predicted"].values
        mae = float((abs(a - p)).mean())
        rmse = float(((a - p) ** 2).mean() ** 0.5)
        sign = float(((a > 0) == (p > 0)).mean())
        return {"mae": mae, "rmse": rmse, "sign_acc_th0": sign}

    def _load_rt(path: Path):
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        a, p = df["actual"].values, df["predicted"].values
        return {
            "rt_mae": float((abs(a - p)).mean()),
            "rt_rmse": float(((a - p) ** 2).mean() ** 0.5),
        }

    v20_dir = OUTPUT_DIR / "v20_spread"
    v19_rt = OUTPUT_DIR / "v19_multitask_rt"

    r20b_sp = _load_spread(v20b_dir / "spread_result.csv")
    if r20b_sp:
        rows.append({"model": "V20b spread (MT)", **r20b_sp})

    r20_sp = _load_spread(v20_dir / "spread_result.csv")
    if r20_sp:
        rows.append({"model": "V20 spread (reg only)", **r20_sp})

    r20b_rt = _load_rt(v20b_dir / "rt_result.csv")
    r20_rt = _load_rt(v20_dir / "rt_result.csv")
    v19_rt_m = _load_rt(v19_rt / "rt_result.csv")

    logging.getLogger(__name__).info("── Comparison (test set) ──")
    for r in rows:
        logging.getLogger(__name__).info(
            "  %-22s spread_MAE=%.2f spread_RMSE=%.2f sign@0=%.3f",
            r["model"],
            r["mae"],
            r["rmse"],
            r["sign_acc_th0"],
        )
    if r20b_rt:
        logging.getLogger(__name__).info(
            "  %-22s recon_RT_MAE=%.2f recon_RT_RMSE=%.2f",
            "V20b→RT",
            r20b_rt["rt_mae"],
            r20b_rt["rt_rmse"],
        )
    if r20_rt:
        logging.getLogger(__name__).info(
            "  %-22s recon_RT_MAE=%.2f recon_RT_RMSE=%.2f",
            "V20→RT",
            r20_rt["rt_mae"],
            r20_rt["rt_rmse"],
        )
    if v19_rt_m:
        logging.getLogger(__name__).info(
            "  %-22s direct_RT_MAE=%.2f direct_RT_RMSE=%.2f",
            "V19-MT-RT",
            v19_rt_m["rt_mae"],
            v19_rt_m["rt_rmse"],
        )


def main() -> None:
    out_sub = os.environ.get("V20B_OUT_DIR", "").strip()
    resolved = OUTPUT_DIR / out_sub if out_sub else OUTPUT_DIR / "v20b_spread_mt"
    run_v20b(out_dir=resolved)
    _compare_with_baselines(resolved)


if __name__ == "__main__":
    main()
