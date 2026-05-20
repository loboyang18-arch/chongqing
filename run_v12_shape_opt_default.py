"""V12 Shape-Opt default runner (Variant A) for experiment pipeline."""
import logging
import os
from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR
from price_forecast_eval.viz import run_standard_visualization
from src.model_v12_shape_opt import run_v12_variant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Compatible with run_experiment.py artifact_env pattern.
    out_subdir = os.environ.get("V12_SHAPE_OUT_DIR", "").strip()
    if out_subdir:
        out_dir = OUTPUT_DIR / out_subdir
    else:
        out_dir = OUTPUT_DIR / "v12_shape_opt_default"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output: %s", out_dir)
    summary, result = run_v12_variant("A")

    # Standardize prediction file schema for experiment post_evaluate.
    da = pd.DataFrame(
        {
            "actual": result["actual"].values,
            "predicted": result["pred"].values,
        },
        index=result.index,
    )
    da.index.name = "ts"
    da.to_csv(out_dir / "da_result.csv")

    logger.info("MAE=%.4f RMSE=%.4f", summary.get("MAE", 0), summary.get("RMSE", 0))

    run_standard_visualization(
        out_dir / "da_result.csv",
        out_dir=out_dir / "plots",
        label="V12-shape-opt",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )
    logger.info("Saved: da_result.csv, plots/")


if __name__ == "__main__":
    main()
