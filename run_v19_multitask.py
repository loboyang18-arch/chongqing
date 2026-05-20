"""V19 Multi-Task Conv2D runner for experiment pipeline."""
import logging
import os

from src.config import OUTPUT_DIR
from src.model_v19_multitask import run_v19

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASK_TARGETS = {
    "da": "da_clearing_price",
    "rt": "settlement_rt_price",
}


def main() -> None:
    task = os.environ.get("V19_TASK", "da").strip().lower()
    target_col = TASK_TARGETS.get(task, TASK_TARGETS["da"])
    out_subdir = os.environ.get("V19_OUT_DIR", "").strip()
    out_dir = OUTPUT_DIR / out_subdir if out_subdir else None
    run_v19(out_dir=out_dir, target_col=target_col, task=task)


if __name__ == "__main__":
    main()
