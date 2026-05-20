"""V18-Multi：V18 窗口 + 方向辅助头，默认输出 output/v18-multi；默认训练 100 epoch（V18_MULTI_EPOCHS 可改）。"""
import logging
import os

from src.config import OUTPUT_DIR
from src.model_v18_multitask import run_v18_multi

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
    task = os.environ.get("V18_MULTI_TASK", "da").strip().lower()
    target_col = TASK_TARGETS.get(task, TASK_TARGETS["da"])
    out_subdir = os.environ.get("V18_MULTI_OUT_DIR", "v18-multi").strip()
    out_dir = OUTPUT_DIR / out_subdir if out_subdir else None
    no_eval = os.environ.get("V18_MULTI_NO_EVAL", "").strip() in ("1", "true", "yes")
    run_v18_multi(out_dir=out_dir, target_col=target_col, task=task, run_eval=not no_eval)


if __name__ == "__main__":
    main()
