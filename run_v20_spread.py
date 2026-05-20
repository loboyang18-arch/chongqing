"""V20 Spread Conv2D runner."""
import logging
import os

from src.config import OUTPUT_DIR
from src.model_v20_spread import run_v20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    out_subdir = os.environ.get("V20_OUT_DIR", "").strip()
    out_dir = OUTPUT_DIR / out_subdir if out_subdir else None
    run_v20(out_dir=out_dir)


if __name__ == "__main__":
    main()
