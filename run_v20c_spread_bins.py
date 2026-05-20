"""V20c spread bin classification + RT reconstruction runner."""
import logging
import os
from pathlib import Path

from src.config import OUTPUT_DIR
from src.model_v20c_spread_bins import run_v20c

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    out_sub = os.environ.get("V20C_OUT_DIR", "").strip()
    resolved = OUTPUT_DIR / out_sub if out_sub else OUTPUT_DIR / "v20c_spread_bins"
    run_v20c(out_dir=resolved)


if __name__ == "__main__":
    main()
