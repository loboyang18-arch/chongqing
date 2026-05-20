#!/usr/bin/env python3
"""兼容入口：转调 `price_forecast_eval` 包的 `eval` 子命令。"""

from __future__ import annotations

import sys

from price_forecast_eval.cli import main as _pkg_main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "eval", *sys.argv[1:]]
    _pkg_main()
