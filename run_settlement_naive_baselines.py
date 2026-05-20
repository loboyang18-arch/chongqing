#!/usr/bin/env python3
"""评估官方结算价 naive 基线（日前 lag24、实时 lag24、实时=日前结算）。"""

import json
import logging
import os
from pathlib import Path

from price_forecast_eval import evaluate_model_predictions
from src.settlement_naive_baselines import COL_DA, COL_RT, DEFAULT_OUT_DIR, run_all
from price_forecast_eval import to_eval_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            return str(obj)
    return obj


def main() -> None:
    out = os.environ.get("NAIVE_SETTLEMENT_OUT_DIR")
    out_dir = Path(out) if out else DEFAULT_OUT_DIR
    summary, detail = run_all(out_dir=out_dir)
    print(summary.to_string(index=False))

    idx = detail.index
    ev_da = evaluate_model_predictions(
        to_eval_frame(idx, detail[COL_DA].values, detail["da_naive_lag24h"].values),
        baseline_metrics=None,
        task_type="da",
        include_extended=True,
    )
    with open(out_dir / "metrics_eval_standard_da_naive.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(ev_da), f, ensure_ascii=False, indent=2)
    ev_rt = evaluate_model_predictions(
        to_eval_frame(idx, detail[COL_RT].values, detail["rt_naive_lag24h"].values),
        baseline_metrics=None,
        task_type="rt",
        include_extended=True,
    )
    with open(out_dir / "metrics_eval_standard_rt_naive.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(ev_rt), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
