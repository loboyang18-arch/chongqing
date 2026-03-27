"""V16d no-engfeat sweep: 3 model scales x multi-seed with summary."""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.model_v16_nhits import (
    EFFECTIVE_START,
    FUTR_COLS,
    HIST_COLS,
    VAL_END,
    build_feature_matrix,
)
from src.model_v16d_hourly_settlement import (
    C_POINT,
    TARGET_SETTLE,
    predict_period,
    predict_test,
    train_one,
)
from src.shape_metrics import compute_shape_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_seeds() -> list:
    raw = os.environ.get("V16_SWEEP_SEEDS", "0,1,2")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _preset_configs() -> list:
    # label, d_model, dim_ff, nlayers, nhead, batch_size
    return [
        ("base", 128, 384, 3, 4, 64),
        ("xl", 192, 576, 4, 8, 32),
        ("xxl", 256, 768, 4, 8, 16),
    ]


def _daily_metrics_from_pred(p24, a24, dates):
    if len(dates) == 0:
        return {"mae": np.nan, "rmse": np.nan, "profile_corr": np.nan}
    idx = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(hours=h) for d in dates for h in range(24)])
    af = a24.reshape(-1)
    pf = p24.reshape(-1)
    return {
        "mae": float(np.mean(np.abs(af - pf))),
        "rmse": float(np.sqrt(np.mean((af - pf) ** 2))),
        **compute_shape_report(af, pf, idx),
    }


def main():
    epochs = int(os.environ.get("V16D_EPOCHS", "300"))
    dropout = float(os.environ.get("V16D_DROPOUT", "0.2"))
    seeds = _parse_seeds()
    configs = _preset_configs()

    out_root = OUTPUT_DIR / os.environ.get("V16_SWEEP_OUT_DIR", f"v16d_noengfeat_sweep_ep{epochs}")
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Sweep start: epochs=%d dropout=%.2f seeds=%s", epochs, dropout, seeds)
    logger.info("Output root: %s", out_root)

    # Build & normalize once
    df = build_feature_matrix()
    raw_y = df[TARGET_SETTLE].values.astype(np.float32)
    hist = df[HIST_COLS].values.T.astype(np.float32)
    futr = df[FUTR_COLS].values.T.astype(np.float32)
    ts = df.index.values

    fit_mask = df.index <= VAL_END
    y_mean = float(raw_y[fit_mask].mean())
    y_std = float(raw_y[fit_mask].std()) + 1e-8
    y_norm = ((raw_y - y_mean) / y_std).astype(np.float32)
    h_mean = hist[:, fit_mask].mean(axis=1, keepdims=True)
    h_std = hist[:, fit_mask].std(axis=1, keepdims=True) + 1e-8
    hist_norm = ((hist - h_mean) / h_std).astype(np.float32)
    f_mean = futr[:, fit_mask].mean(axis=1, keepdims=True)
    f_std = futr[:, fit_mask].std(axis=1, keepdims=True) + 1e-8
    futr_norm = ((futr - f_mean) / f_std).astype(np.float32)

    rows = []
    agg_rows = []

    for label, d_model, dim_ff, nlayers, nhead, bs in configs:
        cfg_dir = out_root / f"{label}_d{d_model}_ff{dim_ff}_l{nlayers}_h{nhead}_bs{bs}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        model_kw = dict(
            d_model=d_model,
            nhead=nhead,
            nlayers=nlayers,
            dim_ff=dim_ff,
            dropout=dropout,
            c_point=C_POINT,
        )
        logger.info(
            "Config %s: d=%d ff=%d L=%d H=%d bs=%d",
            label, d_model, dim_ff, nlayers, nhead, bs
        )

        per_seed_metrics = []
        for seed in seeds:
            seed_dir = cfg_dir / f"seed{seed}"
            seed_dir.mkdir(exist_ok=True)
            res = train_one(
                seed=seed,
                y_norm=y_norm,
                hist_norm=hist_norm,
                futr_norm=futr_norm,
                ts=ts,
                raw_y=raw_y,
                y_mean=y_mean,
                y_std=y_std,
                epochs=epochs,
                bs=bs,
                model_kw=model_kw,
                out_dir=seed_dir,
            )
            p24_te, a24_te, dates_te = predict_test(
                [res["path"]], y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std, model_kw=model_kw
            )
            mte = _daily_metrics_from_pred(p24_te, a24_te, dates_te)

            p24_tr, a24_tr, dates_tr = predict_period(
                [res["path"]], y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std,
                EFFECTIVE_START, VAL_END, model_kw=model_kw
            )
            mtr = _daily_metrics_from_pred(p24_tr, a24_tr, dates_tr)

            row = {
                "config": label,
                "seed": seed,
                "d_model": d_model,
                "dim_ff": dim_ff,
                "nlayers": nlayers,
                "nhead": nhead,
                "bs": bs,
                "epochs": epochs,
                "train_mae": mtr["mae"],
                "train_pcorr": mtr["profile_corr"],
                "test_mae": mte["mae"],
                "test_pcorr": mte["profile_corr"],
                "test_amp_err": mte.get("amplitude_err", np.nan),
                "test_peak_err": mte.get("peak_hour_err", np.nan),
                "test_valley_err": mte.get("valley_hour_err", np.nan),
                "ckpt": str(res["path"]),
            }
            rows.append(row)
            per_seed_metrics.append(row)
            logger.info(
                "  seed=%d -> test_pcorr=%.4f test_mae=%.2f amp_err=%.2f",
                seed, row["test_pcorr"], row["test_mae"], row["test_amp_err"]
            )

        df_cfg = pd.DataFrame(per_seed_metrics)
        agg = {
            "config": label,
            "d_model": d_model,
            "dim_ff": dim_ff,
            "nlayers": nlayers,
            "nhead": nhead,
            "bs": bs,
            "epochs": epochs,
            "seeds": ",".join(map(str, seeds)),
            "test_pcorr_median": float(df_cfg["test_pcorr"].median()),
            "test_pcorr_mean": float(df_cfg["test_pcorr"].mean()),
            "test_mae_mean": float(df_cfg["test_mae"].mean()),
            "test_amp_err_mean": float(df_cfg["test_amp_err"].mean()),
            "train_pcorr_mean": float(df_cfg["train_pcorr"].mean()),
            "train_mae_mean": float(df_cfg["train_mae"].mean()),
        }
        agg_rows.append(agg)

    detail = pd.DataFrame(rows).sort_values(["config", "seed"]).reset_index(drop=True)
    summary = pd.DataFrame(agg_rows).sort_values(
        ["test_pcorr_median", "test_mae_mean"], ascending=[False, True]
    ).reset_index(drop=True)

    detail.to_csv(out_root / "detail_per_seed.csv", index=False)
    summary.to_csv(out_root / "summary_by_config.csv", index=False)
    logger.info("Saved detail: %s", out_root / "detail_per_seed.csv")
    logger.info("Saved summary: %s", out_root / "summary_by_config.csv")
    logger.info("Top configs:\n%s", summary.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
