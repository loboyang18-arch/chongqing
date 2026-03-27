"""V16d scalable runner WITHOUT V12 engineered features."""
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import OUTPUT_DIR
from src.model_v16d_hourly_settlement import (
    train_one,
    predict_period,
    predict_test,
    plot_train_week,
    plot_hourly_typical,
    plot_24step,
    TARGET_SETTLE,
    C_POINT,
)
from src.model_v16_nhits import (
    build_feature_matrix,
    EFFECTIVE_START,
    VAL_END,
    HIST_COLS,
    FUTR_COLS,
)
from src.shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

# Scalable hyperparameters (all env-overridable)
EPOCHS = int(os.environ.get("V16D_EPOCHS", "500"))
BS = int(os.environ.get("V16D_BS", "64"))
D_MODEL = int(os.environ.get("V16D_D_MODEL", "128"))
N_HEAD = int(os.environ.get("V16D_N_HEAD", "4"))
N_LAYERS = int(os.environ.get("V16D_N_LAYERS", "3"))
DIM_FF = int(os.environ.get("V16D_DIM_FF", "384"))
DROPOUT = float(os.environ.get("V16D_DROPOUT", "0.2"))

default_out = (
    f"v16d_noengfeat_"
    f"d{D_MODEL}_ff{DIM_FF}_l{N_LAYERS}_h{N_HEAD}_bs{BS}_ep{EPOCHS}"
)
OUT_DIR = OUTPUT_DIR / os.environ.get("V16D_OUT_DIR", default_out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KW = dict(
    d_model=D_MODEL,
    nhead=N_HEAD,
    nlayers=N_LAYERS,
    dim_ff=DIM_FF,
    dropout=DROPOUT,
    c_point=C_POINT,
)

# Load sequence data
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

logger.info("=" * 60)
logger.info(
    "V16d NO-ENGFEAT  ep=%d bs=%d d=%d ff=%d L=%d H=%d drop=%.2f",
    EPOCHS,
    BS,
    D_MODEL,
    DIM_FF,
    N_LAYERS,
    N_HEAD,
    DROPOUT,
)
logger.info("OUT_DIR: %s", OUT_DIR)
logger.info("=" * 60)

res = train_one(
    seed=0,
    y_norm=y_norm,
    hist_norm=hist_norm,
    futr_norm=futr_norm,
    ts=ts,
    raw_y=raw_y,
    y_mean=y_mean,
    y_std=y_std,
    epochs=EPOCHS,
    bs=BS,
    model_kw=MODEL_KW,
    out_dir=OUT_DIR,
)

paths = [res["path"]]

logger.info("── Predicting on training set ──")
p24_tr, a24_tr, dates_tr = predict_period(
    paths,
    y_norm,
    hist_norm,
    futr_norm,
    ts,
    raw_y,
    y_mean,
    y_std,
    EFFECTIVE_START,
    VAL_END,
    model_kw=MODEL_KW,
)
plot_train_week(p24_tr, a24_tr, dates_tr, OUT_DIR)

logger.info("── Predicting on test set ──")
p24, a24, dates = predict_test(
    paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std, model_kw=MODEL_KW
)
plot_hourly_typical(p24, a24, dates, OUT_DIR)
result = plot_24step(p24, a24, dates, OUT_DIR)
result.to_csv(OUT_DIR / "da_result.csv")

af = result["actual"].values
pf = result["predicted"].values
mae = np.mean(np.abs(af - pf))
rmse = np.sqrt(np.mean((af - pf) ** 2))
shape = compute_shape_report(af, pf, result.index)

logger.info("=" * 60)
logger.info("V16d-NOENGFEAT RESULTS")
logger.info("  MAE:          %.2f", mae)
logger.info("  RMSE:         %.2f", rmse)
for k, v in shape.items():
    logger.info("  %-18s %.4f", k, v)
logger.info("=" * 60)

summary = {"mae": mae, "rmse": rmse, **shape}
pd.Series(summary).to_csv(OUT_DIR / "metrics_summary.csv")
