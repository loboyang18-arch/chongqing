"""V17 Day-level Transformer: one-day sample -> 24 hourly settlement prices."""
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import OUTPUT_DIR
from src.model_v16_nhits import (
    EFFECTIVE_END,
    EFFECTIVE_START,
    TEST_START,
    VAL_END,
    FUTR_COLS,
    HIST_COLS,
    build_feature_matrix,
)
from src.model_v16d_hourly_settlement import plot_24step, plot_hourly_typical, plot_train_week
from src.shape_metrics import compute_shape_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


@dataclass
class Cfg:
    epochs: int = int(os.environ.get("V17_EPOCHS", "200"))
    bs: int = int(os.environ.get("V17_BS", "32"))
    lr: float = float(os.environ.get("V17_LR", "2e-4"))
    d_model: int = int(os.environ.get("V17_D_MODEL", "128"))
    nhead: int = int(os.environ.get("V17_N_HEAD", "4"))
    nlayers: int = int(os.environ.get("V17_N_LAYERS", "3"))
    dim_ff: int = int(os.environ.get("V17_DIM_FF", "384"))
    dropout: float = float(os.environ.get("V17_DROPOUT", "0.2"))
    past_days: int = int(os.environ.get("V17_PAST_DAYS", "14"))
    # 训练样本起点步长（小时）：1=每天24个起点，3=每天8个起点，24=每天1个起点
    train_anchor_stride_h: int = int(os.environ.get("V17_TRAIN_ANCHOR_STRIDE_H", "1"))
    # 评估起点步长（小时）：默认24，仅评估每天00:00的24点预测
    eval_anchor_stride_h: int = int(os.environ.get("V17_EVAL_ANCHOR_STRIDE_H", "24"))
    out_dir_name: str = os.environ.get("V17_OUT_DIR", "")


class Day24Dataset(Dataset):
    def __init__(
        self,
        dfh: pd.DataFrame,
        raw_target: pd.Series,
        day_list,
        past_hours,
        target_col,
        past_cols,
        futr_cols,
        y_mean,
        y_std,
        anchor_stride_h=24,
    ):
        self.dfh = dfh
        self.raw_target = raw_target
        self.day_list = day_list
        self.past_hours = past_hours
        self.target_col = target_col
        self.past_cols = past_cols
        self.futr_cols = futr_cols
        self.y_mean = y_mean
        self.y_std = y_std
        self.anchor_stride_h = max(1, int(anchor_stride_h))
        self.starts = []

        for d in day_list:
            day0 = pd.Timestamp(d)
            for hh in range(0, 24, self.anchor_stride_h):
                t0 = day0 + pd.Timedelta(hours=hh)
                p0 = t0 - pd.Timedelta(hours=past_hours)
                t1 = t0 + pd.Timedelta(hours=23)
                if p0 not in dfh.index or t1 not in dfh.index:
                    continue
                past = dfh.loc[p0 : t0 - pd.Timedelta(hours=1)]
                dayf = dfh.loc[t0:t1]
                if len(past) != past_hours or len(dayf) != 24:
                    continue
                raw_day = self.raw_target.loc[t0:t1]
                if len(raw_day) != 24 or (not np.isfinite(raw_day.values).all()):
                    continue
                self.starts.append(t0)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t0 = self.starts[i]
        past = self.dfh.loc[t0 - pd.Timedelta(hours=self.past_hours) : t0 - pd.Timedelta(hours=1), self.past_cols].values
        futr = self.dfh.loc[t0 : t0 + pd.Timedelta(hours=23), self.futr_cols].values
        y = self.raw_target.loc[t0 : t0 + pd.Timedelta(hours=23)].values
        y = (y - self.y_mean) / self.y_std
        return (
            torch.tensor(past, dtype=torch.float32),
            torch.tensor(futr, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class Day24Transformer(nn.Module):
    def __init__(self, c_past, c_futr, past_hours, d_model=128, nhead=4, nlayers=3, dim_ff=384, dropout=0.2):
        super().__init__()
        self.past_hours = past_hours
        self.futr_hours = 24
        self.total_len = past_hours + 24
        self.past_proj = nn.Linear(c_past, d_model)
        self.futr_proj = nn.Linear(c_futr, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.total_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, past, futr):
        x_p = self.past_proj(past)
        x_f = self.futr_proj(futr)
        x = torch.cat([x_p, x_f], dim=1)
        x = x + self.pos[:, : x.shape[1], :]
        h = self.encoder(x)
        out = self.head(h[:, -24:, :]).squeeze(-1)
        return out


def _to_day_arrays(model, ds, y_mean, y_std):
    if len(ds) == 0:
        return np.zeros((0, 24)), np.zeros((0, 24)), []
    tl = DataLoader(ds, batch_size=128, shuffle=False)
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for past, futr, y in tl:
            p = model(past.to(DEVICE), futr.to(DEVICE)).cpu().numpy()
            preds.append(p)
            acts.append(y.numpy())
    p24 = np.concatenate(preds, axis=0) * y_std + y_mean
    a24 = np.concatenate(acts, axis=0) * y_std + y_mean
    dates = [pd.Timestamp(t).date() for t in ds.starts]
    return p24, a24, dates


def _daily_metrics(p24, a24, dates):
    if len(dates) == 0:
        return dict(mae=np.nan, rmse=np.nan, profile_corr=np.nan)
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d) + pd.Timedelta(hours=h) for d in dates for h in range(24)]
    )
    af = a24.reshape(-1)
    pf = p24.reshape(-1)
    shape = compute_shape_report(af, pf, idx)
    return {
        "mae": float(np.mean(np.abs(af - pf))),
        "rmse": float(np.sqrt(np.mean((af - pf) ** 2))),
        **shape,
    }


def main():
    cfg = Cfg()
    out_name = cfg.out_dir_name or (
        f"v17_day24_d{cfg.d_model}_ff{cfg.dim_ff}_l{cfg.nlayers}_h{cfg.nhead}_"
        f"bs{cfg.bs}_ep{cfg.epochs}_pd{cfg.past_days}_tas{cfg.train_anchor_stride_h}"
    )
    out_dir = OUTPUT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", DEVICE)
    logger.info("Output: %s", out_dir)

    # 15-min -> hourly grid (minute==0)
    df15 = build_feature_matrix()
    dfh = df15[df15.index.minute == 0].copy()
    target_col = "settlement_da_price"

    # Keep only columns with full hour-level meaning for day-level modeling
    past_cols = [target_col] + HIST_COLS + FUTR_COLS
    past_cols = [c for c in past_cols if c in dfh.columns]
    futr_cols = [c for c in FUTR_COLS if c in dfh.columns]
    # 去重（保序），避免重复列导致 DataFrame 赋值长度不一致
    past_cols = list(dict.fromkeys(past_cols))
    futr_cols = list(dict.fromkeys(futr_cols))

    train_start = pd.Timestamp(EFFECTIVE_START).floor("D")
    train_end = pd.Timestamp(VAL_END).floor("D")
    test_start = pd.Timestamp(TEST_START).floor("D")
    test_end = pd.Timestamp(EFFECTIVE_END).floor("D")

    fit_mask = dfh.index <= pd.Timestamp(VAL_END)
    y_mean = float(dfh.loc[fit_mask, target_col].mean())
    y_std = float(dfh.loc[fit_mask, target_col].std()) + 1e-8

    raw_target = dfh[target_col].copy()

    # Per-channel normalization (features only; labels keep raw_target then normalize once in Dataset)
    pm = dfh.loc[fit_mask, past_cols].mean()
    ps = dfh.loc[fit_mask, past_cols].std().replace(0, np.nan).fillna(1.0)
    fm = dfh.loc[fit_mask, futr_cols].mean()
    fs = dfh.loc[fit_mask, futr_cols].std().replace(0, np.nan).fillna(1.0)
    dfh[past_cols] = ((dfh[past_cols] - pm) / ps).astype(np.float32)
    dfh[futr_cols] = ((dfh[futr_cols] - fm) / fs).astype(np.float32)
    dfh[past_cols] = dfh[past_cols].fillna(0.0)
    dfh[futr_cols] = dfh[futr_cols].fillna(0.0)

    all_days = sorted(pd.unique(dfh.index.floor("D")))
    train_days = [d for d in all_days if train_start <= d <= train_end]
    test_days = [d for d in all_days if test_start <= d <= test_end]
    past_hours = cfg.past_days * 24

    train_ds = Day24Dataset(
        dfh, raw_target, train_days, past_hours, target_col, past_cols, futr_cols, y_mean, y_std,
        anchor_stride_h=cfg.train_anchor_stride_h,
    )
    # 训练监控集：测试期也按训练起点密度看趋势
    test_ds = Day24Dataset(
        dfh, raw_target, test_days, past_hours, target_col, past_cols, futr_cols, y_mean, y_std,
        anchor_stride_h=cfg.train_anchor_stride_h,
    )
    # 最终评估与出图：默认每天00:00起点（可对齐原有日报告）
    train_eval_ds = Day24Dataset(
        dfh, raw_target, train_days, past_hours, target_col, past_cols, futr_cols, y_mean, y_std,
        anchor_stride_h=cfg.eval_anchor_stride_h,
    )
    test_eval_ds = Day24Dataset(
        dfh, raw_target, test_days, past_hours, target_col, past_cols, futr_cols, y_mean, y_std,
        anchor_stride_h=cfg.eval_anchor_stride_h,
    )

    logger.info(
        "Samples: train=%d test(mon)=%d train(eval)=%d test(eval)=%d past_hours=%d",
        len(train_ds), len(test_ds), len(train_eval_ds), len(test_eval_ds), past_hours
    )
    logger.info("Input dims: c_past=%d c_futr=%d", len(past_cols), len(futr_cols))

    dl_kw = {"pin_memory": DEVICE.type == "cuda", "num_workers": 0}
    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=False, **dl_kw)
    model = Day24Transformer(
        c_past=len(past_cols),
        c_futr=len(futr_cols),
        past_hours=past_hours,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        nlayers=cfg.nlayers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %d", n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, 1e-6)

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for past, futr, y in train_loader:
            past = past.to(DEVICE)
            futr = futr.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            p = model(past, futr)
            loss = F.l1_loss(p, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sch.step()

        p24_tr, a24_tr, dates_tr = _to_day_arrays(model, train_eval_ds, y_mean, y_std)
        p24_te, a24_te, dates_te = _to_day_arrays(model, test_eval_ds, y_mean, y_std)
        mtr = _daily_metrics(p24_tr, a24_tr, dates_tr)
        mte = _daily_metrics(p24_te, a24_te, dates_te)
        logger.info(
            "ep%3d loss=%.4f | train_mae=%.2f train_pcorr=%.3f | test_mae=%.2f test_pcorr=%.3f",
            ep,
            float(np.mean(losses) if losses else np.nan),
            mtr["mae"],
            mtr["profile_corr"],
            mte["mae"],
            mte["profile_corr"],
        )

    torch.save(model.state_dict(), out_dir / "seed0.pt")
    logger.info("Saved: %s", out_dir / "seed0.pt")

    # Final outputs/plots
    p24_tr, a24_tr, dates_tr = _to_day_arrays(model, train_eval_ds, y_mean, y_std)
    p24_te, a24_te, dates_te = _to_day_arrays(model, test_eval_ds, y_mean, y_std)
    plot_train_week(p24_tr, a24_tr, dates_tr, out_dir)
    plot_hourly_typical(p24_te, a24_te, dates_te, out_dir)
    res = plot_24step(p24_te, a24_te, dates_te, out_dir)
    res.to_csv(out_dir / "da_result.csv")

    af = res["actual"].values
    pf = res["predicted"].values
    summary = {
        "mae": float(np.mean(np.abs(af - pf))),
        "rmse": float(np.sqrt(np.mean((af - pf) ** 2))),
        **compute_shape_report(af, pf, res.index),
    }
    pd.Series(summary).to_csv(out_dir / "metrics_summary.csv")
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
