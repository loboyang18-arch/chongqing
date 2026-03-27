"""
V16 N-HiTS — 原始15分钟多元序列电价预测

Architecture:
  Input: Conv1d channel encoder (hist→8ch, futr→6ch)
  Core:  3-Stack N-HiTS (pool [16,4,2], downsample [16,4,1])
  Norm:  RevIN (target only) + global z-score (exogenous)
  Loss:  0.4*MAE_96 + 0.4*(1-PearsonCorr_24) + 0.2*AmpError_24

Data:
  lookback=672 (7d@15min), horizon=96 (1d@15min)
  stride=4 (training), day-aligned (val/test)
  Lag1 hist: last (tod) steps forward-fill
  Lag2 hist: last (96+tod) steps forward-fill
"""

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from .config import SOURCE_DIR, OUTPUT_DIR, KEY_SECTIONS
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V16_DIR = OUTPUT_DIR / "v16_nhits"
V16_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LOOKBACK = 672
HORIZON = 96
SPD = 96

EFFECTIVE_START = pd.Timestamp("2025-11-01")
EFFECTIVE_END = pd.Timestamp("2026-03-10 23:45:00")
TRAIN_END = pd.Timestamp("2026-01-25 23:45:00")
VAL_END = pd.Timestamp("2026-02-08 23:45:00")
TEST_START = pd.Timestamp("2026-02-09")

TARGET = "da_clearing_price"
LAG1_HIST = [
    "rt_clearing_price", "reliability_da_price",
    "da_clearing_power", "rt_clearing_volume",
]
LAG2_HIST = [
    "actual_load", "total_gen", "hydro_gen", "non_market_gen", "renewable_gen",
    "tie_line_power", "settlement_da_price", "settlement_rt_price",
    "section_flow_hongban", "section_flow_zitong", "avg_bid_price",
]
FUTR_COLS = [
    "load_forecast", "renewable_fcst", "total_gen_fcst_pm",
    "hydro_gen_fcst_pm", "hydro_gen_fcst_am", "non_market_gen_fcst_pm",
    "renewable_fcst_total_pm", "tie_line_fcst_pm",
    "maintenance_gen_count", "maintenance_grid_count",
    "minute_of_day_sin", "minute_of_day_cos", "dow_sin", "dow_cos",
]
HIST_COLS = LAG1_HIST + LAG2_HIST
N_LAG1 = len(LAG1_HIST)
N_HIST = len(HIST_COLS)
N_FUTR = len(FUTR_COLS)


# =====================================================================
# 1. Data Loading
# =====================================================================

def _load_ts(fname: str, dcol: str, vcol: str, gran: int = 15) -> pd.Series:
    path = SOURCE_DIR / fname
    df = pd.read_csv(path, parse_dates=[dcol])
    s = pd.to_numeric(df[vcol], errors="coerce")
    s.index = df[dcol]
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if gran == 5:
        s = s.resample("15min").mean()
    return s


def build_feature_matrix() -> pd.DataFrame:
    """Load all raw features → unified 15-min DataFrame."""
    logger.info("Building 15-min feature matrix...")
    idx = pd.date_range(EFFECTIVE_START, EFFECTIVE_END, freq="15min")
    df = pd.DataFrame(index=idx)
    df.index.name = "ts"

    spec_15 = {
        "da_clearing_price": ("日前市场交易出清结果.csv", "datetime", "现货日前市场出清电价"),
        "rt_clearing_price": ("实时出清结果.csv", "datetime", "现货实时出清电价"),
        "da_clearing_power": ("日前市场交易出清结果.csv", "datetime", "现货日前市场出清电力"),
        "rt_clearing_volume": ("实时出清结果.csv", "datetime", "现货实时出清电量"),
        "actual_load": ("实际负荷.csv", "datetime", "实际负荷"),
        "total_gen": ("发电总出力.csv", "datetime", "发电总出力"),
        "hydro_gen": ("水电（含抽蓄）总出力.csv", "datetime", "水电（含抽蓄）总出力"),
        "non_market_gen": ("非市场机组总出力.csv", "datetime", "非市场机组总出力"),
        "renewable_gen": ("新能源总出力.csv", "datetime", "新能源总出力"),
        "reliability_da_price": (
            "现货交易可靠性结果查询（用电侧）.csv", "datetime",
            "日前加权平均节点电价(元/MWh)",
        ),
        "load_forecast": ("系统负荷预测.csv", "datetime", "系统负荷预测"),
        "renewable_fcst": ("新能源预测.csv", "datetime", "新能源预测"),
        "total_gen_fcst_pm": ("发电总出力预测.csv", "datetime", "发电总出力预测下午"),
        "hydro_gen_fcst_pm": ("水电（含抽蓄）总出力预测.csv", "datetime", "水电（含抽蓄）总出力预测下午"),
        "hydro_gen_fcst_am": ("水电（含抽蓄）总出力预测.csv", "datetime", "水电（含抽蓄）总出力预测上午"),
        "non_market_gen_fcst_pm": ("非市场机组总出力预测.csv", "datetime", "非市场机组总出力预测下午传输"),
        "renewable_fcst_total_pm": ("新能源总出力预测.csv", "datetime", "新能源总出力预测下午"),
        "tie_line_fcst_pm": ("省间联络线输电曲线预测.csv", "datetime", "省间联络线输电曲线预测下午"),
    }
    for metric, (fn, dc, vc) in spec_15.items():
        try:
            s = _load_ts(fn, dc, vc, 15)
            df[metric] = s.reindex(idx)
            logger.info("  %-30s %d vals", metric, s.reindex(idx).notna().sum())
        except Exception as e:
            logger.error("  FAIL %-30s %s", metric, e)

    try:
        s = _load_ts("省间联络线输电.csv", "datetime", "省间联络线输电", 5)
        df["tie_line_power"] = s.reindex(idx)
        logger.info("  %-30s %d vals (5m→15m)", "tie_line_power",
                    s.reindex(idx).notna().sum())
    except Exception as e:
        logger.error("  FAIL tie_line_power %s", e)

    hourly = pd.read_csv(
        OUTPUT_DIR / "dws_hourly_features.csv", parse_dates=["ts"], index_col="ts",
    )
    for col in ["settlement_da_price", "settlement_rt_price",
                "maintenance_gen_count", "maintenance_grid_count"]:
        if col in hourly.columns:
            s_h = hourly[col].dropna()
            h_end = min(s_h.index.max() + pd.Timedelta(minutes=45), EFFECTIVE_END)
            h_idx = pd.date_range(s_h.index.min(), h_end, freq="15min")
            df[col] = s_h.reindex(h_idx, method="ffill").reindex(idx)
            logger.info("  %-30s hourly→15min", col)

    sec_map = {
        "section_flow_hongban": KEY_SECTIONS[0],
        "section_flow_zitong": KEY_SECTIONS[1],
    }
    try:
        sec_df = pd.read_csv(SOURCE_DIR / "实际运行输电断面约束情况.csv")
        sec_df["时点"] = pd.to_datetime(sec_df["时点"], errors="coerce")
        sec_df = sec_df.dropna(subset=["时点"])
        for short, full in sec_map.items():
            mask = (sec_df["设备名称"] == full) & (sec_df["设备类型"] == "实际潮流")
            sub = sec_df.loc[mask].set_index("时点").sort_index()
            s = pd.to_numeric(sub["值"], errors="coerce")
            s = s[~s.index.duplicated(keep="first")]
            h_end = min(s.index.max() + pd.Timedelta(minutes=45), EFFECTIVE_END)
            h_idx = pd.date_range(s.index.min(), h_end, freq="15min")
            df[short] = s.reindex(h_idx, method="ffill").reindex(idx)
            logger.info("  %-30s hourly→15min", short)
    except Exception as e:
        logger.error("  FAIL section data: %s", e)

    if "avg_bid_price" in hourly.columns:
        s_h = hourly["avg_bid_price"].dropna()
        h_end = min(s_h.index.max() + pd.Timedelta(minutes=45), EFFECTIVE_END)
        h_idx = pd.date_range(s_h.index.min(), h_end, freq="15min")
        df["avg_bid_price"] = s_h.reindex(h_idx, method="ffill").reindex(idx)
        logger.info("  %-30s daily→15min", "avg_bid_price")

    mins = df.index.hour * 60 + df.index.minute
    df["minute_of_day_sin"] = np.sin(2 * np.pi * mins / 1440)
    df["minute_of_day_cos"] = np.cos(2 * np.pi * mins / 1440)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df = df.ffill().bfill().fillna(0.0)

    required = [TARGET] + HIST_COLS + FUTR_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    logger.info("Feature matrix: %d × %d (%s ~ %s)",
                len(df), len(required), df.index[0], df.index[-1])
    return df


# =====================================================================
# 2. Dataset
# =====================================================================

class TSDataset(Dataset):
    def __init__(self, y, hist, futr, ts,
                 period_start, period_end,
                 stride=1, day_aligned=False):
        self.y = y.astype(np.float32)
        self.hist = hist.astype(np.float32)
        self.futr = futr.astype(np.float32)
        self.ts = ts
        self.starts = []
        for t in range(LOOKBACK, len(y) - HORIZON + 1, stride):
            ts_s = pd.Timestamp(ts[t])
            ts_e = pd.Timestamp(ts[t + HORIZON - 1])
            if ts_s < period_start or ts_e > period_end:
                continue
            if day_aligned and (ts_s.hour != 0 or ts_s.minute != 0):
                continue
            if np.isnan(y[t:t + HORIZON]).any():
                continue
            self.starts.append(t)
        logger.info("Dataset: %d windows (%s~%s stride=%d day_align=%s)",
                    len(self.starts), period_start.date(), period_end.date(),
                    stride, day_aligned)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t = self.starts[idx]
        ts_t = pd.Timestamp(self.ts[t])
        tod = ts_t.hour * 4 + ts_t.minute // 15

        y_p = self.y[t - LOOKBACK:t].copy()
        y_t = self.y[t:t + HORIZON].copy()
        h_p = self.hist[:, t - LOOKBACK:t].copy()
        f_p = self.futr[:, t - LOOKBACK:t].copy()
        f_f = self.futr[:, t:t + HORIZON].copy()

        if tod > 0:
            c = LOOKBACK - tod
            for i in range(N_LAG1):
                h_p[i, c:] = h_p[i, c - 1]
            y_p[c:] = y_p[c - 1]

        lag2_ff = SPD + tod
        c2 = LOOKBACK - lag2_ff
        if c2 > 0:
            for i in range(N_LAG1, N_HIST):
                h_p[i, c2:] = h_p[i, c2 - 1]

        return (torch.from_numpy(y_p), torch.from_numpy(h_p),
                torch.from_numpy(f_p), torch.from_numpy(f_f),
                torch.from_numpy(y_t))


# =====================================================================
# 3. RevIN
# =====================================================================

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x):
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        return (x - self._mean) / self._std

    def denormalize(self, x):
        return x * self._std + self._mean


# =====================================================================
# 4. N-HiTS Model
# =====================================================================

class NHiTSBlock(nn.Module):
    def __init__(self, n_past_ch, n_futr_ch, L, H,
                 pool_k, freq_ds, hidden, dropout=0.1):
        super().__init__()
        self.pool = nn.MaxPool1d(pool_k, ceil_mode=True)
        pL = math.ceil(L / pool_k)
        pH = math.ceil(H / pool_k)
        flat = n_past_ch * pL + n_futr_ch * pH
        layers = []
        d = flat
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.proj_b = nn.Linear(d, math.ceil(L / freq_ds))
        self.proj_f = nn.Linear(d, math.ceil(H / freq_ds))
        self.L, self.H = L, H

    def forward(self, x_past, x_futr):
        pp = self.pool(x_past).flatten(1)
        pf = self.pool(x_futr).flatten(1)
        h = self.mlp(torch.cat([pp, pf], 1))
        tb = self.proj_b(h)
        tf = self.proj_f(h)
        bc = F.interpolate(
            tb.unsqueeze(1), self.L, mode="linear", align_corners=False,
        ).squeeze(1)
        fc = F.interpolate(
            tf.unsqueeze(1), self.H, mode="linear", align_corners=False,
        ).squeeze(1)
        return bc, fc


class NHiTS(nn.Module):
    def __init__(self, n_hist=N_HIST, n_futr=N_FUTR,
                 L=LOOKBACK, H=HORIZON,
                 d_hist=4, d_futr=3,
                 pool_ks=(16, 4, 2), freq_ds=(16, 4, 1),
                 hiddens=([128], [128], [64]),
                 dropout=0.2,
                 hist_mean=None, hist_std=None,
                 futr_mean=None, futr_std=None):
        super().__init__()
        self.L, self.H = L, H
        for name, arr in [("hm", hist_mean), ("hs", hist_std),
                          ("fm", futr_mean), ("fs", futr_std)]:
            if arr is not None:
                self.register_buffer(
                    name, torch.tensor(arr, dtype=torch.float32).view(1, -1, 1))

        self.revin = RevIN()
        self.hist_enc = nn.Sequential(
            nn.Conv1d(n_hist, d_hist * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_hist * 2, d_hist, 1),
        )
        self.futr_enc = nn.Sequential(
            nn.Conv1d(n_futr, d_futr * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_futr * 2, d_futr, 1),
        )
        npc = 1 + d_hist + d_futr
        nfc = d_futr
        self.blocks = nn.ModuleList([
            NHiTSBlock(npc, nfc, L, H, pk, fd, hid, dropout)
            for pk, fd, hid in zip(pool_ks, freq_ds, hiddens)
        ])

    def forward(self, y_past, hist_past, futr_past, futr_future):
        y_n = self.revin.normalize(y_past)
        if hasattr(self, "hm"):
            hist_past = (hist_past - self.hm) / (self.hs + 1e-8)
        if hasattr(self, "fm"):
            futr_past = (futr_past - self.fm) / (self.fs + 1e-8)
            futr_future = (futr_future - self.fm) / (self.fs + 1e-8)

        he = self.hist_enc(hist_past)
        fep = self.futr_enc(futr_past)
        fef = self.futr_enc(futr_future)

        y_res = y_n
        fc_sum = torch.zeros(y_past.shape[0], self.H, device=y_past.device)
        for block in self.blocks:
            past_in = torch.cat([y_res.unsqueeze(1), he, fep], dim=1)
            bc, fc = block(past_in, fef)
            y_res = y_res - bc
            fc_sum = fc_sum + fc
        return self.revin.denormalize(fc_sum)


# =====================================================================
# 5. Composite Loss
# =====================================================================

class CompositeLoss(nn.Module):
    def __init__(self, w_mae=0.4, w_corr=0.4, w_amp=0.2):
        super().__init__()
        self.w_mae, self.w_corr, self.w_amp = w_mae, w_corr, w_amp

    def forward(self, pred, actual):
        mae = (pred - actual).abs().mean()
        p24 = pred.reshape(-1, 24, 4).mean(2)
        a24 = actual.reshape(-1, 24, 4).mean(2)
        pc = p24 - p24.mean(1, keepdim=True)
        ac = a24 - a24.mean(1, keepdim=True)
        ps = pc.pow(2).sum(1).sqrt()
        as_ = ac.pow(2).sum(1).sqrt()
        ok = (ps > 1e-4) & (as_ > 1e-4)
        if ok.sum() > 0:
            corr = ((pc[ok] * ac[ok]).sum(1) / (ps[ok] * as_[ok] + 1e-8)).mean()
        else:
            corr = torch.tensor(0.0, device=pred.device)
        amp_p = p24.max(1).values - p24.min(1).values
        amp_a = a24.max(1).values - a24.min(1).values
        amp_err = (amp_p - amp_a).abs().mean()
        loss = self.w_mae * mae + self.w_corr * (1 - corr) + self.w_amp * amp_err / 100
        return loss, mae.item(), corr.item(), amp_err.item()


# =====================================================================
# 6. Training
# =====================================================================

def _seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def _val_metrics(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for yp, hp, fp, ff, yt in loader:
            yp, hp, fp, ff, yt = [x.to(device) for x in (yp, hp, fp, ff, yt)]
            p = model(yp, hp, fp, ff)
            preds.append(p.reshape(-1, 24, 4).mean(2).cpu().numpy())
            actuals.append(yt.reshape(-1, 24, 4).mean(2).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mae = np.mean(np.abs(preds - actuals))
    corrs = []
    for i in range(len(preds)):
        if np.std(actuals[i]) > 1e-6 and np.std(preds[i]) > 1e-6:
            c = np.corrcoef(actuals[i], preds[i])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    pcorr = np.mean(corrs) if corrs else 0.0
    return mae, pcorr


def train_one(seed, y, hist, futr, ts, hm, hs, fm, fs,
              epochs=500, patience=60, lr=5e-4, bs=64):
    _seed(seed)
    train_ds = TSDataset(y, hist, futr, ts,
                         EFFECTIVE_START, TRAIN_END, stride=1)
    val_ds = TSDataset(y, hist, futr, ts,
                       TRAIN_END + pd.Timedelta(minutes=15), VAL_END,
                       stride=SPD, day_aligned=True)

    tl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, max(len(val_ds), 1), shuffle=False)

    model = NHiTS(hist_mean=hm, hist_std=hs,
                  futr_mean=fm, futr_std=fs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
    loss_fn = CompositeLoss()

    best_sc, best_ep = 1e9, 0
    ckpt = V16_DIR / f"seed{seed}.pt"

    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for yp, hp, fp, ff, yt in tl:
            yp, hp, fp, ff, yt = [x.to(DEVICE) for x in (yp, hp, fp, ff, yt)]
            opt.zero_grad()
            p = model(yp, hp, fp, ff)
            loss, _, _, _ = loss_fn(p, yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        mae, pcorr = _val_metrics(model, vl, DEVICE)
        sc = mae / 100 + 0.5 * (1 - pcorr)
        if sc < best_sc:
            best_sc, best_ep = sc, ep
            torch.save(model.state_dict(), ckpt)
        if ep % 20 == 0 or ep == epochs - 1:
            logger.info(
                "  [seed%d] ep%3d  loss=%.4f  val_mae=%.2f  val_pcorr=%.3f  "
                "sc=%.4f%s",
                seed, ep, ep_loss / max(nb, 1), mae, pcorr, sc,
                " *" if ep == best_ep else "",
            )
        if ep - best_ep >= patience:
            logger.info("  [seed%d] early stop ep%d (best=%d)", seed, ep, best_ep)
            break

    return {"seed": seed, "best_epoch": best_ep,
            "best_score": best_sc, "path": ckpt}


# =====================================================================
# 7. Inference
# =====================================================================

def predict_ensemble(paths, y, hist, futr, ts, hm, hs, fm, fs):
    test_ds = TSDataset(y, hist, futr, ts,
                        TEST_START, EFFECTIVE_END,
                        stride=SPD, day_aligned=True)
    if len(test_ds) == 0:
        logger.warning("No test windows!")
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    tl = DataLoader(test_ds, len(test_ds), shuffle=False)
    for batch in tl:
        actual_96 = batch[4].numpy()
        break

    model_preds = []
    for mp in paths:
        model = NHiTS(hist_mean=hm, hist_std=hs,
                      futr_mean=fm, futr_std=fs).to(DEVICE)
        model.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
        model.eval()
        with torch.no_grad():
            for batch in tl:
                yp, hp, fp, ff = [x.to(DEVICE) for x in batch[:4]]
                model_preds.append(model(yp, hp, fp, ff).cpu().numpy())
                break

    p96 = np.stack(model_preds).mean(0)
    p24 = p96.reshape(-1, 24, 4).mean(2)
    a24 = actual_96.reshape(-1, 24, 4).mean(2)
    dates = [pd.Timestamp(ts[t]).date() for t in test_ds.starts]
    return p96, actual_96, p24, a24, dates


# =====================================================================
# 8. Plotting
# =====================================================================

def _cn_font():
    for p in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Medium.ttc"]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _week_groups(dates):
    weeks, week = [], [0]
    for i in range(1, len(dates)):
        if (dates[i] - dates[week[0]]).days >= 7:
            weeks.append(week)
            week = [i]
        else:
            week.append(i)
    if week:
        weeks.append(week)
    return weeks


def plot_96step(p96, a96, dates, out_dir):
    _cn_font()
    n = len(dates)
    if n == 0:
        return

    for wi, w_idxs in enumerate(_week_groups(dates)):
        fig, ax = plt.subplots(figsize=(22, 5))
        a_all = np.concatenate([a96[i] for i in w_idxs])
        p_all = np.concatenate([p96[i] for i in w_idxs])
        x = np.arange(len(a_all))
        ax.plot(x, a_all, "k-", lw=1.5, label="实际(15min)", zorder=3)
        ax.plot(x, p_all, "#E91E63", lw=1.0, alpha=0.85, label="V16(15min)")

        ticks, labels = [], []
        pos = 0
        for i in w_idxs:
            if pos > 0:
                ax.axvline(pos, color="gray", ls="--", alpha=0.3, lw=0.8)
            ticks.append(pos + 48)
            if np.std(a96[i]) > 1e-6 and np.std(p96[i]) > 1e-6:
                r = np.corrcoef(a96[i], p96[i])[0, 1]
                labels.append(f"{dates[i]}\nr={r:.2f}")
            else:
                labels.append(str(dates[i]))
            pos += SPD
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"V16 N-HiTS 96步预测 — 第{wi+1}周",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"da_96step_week{wi+1}.png",
                    dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_96step_week%d.png", wi + 1)

    daily_mae = [np.mean(np.abs(a96[i] - p96[i])) for i in range(n)]
    med = np.median(daily_mae)
    typical = sorted(range(n), key=lambda i: abs(daily_mae[i] - med))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    hours_15 = np.arange(96) * 0.25
    for j, di in enumerate(sorted(typical)):
        ax = axes[j // 3][j % 3]
        ax.plot(hours_15, a96[di], "k-", lw=2, label="实际")
        ax.plot(hours_15, p96[di], "#E91E63", lw=1.5, label="V16")
        r = (np.corrcoef(a96[di], p96[di])[0, 1]
             if np.std(a96[di]) > 1e-6 and np.std(p96[di]) > 1e-6 else 0)
        ax.set_title(f"{dates[di]}  r={r:.2f}  MAE={daily_mae[di]:.1f}",
                     fontsize=10)
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("V16 N-HiTS 96步预测 — 典型日",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_96step_typical.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_96step_typical.png")


def plot_24step(p24, a24, dates, out_dir):
    _cn_font()
    n = len(dates)
    rows = []
    for i in range(n):
        for h in range(24):
            ts = pd.Timestamp(dates[i]) + pd.Timedelta(hours=h)
            rows.append({"ts": ts, "actual": a24[i, h], "predicted": p24[i, h]})
    result = pd.DataFrame(rows).set_index("ts")

    for wi, w_idxs in enumerate(_week_groups(dates)):
        start_d, end_d = dates[w_idxs[0]], dates[w_idxs[-1]]
        mask = (result.index.date >= start_d) & (result.index.date <= end_d)
        chunk = result.loc[mask]
        if len(chunk) < 24:
            continue
        fig, ax = plt.subplots(figsize=(18, 5))
        x = range(len(chunk))
        ax.plot(x, chunk["actual"].values, "k-", lw=1.8,
                label="实际", zorder=3)
        ax.plot(x, chunk["predicted"].values, "#E91E63", lw=1.3,
                alpha=0.85, label="V16-24h")
        ticks, labels = [], []
        for di in w_idxs:
            d = dates[di]
            d_mask = chunk.index.date == d
            idxs = np.where(d_mask)[0]
            if len(idxs) > 0:
                if idxs[0] > 0:
                    ax.axvline(idxs[0], color="gray", ls="--",
                               alpha=0.3, lw=0.8)
                ticks.append(idxs[0] + 12)
                if np.std(a24[di]) > 1e-6 and np.std(p24[di]) > 1e-6:
                    r = np.corrcoef(a24[di], p24[di])[0, 1]
                    labels.append(f"{d}\nr={r:.2f}")
                else:
                    labels.append(str(d))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(200, 600)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"V16 N-HiTS 24步预测 — 第{wi+1}周",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"da_week{wi+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_week%d.png", wi + 1)

    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(result["actual"].values, "k-", lw=1.5, label="实际", zorder=3)
    ax.plot(result["predicted"].values, "#E91E63", lw=1.0,
            alpha=0.85, label="V16")
    ax.set_ylim(200, 600)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title("V16 N-HiTS — 全测试集", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_full_test.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_full_test.png")
    return result


# =====================================================================
# 9. Main
# =====================================================================

def run_v16(n_seeds=5, top_k=3):
    logger.info("=" * 60)
    logger.info("V16 N-HiTS — Start")
    logger.info("=" * 60)

    df = build_feature_matrix()
    y = df[TARGET].values.astype(np.float32)
    hist = df[HIST_COLS].values.T.astype(np.float32)
    futr = df[FUTR_COLS].values.T.astype(np.float32)
    ts = df.index.values

    train_mask = df.index <= TRAIN_END
    hm = hist[:, train_mask].mean(axis=1)
    hs = hist[:, train_mask].std(axis=1) + 1e-8
    fm = futr[:, train_mask].mean(axis=1)
    fs = futr[:, train_mask].std(axis=1) + 1e-8

    results = []
    for seed in range(n_seeds):
        logger.info("─── Training seed %d/%d ───", seed + 1, n_seeds)
        res = train_one(seed, y, hist, futr, ts, hm, hs, fm, fs)
        results.append(res)

    results.sort(key=lambda r: r["best_score"])
    top = results[:top_k]
    paths = [r["path"] for r in top]
    logger.info("Top-%d seeds: %s", top_k,
                [(r["seed"], f"sc={r['best_score']:.4f}") for r in top])

    p96, a96, p24, a24, dates = predict_ensemble(
        paths, y, hist, futr, ts, hm, hs, fm, fs)

    plot_96step(p96, a96, dates, V16_DIR)
    result = plot_24step(p24, a24, dates, V16_DIR)
    result.to_csv(V16_DIR / "da_result.csv")
    logger.info("Saved: da_result.csv (%d rows)", len(result))

    rows_96 = []
    for i, d in enumerate(dates):
        for s in range(96):
            t96 = pd.Timestamp(d) + pd.Timedelta(minutes=s * 15)
            rows_96.append({"ts": t96, "actual": a96[i, s],
                            "predicted": p96[i, s]})
    pd.DataFrame(rows_96).set_index("ts").to_csv(
        V16_DIR / "da_result_96step.csv")

    af = result["actual"].values
    pf = result["predicted"].values
    mae = np.mean(np.abs(af - pf))
    rmse = np.sqrt(np.mean((af - pf) ** 2))
    shape = compute_shape_report(af, pf, result.index)

    logger.info("=" * 60)
    logger.info("V16 RESULTS")
    logger.info("  MAE:          %.2f", mae)
    logger.info("  RMSE:         %.2f", rmse)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    summary = {"mae": mae, "rmse": rmse, **shape}
    pd.Series(summary).to_csv(V16_DIR / "metrics_summary.csv")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v16()
