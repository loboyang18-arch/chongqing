"""
V16d — 日前小时结算价预测（settlement_da_price）

实验设定：
  - 目标为官方小时 settlement_da_price（整点样本，24点/日）。
  - 保留历史 settlement_da_price 作为可用历史特征（Lag2 轴）。
  - 使用 V16c 的双轴对齐滑窗（无 forward-fill），固定训练 50 epoch。
  - 每轮打印 train/test 的 MAE 与 profile_corr，结束后输出测试集曲线与 CSV。
"""

import logging
import os
import subprocess
from pathlib import Path

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

from .config import OUTPUT_DIR
from .model_v16_nhits import (
    build_feature_matrix,
    EFFECTIVE_START, EFFECTIVE_END, VAL_END, TEST_START,
    HIST_COLS, FUTR_COLS, N_LAG1, N_HIST, N_FUTR, SPD,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

TARGET_SETTLE = "settlement_da_price"
V16D_DIR = OUTPUT_DIR / "v16d_hourly_settlement"

V16D_DIR.mkdir(exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MAX_EPOCHS = 50

LOOKBACK = 672
LAG1_END_OFF = SPD * 1   # t - 96 → D-1 同时刻
LAG2_END_OFF = SPD * 2   # t - 192 → D-2 同时刻
LAG1_START_OFF = LAG1_END_OFF + LOOKBACK - 1  # 767
LAG2_START_OFF = LAG2_END_OFF + LOOKBACK - 1  # 863
MIN_T = LAG2_START_OFF

C_SEQ = 1 + N_HIST + N_FUTR
C_POINT = N_FUTR + 4


def _dataloader_kw():
    """CUDA 下启用 pin_memory 与多进程加载；可通过 V16D_DL_NUM_WORKERS 覆盖。"""
    if DEVICE.type == "cuda":
        default_nw = "4"
    else:
        default_nw = "0"
    nw = int(os.environ.get("V16D_DL_NUM_WORKERS", default_nw))
    nw = max(0, nw)
    kw = {
        "pin_memory": DEVICE.type == "cuda",
        "num_workers": nw,
    }
    if nw > 0:
        kw["persistent_workers"] = True
    return kw


def _log_device_info():
    logger.info("PyTorch %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("MPS available: %s", torch.backends.mps.is_available())
    if DEVICE.type == "mps":
        logger.info("Using device: MPS (Apple GPU)")
    elif DEVICE.type == "cuda":
        logger.info("Using device: CUDA")
    else:
        logger.info("Using device: CPU")
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        logger.info("CPU brand: %s", brand)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass


# =====================================================================
# Dataset
# =====================================================================

OUTLIER_FLOOR = 200.0  # 低于此价格的样本视为向下炸点

class HourlySettlementDataset(Dataset):
    """对齐双轴序列 + 整点 settlement 标签。"""

    def __init__(self, y_norm, hist_norm, futr_norm, ts,
                 period_start, period_end, raw_y=None, filter_outliers=False,
                 eng_feats=None, eng_ts_map=None):
        self.y = y_norm
        self.hist = hist_norm
        self.futr = futr_norm
        self.ts = ts
        self.eng_feats = eng_feats      # (N_hourly, D_eng) normalised
        self.eng_ts_map = eng_ts_map    # {pd.Timestamp -> row_idx}
        self.indices = []
        n_outlier = 0
        n = len(y_norm)
        for t in range(MIN_T, n):
            ts_t = pd.Timestamp(ts[t])
            if ts_t < period_start or ts_t > period_end:
                continue
            if ts_t.minute != 0:
                continue
            if not np.isfinite(y_norm[t]):
                continue
            if filter_outliers and raw_y is not None and raw_y[t] < OUTLIER_FLOOR:
                n_outlier += 1
                continue
            self.indices.append(t)
        logger.info(
            "HourlySettlementDataset: %d samples (%s ~ %s), MIN_T=%d%s",
            len(self.indices), period_start.date(), period_end.date(), MIN_T,
            f", filtered {n_outlier} outliers(<{OUTLIER_FLOOR})" if n_outlier else "",
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        ts_t = pd.Timestamp(self.ts[t])
        hour = ts_t.hour

        seq = np.zeros((C_SEQ, LOOKBACK), dtype=np.float32)
        for i in range(LOOKBACK):
            i_lag1 = t - LAG1_START_OFF + i
            i_lag2 = t - LAG2_START_OFF + i
            seq[0, i] = self.y[i_lag1]
            seq[1 : 1 + N_LAG1, i] = self.hist[:N_LAG1, i_lag1]
            seq[1 + N_LAG1 : 1 + N_HIST, i] = self.hist[N_LAG1:N_HIST, i_lag2]
            seq[1 + N_HIST :, i] = self.futr[:, i_lag1]

        futr_t = self.futr[:, t].copy()
        step_norm = hour / 23.0
        is_wknd = 1.0 if ts_t.dayofweek >= 5 else 0.0
        month_sin = np.sin(2.0 * np.pi * ts_t.month / 12.0)
        month_cos = np.cos(2.0 * np.pi * ts_t.month / 12.0)
        parts = [futr_t, [step_norm, is_wknd, month_sin, month_cos]]

        if self.eng_feats is not None and self.eng_ts_map is not None:
            row_i = self.eng_ts_map.get(ts_t)
            if row_i is not None:
                parts.append(self.eng_feats[row_i])
            else:
                parts.append(np.zeros(self.eng_feats.shape[1], dtype=np.float32))

        point = np.concatenate(parts).astype(np.float32)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(point),
            torch.tensor(self.y[t], dtype=torch.float32),
        )


# =====================================================================
# Model
# =====================================================================

class PatchTransformerPrice(nn.Module):
    def __init__(
        self,
        c_seq=C_SEQ,
        c_point=C_POINT,
        d_model=32,
        nhead=4,
        nlayers=2,
        dim_ff=96,
        patch=4,
        dropout=0.2,
    ):
        super().__init__()
        self.patch = patch
        self.seq_len_tok = (LOOKBACK - patch) // patch + 1
        self.patch_proj = nn.Conv1d(c_seq, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len_tok, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.point_mlp = nn.Sequential(
            nn.Linear(c_point, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, seq, point):
        # seq: (B, C, L)
        x = self.patch_proj(seq).transpose(1, 2)
        x = x + self.pos
        x = self.encoder(x)
        x = x.mean(dim=1)
        p = self.point_mlp(point)
        h = torch.cat([x, p], dim=1)
        return self.head(h).squeeze(-1)


# =====================================================================
# Training / validation
# =====================================================================

def _seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def _day_profile_metrics(model, loader, device, y_mean, y_std, raw_y, indices, ts):
    model.eval()
    all_p = []
    with torch.no_grad():
        for seq, point, _ in loader:
            pred = model(seq.to(device), point.to(device))
            all_p.append(pred.cpu().numpy())
    preds = np.concatenate(all_p) * y_std + y_mean

    days = {}
    for i, t in enumerate(indices):
        d = pd.Timestamp(ts[t]).date()
        s = pd.Timestamp(ts[t]).hour
        if d not in days:
            days[d] = {"p": np.full(24, np.nan), "a": np.full(24, np.nan)}
        days[d]["p"][s] = preds[i]
        days[d]["a"][s] = raw_y[t]

    maes, corrs = [], []
    for d in sorted(days):
        p, a = days[d]["p"], days[d]["a"]
        if np.isnan(p).any() or np.isnan(a).any():
            continue
        maes.append(np.mean(np.abs(p - a)))
        if np.std(a) > 1e-6 and np.std(p) > 1e-6:
            c = np.corrcoef(a, p)[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    mae = np.mean(maes) if maes else 999.0
    pcorr = np.mean(corrs) if corrs else 0.0
    return mae, pcorr


def train_one(
    seed,
    y_norm,
    hist_norm,
    futr_norm,
    ts,
    raw_y,
    y_mean,
    y_std,
    epochs=None,
    lr=1e-4,
    bs=64,
    model_kw=None,
    out_dir=None,
    eng_feats=None,
    eng_ts_map=None,
):
    epochs = MAX_EPOCHS if epochs is None else epochs
    if out_dir is None:
        out_dir = V16D_DIR
    _seed(seed)

    ds_kw = dict(eng_feats=eng_feats, eng_ts_map=eng_ts_map)
    train_ds = HourlySettlementDataset(
        y_norm, hist_norm, futr_norm, ts, EFFECTIVE_START, VAL_END,
        raw_y=raw_y, filter_outliers=True, **ds_kw,
    )
    test_ds = HourlySettlementDataset(
        y_norm, hist_norm, futr_norm, ts, TEST_START, EFFECTIVE_END,
        **ds_kw,
    )

    _dl = _dataloader_kw()
    tl = DataLoader(train_ds, bs, shuffle=True, drop_last=True, **_dl)
    train_eval_l = DataLoader(
        train_ds, min(256, max(len(train_ds), 1)), shuffle=False, **_dl
    )
    test_eval_l = DataLoader(
        test_ds, min(256, max(len(test_ds), 1)), shuffle=False, **_dl
    )

    model = PatchTransformerPrice(**(model_kw or {})).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if seed == 0:
        logger.info("Model params: %d", n_params)
        logger.info(
            "Train (merged): %d samples | Test (monitor): %d samples",
            len(train_ds),
            len(test_ds),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)

    ckpt = out_dir / f"seed{seed}.pt"
    last_test_mae, last_test_pcorr = 0.0, 0.0

    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for seq, point, price_t in tl:
            seq, point = seq.to(DEVICE), point.to(DEVICE)
            price_t = price_t.to(DEVICE)
            opt.zero_grad()
            pred = model(seq, point)
            loss = F.l1_loss(pred, price_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        train_mae, train_pcorr = _day_profile_metrics(
            model,
            train_eval_l,
            DEVICE,
            y_mean,
            y_std,
            raw_y,
            train_ds.indices,
            ts,
        )
        last_test_mae, last_test_pcorr = _day_profile_metrics(
            model,
            test_eval_l,
            DEVICE,
            y_mean,
            y_std,
            raw_y,
            test_ds.indices,
            ts,
        )

        logger.info(
            "  [seed%d] ep%3d  loss=%.4f  train_mae=%.2f  train_pcorr=%.3f  "
            "test_mae=%.2f  test_pcorr=%.3f",
            seed,
            ep,
            ep_loss / max(nb, 1),
            train_mae,
            train_pcorr,
            last_test_mae,
            last_test_pcorr,
        )

    torch.save(model.state_dict(), ckpt)
    logger.info(
        "  [seed%d] finished %d epochs, saved last epoch -> %s",
        seed,
        epochs,
        ckpt,
    )

    return {
        "seed": seed,
        "last_epoch": epochs - 1,
        "last_test_mae": last_test_mae,
        "last_test_pcorr": last_test_pcorr,
        "path": ckpt,
    }


def predict_period(paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std,
                    period_start, period_end, model_kw=None,
                    eng_feats=None, eng_ts_map=None):
    """Predict over an arbitrary date range and return (p24, a24, dates)."""
    ds = HourlySettlementDataset(
        y_norm, hist_norm, futr_norm, ts, period_start, period_end,
        eng_feats=eng_feats, eng_ts_map=eng_ts_map,
    )
    tl = DataLoader(ds, 256, shuffle=False, **_dataloader_kw())

    stacks = []
    for mp in paths:
        model = PatchTransformerPrice(**(model_kw or {})).to(DEVICE)
        model.load_state_dict(
            torch.load(mp, map_location=DEVICE, weights_only=True)
        )
        model.eval()
        preds = []
        with torch.no_grad():
            for seq, point, _ in tl:
                pred = model(seq.to(DEVICE), point.to(DEVICE))
                preds.append(pred.cpu().numpy())
        stacks.append(np.concatenate(preds))

    ens = np.stack(stacks).mean(0) * y_std + y_mean

    days = {}
    for i, t in enumerate(ds.indices):
        d = pd.Timestamp(ts[t]).date()
        s = pd.Timestamp(ts[t]).hour
        if d not in days:
            days[d] = {"p": np.full(24, np.nan), "a": np.full(24, np.nan)}
        days[d]["p"][s] = ens[i]
        days[d]["a"][s] = raw_y[t]

    dates = sorted(
        d
        for d in days
        if not np.isnan(days[d]["p"]).any() and not np.isnan(days[d]["a"]).any()
    )

    p24 = np.array([days[d]["p"] for d in dates])
    a24 = np.array([days[d]["a"] for d in dates])
    return p24, a24, dates


def predict_test(paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std,
                  model_kw=None, eng_feats=None, eng_ts_map=None):
    return predict_period(
        paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std,
        TEST_START, EFFECTIVE_END, model_kw=model_kw,
        eng_feats=eng_feats, eng_ts_map=eng_ts_map,
    )


# =====================================================================
# Plotting
# =====================================================================

def _cn_font():
    for p in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]:
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


def plot_hourly_typical(p24, a24, dates, out_dir):
    _cn_font()
    n = len(dates)
    if n == 0:
        return

    for wi, w_idxs in enumerate(_week_groups(dates)):
        fig, ax = plt.subplots(figsize=(22, 5))
        a_all = np.concatenate([a24[i] for i in w_idxs])
        p_all = np.concatenate([p24[i] for i in w_idxs])
        x = np.arange(len(a_all))
        ax.plot(x, a_all, "k-", lw=1.5, label="实际(1h)", zorder=3)
        ax.plot(x, p_all, "#2196F3", lw=1.0, alpha=0.85, label="V16d(1h)")
        ticks, labels = [], []
        pos = 0
        for i in w_idxs:
            if pos > 0:
                ax.axvline(pos, color="gray", ls="--", alpha=0.3, lw=0.8)
            ticks.append(pos + 12)
            if np.std(a24[i]) > 1e-6 and np.std(p24[i]) > 1e-6:
                r = np.corrcoef(a24[i], p24[i])[0, 1]
                labels.append(f"{dates[i]}\nr={r:.2f}")
            else:
                labels.append(str(dates[i]))
            pos += 24
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(200, 500)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"V16d 小时预测 — 第{wi+1}周", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"da_hour_week{wi+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_hour_week%d.png", wi + 1)

    daily_mae = [np.mean(np.abs(a24[i] - p24[i])) for i in range(n)]
    med = np.median(daily_mae)
    typical = sorted(range(n), key=lambda i: abs(daily_mae[i] - med))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    hours_24 = np.arange(24)
    for j, di in enumerate(sorted(typical)):
        ax = axes[j // 3][j % 3]
        ax.plot(hours_24, a24[di], "k-", lw=2, label="实际")
        ax.plot(hours_24, p24[di], "#2196F3", lw=1.5, label="V16d")
        r = (
            np.corrcoef(a24[di], p24[di])[0, 1]
            if np.std(a24[di]) > 1e-6 and np.std(p24[di]) > 1e-6
            else 0
        )
        ax.set_title(
            f"{dates[di]}  r={r:.2f}  MAE={daily_mae[di]:.1f}", fontsize=10
        )
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_ylim(200, 500)
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("V16d 小时预测 — 典型日", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_hour_typical.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_hour_typical.png")


def plot_train_week(p24, a24, dates, out_dir):
    """绘制训练集最后一周的逐小时预测 vs 实际，评估拟合能力。"""
    _cn_font()
    n = len(dates)
    if n == 0:
        logger.info("No training dates to plot.")
        return

    last7_start = max(0, n - 7)
    sel = list(range(last7_start, n))

    fig, ax = plt.subplots(figsize=(22, 5))
    a_all = np.concatenate([a24[i] for i in sel])
    p_all = np.concatenate([p24[i] for i in sel])
    x = np.arange(len(a_all))
    ax.plot(x, a_all, "k-", lw=1.5, label="实际(1h)", zorder=3)
    ax.plot(x, p_all, "#E91E63", lw=1.0, alpha=0.85, label="V16d训练集(1h)")

    ticks, labels = [], []
    pos = 0
    for i in sel:
        if pos > 0:
            ax.axvline(pos, color="gray", ls="--", alpha=0.3, lw=0.8)
        ticks.append(pos + 12)
        if np.std(a24[i]) > 1e-6 and np.std(p24[i]) > 1e-6:
            r = np.corrcoef(a24[i], p24[i])[0, 1]
            mae_d = np.mean(np.abs(a24[i] - p24[i]))
            labels.append(f"{dates[i]}\nr={r:.2f} MAE={mae_d:.1f}")
        else:
            labels.append(str(dates[i]))
        pos += 24
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(200, 500)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"V16d 训练集最后一周拟合 ({dates[sel[0]]} ~ {dates[sel[-1]]})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "da_train_last_week.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_train_last_week.png")

    fig, axes = plt.subplots(2, 4, figsize=(22, 8)) if len(sel) > 4 else plt.subplots(1, min(len(sel), 4), figsize=(5.5 * min(len(sel), 4), 4))
    if len(sel) <= 4:
        axes = np.atleast_1d(axes)
    else:
        axes = axes.flatten()
    hours_24 = np.arange(24)
    for j, di in enumerate(sel):
        if j >= len(axes):
            break
        ax = axes[j]
        ax.plot(hours_24, a24[di], "k-", lw=2, label="实际")
        ax.plot(hours_24, p24[di], "#E91E63", lw=1.5, label="V16d训练")
        r = (
            np.corrcoef(a24[di], p24[di])[0, 1]
            if np.std(a24[di]) > 1e-6 and np.std(p24[di]) > 1e-6
            else 0
        )
        mae_d = np.mean(np.abs(a24[di] - p24[di]))
        ax.set_title(f"{dates[di]}  r={r:.2f}  MAE={mae_d:.1f}", fontsize=9)
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_ylim(200, 500)
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(len(sel), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("V16d 训练集最后一周 — 逐日拟合", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_train_last_week_daily.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_train_last_week_daily.png")

    daily_mae = [np.mean(np.abs(a24[i] - p24[i])) for i in range(n)]
    daily_corr = []
    for i in range(n):
        if np.std(a24[i]) > 1e-6 and np.std(p24[i]) > 1e-6:
            daily_corr.append(np.corrcoef(a24[i], p24[i])[0, 1])
        else:
            daily_corr.append(0.0)
    train_mae_mean = np.mean(daily_mae)
    train_pcorr_mean = np.mean(daily_corr)
    logger.info(
        "Train set (all %d days): mean_daily_MAE=%.2f  mean_daily_pcorr=%.3f",
        n, train_mae_mean, train_pcorr_mean,
    )


def plot_24step(p24, a24, dates, out_dir):
    _cn_font()
    n = len(dates)
    rows = []
    for i in range(n):
        for h in range(24):
            ts_row = pd.Timestamp(dates[i]) + pd.Timedelta(hours=h)
            rows.append({"ts": ts_row, "actual": a24[i, h], "predicted": p24[i, h]})
    result = pd.DataFrame(rows).set_index("ts")

    for wi, w_idxs in enumerate(_week_groups(dates)):
        start_d, end_d = dates[w_idxs[0]], dates[w_idxs[-1]]
        mask = (result.index.date >= start_d) & (result.index.date <= end_d)
        chunk = result.loc[mask]
        if len(chunk) < 24:
            continue
        fig, ax = plt.subplots(figsize=(18, 5))
        x = range(len(chunk))
        ax.plot(x, chunk["actual"].values, "k-", lw=1.8, label="实际", zorder=3)
        ax.plot(
            x,
            chunk["predicted"].values,
            "#2196F3",
            lw=1.3,
            alpha=0.85,
            label="V16d-24h",
        )
        ticks, labels = [], []
        for di in w_idxs:
            d = dates[di]
            d_mask = chunk.index.date == d
            idxs = np.where(d_mask)[0]
            if len(idxs) > 0:
                if idxs[0] > 0:
                    ax.axvline(
                        idxs[0], color="gray", ls="--", alpha=0.3, lw=0.8
                    )
                ticks.append(idxs[0] + 12)
                if np.std(a24[di]) > 1e-6 and np.std(p24[di]) > 1e-6:
                    r = np.corrcoef(a24[di], p24[di])[0, 1]
                    labels.append(f"{d}\nr={r:.2f}")
                else:
                    labels.append(str(d))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(200, 500)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"V16d 小时预测(24点) — 第{wi+1}周", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"da_week{wi+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_week%d.png", wi + 1)

    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(result["actual"].values, "k-", lw=1.5, label="实际", zorder=3)
    ax.plot(
        result["predicted"].values, "#2196F3", lw=1.0, alpha=0.85, label="V16d"
    )
    ax.set_ylim(200, 500)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title("V16d 小时结算价预测 — 全测试集", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_full_test.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_full_test.png")
    return result


# =====================================================================
# Main
# =====================================================================

def run_v16d(n_seeds=1, top_k=1):
    logger.info("=" * 60)
    logger.info(
        "V16d Hourly Settlement — Start (seeds=%d, max_epochs=%d)",
        n_seeds,
        MAX_EPOCHS,
    )
    logger.info("=" * 60)
    _log_device_info()

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

    results = []
    for seed in range(n_seeds):
        logger.info("─── Training seed %d/%d ───", seed + 1, n_seeds)
        results.append(
            train_one(seed, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std)
        )

    results.sort(key=lambda r: r["seed"])
    top = results[:top_k]
    paths = [r["path"] for r in top]
    logger.info(
        "Checkpoints (last epoch, seed order): %s",
        [
            (
                r["seed"],
                f"last_ep_test_pcorr={r['last_test_pcorr']:.3f}",
                f"last_ep_test_mae={r['last_test_mae']:.2f}",
            )
            for r in top
        ],
    )

    # ---- 训练集预测 & 绘图（最后一周拟合） ----
    logger.info("── Predicting on training set ──")
    p24_tr, a24_tr, dates_tr = predict_period(
        paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std,
        EFFECTIVE_START, VAL_END,
    )
    plot_train_week(p24_tr, a24_tr, dates_tr, V16D_DIR)

    # ---- 测试集预测 & 绘图 ----
    p24, a24, dates = predict_test(
        paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std
    )

    plot_hourly_typical(p24, a24, dates, V16D_DIR)
    result = plot_24step(p24, a24, dates, V16D_DIR)
    result.to_csv(V16D_DIR / "da_result.csv")
    logger.info("Saved: da_result.csv (%d rows)", len(result))

    rows_24 = []
    for i, d in enumerate(dates):
        for h in range(24):
            t_h = pd.Timestamp(d) + pd.Timedelta(hours=h)
            rows_24.append(
                {"ts": t_h, "actual": a24[i, h], "predicted": p24[i, h]}
            )
    pd.DataFrame(rows_24).set_index("ts").to_csv(V16D_DIR / "da_result_24h.csv")

    af = result["actual"].values
    pf = result["predicted"].values
    mae = np.mean(np.abs(af - pf))
    rmse = np.sqrt(np.mean((af - pf) ** 2))
    shape = compute_shape_report(af, pf, result.index)

    logger.info("=" * 60)
    logger.info("V16d RESULTS")
    logger.info("  MAE:          %.2f", mae)
    logger.info("  RMSE:         %.2f", rmse)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    summary = {"mae": mae, "rmse": rmse, **shape}
    pd.Series(summary).to_csv(V16D_DIR / "metrics_summary.csv")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v16d()
