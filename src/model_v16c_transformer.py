"""
V16c — 对齐滑动窗口 + 小规模 Transformer（仅预测电价）

默认运行：单 seed、固定 MAX_EPOCHS（50）轮。
  - 原验证集时间段并入训练（EFFECTIVE_START～VAL_END）。
  - 每轮在 eval 下对「完整训练集」与「测试集」各算一遍按日 24h 聚合的 MAE / profile_corr
    并打印（与测试集 output 口径一致）。
  - 第 50 轮结束后保存最后一轮权重，再生成测试集曲线与 CSV。

注意：训练过程中观察 test 指标会导致测试集被反复查看，若据此改模型则存在泄露风险；
此处仅按你的实验设定实现，不做基于 test 的选模。

序列构造（无 forward-fill）：
  预测时刻索引 t（day D, step s）：
    Lag1 轴（target + Lag1 hist + Lag0 futr）：672 步，时刻 [t-767, t-96]（止于 D-1 同时刻）
    Lag2 轴（Lag2 hist）：672 步，时刻 [t-863, t-192]（止于 D-2 同时刻）
  每个位置 i 堆叠：y(lag1_i), hist_lag1(lag1_i), hist_lag2(lag2_i), futr(lag1_i)

模型：
  Conv1d patch(k=4,s=4): 672 → 168 tokens, d_model=32
  可学习位置编码 + TransformerEncoder×2, nhead=4, dim_ff=96
  序列 mean pool + 点特征 MLP → 回归头 → 1（归一化电价）

最早有效样本：t >= 863（保证 lag2 窗口起点 >= 0）
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
    TARGET, HIST_COLS, FUTR_COLS, N_LAG1, N_HIST, N_FUTR, SPD,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V16C_DIR = OUTPUT_DIR / "v16c_transformer"
V16C_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MAX_EPOCHS = 50

LOOKBACK = 672
LAG1_END_OFF = SPD * 1   # t - 96 → D-1 同时刻
LAG2_END_OFF = SPD * 2   # t - 192 → D-2 同时刻
LAG1_START_OFF = LAG1_END_OFF + LOOKBACK - 1  # 767
LAG2_START_OFF = LAG2_END_OFF + LOOKBACK - 1  # 863
MIN_T = LAG2_START_OFF

C_SEQ = 1 + N_HIST + N_FUTR
C_POINT = N_FUTR + 4


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

class AlignedSeqDataset(Dataset):
    """对齐双轴序列 + 预测日 Lag0 点特征。"""

    def __init__(self, y_norm, hist_norm, futr_norm, ts,
                 period_start, period_end):
        self.y = y_norm
        self.hist = hist_norm
        self.futr = futr_norm
        self.ts = ts
        self.indices = []
        n = len(y_norm)
        for t in range(MIN_T, n):
            ts_t = pd.Timestamp(ts[t])
            if ts_t < period_start or ts_t > period_end:
                continue
            if not np.isfinite(y_norm[t]):
                continue
            self.indices.append(t)
        logger.info(
            "AlignedSeqDataset: %d samples (%s ~ %s), MIN_T=%d",
            len(self.indices), period_start.date(), period_end.date(), MIN_T,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        ts_t = pd.Timestamp(self.ts[t])
        tod = ts_t.hour * 4 + ts_t.minute // 15

        seq = np.zeros((C_SEQ, LOOKBACK), dtype=np.float32)
        for i in range(LOOKBACK):
            i_lag1 = t - LAG1_START_OFF + i
            i_lag2 = t - LAG2_START_OFF + i
            seq[0, i] = self.y[i_lag1]
            seq[1 : 1 + N_LAG1, i] = self.hist[:N_LAG1, i_lag1]
            seq[1 + N_LAG1 : 1 + N_HIST, i] = self.hist[N_LAG1:N_HIST, i_lag2]
            seq[1 + N_HIST :, i] = self.futr[:, i_lag1]

        futr_t = self.futr[:, t].copy()
        step_norm = tod / 95.0
        is_wknd = 1.0 if ts_t.dayofweek >= 5 else 0.0
        month_sin = np.sin(2.0 * np.pi * ts_t.month / 12.0)
        month_cos = np.cos(2.0 * np.pi * ts_t.month / 12.0)
        point = np.concatenate(
            [futr_t, [step_norm, is_wknd, month_sin, month_cos]]
        ).astype(np.float32)

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
        s = pd.Timestamp(ts[t]).hour * 4 + pd.Timestamp(ts[t]).minute // 15
        if d not in days:
            days[d] = {"p": np.full(SPD, np.nan), "a": np.full(SPD, np.nan)}
        days[d]["p"][s] = preds[i]
        days[d]["a"][s] = raw_y[t]

    maes, corrs = [], []
    for d in sorted(days):
        p, a = days[d]["p"], days[d]["a"]
        if np.isnan(p).any() or np.isnan(a).any():
            continue
        p24 = p.reshape(24, 4).mean(1)
        a24 = a.reshape(24, 4).mean(1)
        maes.append(np.mean(np.abs(p24 - a24)))
        if np.std(a24) > 1e-6 and np.std(p24) > 1e-6:
            c = np.corrcoef(a24, p24)[0, 1]
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
):
    epochs = MAX_EPOCHS if epochs is None else epochs
    _seed(seed)

    train_ds = AlignedSeqDataset(
        y_norm, hist_norm, futr_norm, ts, EFFECTIVE_START, VAL_END
    )
    test_ds = AlignedSeqDataset(
        y_norm, hist_norm, futr_norm, ts, TEST_START, EFFECTIVE_END
    )

    tl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)
    train_eval_l = DataLoader(train_ds, min(256, max(len(train_ds), 1)), shuffle=False)
    test_eval_l = DataLoader(test_ds, min(256, max(len(test_ds), 1)), shuffle=False)

    model = PatchTransformerPrice().to(DEVICE)
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

    ckpt = V16C_DIR / f"seed{seed}.pt"
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


def predict_test(paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std):
    test_ds = AlignedSeqDataset(
        y_norm, hist_norm, futr_norm, ts, TEST_START, EFFECTIVE_END
    )
    tl = DataLoader(test_ds, 256, shuffle=False)

    stacks = []
    for mp in paths:
        model = PatchTransformerPrice().to(DEVICE)
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
    for i, t in enumerate(test_ds.indices):
        d = pd.Timestamp(ts[t]).date()
        s = pd.Timestamp(ts[t]).hour * 4 + pd.Timestamp(ts[t]).minute // 15
        if d not in days:
            days[d] = {"p": np.full(SPD, np.nan), "a": np.full(SPD, np.nan)}
        days[d]["p"][s] = ens[i]
        days[d]["a"][s] = raw_y[t]

    dates = sorted(
        d
        for d in days
        if not np.isnan(days[d]["p"]).any() and not np.isnan(days[d]["a"]).any()
    )

    p96 = np.array([days[d]["p"] for d in dates])
    a96 = np.array([days[d]["a"] for d in dates])
    p24 = p96.reshape(-1, 24, 4).mean(2)
    a24 = a96.reshape(-1, 24, 4).mean(2)
    return p96, a96, p24, a24, dates


# =====================================================================
# Plotting (V16c labels)
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
        ax.plot(x, p_all, "#2196F3", lw=1.0, alpha=0.85, label="V16c(15min)")
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
        ax.set_title(
            f"V16c Transformer 96步 — 第{wi+1}周", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"da_96step_week{wi+1}.png", dpi=120, bbox_inches="tight")
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
        ax.plot(hours_15, p96[di], "#2196F3", lw=1.5, label="V16c")
        r = (
            np.corrcoef(a96[di], p96[di])[0, 1]
            if np.std(a96[di]) > 1e-6 and np.std(p96[di]) > 1e-6
            else 0
        )
        ax.set_title(
            f"{dates[di]}  r={r:.2f}  MAE={daily_mae[di]:.1f}", fontsize=10
        )
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("V16c Transformer 96步 — 典型日", fontsize=14, fontweight="bold")
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
            label="V16c-24h",
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
        ax.set_ylim(200, 600)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"V16c Transformer 24步 — 第{wi+1}周", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"da_week{wi+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_week%d.png", wi + 1)

    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(result["actual"].values, "k-", lw=1.5, label="实际", zorder=3)
    ax.plot(
        result["predicted"].values, "#2196F3", lw=1.0, alpha=0.85, label="V16c"
    )
    ax.set_ylim(200, 600)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title("V16c Transformer — 全测试集", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_full_test.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_full_test.png")
    return result


# =====================================================================
# Main
# =====================================================================

def run_v16c(n_seeds=1, top_k=1):
    logger.info("=" * 60)
    logger.info(
        "V16c Patch Transformer — Start (seeds=%d, max_epochs=%d)",
        n_seeds,
        MAX_EPOCHS,
    )
    logger.info("=" * 60)
    _log_device_info()

    df = build_feature_matrix()
    raw_y = df[TARGET].values.astype(np.float32)
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

    p96, a96, p24, a24, dates = predict_test(
        paths, y_norm, hist_norm, futr_norm, ts, raw_y, y_mean, y_std
    )

    plot_96step(p96, a96, dates, V16C_DIR)
    result = plot_24step(p24, a24, dates, V16C_DIR)
    result.to_csv(V16C_DIR / "da_result.csv")
    logger.info("Saved: da_result.csv (%d rows)", len(result))

    rows_96 = []
    for i, d in enumerate(dates):
        for s in range(96):
            t96 = pd.Timestamp(d) + pd.Timedelta(minutes=s * 15)
            rows_96.append(
                {"ts": t96, "actual": a96[i, s], "predicted": p96[i, s]}
            )
    pd.DataFrame(rows_96).set_index("ts").to_csv(V16C_DIR / "da_result_96step.csv")

    af = result["actual"].values
    pf = result["predicted"].values
    mae = np.mean(np.abs(af - pf))
    rmse = np.sqrt(np.mean((af - pf) ** 2))
    shape = compute_shape_report(af, pf, result.index)

    logger.info("=" * 60)
    logger.info("V16c RESULTS")
    logger.info("  MAE:          %.2f", mae)
    logger.info("  RMSE:         %.2f", rmse)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    summary = {"mae": mae, "rmse": rmse, **shape}
    pd.Series(summary).to_csv(V16C_DIR / "metrics_summary.csv")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v16c()
