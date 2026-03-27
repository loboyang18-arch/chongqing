"""
V16b — Dual-Head Single-Point Prediction (双头单点预测)

Architecture:
  每个15分钟时刻是一个独立样本，模型输出2个值：
    - Price Head: 该时刻的电价
    - Delta Head: 与下一时刻的电价差 (price[t+1] - price[t])

  Sequence encoder: 3-layer Conv1d + AvgPool → global pool → 48-d
  Point encoder: Linear → GELU → 48-d
  Trunk: concat(96) → MLP → 32-d
  Heads: Price(32→1) + Delta(32→1)
  Loss: 0.5*MAE(price) + 0.5*MAE(delta)

  ~43K params (vs V16's ~490K)
  ~7,600 training samples × 2 targets (vs V16's ~7,500 × 96 targets)
  ~1,344 validation samples (vs V16's 14)
"""

import logging
import os
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
    EFFECTIVE_START, EFFECTIVE_END, TRAIN_END, VAL_END, TEST_START,
    TARGET, HIST_COLS, FUTR_COLS, LAG1_HIST, LAG2_HIST,
    N_LAG1, N_HIST, N_FUTR, SPD,
)
from .shape_metrics import compute_shape_report

logger = logging.getLogger(__name__)

V16B_DIR = OUTPUT_DIR / "v16b_dual"
V16B_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LOOKBACK = 672          # 7 days @ 15min
C_SEQ = 1 + N_HIST + N_FUTR   # 30 channels in sequence
C_POINT = N_FUTR + 4          # 14 futr + step_norm + is_weekend + month_sin + month_cos = 18


# =====================================================================
# 1. Dataset
# =====================================================================

class PointDataset(Dataset):
    def __init__(self, y, hist, futr, ts, delta,
                 period_start, period_end):
        self.y, self.hist, self.futr = y, hist, futr
        self.ts, self.delta = ts, delta

        self.indices = []
        for t in range(LOOKBACK, len(y)):
            ts_t = pd.Timestamp(ts[t])
            if ts_t < period_start or ts_t > period_end:
                continue
            if np.isnan(y[t]) or np.isnan(delta[t]):
                continue
            self.indices.append(t)

        logger.info("PointDataset: %d samples (%s ~ %s)",
                    len(self.indices), period_start.date(), period_end.date())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        ts_t = pd.Timestamp(self.ts[t])
        tod = ts_t.hour * 4 + ts_t.minute // 15

        y_seq = self.y[t - LOOKBACK:t].copy()
        h_seq = self.hist[:, t - LOOKBACK:t].copy()
        f_seq = self.futr[:, t - LOOKBACK:t].copy()

        if tod > 0:
            c = LOOKBACK - tod
            for i in range(N_LAG1):
                h_seq[i, c:] = h_seq[i, c - 1]
            y_seq[c:] = y_seq[c - 1]

        lag2_ff = SPD + tod
        c2 = LOOKBACK - lag2_ff
        if c2 > 0:
            for i in range(N_LAG1, N_HIST):
                h_seq[i, c2:] = h_seq[i, c2 - 1]

        seq = np.vstack([y_seq[np.newaxis, :], h_seq, f_seq])

        futr_t = self.futr[:, t].copy()
        step_norm = tod / 95.0
        is_wknd = 1.0 if ts_t.dayofweek >= 5 else 0.0
        month_sin = np.sin(2.0 * np.pi * ts_t.month / 12.0)
        month_cos = np.cos(2.0 * np.pi * ts_t.month / 12.0)
        point = np.concatenate([futr_t,
                                [step_norm, is_wknd, month_sin, month_cos]])

        return (
            torch.from_numpy(seq.astype(np.float32)),
            torch.from_numpy(point.astype(np.float32)),
            torch.tensor(self.y[t], dtype=torch.float32),
            torch.tensor(self.delta[t], dtype=torch.float32),
        )


# =====================================================================
# 2. Model
# =====================================================================

class DualHeadModel(nn.Module):
    def __init__(self, c_seq=C_SEQ, c_point=C_POINT,
                 d_seq=48, d_shared=64, dropout=0.2):
        super().__init__()
        self.seq_enc = nn.Sequential(
            nn.Conv1d(c_seq, d_seq, 7, padding=3),
            nn.BatchNorm1d(d_seq), nn.GELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(d_seq, d_seq, 7, padding=3),
            nn.BatchNorm1d(d_seq), nn.GELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(d_seq, d_seq, 3, padding=1),
            nn.BatchNorm1d(d_seq), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.point_enc = nn.Sequential(
            nn.Linear(c_point, d_seq), nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d_seq * 2, d_shared),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_shared, d_shared // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.price_head = nn.Linear(d_shared // 2, 1)
        self.delta_head = nn.Linear(d_shared // 2, 1)

    def forward(self, seq, point):
        s = self.seq_enc(seq).squeeze(-1)
        p = self.point_enc(point)
        h = self.trunk(torch.cat([s, p], dim=1))
        return self.price_head(h).squeeze(-1), self.delta_head(h).squeeze(-1)


# =====================================================================
# 3. Training
# =====================================================================

def _seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def _val_day_metrics(model, loader, device, y_mean, y_std,
                     raw_y, indices, ts):
    model.eval()
    all_p = []
    with torch.no_grad():
        for seq, point, _, _ in loader:
            p_pred, _ = model(seq.to(device), point.to(device))
            all_p.append(p_pred.cpu().numpy())
    preds = np.concatenate(all_p) * y_std + y_mean

    days = {}
    for i, t in enumerate(indices):
        d = pd.Timestamp(ts[t]).date()
        s = pd.Timestamp(ts[t]).hour * 4 + pd.Timestamp(ts[t]).minute // 15
        if d not in days:
            days[d] = {"p": np.full(SPD, np.nan),
                       "a": np.full(SPD, np.nan)}
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


def train_one(seed, y_norm, hist_norm, futr_norm, delta_norm, ts,
              raw_y, y_mean, y_std,
              epochs=300, patience=30, lr=1e-3, bs=128):
    _seed(seed)

    train_ds = PointDataset(y_norm, hist_norm, futr_norm, ts, delta_norm,
                            EFFECTIVE_START, TRAIN_END)
    val_ds = PointDataset(y_norm, hist_norm, futr_norm, ts, delta_norm,
                          TRAIN_END + pd.Timedelta(minutes=15), VAL_END)

    tl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)

    model = DualHeadModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if seed == 0:
        logger.info("Model params: %d", n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)

    best_sc, best_ep = 1e9, 0
    ckpt = V16B_DIR / f"seed{seed}.pt"

    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for seq, point, price_t, delta_t in tl:
            seq, point = seq.to(DEVICE), point.to(DEVICE)
            price_t, delta_t = price_t.to(DEVICE), delta_t.to(DEVICE)

            opt.zero_grad()
            p_pred, d_pred = model(seq, point)
            loss = 0.5 * F.l1_loss(p_pred, price_t) + \
                   0.5 * F.l1_loss(d_pred, delta_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        mae, pcorr = _val_day_metrics(
            model, vl, DEVICE, y_mean, y_std,
            raw_y, val_ds.indices, ts)
        sc = mae / 100 + 0.5 * (1 - pcorr)
        if sc < best_sc:
            best_sc, best_ep = sc, ep
            torch.save(model.state_dict(), ckpt)
        if ep % 20 == 0 or ep == epochs - 1:
            logger.info(
                "  [seed%d] ep%3d  loss=%.4f  val_mae=%.2f  "
                "val_pcorr=%.3f  sc=%.4f%s",
                seed, ep, ep_loss / max(nb, 1), mae, pcorr, sc,
                " *" if ep == best_ep else "")
        if ep - best_ep >= patience:
            logger.info("  [seed%d] early stop ep%d (best=%d)",
                        seed, ep, best_ep)
            break

    return {"seed": seed, "best_epoch": best_ep,
            "best_score": best_sc, "path": ckpt}


# =====================================================================
# 4. Inference
# =====================================================================

def predict_test(paths, y_norm, hist_norm, futr_norm, delta_norm, ts,
                 raw_y, y_mean, y_std):
    test_ds = PointDataset(y_norm, hist_norm, futr_norm, ts, delta_norm,
                           TEST_START, EFFECTIVE_END)
    tl = DataLoader(test_ds, 512, shuffle=False)

    model_preds = []
    for mp in paths:
        model = DualHeadModel().to(DEVICE)
        model.load_state_dict(
            torch.load(mp, map_location=DEVICE, weights_only=True))
        model.eval()
        preds = []
        with torch.no_grad():
            for seq, point, _, _ in tl:
                p_pred, _ = model(seq.to(DEVICE), point.to(DEVICE))
                preds.append(p_pred.cpu().numpy())
        model_preds.append(np.concatenate(preds))

    ens = np.stack(model_preds).mean(0) * y_std + y_mean

    days = {}
    for i, t in enumerate(test_ds.indices):
        d = pd.Timestamp(ts[t]).date()
        s = pd.Timestamp(ts[t]).hour * 4 + pd.Timestamp(ts[t]).minute // 15
        if d not in days:
            days[d] = {"p": np.full(SPD, np.nan),
                       "a": np.full(SPD, np.nan)}
        days[d]["p"][s] = ens[i]
        days[d]["a"][s] = raw_y[t]

    dates = sorted(d for d in days
                   if not np.isnan(days[d]["p"]).any()
                   and not np.isnan(days[d]["a"]).any())

    p96 = np.array([days[d]["p"] for d in dates])
    a96 = np.array([days[d]["a"] for d in dates])
    p24 = p96.reshape(-1, 24, 4).mean(2)
    a24 = a96.reshape(-1, 24, 4).mean(2)
    return p96, a96, p24, a24, dates


# =====================================================================
# 5. Plotting
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
        ax.plot(x, p_all, "#E91E63", lw=1.0, alpha=0.85, label="V16b(15min)")
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
        ax.set_title(f"V16b 96步预测 — 第{wi+1}周",
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
        ax.plot(hours_15, p96[di], "#E91E63", lw=1.5, label="V16b")
        r = (np.corrcoef(a96[di], p96[di])[0, 1]
             if np.std(a96[di]) > 1e-6 and np.std(p96[di]) > 1e-6 else 0)
        ax.set_title(f"{dates[di]}  r={r:.2f}  MAE={daily_mae[di]:.1f}",
                     fontsize=10)
        ax.set_xlabel("小时")
        ax.set_ylabel("元/MWh")
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("V16b 96步预测 — 典型日",
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
            rows.append({"ts": ts, "actual": a24[i, h],
                         "predicted": p24[i, h]})
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
                alpha=0.85, label="V16b-24h")
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
        ax.set_title(f"V16b 24步预测 — 第{wi+1}周",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"da_week{wi+1}.png",
                    dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_week%d.png", wi + 1)

    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(result["actual"].values, "k-", lw=1.5, label="实际", zorder=3)
    ax.plot(result["predicted"].values, "#E91E63", lw=1.0,
            alpha=0.85, label="V16b")
    ax.set_ylim(200, 600)
    ax.set_ylabel("元/MWh")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_title("V16b — 全测试集", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "da_full_test.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: da_full_test.png")
    return result


# =====================================================================
# 6. Main
# =====================================================================

def run_v16b(n_seeds=5, top_k=3):
    logger.info("=" * 60)
    logger.info("V16b Dual-Head Single-Point — Start")
    logger.info("=" * 60)

    df = build_feature_matrix()
    raw_y = df[TARGET].values.astype(np.float32)

    delta_raw = np.zeros_like(raw_y)
    delta_raw[:-1] = raw_y[1:] - raw_y[:-1]
    delta_raw[-1] = delta_raw[-2]

    hist = df[HIST_COLS].values.T.astype(np.float32)
    futr = df[FUTR_COLS].values.T.astype(np.float32)
    ts = df.index.values

    train_mask = df.index <= TRAIN_END
    y_mean = float(raw_y[train_mask].mean())
    y_std = float(raw_y[train_mask].std()) + 1e-8
    d_mean = float(delta_raw[train_mask].mean())
    d_std = float(delta_raw[train_mask].std()) + 1e-8

    y_norm = ((raw_y - y_mean) / y_std).astype(np.float32)
    delta_norm = ((delta_raw - d_mean) / d_std).astype(np.float32)

    h_mean = hist[:, train_mask].mean(axis=1, keepdims=True)
    h_std = hist[:, train_mask].std(axis=1, keepdims=True) + 1e-8
    hist_norm = ((hist - h_mean) / h_std).astype(np.float32)

    f_mean = futr[:, train_mask].mean(axis=1, keepdims=True)
    f_std = futr[:, train_mask].std(axis=1, keepdims=True) + 1e-8
    futr_norm = ((futr - f_mean) / f_std).astype(np.float32)

    logger.info("Normalization: y_mean=%.1f y_std=%.1f d_mean=%.2f d_std=%.2f",
                y_mean, y_std, d_mean, d_std)

    results = []
    for seed in range(n_seeds):
        logger.info("─── Training seed %d/%d ───", seed + 1, n_seeds)
        res = train_one(seed, y_norm, hist_norm, futr_norm, delta_norm,
                        ts, raw_y, y_mean, y_std)
        results.append(res)

    results.sort(key=lambda r: r["best_score"])
    top = results[:top_k]
    paths = [r["path"] for r in top]
    logger.info("Top-%d seeds: %s", top_k,
                [(r["seed"], f"sc={r['best_score']:.4f}") for r in top])

    p96, a96, p24, a24, dates = predict_test(
        paths, y_norm, hist_norm, futr_norm, delta_norm,
        ts, raw_y, y_mean, y_std)

    plot_96step(p96, a96, dates, V16B_DIR)
    result = plot_24step(p24, a24, dates, V16B_DIR)
    result.to_csv(V16B_DIR / "da_result.csv")
    logger.info("Saved: da_result.csv (%d rows)", len(result))

    rows_96 = []
    for i, d in enumerate(dates):
        for s in range(96):
            t96 = pd.Timestamp(d) + pd.Timedelta(minutes=s * 15)
            rows_96.append({"ts": t96, "actual": a96[i, s],
                            "predicted": p96[i, s]})
    pd.DataFrame(rows_96).set_index("ts").to_csv(
        V16B_DIR / "da_result_96step.csv")

    af = result["actual"].values
    pf = result["predicted"].values
    mae = np.mean(np.abs(af - pf))
    rmse = np.sqrt(np.mean((af - pf) ** 2))
    shape = compute_shape_report(af, pf, result.index)

    logger.info("=" * 60)
    logger.info("V16b RESULTS")
    logger.info("  MAE:          %.2f", mae)
    logger.info("  RMSE:         %.2f", rmse)
    for k, v in shape.items():
        logger.info("  %-18s %.4f", k, v)
    logger.info("=" * 60)

    summary = {"mae": mae, "rmse": rmse, **shape}
    pd.Series(summary).to_csv(V16B_DIR / "metrics_summary.csv")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v16b()
