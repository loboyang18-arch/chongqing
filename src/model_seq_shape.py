"""
Transformer 24维形状向量预测模型 — 序列形状 (SeqShape)。

以 Transformer Encoder 直接预测24维日内形状向量，使用余弦相似度损失
只关心形状方向一致性。与现有 S1/S2/S3 信号一起通过 grid search
寻找最优混合权重（S4 = Transformer），以 shape_r 为目标。

架构: Linear(n_feat→32) + PositionalEncoding + TransformerEncoder(2层) + FC→24
损失: 1 - cosine_similarity(centered_pred, centered_actual)

产出:
  - output/viz/seqshape_{da,rt}_result.csv / summary.csv
  - output/viz/seqshape_{da,rt}_*.png (8张可视化)
"""

import json
import logging
import math
import os
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR, PARAMS_DIR
from .model_baseline import (
    EARLY_STOPPING_ROUNDS,
    NUM_BOOST_ROUND,
    TEST_START,
    TRAIN_END,
    _compute_metrics,
    _load_dataset,
)

logger = logging.getLogger(__name__)
VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)

# ── 序列 / 训练超参 ──────────────────────────────────────
WINDOW = 168
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

D_MODEL = 32
N_HEAD = 2
N_LAYERS = 2
D_FF = 64
DROPOUT = 0.3


# =====================================================================
# 1. 模型定义
# =====================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerShapeModel(nn.Module):
    def __init__(
        self, n_features: int, d_model: int = D_MODEL, nhead: int = N_HEAD,
        num_layers: int = N_LAYERS, dim_ff: int = D_FF, dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=WINDOW + 10, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 24),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.fc(x)


def shape_cosine_loss(pred_24h: torch.Tensor, actual_24h: torch.Tensor) -> torch.Tensor:
    pred_c = pred_24h - pred_24h.mean(dim=1, keepdim=True)
    actual_c = actual_24h - actual_24h.mean(dim=1, keepdim=True)
    cos_sim = F.cosine_similarity(pred_c, actual_c, dim=1)
    return (1 - cos_sim).mean()


# =====================================================================
# 2. 数据构造 / 归一化
# =====================================================================

def _build_shape_sequences(
    df: pd.DataFrame, feature_cols: List[str], target_col: str,
    window: int = WINDOW, stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """滑窗构造 (X_window, Y_24h_dev, timestamp, target_date) 序列。"""
    X_data = np.nan_to_num(df[feature_cols].values, nan=0.0).astype(np.float32)

    df_dates = df.index.date
    daily_mean = df.groupby(df.index.date)[target_col].mean()

    dev_by_date: Dict = {}
    for d in sorted(set(df_dates)):
        mask = df_dates == d
        day_vals = df.loc[mask, target_col].values
        if len(day_vals) == 24:
            dev_by_date[d] = (day_vals - daily_mean[d]).astype(np.float32)

    X_list, y_list, idx_list, date_list = [], [], [], []
    for i in range(window, len(df), stride):
        d = df_dates[i]
        if d not in dev_by_date:
            continue
        X_list.append(X_data[i - window : i])
        y_list.append(dev_by_date[d])
        idx_list.append(df.index[i])
        date_list.append(d)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        pd.DatetimeIndex(idx_list),
        np.array(date_list),
    )


def _normalize_features(X_train, X_test):
    n_feat = X_train.shape[2]
    flat = X_train.reshape(-1, n_feat)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


# =====================================================================
# 3. 训练 / 推理
# =====================================================================

def _train_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_features: int, tag: str = "",
) -> Tuple[TransformerShapeModel, dict]:
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerShapeModel(n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  [%s] Transformer params: %d | device: %s", tag, n_params, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = shape_cosine_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_losses.append(shape_cosine_loss(model(xb), yb).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "  [%s] Epoch %3d  train=%.4f  val=%.4f  lr=%.1e",
                tag, epoch, t_loss, v_loss, lr_now,
            )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("  [%s] Early stop at epoch %d (best=%.4f)", tag, epoch, best_val_loss)
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _predict_daily_shape(
    model: TransformerShapeModel, X: np.ndarray, target_dates: np.ndarray,
) -> Dict:
    """推理并按日聚合（多窗口平均）→ {date: 24-dim array}。"""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X)), batch_size=BATCH_SIZE, shuffle=False,
    )
    parts = []
    with torch.no_grad():
        for (xb,) in loader:
            parts.append(model(xb.to(DEVICE)).cpu().numpy())
    preds = np.concatenate(parts, axis=0)

    daily: Dict = {}
    for d in sorted(set(target_dates)):
        daily[d] = preds[target_dates == d].mean(axis=0)
    return daily


# =====================================================================
# 4. 辅助函数：字体 / 调参 / 日级特征 / shape_r
# =====================================================================

def _setup_cn_font():
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


def _load_tuned_params(name: str) -> Dict:
    with open(PARAMS_DIR / f"tuning_{name}_best_params.json") as f:
        return json.load(f)


def _build_daily_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df_feat = df[feature_cols].copy()
    df_feat["date"] = df_feat.index.date
    agg = {}
    for c in feature_cols:
        if c not in ("hour", "day_of_week", "is_weekend", "month"):
            agg[c] = ["mean"]
    daily = df_feat.groupby("date").agg(agg)
    daily.columns = [f"{c[0]}_dmean" for c in daily.columns]
    for cal in ("day_of_week", "is_weekend", "month"):
        if cal in feature_cols:
            daily[cal] = df_feat.groupby("date")[cal].first()
    return daily


def _compute_shape_corr_arrays(
    actual: np.ndarray, pred: np.ndarray, dates: np.ndarray,
) -> float:
    corr_daily = []
    for d in sorted(set(dates)):
        m = dates == d
        if m.sum() == 24:
            a, p = actual[m], pred[m]
            if np.std(a) > 1e-6 and np.std(p) > 1e-6:
                c = np.corrcoef(a, p)[0, 1]
                if np.isfinite(c):
                    corr_daily.append(c)
    return float(np.mean(corr_daily)) if corr_daily else -1.0


def _compute_s1_s3_hourly(
    target_df: pd.DataFrame, full_df: pd.DataFrame, target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 S1(近3日均值形状) 和 S3(昨日形状) 的逐小时数组。"""
    all_dates = sorted(set(full_df.index.date))
    date2idx = {d: i for i, d in enumerate(all_dates)}
    daily_mean_actual = full_df.groupby(full_df.index.date)[target_col].mean()

    dev_by_date: Dict = {}
    for d in all_dates:
        vals = full_df.loc[full_df.index.date == d, target_col].values
        dev_by_date[d] = vals - daily_mean_actual.get(d, np.nan)

    dates_arr = target_df.index.date
    target_dates = sorted(set(dates_arr))
    s1 = np.full(len(target_df), np.nan)
    s3 = np.full(len(target_df), np.nan)

    for d in target_dates:
        if d not in date2idx:
            continue
        idx = date2idx[d]
        mask = dates_arr == d
        n_h = mask.sum()

        # S1: 近3日均值
        prev = [all_dates[idx - k] for k in range(1, 4) if idx - k >= 0]
        devs = [dev_by_date[p] for p in prev if len(dev_by_date.get(p, [])) == n_h]
        if devs:
            s1[mask] = np.mean(devs, axis=0)

        # S3: 昨日
        if idx > 0:
            yd = all_dates[idx - 1]
            if len(dev_by_date.get(yd, [])) == n_h:
                s3[mask] = dev_by_date[yd]

    return s1, s3


def _daily_dict_to_hourly(
    daily_dict: Dict, target_df: pd.DataFrame,
) -> np.ndarray:
    """将 {date: 24-dim} 字典展开为逐小时数组。"""
    arr = np.full(len(target_df), np.nan)
    dates_arr = target_df.index.date
    for d in sorted(set(dates_arr)):
        mask = dates_arr == d
        n_h = mask.sum()
        if d in daily_dict and n_h == 24 and len(daily_dict[d]) == 24:
            arr[mask] = daily_dict[d]
    return arr


# =====================================================================
# 5. 主函数: 训练 + 集成 + 评估
# =====================================================================

def run_seq_shape(name: str, target_col: str, naive_col: str):
    logger.info("=" * 60)
    logger.info("SEQ-SHAPE (Transformer): %s | device: %s", name.upper(), DEVICE)
    logger.info("=" * 60)

    df = _load_dataset(name)
    params = _load_tuned_params(name)
    feature_cols = [c for c in df.columns if c != target_col]

    df["date"] = df.index.date
    daily_mean = df.groupby("date")[target_col].transform("mean")
    df["target_daily_mean"] = daily_mean
    df["target_hourly_dev"] = df[target_col] - daily_mean

    train_df = df.loc[:TRAIN_END].copy()
    test_df = df.loc[TEST_START:].copy()
    full_df = pd.concat([train_df, test_df])

    # ── Stage 1: Level LGB (日均价预测) ──────────────────
    logger.info("--- Stage 1: Level Model ---")
    daily_train = _build_daily_features(train_df, feature_cols)
    daily_test = _build_daily_features(test_df, feature_cols)
    daily_target_train = train_df.groupby("date")[target_col].mean()
    daily_target_test = test_df.groupby("date")[target_col].mean()
    daily_train = daily_train.loc[daily_target_train.index]
    daily_test = daily_test.loc[daily_target_test.index]
    level_feat_cols = list(daily_train.columns)

    p_lv = params.copy()
    p_lv.update(objective="regression", min_child_samples=5)
    p_lv["num_leaves"] = min(p_lv.get("num_leaves", 31), 31)
    dt_tr = lgb.Dataset(daily_train[level_feat_cols], label=daily_target_train)
    dt_va = lgb.Dataset(daily_test[level_feat_cols], label=daily_target_test, reference=dt_tr)
    level_model = lgb.train(
        p_lv, dt_tr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dt_tr, dt_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )
    level_pred_train = level_model.predict(daily_train[level_feat_cols])
    level_pred_test = level_model.predict(daily_test[level_feat_cols])
    logger.info(
        "  Level MAE: %.2f",
        np.mean(np.abs(daily_target_test.values - level_pred_test)),
    )

    # ── Stage 2: Shape LGB (S2 信号) ────────────────────
    logger.info("--- Stage 2: Shape LGB (S2) ---")
    shape_feat_cols = list(feature_cols)
    p_sh = params.copy()
    p_sh["objective"] = "huber"
    ds_tr = lgb.Dataset(train_df[shape_feat_cols], label=train_df["target_hourly_dev"])
    ds_va = lgb.Dataset(test_df[shape_feat_cols], label=test_df["target_hourly_dev"], reference=ds_tr)
    shape_lgb = lgb.train(
        p_sh, ds_tr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(200)],
    )
    shape_pred_train = shape_lgb.predict(train_df[shape_feat_cols])
    actual_dev_std = train_df["target_hourly_dev"].std()
    pred_dev_std = float(np.std(shape_pred_train))
    scale_factor = min(actual_dev_std / max(pred_dev_std, 1e-6), 5.0)
    logger.info("  Shape LGB scale: %.2f (actual_std=%.2f, pred_std=%.2f)",
                scale_factor, actual_dev_std, pred_dev_std)

    # ── Stage 3: Transformer Shape (S4 信号) ────────────
    logger.info("--- Stage 3: Transformer Shape (S4) ---")
    X_all, y_all, idx_all, dates_all = _build_shape_sequences(
        df, feature_cols, target_col, WINDOW, stride=1,
    )
    tr_mask = idx_all <= pd.Timestamp(TRAIN_END)
    te_mask = idx_all >= pd.Timestamp(TEST_START)

    X_tr_seq, y_tr_seq, d_tr_seq = X_all[tr_mask], y_all[tr_mask], dates_all[tr_mask]
    X_te_seq, y_te_seq, d_te_seq = X_all[te_mask], y_all[te_mask], dates_all[te_mask]
    logger.info("  Sequences: train=%d  test=%d", len(X_tr_seq), len(X_te_seq))

    X_tr_n, X_te_n, feat_mu, feat_sig = _normalize_features(X_tr_seq, X_te_seq)

    model_tf, history = _train_transformer(
        X_tr_n, y_tr_seq, X_te_n, y_te_seq, X_tr_n.shape[2], tag="FULL",
    )
    s4_test_dict = _predict_daily_shape(model_tf, X_te_n, d_te_seq)

    # ── Stage 4: OOF (75/25 split) → grid search ───────
    logger.info("--- Stage 4: OOF + Grid Search ---")
    n_train = len(train_df)
    val_start = int(n_train * 0.75)
    pre_val_df = train_df.iloc[:val_start]
    val_df = train_df.iloc[val_start:]

    # 4a. OOF Shape LGB
    p_oof = params.copy()
    p_oof["objective"] = "huber"
    ds_pv = lgb.Dataset(pre_val_df[shape_feat_cols], label=pre_val_df["target_hourly_dev"])
    ds_vv = lgb.Dataset(val_df[shape_feat_cols], label=val_df["target_hourly_dev"], reference=ds_pv)
    shape_oof_lgb = lgb.train(
        p_oof, ds_pv, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_pv, ds_vv], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    oof_pred_std = float(np.std(shape_oof_lgb.predict(pre_val_df[shape_feat_cols])))
    oof_scale = min(pre_val_df["target_hourly_dev"].std() / max(oof_pred_std, 1e-6), 5.0)
    s2_val_oof = shape_oof_lgb.predict(val_df[shape_feat_cols]) * oof_scale

    # 4b. OOF Transformer
    pre_val_end_ts = pre_val_df.index[-1]
    val_start_ts = val_df.index[0]
    pv_mask = (idx_all <= pre_val_end_ts) & tr_mask
    vv_mask = (idx_all >= val_start_ts) & tr_mask

    X_pv, y_pv, d_pv = X_all[pv_mask], y_all[pv_mask], dates_all[pv_mask]
    X_vv, y_vv, d_vv = X_all[vv_mask], y_all[vv_mask], dates_all[vv_mask]

    if len(X_pv) > 0 and len(X_vv) > 0:
        X_pv_n, X_vv_n, _, _ = _normalize_features(X_pv, X_vv)
        model_oof_tf, _ = _train_transformer(
            X_pv_n, y_pv, X_vv_n, y_vv, X_pv_n.shape[2], tag="OOF",
        )
        s4_val_dict = _predict_daily_shape(model_oof_tf, X_vv_n, d_vv)
    else:
        s4_val_dict = {}
        logger.warning("  OOF Transformer skipped (insufficient data)")

    # 4c. 组装验证集的 S1/S2/S3/S4 逐小时数组
    level_train_map = dict(zip(daily_target_train.index, level_pred_train))
    val_level = np.array([level_train_map.get(d, np.nan) for d in val_df["date"]])
    val_actual = val_df[target_col].values
    val_dates_arr = val_df.index.date

    val_s1, val_s3 = _compute_s1_s3_hourly(val_df, full_df, target_col)
    val_s2 = s2_val_oof
    val_s4 = _daily_dict_to_hourly(s4_val_dict, val_df)

    # 4d. Grid search: w1*S1 + w2*S2 + w3*S3 + w4*S4, max shape_r
    best_w, best_sr = (0.25, 0.25, 0.25, 0.25), -1.0
    step = 0.05
    for w1 in np.arange(0, 1.01, step):
        for w2 in np.arange(0, 1.01 - w1, step):
            for w3 in np.arange(0, 1.01 - w1 - w2, step):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < -1e-6:
                    continue
                blend = w1 * val_s1 + w2 * val_s2 + w3 * val_s3 + w4 * val_s4
                pred = val_level + blend
                fin = np.isfinite(pred)
                if fin.sum() < 48:
                    continue
                sr = _compute_shape_corr_arrays(val_actual[fin], pred[fin], val_dates_arr[fin])
                if sr > best_sr:
                    best_sr = sr
                    best_w = (round(w1, 2), round(w2, 2), round(w3, 2), round(w4, 2))

    w1, w2, w3, w4 = best_w
    logger.info(
        "  Optimal: S1=%.2f  S2=%.2f  S3=%.2f  S4=%.2f  val_shape_r=%.3f",
        w1, w2, w3, w4, best_sr,
    )

    # ── Stage 5: 测试集最终预测 ─────────────────────────
    logger.info("--- Stage 5: Test prediction ---")
    test_dates_arr = test_df.index.date

    s1_test, s3_test = _compute_s1_s3_hourly(test_df, full_df, target_col)
    s2_test = shape_lgb.predict(test_df[shape_feat_cols]) * scale_factor
    s4_test = _daily_dict_to_hourly(s4_test_dict, test_df)

    can_mix = np.isfinite(s1_test) & np.isfinite(s3_test) & np.isfinite(s4_test)
    shape_mix = np.where(
        can_mix,
        w1 * s1_test + w2 * s2_test + w3 * s3_test + w4 * s4_test,
        w1 * np.nan_to_num(s1_test) + w2 * s2_test
        + w3 * np.nan_to_num(s3_test) + w4 * np.nan_to_num(s4_test),
    )

    level_test_map = dict(zip(daily_target_test.index, level_pred_test))
    level_test_h = np.array([level_test_map.get(d, np.nan) for d in test_df["date"]])
    pred_final = level_test_h + shape_mix

    # ── 对比方法 ────────────────────────────────────────
    actual_test = test_df[target_col].values
    pred_naive = test_df[naive_col].values

    p_base = params.copy()
    dtb = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    dvb = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=dtb)
    base_model = lgb.train(
        p_base, dtb, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtb, dvb], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    pred_base = base_model.predict(test_df[feature_cols])

    pred_s1 = level_test_h + np.where(np.isfinite(s1_test), s1_test, 0)
    s1s3 = np.where(
        np.isfinite(s1_test) & np.isfinite(s3_test),
        0.7 * s1_test + 0.3 * s3_test,
        np.where(np.isfinite(s1_test), s1_test, np.nan_to_num(s3_test)),
    )
    pred_s1s3 = level_test_h + s1s3
    pred_s4_only = level_test_h + np.nan_to_num(s4_test)

    res_dict = {
        "actual": actual_test, "naive": pred_naive, "pred_base": pred_base,
        "pred_s1_3day": pred_s1, "pred_s1s3": pred_s1s3,
        "pred_s4_transformer": pred_s4_only, "pred_seqshape": pred_final,
    }
    prev_path = VIZ_DIR / f"final_{name}_result.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, parse_dates=["ts"], index_col="ts")
        if "pred_final" in prev.columns:
            common_idx = test_df.index.intersection(prev.index)
            if len(common_idx) == len(test_df):
                res_dict["pred_prev_final"] = prev.loc[test_df.index, "pred_final"].values

    results = pd.DataFrame(res_dict, index=test_df.index)
    results.index.name = "ts"

    methods = {
        "标准LGB": "pred_base",
        "Naive": "naive",
        "S1_3日均值形状": "pred_s1_3day",
        "S1S3_历史惯性": "pred_s1s3",
        "S4_Transformer": "pred_s4_transformer",
        f"SeqShape(w={w1},{w2},{w3},{w4})": "pred_seqshape",
    }
    if "pred_prev_final" in results.columns:
        methods["原Final(S1S2S3)"] = "pred_prev_final"

    # ── 评估 ────────────────────────────────────────────
    logger.info("\n=== %s Transformer形状模型对比 ===", name.upper())
    summary_rows = []
    for label, col in methods.items():
        m = _compute_metrics(actual_test, results[col].values)
        sr = _compute_shape_corr_arrays(actual_test, results[col].values, test_dates_arr)
        pstd = float(results[col].std())
        logger.info(
            "  %-45s MAE=%6.2f  RMSE=%6.2f  std=%5.1f  shape_r=%.3f",
            label, m["MAE"], m["RMSE"], pstd, sr,
        )
        summary_rows.append({
            "method": label, "MAE": m["MAE"], "RMSE": m["RMSE"],
            "sMAPE(%)": m["sMAPE(%)"], "wMAPE(%)": m["wMAPE(%)"],
            "pred_std": round(pstd, 2), "shape_corr": round(sr, 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(VIZ_DIR / f"seqshape_{name}_summary.csv", index=False)
    results.to_csv(VIZ_DIR / f"seqshape_{name}_result.csv")

    shape_imp = pd.DataFrame({
        "feature": shape_feat_cols,
        "importance": shape_lgb.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    level_imp = pd.DataFrame({
        "feature": level_feat_cols,
        "importance": level_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    extra = {
        "weights": best_w, "scale_factor": scale_factor,
        "best_shape_r_val": best_sr,
        "transformer_params": sum(p.numel() for p in model_tf.parameters()),
        "training_history": history,
    }
    return results, summary, methods, shape_imp, level_imp, extra


# =====================================================================
# 6. 可视化
# =====================================================================

def plot_seq_shape(results, summary, methods, shape_imp, level_imp, extra, name):
    _setup_cn_font()

    colors = {"标准LGB": "#90CAF9", "Naive": "#BDBDBD"}
    palette = ["#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    ci = 0
    for lab in methods:
        if lab not in colors:
            colors[lab] = palette[ci % len(palette)]
            ci += 1

    summary_s = summary.sort_values("shape_corr", ascending=False)

    # ── 1. 全时段 ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(results.index, results["actual"], color="black", lw=1.2, label="实际", zorder=5)
    for lab, col in methods.items():
        if lab in ("标准LGB", "Naive"):
            ax.plot(results.index, results[col], "--", color=colors[lab], lw=0.6, alpha=0.4, label=lab)
            continue
        sr_v = summary.loc[summary["method"] == lab, "shape_corr"].values[0]
        ax.plot(results.index, results[col], color=colors[lab], lw=0.9, alpha=0.85,
                label=f"{lab} (r={sr_v:.3f})")
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — Transformer形状模型全时段对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_full.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. 典型日 ──────────────────────────────────────
    rp = results.copy()
    rp["date"] = rp.index.date
    dstd = rp.groupby("date")["actual"].std()
    days = {
        "平稳日": dstd.idxmin(),
        "中等日": dstd.index[(dstd - dstd.median()).abs().argsort().iloc[0]],
        "高波动日": dstd.idxmax(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (title, day) in zip(axes, days.items()):
        dd = rp[rp["date"] == day]
        hrs = dd.index.hour
        ax.plot(hrs, dd["actual"], "o-", color="black", lw=2, ms=4, label="实际", zorder=5)
        for lab, col in methods.items():
            ls = "--" if lab in ("标准LGB", "Naive") else "-"
            lw = 0.8 if lab in ("标准LGB", "Naive") else 1.4
            ax.plot(hrs, dd[col], ls, color=colors[lab], lw=lw, alpha=0.85, label=lab)
        ax.set_title(f"{title} {str(day)[5:]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("电价 (元/MWh)")
    axes[2].legend(fontsize=6, loc="best")
    plt.suptitle(f"{name.upper()} — Transformer形状模型典型日", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_typical.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. 缩放周 ──────────────────────────────────────
    wk = results.loc["2026-03-01":"2026-03-07 23:00:00"]
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(wk.index, wk["actual"], "o-", color="black", ms=2, lw=1.5, label="实际", zorder=5)
    for lab, col in methods.items():
        ls = "--" if lab in ("标准LGB", "Naive") else "-"
        lw = 0.8 if lab in ("标准LGB", "Naive") else 1.2
        ax.plot(wk.index, wk[col], ls, color=colors[lab], lw=lw, alpha=0.85, label=lab)
    ax.set_ylabel("电价 (元/MWh)")
    ax.set_title(f"{name.upper()} — 缩放周 03-01~03-07", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H时"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_zoom_week.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 4. shape_r 柱状图 ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(summary_s))
    bc = [colors.get(m, "#42A5F5") for m in summary_s["method"]]
    ax.bar(x, summary_s["shape_corr"], color=bc, alpha=0.85, edgecolor="white")
    for i, (v, m) in enumerate(zip(summary_s["shape_corr"], summary_s["MAE"])):
        ax.text(i, v + 0.01, f"r={v:.3f}\nMAE={m:.1f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_s["method"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("日内形状相关性 (shape_r)")
    ax.set_ylim(0, max(summary_s["shape_corr"].max() * 1.25, 0.1))
    ax.set_title(f"{name.upper()} — shape_r 排名", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 5. 散点图 ──────────────────────────────────────
    nm = len(methods)
    ncols = 3
    nrows = max((nm + ncols - 1) // ncols, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)
    af = axes.flatten()
    for i, (lab, col) in enumerate(methods.items()):
        ax = af[i]
        ax.scatter(results["actual"], results[col], alpha=0.3, s=8, color=colors[lab])
        lims = [results["actual"].min() - 20, results["actual"].max() + 20]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("实际"); ax.set_ylabel("预测")
        sr_v = summary.loc[summary["method"] == lab, "shape_corr"].values[0]
        mae_v = summary.loc[summary["method"] == lab, "MAE"].values[0]
        ax.set_title(f"{lab}\nshape_r={sr_v:.3f}  MAE={mae_v:.1f}", fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
    for j in range(nm, len(af)):
        af[j].set_visible(False)
    plt.suptitle(f"{name.upper()} — 预测 vs 实际", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 6. 逐日 shape_r ───────────────────────────────
    rp2 = results.copy()
    rp2["date"] = rp2.index.date
    daily_r = {}
    for lab, col in methods.items():
        daily_r[lab] = {}
        for d in rp2["date"].unique():
            dd = rp2[rp2["date"] == d]
            if len(dd) == 24 and np.std(dd[col].values) > 1e-6:
                c = np.corrcoef(dd["actual"].values, dd[col].values)[0, 1]
                daily_r[lab][d] = c if np.isfinite(c) else np.nan

    fig, ax = plt.subplots(figsize=(16, 5))
    for lab in methods:
        if lab == "标准LGB":
            continue
        s = pd.Series(daily_r[lab]).sort_index()
        ls = "--" if lab == "Naive" else "-"
        ax.plot(s.index, s.values, ls, color=colors[lab], lw=1.0, alpha=0.85,
                label=f"{lab} (mean={s.mean():.3f})", marker=".", ms=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("日内形状相关性 (r)")
    ax.set_title(f"{name.upper()} — 逐日shape_r走势", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_daily_shape_r.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 7. 特征重要度 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, imp, title, color in [
        (axes[0], shape_imp.head(15), "Shape LGB Top-15", "#E91E63"),
        (axes[1], level_imp.head(15), "Level LGB Top-15", "#42A5F5"),
    ]:
        ax.barh(range(len(imp)), imp["importance"].values, color=color, alpha=0.8)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Gain")
    plt.suptitle(f"{name.upper()} — 特征重要度", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"seqshape_{name}_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 8. 训练曲线 ────────────────────────────────────
    hist = extra.get("training_history")
    if hist:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist["train_loss"], label="训练集", color="#E91E63")
        ax.plot(hist["val_loss"], label="验证集", color="#42A5F5")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cosine Loss")
        ax.set_title(f"{name.upper()} — Transformer训练曲线", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(VIZ_DIR / f"seqshape_{name}_train_loss.png", dpi=120, bbox_inches="tight")
        plt.close()

    logger.info("Saved plots → output/viz/seqshape_%s_*.png", name)


# =====================================================================
# 7. 入口
# =====================================================================

def run_all():
    _setup_cn_font()
    all_res = {}
    for name, tc, nc in [
        ("da", "target_da_clearing_price", "da_clearing_price_lag24h"),
        ("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h"),
    ]:
        results, summary, methods, si, li, extra = run_seq_shape(name, tc, nc)
        plot_seq_shape(results, summary, methods, si, li, extra, name)
        all_res[name] = {"results": results, "summary": summary, "methods": methods, "extra": extra}

    logger.info("=" * 60)
    logger.info("ALL SEQ-SHAPE EXPERIMENTS COMPLETE → output/viz/seqshape_*")
    logger.info("=" * 60)
    return all_res


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
