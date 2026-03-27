"""
序列模型 — 1D-CNN + LSTM 混合模型。

输入：过去 168h（7天）的多变量时序窗口
输出：下一时刻的目标电价

与树模型对比，验证序列模型是否能捕获更复杂的时序模式。

产出：
  - output/sequence_da_result.csv
  - output/sequence_rt_result.csv
  - output/sequence_metrics.csv
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR
from .model_baseline import (
    TEST_START,
    TRAIN_END,
    _compute_metrics,
    _evaluate_model,
    _load_dataset,
)

logger = logging.getLogger(__name__)

WINDOW = 168
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CNN_LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window, features)
        x = x.permute(0, 2, 1)  # (batch, features, window)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)


def _build_sequences(
    df: pd.DataFrame,
    target_col: str,
    window: int = WINDOW,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    feature_cols = [c for c in df.columns if c != target_col]
    X_data = df[feature_cols].values
    y_data = df[target_col].values

    X_data = np.nan_to_num(X_data, nan=0.0)

    X_list, y_list, idx_list = [], [], []
    for i in range(window, len(df)):
        X_list.append(X_data[i - window: i])
        y_list.append(y_data[i])
        idx_list.append(df.index[i])

    return np.array(X_list), np.array(y_list), pd.DatetimeIndex(idx_list)


def _normalize(X_train, X_test):
    n_samples_tr, window, n_feat = X_train.shape
    flat_tr = X_train.reshape(-1, n_feat)
    mu = flat_tr.mean(axis=0)
    sigma = flat_tr.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_train_n = (X_train - mu) / sigma
    X_test_n = (X_test - mu) / sigma
    return X_train_n, X_test_n, mu, sigma


def run_sequence_model(
    name: str,
    target_col: str,
    naive_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("SEQUENCE MODEL: %s | device: %s", name.upper(), DEVICE)
    logger.info("=" * 60)

    df = _load_dataset(name)
    X_all, y_all, idx_all = _build_sequences(df, target_col, WINDOW)

    train_mask = idx_all <= pd.Timestamp(TRAIN_END)
    test_mask = idx_all >= pd.Timestamp(TEST_START)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    idx_test = idx_all[test_mask]

    logger.info("Train: %d sequences | Test: %d sequences", len(X_train), len(X_test))

    X_train_n, X_test_n, _, _ = _normalize(X_train, X_test)

    # ── y normalization ──────────────────────────
    y_mu, y_sigma = y_train.mean(), y_train.std()
    if y_sigma < 1e-8:
        y_sigma = 1.0
    y_train_n = (y_train - y_mu) / y_sigma
    y_test_n = (y_test - y_mu) / y_sigma

    # ── DataLoaders ──────────────────────────────
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_n),
        torch.FloatTensor(y_train_n),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test_n),
        torch.FloatTensor(y_test_n),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ────────────────────────────────────
    n_features = X_train_n.shape[2]
    model = CNN_LSTM(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("  Epoch %3d: train=%.4f  val=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model.eval()

    # ── Final prediction ─────────────────────────
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            all_preds.append(pred.cpu().numpy())
    y_pred_n = np.concatenate(all_preds)
    y_pred = y_pred_n * y_sigma + y_mu

    # ── Naive baseline ───────────────────────────
    test_full = df.loc[idx_test]
    naive_vals = test_full[naive_col].values if naive_col in test_full.columns else np.full(len(idx_test), np.nan)

    results = pd.DataFrame(
        {"actual": y_test, "pred_seq": y_pred, "pred_naive": naive_vals},
        index=idx_test,
    )
    results.index.name = "ts"
    results.to_csv(OUTPUT_DIR / f"sequence_{name}_result.csv")

    m = _compute_metrics(y_test, y_pred)
    logger.info("  Test: MAE=%.2f  RMSE=%.2f  sMAPE=%.2f%%  wMAPE=%.2f%%",
                m["MAE"], m["RMSE"], m["sMAPE(%)"], m["wMAPE(%)"])

    metrics_records = []
    metrics_records.extend(_evaluate_model(results, name, "cnn_lstm", "actual", "pred_seq"))
    metrics_records.extend(_evaluate_model(results, name, "naive", "actual", "pred_naive"))
    metrics_df = pd.DataFrame(metrics_records)

    return results, metrics_df


def run_all_sequence() -> Dict:
    da_res, da_met = run_sequence_model("da", "target_da_clearing_price", "da_clearing_price_lag24h")
    rt_res, rt_met = run_sequence_model("rt", "target_rt_clearing_price", "rt_clearing_price_lag24h")

    all_metrics = pd.concat([da_met, rt_met], ignore_index=True)
    col_order = [
        "model", "method", "group_type", "group_value",
        "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "wMAPE(%)", "MAPE_filtered(%)", "count",
    ]
    all_metrics = all_metrics[col_order]
    all_metrics.to_csv(OUTPUT_DIR / "sequence_metrics.csv", index=False)

    logger.info("=" * 60)
    logger.info("SEQUENCE MODEL SUMMARY")
    logger.info("=" * 60)
    overall = all_metrics[all_metrics["group_type"] == "overall"]
    for _, row in overall.iterrows():
        logger.info(
            "  %s %-10s | MAE=%.2f  RMSE=%.2f  sMAPE=%.1f%%  wMAPE=%.1f%%",
            row["model"].upper(), row["method"],
            row["MAE"], row["RMSE"], row["sMAPE(%)"], row["wMAPE(%)"],
        )

    return {"da": {"results": da_res, "metrics": da_met}, "rt": {"results": rt_res, "metrics": rt_met}}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_sequence()
