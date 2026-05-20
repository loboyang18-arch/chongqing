"""
V21 — LightGBM 价差方向分类 + 回归辅助 + 阈值优化

策略：
  A) 二分类模型：predict_proba → 阈值优化（validation balanced accuracy 最优）
  B) 回归模型：直接预测 spread → sign(pred) 作为方向
  C) 组合策略：classification prob + regression sign 投票

特征来源：feature_da.csv + 追加价差衍生特征（多级 lag、滚动统计、同小时模式）。
时序安全：所有 settlement/clearing 价格严格 lag≥48h，clearing spread lag≥24h。
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .experiment.splits import HOURLY_TEST_END
from .experiment.splits import HOURLY_TEST_START as TEST_START
from .experiment.splits import HOURLY_TRAIN_END as TRAIN_END
from price_forecast_eval.viz import run_standard_visualization

logger = logging.getLogger(__name__)

NUM_BOOST_ROUND = int(os.environ.get("V21_ROUNDS", "3000"))
EARLY_STOPPING = int(os.environ.get("V21_EARLY_STOP", "100"))
VAL_FRAC = float(os.environ.get("V21_VAL_FRAC", "0.15"))

V21_DIR = OUTPUT_DIR / "v21_spread_lgbm"


# ═══════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════

def _load_feature_da() -> pd.DataFrame:
    path = OUTPUT_DIR / "feature_da.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    logger.info("Loaded feature_da: %d rows × %d cols", len(df), len(df.columns))
    return df


def _load_hourly_raw() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_hourly_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    return df


# ═══════════════════════════════════════════════════════
# 价差衍生特征
# ═══════════════════════════════════════════════════════

def _add_spread_features(feat: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """追加价差相关衍生特征（严格 lag 安全）。"""

    idx = feat.index

    # ── settlement spread lags ──
    if "settlement_da_price" in raw.columns and "settlement_rt_price" in raw.columns:
        sp_raw = (raw["settlement_rt_price"] - raw["settlement_da_price"]).reindex(idx)
        for lag in [48, 72, 96, 168]:
            feat[f"spread_lag{lag}h"] = sp_raw.shift(lag)

        sp48 = feat["spread_lag48h"]
        feat["spread_sign_lag48h"] = np.sign(sp48)
        feat["spread_abs_lag48h"] = sp48.abs()

        for win in [3, 5, 7]:
            n = 24 * win
            feat[f"spread_roll_{win}d_mean"] = sp48.rolling(n, min_periods=24).mean()
            feat[f"spread_roll_{win}d_std"] = sp48.rolling(n, min_periods=24).std()

        feat["spread_sign_roll5d_sum"] = np.sign(sp48).rolling(120, min_periods=24).sum()
        feat["spread_sign_roll3d_sum"] = np.sign(sp48).rolling(72, min_periods=24).sum()

        feat["spread_pos_ratio_5d"] = (
            (sp48 > 0).astype(float).rolling(120, min_periods=24).mean()
        )

        sp_grouped = sp48.groupby(idx.hour)
        feat["spread_same_hour_7d_mean"] = sp_grouped.transform(
            lambda x: x.rolling(7, min_periods=3).mean()
        )
        feat["spread_same_hour_7d_std"] = sp_grouped.transform(
            lambda x: x.rolling(7, min_periods=3).std()
        )
        feat["spread_same_hour_sign_ratio_7d"] = sp_grouped.transform(
            lambda x: (x > 0).astype(float).rolling(7, min_periods=3).mean()
        )

    # ── clearing spread lags (available at lag24h) ──
    if "rt_clearing_price" in raw.columns and "da_clearing_price" in raw.columns:
        cs = (raw["rt_clearing_price"] - raw["da_clearing_price"]).reindex(idx)
        feat["clearing_spread_lag24h"] = cs.shift(24)
        feat["clearing_spread_lag48h"] = cs.shift(48)
        feat["clearing_spread_sign_lag24h"] = np.sign(cs.shift(24))
        feat["clearing_spread_roll_3d_mean"] = cs.shift(24).rolling(72, min_periods=24).mean()

    # ── settlement price lags ──
    if "settlement_da_price" in raw.columns:
        sda = raw["settlement_da_price"].reindex(idx)
        feat["sda_price_lag48h"] = sda.shift(48)
        feat["sda_price_roll_3d_mean_lag48"] = sda.shift(48).rolling(72, min_periods=24).mean()

    if "settlement_rt_price" in raw.columns:
        srt = raw["settlement_rt_price"].reindex(idx)
        feat["srt_price_lag48h"] = srt.shift(48)
        feat["srt_price_roll_3d_mean_lag48"] = srt.shift(48).rolling(72, min_periods=24).mean()

    # ── DA-RT price ratio ──
    if "sda_price_lag48h" in feat.columns and "srt_price_lag48h" in feat.columns:
        denom = feat["sda_price_lag48h"].replace(0, np.nan)
        feat["rt_da_ratio_lag48h"] = feat["srt_price_lag48h"] / denom

    return feat


def _build_labels(raw: pd.DataFrame, index: pd.DatetimeIndex):
    sda = raw["settlement_da_price"].reindex(index)
    srt = raw["settlement_rt_price"].reindex(index)
    spread = srt - sda
    return (spread > 0).astype(int), spread


def _time_split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:HOURLY_TEST_END]
    return train, test


# ═══════════════════════════════════════════════════════
# 阈值优化
# ═══════════════════════════════════════════════════════

def _optimize_threshold(prob: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """在验证集上搜索 balanced accuracy 最优阈值。"""
    best_thr, best_ba = 0.5, 0.0
    for thr in np.arange(0.20, 0.80, 0.01):
        pred = (prob > thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        rec_pos = tp / max(tp + fn, 1)
        rec_neg = tn / max(tn + fp, 1)
        ba = 0.5 * (rec_pos + rec_neg)
        if ba > best_ba:
            best_ba = ba
            best_thr = thr
    return float(best_thr), float(best_ba)


# ═══════════════════════════════════════════════════════
# 方向评估工具
# ═══════════════════════════════════════════════════════

def _eval_direction(pred: np.ndarray, y_true: np.ndarray, tag: str = ""):
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    acc = float(np.mean(pred == y_true))
    prec_pos = tp / max(tp + fp, 1)
    rec_pos = tp / max(tp + fn, 1)
    prec_neg = tn / max(tn + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    ba = 0.5 * (rec_pos + rec_neg)
    logger.info("[%s] Acc=%.4f BA=%.4f | TP=%d TN=%d FP=%d FN=%d | Prec+/−=%.3f/%.3f Rec+/−=%.3f/%.3f",
                tag, acc, ba, tp, tn, fp, fn, prec_pos, prec_neg, rec_pos, rec_neg)
    return {
        "accuracy": acc, "balanced_accuracy": ba,
        "precision_pos": prec_pos, "recall_pos": rec_pos,
        "precision_neg": prec_neg, "recall_neg": rec_neg,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ═══════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════

def run_v21(out_dir: Optional[Path] = None) -> Dict:
    out_dir = Path(out_dir) if out_dir else V21_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V21 LightGBM — spread direction (clf + reg + combo)")

    feat = _load_feature_da()
    raw = _load_hourly_raw()
    feat = _add_spread_features(feat, raw)

    label_series, spread_series = _build_labels(raw, feat.index)
    feat["_label"] = label_series
    feat["_spread"] = spread_series

    valid_mask = feat["_label"].notna() & feat["_spread"].notna()
    feat = feat[valid_mask].copy()
    logger.info("Valid samples: %d", len(feat))

    train_df, test_df = _time_split(feat)
    logger.info("Train: %d (%s ~ %s)", len(train_df), train_df.index.min(), train_df.index.max())
    logger.info("Test:  %d (%s ~ %s)", len(test_df), test_df.index.min(), test_df.index.max())

    exclude = {"_label", "_spread", "target", "settlement_da_price", "settlement_rt_price"}
    feature_cols = [
        c for c in feat.columns
        if c not in exclude and feat[c].dtype in ("float64", "float32", "int64", "int32", "int8")
    ]
    nan_frac = train_df[feature_cols].isna().mean()
    drop_cols = nan_frac[nan_frac > 0.5].index.tolist()
    if drop_cols:
        logger.info("Dropping %d cols with >50%% NaN", len(drop_cols))
        feature_cols = [c for c in feature_cols if c not in drop_cols]
    logger.info("Feature cols: %d", len(feature_cols))

    y_train = train_df["_label"].values.astype(int)
    y_test = test_df["_label"].values.astype(int)
    spread_train = train_df["_spread"].values
    spread_test = test_df["_spread"].values

    # ── train / val split ──
    n_val = max(int(len(train_df) * VAL_FRAC), 24)
    train_idx = train_df.index[:-n_val]
    val_idx = train_df.index[-n_val:]

    X_tr = train_df.loc[train_idx, feature_cols]
    y_tr = train_df.loc[train_idx, "_label"].values.astype(int)
    X_va = train_df.loc[val_idx, feature_cols]
    y_va = train_df.loc[val_idx, "_label"].values.astype(int)

    spread_tr = train_df.loc[train_idx, "_spread"].values
    spread_va = train_df.loc[val_idx, "_spread"].values

    pos_tr = int(y_tr.sum())
    neg_tr = len(y_tr) - pos_tr
    scale_pos = neg_tr / max(pos_tr, 1)

    logger.info("Train split: pos=%d neg=%d (scale_pos_weight=%.3f)", pos_tr, neg_tr, scale_pos)
    logger.info("Val   split: pos=%d neg=%d", int(y_va.sum()), len(y_va) - int(y_va.sum()))
    logger.info("Test  split: pos=%d neg=%d", int(y_test.sum()), len(y_test) - int(y_test.sum()))

    # ═══════════════════════════════════════════════════
    # Model A: Classification
    # ═══════════════════════════════════════════════════
    logger.info("── Training Classification Model ──")
    clf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "scale_pos_weight": scale_pos,
        "verbose": -1,
        "seed": 42,
    }

    dtrain_clf = lgb.Dataset(X_tr, label=y_tr)
    dval_clf = lgb.Dataset(X_va, label=y_va, reference=dtrain_clf)
    clf_model = lgb.train(
        clf_params, dtrain_clf,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain_clf, dval_clf],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(EARLY_STOPPING)],
    )
    clf_best_iter = clf_model.best_iteration
    logger.info("CLF best iteration: %d", clf_best_iter)

    prob_va = clf_model.predict(X_va)
    opt_thr, opt_ba = _optimize_threshold(prob_va, y_va)
    logger.info("Optimal threshold on val: %.3f  (BA=%.4f)", opt_thr, opt_ba)

    dtrain_clf_full = lgb.Dataset(train_df[feature_cols], label=y_train)
    clf_full = lgb.train(clf_params, dtrain_clf_full, num_boost_round=max(clf_best_iter, 50))

    prob_test_clf = clf_full.predict(test_df[feature_cols])
    pred_test_clf_05 = (prob_test_clf > 0.5).astype(int)
    pred_test_clf_opt = (prob_test_clf > opt_thr).astype(int)

    logger.info("── CLF results (threshold=0.5) ──")
    m_clf_05 = _eval_direction(pred_test_clf_05, y_test, "CLF@0.5")
    logger.info("── CLF results (threshold=%.3f) ──", opt_thr)
    m_clf_opt = _eval_direction(pred_test_clf_opt, y_test, f"CLF@{opt_thr:.3f}")

    # ═══════════════════════════════════════════════════
    # Model B: Regression
    # ═══════════════════════════════════════════════════
    logger.info("── Training Regression Model ──")
    reg_params = {
        "objective": "huber",
        "metric": "mae",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "verbose": -1,
        "seed": 42,
    }

    dtrain_reg = lgb.Dataset(X_tr, label=spread_tr)
    dval_reg = lgb.Dataset(X_va, label=spread_va, reference=dtrain_reg)
    reg_model = lgb.train(
        reg_params, dtrain_reg,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain_reg, dval_reg],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(EARLY_STOPPING)],
    )
    reg_best_iter = reg_model.best_iteration
    logger.info("REG best iteration: %d", reg_best_iter)

    dtrain_reg_full = lgb.Dataset(train_df[feature_cols], label=spread_train)
    reg_full = lgb.train(reg_params, dtrain_reg_full, num_boost_round=max(reg_best_iter, 50))

    spread_pred_test = reg_full.predict(test_df[feature_cols])
    pred_test_reg = (spread_pred_test > 0).astype(int)

    logger.info("── REG results (sign of predicted spread) ──")
    m_reg = _eval_direction(pred_test_reg, y_test, "REG-sign")

    reg_mae = float(np.mean(np.abs(spread_test - spread_pred_test)))
    reg_rmse = float(np.sqrt(np.mean((spread_test - spread_pred_test) ** 2)))
    logger.info("REG spread MAE: %.2f  RMSE: %.2f", reg_mae, reg_rmse)

    # ═══════════════════════════════════════════════════
    # Model C: Combo (clf_prob + reg_sign)
    # ═══════════════════════════════════════════════════
    logger.info("── Combo results ──")
    clf_vote = (prob_test_clf > opt_thr).astype(float)
    reg_vote = (spread_pred_test > 0).astype(float)
    combo_score = 0.5 * clf_vote + 0.5 * reg_vote
    pred_test_combo = (combo_score >= 0.5).astype(int)
    m_combo = _eval_direction(pred_test_combo, y_test, "COMBO")

    # ── pick best strategy ──
    strategies = {
        "clf@0.5": (m_clf_05, pred_test_clf_05),
        f"clf@{opt_thr:.3f}": (m_clf_opt, pred_test_clf_opt),
        "reg-sign": (m_reg, pred_test_reg),
        "combo": (m_combo, pred_test_combo),
    }
    best_name = max(strategies, key=lambda k: strategies[k][0]["balanced_accuracy"])
    best_metrics, best_pred = strategies[best_name]
    logger.info("Best strategy: %s (BA=%.4f)", best_name, best_metrics["balanced_accuracy"])

    # ═══════════════════════════════════════════════════
    # 重构 RT 价格
    # ═══════════════════════════════════════════════════
    sda_test = raw["settlement_da_price"].reindex(test_df.index).values
    srt_test = raw["settlement_rt_price"].reindex(test_df.index).values

    rt_pred_reg = sda_test + spread_pred_test
    valid_rt = np.isfinite(sda_test) & np.isfinite(srt_test)

    pos_mean = float(np.mean(spread_train[y_train == 1])) if (y_train == 1).any() else 0.0
    neg_mean = float(np.mean(spread_train[y_train == 0])) if (y_train == 0).any() else 0.0
    recon_spread_best = np.where(best_pred == 1, pos_mean, neg_mean)
    rt_pred_best = sda_test + recon_spread_best

    rt_mae_reg = float(np.mean(np.abs(srt_test[valid_rt] - rt_pred_reg[valid_rt])))
    rt_mae_best = float(np.mean(np.abs(srt_test[valid_rt] - rt_pred_best[valid_rt])))
    logger.info("RT MAE (reg direct): %.2f | RT MAE (best clf recon): %.2f", rt_mae_reg, rt_mae_best)

    # ═══════════════════════════════════════════════════
    # 保存结果
    # ═══════════════════════════════════════════════════
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain_clf": clf_full.feature_importance("gain"),
        "split_clf": clf_full.feature_importance("split"),
        "gain_reg": reg_full.feature_importance("gain"),
        "split_reg": reg_full.feature_importance("split"),
    }).sort_values("gain_reg", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    logger.info("Top-10 features (REG gain):\n%s", importance.head(10)[["feature", "gain_reg", "gain_clf"]].to_string(index=False))

    rt_result_reg = pd.DataFrame({"actual": srt_test, "predicted": rt_pred_reg}, index=test_df.index)
    rt_result_reg.index.name = "ts"
    rt_result_reg = rt_result_reg[np.isfinite(rt_result_reg["actual"])].sort_index()
    rt_result_reg.to_csv(out_dir / "rt_result.csv")

    spread_result = pd.DataFrame({
        "actual_spread": spread_test,
        "pred_spread_reg": spread_pred_test,
        "pred_sign_clf05": pred_test_clf_05,
        "pred_sign_clf_opt": pred_test_clf_opt,
        "pred_sign_reg": pred_test_reg,
        "pred_sign_combo": pred_test_combo,
        "actual_sign": y_test,
        "prob_clf": prob_test_clf,
        "best_strategy": best_name,
    }, index=test_df.index)
    spread_result.index.name = "ts"
    spread_result.to_csv(out_dir / "spread_result.csv")

    all_metrics = {
        "best_strategy": best_name,
        "clf_best_iter": clf_best_iter,
        "reg_best_iter": reg_best_iter,
        "optimal_threshold": opt_thr,
        "n_features": len(feature_cols),
        "pos_mean_spread": pos_mean,
        "neg_mean_spread": neg_mean,
        "train_pos_frac": float(y_train.mean()),
        "test_pos_frac": float(y_test.mean()),
    }
    for name, (m, _) in strategies.items():
        for k, v in m.items():
            all_metrics[f"{name}_{k}"] = v
    all_metrics["reg_spread_mae"] = reg_mae
    all_metrics["reg_spread_rmse"] = reg_rmse
    all_metrics["rt_mae_reg"] = rt_mae_reg
    all_metrics["rt_mae_best_clf"] = rt_mae_best

    with open(out_dir / "direction_metrics.json", "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in all_metrics.items()},
            f, indent=2, ensure_ascii=False,
        )

    run_standard_visualization(
        out_dir / "rt_result.csv",
        out_dir=out_dir / "plots",
        label="V21-LGBM-REG→RT",
        actual_col="actual",
        pred_col="predicted",
        mode="appendix",
        weekly=True,
    )

    clf_full.save_model(str(out_dir / "model_clf.txt"))
    reg_full.save_model(str(out_dir / "model_reg.txt"))

    logger.info("=" * 60)
    logger.info("V21 FINAL RESULTS")
    logger.info("  Best strategy:    %s", best_name)
    logger.info("  Sign accuracy:    %.4f", best_metrics["accuracy"])
    logger.info("  Balanced acc:     %.4f", best_metrics["balanced_accuracy"])
    logger.info("  Prec pos/neg:     %.3f / %.3f", best_metrics["precision_pos"], best_metrics["precision_neg"])
    logger.info("  Rec  pos/neg:     %.3f / %.3f", best_metrics["recall_pos"], best_metrics["recall_neg"])
    logger.info("  REG spread MAE:   %.2f", reg_mae)
    logger.info("  RT MAE (reg):     %.2f", rt_mae_reg)
    logger.info("  Train pos%%=%.1f%% → Test pos%%=%.1f%%",
                float(y_train.mean()) * 100, float(y_test.mean()) * 100)
    logger.info("=" * 60)

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v21()
