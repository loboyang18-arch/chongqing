"""
基线模型 — LightGBM 日前/实时电价预测（v2 迭代版）。

改进点（相比 v1）：
  - learning_rate 0.05→0.01，num_boost_round 1000→3000，early_stopping 50→100
  - RT 模型使用 Huber 损失降低极端值影响
  - 新增 TimeSeriesSplit 5-fold 交叉验证
  - 新增 sMAPE、加权 MAPE、排除近零价格 MAPE 等评估指标

产出：
  - output/baseline_da_result.csv
  - output/baseline_rt_result.csv
  - output/baseline_metrics.csv
"""

import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

# ── 时间划分 ──────────────────────────────────────────────
TRAIN_END = "2026-02-08 23:00:00"
TEST_START = "2026-02-09 00:00:00"

# ── 峰谷时段（重庆电力市场） ─────────────────────────────
PEAK_HOURS = set(range(8, 12)) | set(range(17, 21))
VALLEY_HOURS = set(range(0, 8)) | {23}
FLAT_HOURS = set(range(12, 17)) | set(range(21, 23))

# ── LightGBM 超参数 ──────────────────────────────────────
LGB_PARAMS_DA = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}

LGB_PARAMS_RT = {
    "objective": "huber",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100


def _period_label(hour: int) -> str:
    if hour in PEAK_HOURS:
        return "peak"
    if hour in VALLEY_HOURS:
        return "valley"
    return "flat"


def _load_dataset(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"feature_{name}.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    logger.info("Loaded %s: %d rows × %d cols", path.name, len(df), len(df.columns))
    return df


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:]
    logger.info(
        "Train: %d rows (%s ~ %s)",
        len(train), train.index.min(), train.index.max(),
    )
    logger.info(
        "Test:  %d rows (%s ~ %s)",
        len(test), test.index.min(), test.index.max(),
    )
    return train, test


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE / MAPE / sMAPE / wMAPE / filtered-MAPE."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {
            "MAE": np.nan, "RMSE": np.nan, "MAPE(%)": np.nan,
            "sMAPE(%)": np.nan, "wMAPE(%)": np.nan,
            "MAPE_filtered(%)": np.nan, "count": 0,
        }

    residual = yt - yp
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))

    nonzero = np.abs(yt) > 1e-6
    mape = (
        np.mean(np.abs(residual[nonzero]) / np.abs(yt[nonzero])) * 100
        if nonzero.sum() > 0 else np.nan
    )

    # sMAPE: symmetric MAPE, bounded [0, 200%]
    denom_s = (np.abs(yt) + np.abs(yp)) / 2
    valid_s = denom_s > 1e-6
    smape = (
        np.mean(np.abs(residual[valid_s]) / denom_s[valid_s]) * 100
        if valid_s.sum() > 0 else np.nan
    )

    # wMAPE: weighted MAPE = sum(|residual|) / sum(|actual|)
    sum_abs_actual = np.sum(np.abs(yt))
    wmape = np.sum(np.abs(residual)) / sum_abs_actual * 100 if sum_abs_actual > 1e-6 else np.nan

    # filtered MAPE: exclude hours where |actual| < 50
    high_price = np.abs(yt) >= 50
    mape_filtered = (
        np.mean(np.abs(residual[high_price]) / np.abs(yt[high_price])) * 100
        if high_price.sum() > 0 else np.nan
    )

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE(%)": round(mape, 2) if not np.isnan(mape) else np.nan,
        "sMAPE(%)": round(smape, 2) if not np.isnan(smape) else np.nan,
        "wMAPE(%)": round(wmape, 2) if not np.isnan(wmape) else np.nan,
        "MAPE_filtered(%)": round(mape_filtered, 2) if not np.isnan(mape_filtered) else np.nan,
        "count": int(len(yt)),
    }


def _metrics_by_group(
    actual: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
) -> List[Dict]:
    records = []
    for g in sorted(set(groups)):
        idx = groups == g
        m = _compute_metrics(actual[idx], pred[idx])
        m["group_value"] = g
        records.append(m)
    return records


def _train_lgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    logger.info("Best iteration: %d", model.best_iteration)
    return model


def _ts_cv_score(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Dict,
    n_splits: int = 5,
) -> Dict[str, float]:
    """TimeSeriesSplit CV: return average MAE/RMSE across folds."""
    n = len(df)
    fold_size = n // (n_splits + 1)

    fold_maes = []
    fold_rmses = []
    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(train_end_idx + fold_size, n)
        if val_end_idx <= val_start_idx:
            continue

        train_data = df.iloc[:train_end_idx]
        val_data = df.iloc[val_start_idx:val_end_idx]

        X_tr, y_tr = train_data[feature_cols], train_data[target_col]
        X_va, y_va = val_data[feature_cols], val_data[target_col]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        model = lgb.train(
            params, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        y_pred = model.predict(X_va)
        residual = y_va.values - y_pred
        fold_maes.append(np.mean(np.abs(residual)))
        fold_rmses.append(np.sqrt(np.mean(residual ** 2)))
        logger.info(
            "  CV fold %d: train=%d val=%d iter=%d MAE=%.2f",
            i + 1, len(train_data), len(val_data), model.best_iteration, fold_maes[-1],
        )

    return {
        "cv_mae_mean": round(np.mean(fold_maes), 4),
        "cv_mae_std": round(np.std(fold_maes), 4),
        "cv_rmse_mean": round(np.mean(fold_rmses), 4),
        "cv_rmse_std": round(np.std(fold_rmses), 4),
        "n_folds": len(fold_maes),
    }


def _evaluate_model(
    results: pd.DataFrame,
    model_name: str,
    method: str,
    actual_col: str,
    pred_col: str,
) -> List[Dict]:
    """对单个 (model, method) 组合做全维度评估。"""
    actual = results[actual_col].values
    pred = results[pred_col].values
    hours = results.index.hour
    dates = np.array([str(d.date()) for d in results.index])
    periods = np.array([_period_label(h) for h in hours])

    all_records = []
    base = {"model": model_name, "method": method}

    m = _compute_metrics(actual, pred)
    m.update(base)
    m.update({"group_type": "overall", "group_value": "all"})
    all_records.append(m)

    for rec in _metrics_by_group(actual, pred, hours):
        rec.update(base)
        rec["group_type"] = "hour"
        all_records.append(rec)

    for rec in _metrics_by_group(actual, pred, dates):
        rec.update(base)
        rec["group_type"] = "date"
        all_records.append(rec)

    for rec in _metrics_by_group(actual, pred, periods):
        rec.update(base)
        rec["group_type"] = "period"
        all_records.append(rec)

    return all_records


def run_single_model(
    name: str,
    target_col: str,
    naive_col: str,
    params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, lgb.Booster]:
    """训练并评估单个模型（DA 或 RT）。"""
    logger.info("=" * 60)
    logger.info("Model: %s | target: %s", name.upper(), target_col)
    logger.info("=" * 60)

    df = _load_dataset(name)
    train_df, test_df = _time_split(df)

    feature_cols = [c for c in df.columns if c != target_col]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # ── TimeSeriesSplit CV on training set ────────
    logger.info("Running TimeSeriesSplit CV (5 folds) on training set...")
    cv_scores = _ts_cv_score(train_df, feature_cols, target_col, params, n_splits=5)
    logger.info(
        "CV: MAE=%.2f±%.2f  RMSE=%.2f±%.2f",
        cv_scores["cv_mae_mean"], cv_scores["cv_mae_std"],
        cv_scores["cv_rmse_mean"], cv_scores["cv_rmse_std"],
    )

    # ── Final LightGBM on full training set ──────
    model = _train_lgb(X_train, y_train, X_test, y_test, params)
    y_pred_lgb = model.predict(X_test)

    # ── Naive baseline: lag24h ────────────────────
    y_pred_naive = test_df[naive_col].values

    # ── 结果表 ────────────────────────────────────
    results = pd.DataFrame(
        {
            "actual": y_test.values,
            "pred_lgb": y_pred_lgb,
            "pred_naive": y_pred_naive,
        },
        index=test_df.index,
    )
    results.index.name = "ts"
    out_path = OUTPUT_DIR / f"baseline_{name}_result.csv"
    results.to_csv(out_path)
    logger.info("Saved %s (%d rows)", out_path.name, len(results))

    # ── 评估指标 ──────────────────────────────────
    metrics_records = []
    metrics_records.extend(_evaluate_model(results, name, "lgb", "actual", "pred_lgb"))
    metrics_records.extend(_evaluate_model(results, name, "naive", "actual", "pred_naive"))
    metrics_df = pd.DataFrame(metrics_records)

    # ── 特征重要度 ────────────────────────────────
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        },
    ).sort_values("importance", ascending=False)

    logger.info("Top-10 features (gain):")
    for _, row in importance.head(10).iterrows():
        logger.info("  %-50s %.1f", row["feature"], row["importance"])

    return results, metrics_df, importance, model


def run_baseline() -> Dict:
    """主入口：运行 DA + RT 基线模型，合并输出评估指标。"""
    da_results, da_metrics, da_imp, da_model = run_single_model(
        "da", "target_da_clearing_price", "da_clearing_price_lag24h",
        LGB_PARAMS_DA,
    )
    rt_results, rt_metrics, rt_imp, rt_model = run_single_model(
        "rt", "target_rt_clearing_price", "rt_clearing_price_lag24h",
        LGB_PARAMS_RT,
    )

    # ── 合并指标并保存 ────────────────────────────
    all_metrics = pd.concat([da_metrics, rt_metrics], ignore_index=True)
    col_order = [
        "model", "method", "group_type", "group_value",
        "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "wMAPE(%)", "MAPE_filtered(%)", "count",
    ]
    all_metrics = all_metrics[col_order]
    all_metrics.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)

    # ── 总结 ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE SUMMARY (v2)")
    logger.info("=" * 60)
    overall = all_metrics[all_metrics["group_type"] == "overall"]
    for _, row in overall.iterrows():
        logger.info(
            "  %s %-6s | MAE=%.2f  RMSE=%.2f  MAPE=%.1f%%  sMAPE=%.1f%%  wMAPE=%.1f%%",
            row["model"].upper(), row["method"],
            row["MAE"], row["RMSE"], row["MAPE(%)"],
            row["sMAPE(%)"], row["wMAPE(%)"],
        )
    logger.info(
        "Saved: baseline_da_result.csv, baseline_rt_result.csv, baseline_metrics.csv",
    )

    return {
        "da": {"results": da_results, "metrics": da_metrics, "importance": da_imp, "model": da_model},
        "rt": {"results": rt_results, "metrics": rt_metrics, "importance": rt_imp, "model": rt_model},
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_baseline()
