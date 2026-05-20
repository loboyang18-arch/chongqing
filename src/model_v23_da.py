"""
V23 — 先按 V18 的 Lag0/Lag1/Lag2 日期对齐规则偏移特征，
再对「每个 15min 时刻在决策时能看到的全部信息」做 PCA，
主成分作为新的 Lag0 通道，交给 V18 Conv2D 预测日前出清价。

流程：
  1. build_feature_matrix() → 原始 15min 矩阵
  2. LAG0 列不偏移；LAG1 列 shift(+1天=96行)；LAG2 列 shift(+2天=192行)
     → 每行 = 「该时刻决策时能看到的全部信息」，无未来泄漏
  3. 训练窗内拟合 StandardScaler + PCA → 每个 15min 时刻得到 b 维主成分
  4. PCA 列作为新 Lag0（Lag1/Lag2 清空），V18 按原有
     「前后时刻窗口 × 7 天 lookback」组装张量、训练 Conv2D

环境变量：
  V23_PCA_COMPONENTS   主成分个数（默认 48）
  V23_PCA_RANDOM_STATE 默认 42
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR
from .experiment.splits import TRAIN_END
from .model_v16_nhits import build_feature_matrix
from .model_v18_conv2d import TARGET_COL, _v18_is_spread_target, run_v18

logger = logging.getLogger(__name__)

V23_DIR = OUTPUT_DIR / "v23_da"
V23_PCA_COMPONENTS = int(os.environ.get("V23_PCA_COMPONENTS", "48"))
V23_PCA_RANDOM_STATE = int(os.environ.get("V23_PCA_RANDOM_STATE", "42"))

SPD = 96  # slots per day (15 min)


# ── V18 通道快照 / 恢复 / Patch ────────────────────────────────────

def _snapshot_v18_channels() -> Dict[str, Any]:
    import src.model_v18_conv2d as m18
    return {
        "LAG0_COLS": list(m18.LAG0_COLS),
        "LAG1_COLS": list(m18.LAG1_COLS),
        "LAG2_COLS": list(m18.LAG2_COLS),
        "C_LAG0": int(m18.C_LAG0),
        "C_LAG1": int(m18.C_LAG1),
        "C_LAG2": int(m18.C_LAG2),
        "C_TOTAL": int(m18.C_TOTAL),
    }


def _restore_v18_channels(snap: Dict[str, Any]) -> None:
    import src.model_v18_conv2d as m18
    m18.LAG0_COLS = list(snap["LAG0_COLS"])
    m18.LAG1_COLS = list(snap["LAG1_COLS"])
    m18.LAG2_COLS = list(snap["LAG2_COLS"])
    m18.C_LAG0 = int(snap["C_LAG0"])
    m18.C_LAG1 = int(snap["C_LAG1"])
    m18.C_LAG2 = int(snap["C_LAG2"])
    m18.C_TOTAL = int(snap["C_TOTAL"])


def _apply_pca_patch(pca_names: List[str]) -> None:
    import src.model_v18_conv2d as m18
    k = len(pca_names)
    m18.LAG0_COLS = list(pca_names)
    m18.LAG1_COLS = []
    m18.LAG2_COLS = []
    m18.C_LAG0 = k + 4   # + 4 time encodings
    m18.C_LAG1 = 0
    m18.C_LAG2 = 0
    m18.C_TOTAL = m18.C_LAG0


# ── 核心：Lag 对齐 + PCA ─────────────────────────────────────────

def build_lag_aligned_pca_matrix(
    n_components: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    1. 按 V18 的 lag 规则偏移原始列（LAG1 +1天, LAG2 +2天）
    2. 在训练窗内拟合 StandardScaler + PCA
    3. 返回 (替换后的 df, 元信息)
    """
    import src.model_v18_conv2d as m18

    lag0_cols = list(m18.LAG0_COLS)
    lag1_cols = list(m18.LAG1_COLS)
    lag2_cols = list(m18.LAG2_COLS)

    df = build_feature_matrix()

    # ---- 构建 lag-aligned 矩阵 ----
    parts: List[np.ndarray] = []
    aligned_names: List[str] = []

    for col in lag0_cols:
        if col in df.columns:
            parts.append(df[col].values.astype(np.float64))
            aligned_names.append(col)

    for col in lag1_cols:
        if col in df.columns:
            shifted = df[col].shift(SPD).values.astype(np.float64)
            parts.append(shifted)
            aligned_names.append(f"{col}_lag1")

    for col in lag2_cols:
        if col in df.columns:
            shifted = df[col].shift(2 * SPD).values.astype(np.float64)
            parts.append(shifted)
            aligned_names.append(f"{col}_lag2")

    X = np.column_stack(parts)   # (T, n_aligned_features)
    logger.info("Lag-aligned feature matrix: %d rows × %d cols", X.shape[0], X.shape[1])

    # ---- PCA（仅训练窗拟合）----
    train_m = np.asarray(df.index <= TRAIN_END, dtype=bool)
    n_train = int(train_m.sum())
    if n_train < 10:
        raise RuntimeError("训练窗内样本过少 (%d)，无法拟合 PCA" % n_train)

    col_mean = np.nanmean(X[train_m], axis=0)
    X_filled = np.where(np.isfinite(X), X, col_mean)

    scaler = StandardScaler()
    scaler.fit(X_filled[train_m])
    Z = scaler.transform(X_filled)

    n_feat = Z.shape[1]
    k = max(1, min(int(n_components), n_feat, n_train - 1))
    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(Z[train_m])
    P = pca.transform(Z)

    # ---- 组装输出 df ----
    pca_names = [f"feat_pca_{i}" for i in range(k)]
    pca_df = pd.DataFrame(P, index=df.index, columns=pca_names)

    all_feat_cols = list(dict.fromkeys(lag0_cols + lag1_cols + lag2_cols))
    out = df.drop(columns=[c for c in all_feat_cols if c in df.columns], errors="ignore")
    out = out.join(pca_df, how="left")

    # 确保目标列仍在（它被 _build_daily_arrays 用来组 hourly target）
    target_needs = [TARGET_COL]
    if _v18_is_spread_target():
        target_needs.extend(["rt_clearing_price", "da_clearing_price"])
    for c in dict.fromkeys(target_needs):
        if c in df.columns and c not in out.columns:
            out[c] = df[c]

    meta: Dict[str, Any] = {
        "n_components": k,
        "requested_components": int(n_components),
        "lag0_cols_used": lag0_cols,
        "lag1_cols_shifted_1d": lag1_cols,
        "lag2_cols_shifted_2d": lag2_cols,
        "aligned_feature_names": aligned_names,
        "n_aligned_features": len(aligned_names),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_explained_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
    }
    return out, meta


# ── 评估 ──────────────────────────────────────────────────────────

def _run_standard_eval(out_dir: Path) -> None:
    root = out_dir.resolve()
    summary = root / "evaluation_summary_appendix_v1.csv"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run_evaluate_all_models.py"),
        "--output-root", str(root),
        "--summary", str(summary),
        "--task", "da",
        "--no-baseline",
    ]
    logger.info("Running standard eval: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


# ── 主入口 ────────────────────────────────────────────────────────

def run_v23(out_dir: Optional[Path] = None, run_eval: bool = True) -> Dict[str, Any]:
    out_dir = Path(out_dir or V23_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_comp = V23_PCA_COMPONENTS
    rs = V23_PCA_RANDOM_STATE

    logger.info("=" * 60)
    logger.info("V23 — Lag-aligned 全特征 PCA → V18 Conv2D, da_clearing_price")
    logger.info("  PCA components=%d (V23_PCA_COMPONENTS), random_state=%d", n_comp, rs)

    snap = _snapshot_v18_channels()
    meta: Dict[str, Any] = {}
    try:
        df_pca, meta = build_lag_aligned_pca_matrix(
            n_components=n_comp, random_state=rs,
        )
        k = int(meta["n_components"])
        pca_names = [f"feat_pca_{i}" for i in range(k)]
        _apply_pca_patch(pca_names)
        logger.info(
            "  PCA k=%d (from %d lag-aligned feats) | cumulative explained var=%.4f",
            k, meta["n_aligned_features"], meta["cumulative_explained_variance"],
        )

        meta_path = out_dir / "feature_pca_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s", meta_path.name)

        run_v18(out_dir=out_dir, feature_df=df_pca)
    finally:
        _restore_v18_channels(snap)
        logger.info("Restored V18 Lag0/Lag1/Lag2 channel definitions")

    if run_eval:
        _run_standard_eval(out_dir)
    return meta


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v23()
