"""
V25 — 10 层 Conv2D + 残差连接，预测日前出清价

基于 V24 sql_data 特征 + V18 训练流程，将 3 层 Conv2dPriceNet 替换为
10 层 ResConv2dPriceNet（含 3 个残差块）。

输入: (B, C_TOTAL, H_SLOTS, 7) — 与 V18 完全一致
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OUTPUT_DIR
from .model_v18_conv2d import (
    C_TOTAL, H_SLOTS, DEVICE, SLOTS_PER_HOUR,
    CONTEXT_BEFORE, CONTEXT_AFTER,
    run_v18, train_model,
)
from .model_v24_da import (
    _load_raw_df, _patch_v18_for_v24_direct, _snapshot_v18, _restore_v18,
    V24_USE_WEATHER, V24_PCA_COMPONENTS,
    load_sql_feature_matrix, load_sql_feature_matrix_pca,
    _patch_v18_for_v24_pca,
    LAG0_COLS, LAG1_COLS, LAG2_COLS, TARGET_COL,
    _RT_TARGET_NAMES,
)

logger = logging.getLogger(__name__)

V25_DIR = OUTPUT_DIR / os.environ.get("V25_OUT_DIR", "v25_resconv").strip()


class _ResBlock(nn.Module):
    """Conv-BN-GELU-Conv-BN + skip → GELU (2 conv layers per block)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + x)


class ResConv2dPriceNet(nn.Module):
    """10-layer Conv2D with residual connections.

    Architecture (10 conv layers):
      Stem   Conv(C_in→64) + BN + GELU                    L1
      Res64  _ResBlock(64)                                 L2-L3  skip
             MaxPool(2,1)
      Trans  Conv(64→128)  + BN + GELU                    L4
      Res128a _ResBlock(128)                               L5-L6  skip
      Res128b _ResBlock(128)                               L7-L8  skip
             MaxPool(2,1)
      Final  Conv(128→64, p=0) + BN + GELU                L9
      Shrink Conv(64→64, p=0)  + BN + GELU                L10
      Head:  Flatten → FC(→128) → GELU → Dropout → FC(→1)
    """

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS, dropout=0.1):
        super().__init__()
        # L1: stem
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # L2-L3: res block 64ch
        self.res64 = _ResBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        # L4: channel transition 64→128
        self.trans = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # L5-L6, L7-L8: two res blocks 128ch
        self.res128a = _ResBlock(128)
        self.res128b = _ResBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # L9: shrink spatial (pad=0)
        self.final1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # L10: further shrink (pad=0)
        self.final2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        h_out = h_slots // 2 // 2 - 2 - 2  # 2x pool(2,1) + 2x conv(k=3,p=0)
        w_out = 7 - 2 - 2                   # 7 → 5 → 3
        fc_in = 64 * h_out * w_out
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self._fc_in = fc_in

    def forward(self, x):
        x = self.stem(x)       # L1
        x = self.res64(x)      # L2-L3 + skip
        x = self.pool1(x)
        x = self.trans(x)      # L4
        x = self.res128a(x)    # L5-L6 + skip
        x = self.res128b(x)    # L7-L8 + skip
        x = self.pool2(x)
        x = self.final1(x)     # L9
        x = self.final2(x)     # L10
        return self.head(x).squeeze(-1)


class DualHeadResConv2dPriceNet(nn.Module):
    """10-layer ResConv backbone + dual heads (price & delta).

    Shared backbone identical to ResConv2dPriceNet.
    price_head → absolute price prediction
    delta_head → hour-to-hour price change prediction
    forward returns (price, delta) tuple.
    """

    _dual_head = True

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.res64 = _ResBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.trans = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.res128a = _ResBlock(128)
        self.res128b = _ResBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.final1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=0), nn.BatchNorm2d(64), nn.GELU())
        self.final2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.GELU())

        h_out = h_slots // 2 // 2 - 2 - 2
        w_out = 7 - 2 - 2
        fc_in = 64 * h_out * w_out
        self._fc_in = fc_in

        self.price_head = nn.Sequential(
            nn.Flatten(), nn.Linear(fc_in, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))
        self.delta_head = nn.Sequential(
            nn.Flatten(), nn.Linear(fc_in, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def _backbone(self, x):
        x = self.stem(x)
        x = self.res64(x)
        x = self.pool1(x)
        x = self.trans(x)
        x = self.res128a(x)
        x = self.res128b(x)
        x = self.pool2(x)
        x = self.final1(x)
        x = self.final2(x)
        return x

    def forward(self, x):
        feat = self._backbone(x)
        price = self.price_head(feat).squeeze(-1)
        delta = self.delta_head(feat).squeeze(-1)
        return price, delta


V25_DUAL = os.environ.get("V25_DUAL", "0").strip().lower() in ("1", "true", "yes")


def default_v25_viz_label(use_dual: Optional[bool] = None) -> str:
    """默认图例名，如 V25-ResConv-DA-5+0-λ0.2（可由 V18_VIZ_LABEL 覆盖）。"""
    dual = V25_DUAL if use_dual is None else use_dual
    task = "RT" if TARGET_COL in _RT_TARGET_NAMES else "DA"
    before = os.environ.get("V25_CTX_BEFORE", os.environ.get("V18_CTX_BEFORE", "5"))
    after = os.environ.get("V25_CTX_AFTER", os.environ.get("V18_CTX_AFTER", "0"))
    ctx = f"{before}+{after}"
    if dual:
        lam = os.environ.get("V25_DELTA_LAMBDA", os.environ.get("V18_DELTA_LAMBDA", "0.2"))
        return f"V25-ResConv-{task}-{ctx}-λ{lam}"
    return f"V25-ResConv-{task}-{ctx}"


def run_v25(out_dir: Optional[Path] = None) -> Dict:
    import json
    out_dir = Path(out_dir or V25_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_dual = V25_DUAL
    cls = DualHeadResConv2dPriceNet if use_dual else ResConv2dPriceNet
    cls_name = cls.__name__

    logger.info("=" * 60)
    logger.info("V25 — %s (10-layer) → %s", cls_name, TARGET_COL)
    logger.info("  Using V24 sql_data features, model_cls=%s", cls_name)

    snap = _snapshot_v18()
    try:
        use_pca = V24_PCA_COMPONENTS > 0
        if use_pca:
            df, pca_names, pca_meta = load_sql_feature_matrix_pca(
                V24_PCA_COMPONENTS)
            _patch_v18_for_v24_pca(pca_names)
        else:
            df = load_sql_feature_matrix()
            _patch_v18_for_v24_direct()

        if not os.environ.get("V18_VIZ_LABEL", "").strip():
            os.environ["V18_VIZ_LABEL"] = default_v25_viz_label(use_dual)

        meta = {
            "model": cls_name,
            "layers": 10,
            "residual_connections": "L2-L3, L5-L6, L7-L8 (3 ResBlocks)",
            "dual_task": use_dual,
            "use_weather": V24_USE_WEATHER,
            "use_pca": use_pca,
            "target": TARGET_COL,
            "n_lag2": len(LAG2_COLS),
            "viz_label": os.environ["V18_VIZ_LABEL"],
        }
        with open(out_dir / "v25_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        run_v18(out_dir=out_dir, feature_df=df, model_cls=cls)
    finally:
        _restore_v18(snap)
        logger.info("Restored V18 channel definitions")

    return meta


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_v25()
