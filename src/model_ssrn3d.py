"""
SSRN3D — Spectral–Spatial Residual Network（电价网格版）

正确 SSRN 体积布局（与高光谱原版一致）:
  输入网格 (B, C, H_SLOTS, 7)
  → 体积 (B, 1, C, H_SLOTS, 7)
      深度 D = 特征通道（光谱维）
      空间 H × W = 日内 15min 槽 × 回溯 7 天

  Spectral residual: Conv3d kernel (3, 1, 1) — 沿 D（特征）
  Spatial  residual: Conv3d kernel (1, 3, 3) — 沿 H×W（槽×天）

训练/评估复用 model_v18_conv2d.run_v18 + V24 sql 特征。
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OUTPUT_DIR
from .model_v18_conv2d import (
    C_TOTAL,
    H_SLOTS,
    run_v18,
)
from .model_v24_da import (
    _patch_v18_for_v24_direct,
    _patch_v18_for_v24_pca,
    _restore_v18,
    _snapshot_v18,
    LAG2_COLS,
    TARGET_COL,
    V24_PCA_COMPONENTS,
    V24_USE_WEATHER,
    load_sql_feature_matrix,
    load_sql_feature_matrix_pca,
    _RT_TARGET_NAMES,
)

logger = logging.getLogger(__name__)

SSRN3D_DIR = OUTPUT_DIR / os.environ.get("SSRN3D_OUT_DIR", "ssrn3d_da_5p0").strip()

SSRN3D_DUAL = os.environ.get("SSRN3D_DUAL", "1").strip().lower() in ("1", "true", "yes")
SSRN3D_DELTA_LAMBDA = float(
    os.environ.get("SSRN3D_DELTA_LAMBDA", os.environ.get("V25_DELTA_LAMBDA", "0.2"))
)


def _grid_to_volume(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H_SLOTS, 7) → (B, 1, C, H_SLOTS, 7)；D=特征，H×W=空间。"""
    return x.unsqueeze(1).contiguous()


class _SpectralResBlock3d(nn.Module):
    """光谱残差：核 (3,1,1)，沿深度维（特征通道轴）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + x)


class _SpatialResBlock3d(nn.Module):
    """空间残差：核 (1,3,3)，沿 H（日内槽）× W（回溯天）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + x)


def _build_backbone_modules(c_in: int):
    """与 _SSRN3DBackbone 一致的模块列表，供 dry-run 推断 fc_in。"""
    return nn.ModuleDict({
        "stem": nn.Sequential(
            nn.Conv3d(1, 64, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.GELU(),
        ),
        "spectral64": _SpectralResBlock3d(64),
        "pool1": nn.MaxPool3d(kernel_size=(1, 2, 1)),
        "trans": nn.Sequential(
            nn.Conv3d(64, 128, (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU(),
        ),
        "spatial128a": _SpatialResBlock3d(128),
        "spatial128b": _SpatialResBlock3d(128),
        "pool2": nn.MaxPool3d(kernel_size=(1, 2, 1)),
        "final1": nn.Sequential(
            nn.Conv3d(128, 64, (1, 3, 3), padding=0),
            nn.BatchNorm3d(64),
            nn.GELU(),
        ),
        "final2": nn.Sequential(
            nn.Conv3d(64, 64, (1, 3, 3), padding=0),
            nn.BatchNorm3d(64),
            nn.GELU(),
        ),
    })


BACKBONE_OUT_CH = 64  # final2 输出通道；头部前 GAP，不再 flatten (64,D,H,W)


def _forward_backbone(
    mods: nn.ModuleDict,
    x: torch.Tensor,
    gap: Optional[nn.Module] = None,
) -> torch.Tensor:
    x = _grid_to_volume(x)
    x = mods["stem"](x)
    x = mods["spectral64"](x)
    x = mods["pool1"](x)
    x = mods["trans"](x)
    x = mods["spatial128a"](x)
    x = mods["spatial128b"](x)
    x = mods["pool2"](x)
    x = mods["final1"](x)
    x = mods["final2"](x)
    if gap is not None:
        x = gap(x)  # (B,64,D,H,W) → (B,64,1,1,1)
    return x


class _SSRN3DBackbone(nn.Module):
    """SSRN 骨干：先光谱卷积/残差，再空间 (1,3,3) 残差。"""

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS, n_days: int = 7):
        super().__init__()
        self.n_days = n_days
        mods = _build_backbone_modules(c_in)
        self.stem = mods["stem"]
        self.spectral64 = mods["spectral64"]
        self.pool1 = mods["pool1"]
        self.trans = mods["trans"]
        self.spatial128a = mods["spatial128a"]
        self.spatial128b = mods["spatial128b"]
        self.pool2 = mods["pool2"]
        self.final1 = mods["final1"]
        self.final2 = mods["final2"]
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc_in = BACKBONE_OUT_CH

    def _mods(self) -> nn.ModuleDict:
        return nn.ModuleDict({
            "stem": self.stem,
            "spectral64": self.spectral64,
            "pool1": self.pool1,
            "trans": self.trans,
            "spatial128a": self.spatial128a,
            "spatial128b": self.spatial128b,
            "pool2": self.pool2,
            "final1": self.final1,
            "final2": self.final2,
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward_backbone(self._mods(), x, gap=self.gap)


class SSRN3DPriceNet(nn.Module):
    """SSRN 单头（绝对价）。"""

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS, n_days: int = 7, dropout=0.1):
        super().__init__()
        self.backbone = _SSRN3DBackbone(c_in, h_slots, n_days)
        self._fc_in = self.backbone.fc_in
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._fc_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class DualHeadSSRN3DPriceNet(nn.Module):
    """SSRN 双头：price + hour-delta（同 V25）。"""

    _dual_head = True

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS, n_days: int = 7, dropout=0.1):
        super().__init__()
        self.backbone = _SSRN3DBackbone(c_in, h_slots, n_days)
        self._fc_in = self.backbone.fc_in

        def _make_head():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(self._fc_in, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        self.price_head = _make_head()
        self.delta_head = _make_head()

    def forward(self, x):
        feat = self.backbone(x)
        price = self.price_head(feat).squeeze(-1)
        delta = self.delta_head(feat).squeeze(-1)
        return price, delta


def default_ssrn3d_viz_label(use_dual: Optional[bool] = None) -> str:
    dual = SSRN3D_DUAL if use_dual is None else use_dual
    task = "RT" if TARGET_COL in _RT_TARGET_NAMES else "DA"
    before = os.environ.get("V18_CTX_BEFORE", "5")
    if dual:
        lam = os.environ.get(
            "SSRN3D_DELTA_LAMBDA",
            os.environ.get("V18_DELTA_LAMBDA", str(SSRN3D_DELTA_LAMBDA)),
        )
        return f"SSRN3D-{task}-{before}+0-λ{lam}"
    return f"SSRN3D-{task}-{before}+0"


def run_ssrn3d(out_dir: Optional[Path] = None, use_dual: Optional[bool] = None) -> Dict:
    out_dir = Path(out_dir or SSRN3D_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    dual = SSRN3D_DUAL if use_dual is None else use_dual
    cls = DualHeadSSRN3DPriceNet if dual else SSRN3DPriceNet
    cls_name = cls.__name__

    if dual and not os.environ.get("V18_DELTA_LAMBDA", "").strip():
        os.environ["V18_DELTA_LAMBDA"] = str(SSRN3D_DELTA_LAMBDA)

    logger.info("=" * 60)
    logger.info("SSRN3D — %s → %s (dual=%s)", cls_name, TARGET_COL, dual)
    logger.info(
        "  layout (B,1,C,H,W) | spectral (3,1,1) + spatial (1,3,3)"
    )

    snap = _snapshot_v18()
    try:
        use_pca = V24_PCA_COMPONENTS > 0
        if use_pca:
            df, pca_names, _ = load_sql_feature_matrix_pca(V24_PCA_COMPONENTS)
            _patch_v18_for_v24_pca(pca_names)
        else:
            df = load_sql_feature_matrix()
            _patch_v18_for_v24_direct()

        if not os.environ.get("V18_VIZ_LABEL", "").strip():
            os.environ["V18_VIZ_LABEL"] = default_ssrn3d_viz_label(dual)

        meta = {
            "model": cls_name,
            "dual_task": dual,
            "V18_DELTA_LAMBDA": float(os.environ.get("V18_DELTA_LAMBDA", SSRN3D_DELTA_LAMBDA)),
            "volume_layout": "(B,1,C,H_SLOTS,7) D=spectral C, HW=spatial",
            "spectral_kernel": "(3,1,1)",
            "spatial_kernel": "(1,3,3)",
            "head_pool": "AdaptiveAvgPool3d(1)",
            "fc_in": BACKBONE_OUT_CH,
            "ctx": f"{os.environ.get('V18_CTX_BEFORE', '5')}+0",
            "use_weather": V24_USE_WEATHER,
            "use_pca": use_pca,
            "target": TARGET_COL,
            "n_lag2": len(LAG2_COLS),
            "viz_label": os.environ["V18_VIZ_LABEL"],
        }
        with open(out_dir / "ssrn3d_meta.json", "w") as f:
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
    run_ssrn3d()
