"""
V25 时段三模型：与单模相同的逐小时单点预测，训练/预测仅覆盖该段小时，再拼接 24h。

默认 8+8+8：00–07 / 08–15 / 16–23。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

SEGMENTS_888: Tuple[Tuple[int, int, str], ...] = (
    (0, 8, "h00_07"),
    (8, 16, "h08_15"),
    (16, 24, "h16_23"),
)


@dataclass(frozen=True)
class SegmentSpec:
    hour_start: int
    hour_end: int
    name: str

    @classmethod
    def from_name(cls, name: str) -> "SegmentSpec":
        for h0, h1, nm in SEGMENTS_888:
            if nm == name:
                return cls(h0, h1, nm)
        raise ValueError(f"Unknown segment: {name}")

    @classmethod
    def all_segments(cls) -> List["SegmentSpec"]:
        return [cls(h0, h1, nm) for h0, h1, nm in SEGMENTS_888]


def stitch_hourly_predictions(
    parts: Sequence[np.ndarray],
    segments: Sequence[SegmentSpec] = None,
) -> np.ndarray:
    """parts: 与 segments 顺序对应的 (N, 8) 预测，拼成 (N, 24)。"""
    segs = segments or SegmentSpec.all_segments()
    if not parts:
        return np.zeros((0, 24))
    n = parts[0].shape[0]
    out = np.zeros((n, 24), dtype=np.float64)
    for seg, pred in zip(segs, parts):
        out[:, seg.hour_start : seg.hour_end] = pred
    return out
