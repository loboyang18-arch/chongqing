"""
Moirai backbone 编码器表征导出（方案 B）。

对每日 24h 预测窗：跑 Moirai encoder，将 future patch token 表征上采样到小时，
并与 context 池化向量拼接，供下游 LightGBM 使用。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from uni2ts.common.torch_util import mask_fill, packed_attention_mask

from .config import OUTPUT_DIR
from .model_moirai_da_clearing import (
    FREQ_PRESETS,
    _build_test_inputs,
    _effective_test_bounds,
    _load_series_with_covariates,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Salesforce/moirai-1.1-R-small"


def _hourly_from_future_tokens(
    fut_repr: np.ndarray,
    pred_steps: int,
    patch_size: int,
) -> np.ndarray:
    """将 future patch tokens (T, D) 上采样为 (pred_steps, D)。"""
    n_tok = max(1, int(np.ceil(pred_steps / patch_size)))
    if fut_repr.shape[0] < n_tok:
        pad = np.repeat(fut_repr[-1:], n_tok - fut_repr.shape[0], axis=0)
        fut_repr = np.concatenate([fut_repr, pad], axis=0)
    fut_repr = fut_repr[:n_tok]
    reps_per = int(np.ceil(pred_steps / n_tok))
    hourly = np.repeat(fut_repr, reps_per, axis=0)[:pred_steps]
    return hourly.astype(np.float32)


@torch.no_grad()
def _encoder_repr_batch(
    model,
    past_target: torch.Tensor,
    patch_size: int,
    pred_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
  返回 (batch, pred_steps, d_model) future 上采样表征，
  与 (batch, d_model) context mean 池化。
    """
    device = past_target.device
    bsz, ctx_len, _ = past_target.shape
    obs = torch.ones_like(past_target, dtype=torch.bool)
    pad = torch.zeros(bsz, ctx_len, dtype=torch.bool, device=device)
    future = torch.zeros(bsz, pred_steps, 1, dtype=past_target.dtype, device=device)
    fobs = torch.ones_like(future, dtype=torch.bool)
    fpad = torch.zeros(bsz, pred_steps, dtype=torch.bool, device=device)

    target, observed_mask, sample_id, time_id, variate_id, prediction_mask = model._convert(
        patch_size,
        past_target,
        obs,
        pad,
        future_target=future,
        future_observed_target=fobs,
        future_is_pad=fpad,
    )
    ps = torch.ones_like(time_id, dtype=torch.long) * patch_size
    loc, scale = model.module.scaler(
        target,
        observed_mask * ~prediction_mask.unsqueeze(-1),
        sample_id,
        variate_id,
    )
    scaled = (target - loc) / scale
    reprs_in = model.module.in_proj(scaled, ps)
    masked = mask_fill(reprs_in, prediction_mask, model.module.mask_encoding.weight)
    reprs = model.module.encoder(
        masked,
        packed_attention_mask(sample_id),
        time_id=time_id,
        var_id=variate_id,
    )

    fut_hourly: List[np.ndarray] = []
    ctx_means: List[np.ndarray] = []
    for i in range(bsz):
        pred_m = prediction_mask[i].cpu().numpy()
        r = reprs[i].cpu().numpy()
        fut_tok = r[pred_m]
        ctx_tok = r[~pred_m]
        fut_hourly.append(
            _hourly_from_future_tokens(fut_tok, pred_steps, patch_size),
        )
        ctx_means.append(ctx_tok.mean(axis=0).astype(np.float32) if len(ctx_tok) else r.mean(axis=0))

    return np.stack(fut_hourly, axis=0), np.stack(ctx_means, axis=0)


def export_hourly_encoder_embeddings(
    test_start: str,
    test_end: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    freq: str = "1h",
    context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    patch_size: int = 8,
    batch_size: int = 16,
    out_csv: Optional[Path] = None,
    dws_csv: Optional[str] = None,
) -> pd.DataFrame:
    """导出小时级 Moirai encoder 特征，列 enc_fut_* / enc_ctx_mean_*。"""
    preset = FREQ_PRESETS[freq]
    ctx_steps = context_length if context_length is not None else preset["context_steps"]
    pred_steps = prediction_length if prediction_length is not None else preset["pred_steps"]

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    logger.info("加载 Moirai %s (patch_size=%d) …", model_id, patch_size)
    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=pred_steps,
        context_length=ctx_steps,
        patch_size=patch_size,
        num_samples=10,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    d_model = int(module.d_model)

    df, _ = _load_series_with_covariates([], freq=freq, dws_csv=dws_csv)
    eff_start, eff_end = _effective_test_bounds(df, test_start, test_end, freq)
    all_days = sorted(set(df.index.normalize().date.tolist()))
    test_days = [d for d in all_days if eff_start.date() <= d <= eff_end.date()]
    samples = _build_test_inputs(
        df, test_days, ctx_steps, pred_steps, [], freq=freq,
    )
    if not samples:
        raise RuntimeError("无有效 embedding 样本")

    fut_cols = [f"enc_fut_{i:03d}" for i in range(d_model)]
    ctx_cols = [f"enc_ctx_mean_{i:03d}" for i in range(d_model)]
    rows: List[dict] = []

    for off in range(0, len(samples), batch_size):
        batch = samples[off: off + batch_size]
        past_list = []
        for s in batch:
            hist = s["target_hist"][-ctx_steps:].astype(np.float32)
            past_list.append(hist)
        past_t = torch.tensor(
            np.stack(past_list, axis=0),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        fut_h, ctx_m = _encoder_repr_batch(model, past_t, patch_size, pred_steps)
        for j, s in enumerate(batch):
            d = pd.Timestamp(s["date"])
            for h in range(24):
                row = {"ts": d + pd.Timedelta(hours=h)}
                row.update(dict(zip(fut_cols, fut_h[j, h])))
                row.update(dict(zip(ctx_cols, ctx_m[j])))
                rows.append(row)

    out = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    logger.info(
        "Encoder 特征: %d 行 × %d 列, %s ~ %s",
        len(out), len(fut_cols) + len(ctx_cols), out["ts"].min(), out["ts"].max(),
    )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        logger.info("已写入 %s", out_csv)
    return out


def load_encoder_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    return df.set_index("ts").sort_index()
