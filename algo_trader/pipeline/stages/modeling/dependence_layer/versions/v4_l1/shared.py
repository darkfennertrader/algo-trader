from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass

import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    INDEX_CLASS_ID,
    RuntimeAssetMetadata,
)


@dataclass(frozen=True)
class IndexTCopulaOverlayConfig:
    enabled: bool = True
    df: float = 6.0
    eps: float = 1e-6


def build_index_t_copula_row_scales(
    *,
    assets: RuntimeAssetMetadata,
    mix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    safe_mix = mix.to(device=device, dtype=dtype).clamp_min(float(eps))
    row_scales = torch.ones(
        (int(safe_mix.shape[0]), len(assets.asset_names)),
        device=device,
        dtype=dtype,
    )
    index_mask = (assets.class_ids == INDEX_CLASS_ID).to(device=device)
    if not bool(index_mask.any()):
        return row_scales
    row_scales[:, index_mask] = safe_mix.rsqrt().unsqueeze(-1)
    return row_scales


def apply_index_t_copula_overlay(
    *,
    cov_factor: torch.Tensor,
    cov_diag: torch.Tensor,
    assets: RuntimeAssetMetadata,
    mix: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_scales = build_index_t_copula_row_scales(
        assets=assets,
        mix=mix,
        device=cov_factor.device,
        dtype=cov_factor.dtype,
        eps=eps,
    )
    scaled_factor = cov_factor * row_scales.unsqueeze(-1)
    scaled_diag = cov_diag.unsqueeze(0) * row_scales.square()
    return scaled_factor, scaled_diag


__all__ = [
    "IndexTCopulaOverlayConfig",
    "apply_index_t_copula_overlay",
    "build_index_t_copula_row_scales",
]
