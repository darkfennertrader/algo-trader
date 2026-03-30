from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass

import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    INDEX_CLASS_ID,
    RuntimeAssetMetadata,
)

_US_INDEX_NAMES = frozenset({"IBUS30", "IBUS500", "IBUST100"})


@dataclass(frozen=True)
class IndexRegionalTCopulaOverlayConfig:
    enabled: bool = True
    broad_df: float = 6.0
    us_df: float = 10.0
    europe_df: float = 10.0
    us_strength: float = 0.2
    europe_strength: float = 0.2
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexRegionalTCopulaMixSamples:
    broad: torch.Tensor
    us: torch.Tensor
    europe: torch.Tensor


def build_index_t_copula_factor_row_scales(
    *,
    assets: RuntimeAssetMetadata,
    mixes: IndexRegionalTCopulaMixSamples,
    overlay: IndexRegionalTCopulaOverlayConfig,
) -> torch.Tensor:
    device = mixes.broad.device
    dtype = mixes.broad.dtype
    row_scales = torch.ones(
        (int(mixes.broad.shape[0]), len(assets.asset_names)),
        device=device,
        dtype=dtype,
    )
    index_mask = (assets.class_ids == INDEX_CLASS_ID).to(device=device)
    if not bool(index_mask.any()):
        return row_scales
    row_scales[:, index_mask] = _safe_mix(mixes.broad, overlay.eps).rsqrt().unsqueeze(-1)
    us_mask = _build_us_index_mask(assets, device=device)
    europe_mask = index_mask & ~us_mask
    if bool(us_mask.any()):
        row_scales[:, us_mask] *= _regional_scale(
            mix=mixes.us,
            strength=overlay.us_strength,
            eps=overlay.eps,
        ).unsqueeze(-1)
    if bool(europe_mask.any()):
        row_scales[:, europe_mask] *= _regional_scale(
            mix=mixes.europe,
            strength=overlay.europe_strength,
            eps=overlay.eps,
        ).unsqueeze(-1)
    return row_scales


def apply_index_t_copula_overlay(
    *,
    cov_factor: torch.Tensor,
    cov_diag: torch.Tensor,
    assets: RuntimeAssetMetadata,
    mixes: IndexRegionalTCopulaMixSamples,
    overlay: IndexRegionalTCopulaOverlayConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_scales = build_index_t_copula_factor_row_scales(
        assets=assets,
        mixes=mixes,
        overlay=overlay,
    )
    return cov_factor * row_scales.unsqueeze(-1), cov_diag


def _regional_scale(*, mix: torch.Tensor, strength: float, eps: float) -> torch.Tensor:
    return _safe_mix(mix, eps).pow(-0.5 * strength)


def _safe_mix(mix: torch.Tensor, eps: float) -> torch.Tensor:
    return mix.clamp_min(float(eps))


def _build_us_index_mask(
    assets: RuntimeAssetMetadata, *, device: torch.device
) -> torch.Tensor:
    mask = [asset_name in _US_INDEX_NAMES for asset_name in assets.asset_names]
    return torch.tensor(mask, device=device, dtype=torch.bool)


__all__ = [
    "IndexRegionalTCopulaMixSamples",
    "IndexRegionalTCopulaOverlayConfig",
    "apply_index_t_copula_overlay",
    "build_index_t_copula_factor_row_scales",
]
