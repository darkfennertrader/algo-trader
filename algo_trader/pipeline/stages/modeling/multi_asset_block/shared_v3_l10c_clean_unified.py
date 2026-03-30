from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass

import torch

from .shared_v3_l1_unified import INDEX_CLASS_ID, RuntimeAssetMetadata

_US_INDEX_NAMES = frozenset({"IBUS30", "IBUS500", "IBUST100"})


@dataclass(frozen=True)
class IndexTCopulaOverlayConfig:
    enabled: bool = True
    broad_df: float = 6.0
    us_diff_df: float = 10.0
    us_diff_strength: float = 0.35
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexTCopulaMixSamples:
    broad: torch.Tensor
    us_diff: torch.Tensor


def build_index_t_copula_factor_row_scales(
    *,
    assets: RuntimeAssetMetadata,
    mixes: IndexTCopulaMixSamples,
    overlay: IndexTCopulaOverlayConfig,
) -> torch.Tensor:
    device = mixes.broad.device
    dtype = mixes.broad.dtype
    safe_broad_mix = mixes.broad.clamp_min(
        float(overlay.eps)
    )
    safe_us_diff_mix = mixes.us_diff.to(device=device, dtype=dtype).clamp_min(
        float(overlay.eps)
    )
    row_scales = torch.ones(
        (int(safe_broad_mix.shape[0]), len(assets.asset_names)),
        device=device,
        dtype=dtype,
    )
    index_mask = (assets.class_ids == INDEX_CLASS_ID).to(device=device)
    if not bool(index_mask.any()):
        return row_scales
    row_scales[:, index_mask] = safe_broad_mix.rsqrt().unsqueeze(-1)
    us_index_mask = _build_us_index_mask(assets, device=device)
    if not bool(us_index_mask.any()):
        return row_scales
    us_diff_scale = safe_us_diff_mix.pow(-0.5 * overlay.us_diff_strength)
    row_scales[:, us_index_mask] *= us_diff_scale.unsqueeze(-1)
    return row_scales


def apply_index_t_copula_overlay(
    *,
    cov_factor: torch.Tensor,
    cov_diag: torch.Tensor,
    assets: RuntimeAssetMetadata,
    mixes: IndexTCopulaMixSamples,
    overlay: IndexTCopulaOverlayConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_scales = build_index_t_copula_factor_row_scales(
        assets=assets,
        mixes=mixes,
        overlay=overlay,
    )
    scaled_factor = cov_factor * row_scales.unsqueeze(-1)
    return scaled_factor, cov_diag


def _build_us_index_mask(
    assets: RuntimeAssetMetadata, *, device: torch.device
) -> torch.Tensor:
    mask = [asset_name in _US_INDEX_NAMES for asset_name in assets.asset_names]
    return torch.tensor(mask, device=device, dtype=torch.bool)


__all__ = [
    "IndexTCopulaMixSamples",
    "IndexTCopulaOverlayConfig",
    "apply_index_t_copula_overlay",
    "build_index_t_copula_factor_row_scales",
]
