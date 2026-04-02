from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass

import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    INDEX_CLASS_ID,
    RuntimeAssetMetadata,
)

_US_INDEX_NAMES = frozenset({"IBUS30", "IBUS500", "IBUST100"})
_EUROPE_INDEX_NAMES = frozenset(
    {"IBCH20", "IBDE40", "IBES35", "IBEU50", "IBFR40", "IBGB100", "IBNL25"}
)


@dataclass(frozen=True)
class StressWeightPriorConfig:
    alpha: float = 2.0
    beta: float = 8.0


@dataclass(frozen=True)
class RegionalTailConfig:
    upper_df: float
    lower_df: float
    upper_strength: float
    lower_strength: float


@dataclass(frozen=True)
class StressAsymmetricRegionalOverlayConfig:
    broad_df: float = 4.0
    broad_strength: float = 0.20
    gate_scale: float = 1.0
    us: RegionalTailConfig = RegionalTailConfig(
        upper_df=12.0,
        lower_df=10.0,
        upper_strength=0.12,
        lower_strength=0.10,
    )
    europe: RegionalTailConfig = RegionalTailConfig(
        upper_df=6.0,
        lower_df=10.0,
        upper_strength=0.18,
        lower_strength=0.10,
    )


@dataclass(frozen=True)
class IndexConditionalAsymmetricTCopulaOverlayConfig:
    calm_df: float = 6.0
    enabled: bool = True
    stress_prior: StressWeightPriorConfig = StressWeightPriorConfig()
    stress: StressAsymmetricRegionalOverlayConfig = (
        StressAsymmetricRegionalOverlayConfig()
    )
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexConditionalAsymmetricTCopulaMixSamples:
    calm: torch.Tensor
    stress_weight: torch.Tensor
    stress: torch.Tensor
    us_upper: torch.Tensor
    us_lower: torch.Tensor
    europe_upper: torch.Tensor
    europe_lower: torch.Tensor


@dataclass(frozen=True)
class _DirectionalRegionalMix:
    upper_mix: torch.Tensor
    lower_mix: torch.Tensor
    upper_strength: float
    lower_strength: float


@dataclass(frozen=True)
class _OverlayInputs:
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    index_signal: torch.Tensor


def apply_index_t_copula_overlay(
    *,
    inputs: _OverlayInputs,
    assets: RuntimeAssetMetadata,
    mixes: IndexConditionalAsymmetricTCopulaMixSamples,
    overlay: IndexConditionalAsymmetricTCopulaOverlayConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_scales = build_index_t_copula_factor_row_scales(
        assets=assets,
        mixes=mixes,
        overlay=overlay,
        index_signal=inputs.index_signal,
    )
    return inputs.cov_factor * row_scales.unsqueeze(-1), inputs.cov_diag


def build_index_t_copula_factor_row_scales(
    *,
    assets: RuntimeAssetMetadata,
    mixes: IndexConditionalAsymmetricTCopulaMixSamples,
    overlay: IndexConditionalAsymmetricTCopulaOverlayConfig,
    index_signal: torch.Tensor,
) -> torch.Tensor:
    device = mixes.calm.device
    dtype = mixes.calm.dtype
    row_scales = torch.ones(
        (int(mixes.calm.shape[0]), len(assets.asset_names)),
        device=device,
        dtype=dtype,
    )
    index_mask = (assets.class_ids == INDEX_CLASS_ID).to(device=device)
    if not bool(index_mask.any()):
        return row_scales
    row_scales[:, index_mask] = _broad_scale(mixes=mixes, overlay=overlay).unsqueeze(-1)
    us_mask = _build_region_mask(assets, _US_INDEX_NAMES, device)
    europe_mask = _build_region_mask(assets, _EUROPE_INDEX_NAMES, device)
    positive_weight, negative_weight = _build_directional_weights(
        index_signal=index_signal,
        gate_scale=overlay.stress.gate_scale,
        eps=overlay.eps,
    )
    if bool(us_mask.any()):
        row_scales[:, us_mask] *= _regional_scale(
            mix=_DirectionalRegionalMix(
                upper_mix=mixes.us_upper,
                lower_mix=mixes.us_lower,
                upper_strength=overlay.stress.us.upper_strength,
                lower_strength=overlay.stress.us.lower_strength,
            ),
            stress_weight=mixes.stress_weight,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            eps=overlay.eps,
        ).unsqueeze(-1)
    if bool(europe_mask.any()):
        row_scales[:, europe_mask] *= _regional_scale(
            mix=_DirectionalRegionalMix(
                upper_mix=mixes.europe_upper,
                lower_mix=mixes.europe_lower,
                upper_strength=overlay.stress.europe.upper_strength,
                lower_strength=overlay.stress.europe.lower_strength,
            ),
            stress_weight=mixes.stress_weight,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            eps=overlay.eps,
        ).unsqueeze(-1)
    return row_scales


def _broad_scale(
    *,
    mixes: IndexConditionalAsymmetricTCopulaMixSamples,
    overlay: IndexConditionalAsymmetricTCopulaOverlayConfig,
) -> torch.Tensor:
    calm = _safe_mix(mixes.calm, overlay.eps).rsqrt()
    stress = _safe_mix(mixes.stress, overlay.eps).pow(
        -0.5 * overlay.stress.broad_strength * mixes.stress_weight.clamp(0.0, 1.0)
    )
    return calm * stress


def _regional_scale(
    *,
    mix: _DirectionalRegionalMix,
    stress_weight: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    safe_weight = stress_weight.clamp(0.0, 1.0)
    upper = _safe_mix(mix.upper_mix, eps).pow(
        -0.5 * mix.upper_strength * safe_weight * positive_weight
    )
    lower = _safe_mix(mix.lower_mix, eps).pow(
        -0.5 * mix.lower_strength * safe_weight * negative_weight
    )
    return upper * lower


def _build_directional_weights(
    *,
    index_signal: torch.Tensor,
    gate_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_gate = max(gate_scale, eps)
    positive_weight = torch.sigmoid(index_signal / safe_gate)
    return positive_weight, 1.0 - positive_weight


def _safe_mix(mix: torch.Tensor, eps: float) -> torch.Tensor:
    return mix.clamp_min(float(eps))


def _build_region_mask(
    assets: RuntimeAssetMetadata,
    region_names: frozenset[str],
    device: torch.device,
) -> torch.Tensor:
    mask = [asset_name in region_names for asset_name in assets.asset_names]
    return torch.tensor(mask, device=device, dtype=torch.bool)


__all__ = [
    "IndexConditionalAsymmetricTCopulaMixSamples",
    "IndexConditionalAsymmetricTCopulaOverlayConfig",
    "StressAsymmetricRegionalOverlayConfig",
    "StressWeightPriorConfig",
    "apply_index_t_copula_overlay",
    "build_index_t_copula_factor_row_scales",
]
