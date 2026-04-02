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
class StressMixturePriorConfig:
    alpha: float = 2.0
    beta: float = 8.0


@dataclass(frozen=True)
class StressGateConfig:
    center: float = 1.0
    scale: float = 0.75
    max_weight: float = 1.0


@dataclass(frozen=True)
class RegionalStressOverlayConfig:
    broad_df: float = 4.0
    us_df: float = 10.0
    europe_df: float = 6.0
    broad_strength: float = 0.20
    regional_strength: float = 0.12


@dataclass(frozen=True)
class IndexSoftStateMixtureCopulaOverlayConfig:
    calm_df: float = 6.0
    enabled: bool = True
    stress_prior: StressMixturePriorConfig = StressMixturePriorConfig()
    gate: StressGateConfig = StressGateConfig()
    stress: RegionalStressOverlayConfig = RegionalStressOverlayConfig()
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexSoftStateMixtureCopulaMixSamples:
    calm: torch.Tensor
    mixture_weight: torch.Tensor
    stress: torch.Tensor
    us_stress: torch.Tensor
    europe_stress: torch.Tensor


@dataclass(frozen=True)
class OverlayInputs:
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    index_signal: torch.Tensor


def apply_index_t_copula_overlay(
    *,
    inputs: OverlayInputs,
    assets: RuntimeAssetMetadata,
    mixes: IndexSoftStateMixtureCopulaMixSamples,
    overlay: IndexSoftStateMixtureCopulaOverlayConfig,
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
    mixes: IndexSoftStateMixtureCopulaMixSamples,
    overlay: IndexSoftStateMixtureCopulaOverlayConfig,
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
    stress_weight = _effective_stress_weight(
        mixture_weight=mixes.mixture_weight,
        index_signal=index_signal,
        gate=overlay.gate,
        eps=overlay.eps,
    )
    row_scales[:, index_mask] = _broad_scale(
        mixes=mixes,
        overlay=overlay,
        stress_weight=stress_weight,
    ).unsqueeze(-1)
    us_mask = _build_region_mask(assets, _US_INDEX_NAMES, device)
    europe_mask = _build_region_mask(assets, _EUROPE_INDEX_NAMES, device)
    if bool(us_mask.any()):
        row_scales[:, us_mask] *= _regional_scale(
            mix=mixes.us_stress,
            stress_weight=stress_weight,
            strength=overlay.stress.regional_strength,
            eps=overlay.eps,
        ).unsqueeze(-1)
    if bool(europe_mask.any()):
        row_scales[:, europe_mask] *= _regional_scale(
            mix=mixes.europe_stress,
            stress_weight=stress_weight,
            strength=overlay.stress.regional_strength,
            eps=overlay.eps,
        ).unsqueeze(-1)
    return row_scales


def _effective_stress_weight(
    *,
    mixture_weight: torch.Tensor,
    index_signal: torch.Tensor,
    gate: StressGateConfig,
    eps: float,
) -> torch.Tensor:
    safe_scale = max(gate.scale, eps)
    gate_center = torch.as_tensor(
        gate.center,
        device=index_signal.device,
        dtype=index_signal.dtype,
    )
    gate_weight = torch.sigmoid((index_signal.abs() - gate_center) / safe_scale)
    return mixture_weight.clamp(0.0, 1.0) * gate_weight * float(gate.max_weight)


def _broad_scale(
    *,
    mixes: IndexSoftStateMixtureCopulaMixSamples,
    overlay: IndexSoftStateMixtureCopulaOverlayConfig,
    stress_weight: torch.Tensor,
) -> torch.Tensor:
    calm = _safe_mix(mixes.calm, overlay.eps).rsqrt()
    stress = _regional_scale(
        mix=mixes.stress,
        stress_weight=stress_weight,
        strength=overlay.stress.broad_strength,
        eps=overlay.eps,
    )
    return calm * stress


def _regional_scale(
    *,
    mix: torch.Tensor,
    stress_weight: torch.Tensor,
    strength: float,
    eps: float,
) -> torch.Tensor:
    exponent = -0.5 * strength * stress_weight.clamp(0.0, 1.0)
    return _safe_mix(mix, eps).pow(exponent)


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
    "IndexSoftStateMixtureCopulaMixSamples",
    "IndexSoftStateMixtureCopulaOverlayConfig",
    "OverlayInputs",
    "RegionalStressOverlayConfig",
    "StressGateConfig",
    "StressMixturePriorConfig",
    "apply_index_t_copula_overlay",
    "build_index_t_copula_factor_row_scales",
]
