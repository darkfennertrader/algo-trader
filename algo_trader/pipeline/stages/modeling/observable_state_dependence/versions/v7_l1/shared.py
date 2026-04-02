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
class ObservableStateGateConfig:
    center: float = 0.50
    scale: float = 0.75


@dataclass(frozen=True)
class ObservableStatePriorScaleConfig:
    bias: float = 1.0
    global_weight: float = 0.50
    index_weight: float = 0.50
    broad_strength: float = 0.20
    regional_strength: float = 0.12


@dataclass(frozen=True)
class ObservableStateDependenceConfig:
    enabled: bool = True
    gate: ObservableStateGateConfig = ObservableStateGateConfig()
    prior_scales: ObservableStatePriorScaleConfig = ObservableStatePriorScaleConfig()
    eps: float = 1e-6


@dataclass(frozen=True)
class ObservableStateCoefficients:
    bias: torch.Tensor
    global_weight: torch.Tensor
    index_weight: torch.Tensor
    broad_strength: torch.Tensor
    us_strength: torch.Tensor
    europe_strength: torch.Tensor

    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "ObservableStateCoefficients":
        return ObservableStateCoefficients(
            bias=self.bias.to(device=device, dtype=dtype),
            global_weight=self.global_weight.to(device=device, dtype=dtype),
            index_weight=self.index_weight.to(device=device, dtype=dtype),
            broad_strength=self.broad_strength.to(device=device, dtype=dtype),
            us_strength=self.us_strength.to(device=device, dtype=dtype),
            europe_strength=self.europe_strength.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class ObservableStateOverlayInputs:
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    gate: torch.Tensor


def build_observable_state_gate_series(
    *,
    X_asset: torch.Tensor,
    X_global: torch.Tensor,
    assets: RuntimeAssetMetadata,
    coefficients: ObservableStateCoefficients,
    overlay: ObservableStateDependenceConfig,
) -> torch.Tensor:
    global_signal = _global_feature_rms(
        X_global=X_global,
        eps=overlay.eps,
    )
    index_signal = _index_feature_rms(
        X_asset=X_asset,
        assets=assets,
        eps=overlay.eps,
    )
    safe_scale = max(overlay.gate.scale, overlay.eps)
    raw = (
        coefficients.bias
        + coefficients.global_weight * global_signal
        + coefficients.index_weight * index_signal
    )
    return torch.sigmoid((raw - overlay.gate.center) / safe_scale)


def apply_observable_state_dependence_overlay(
    *,
    inputs: ObservableStateOverlayInputs,
    assets: RuntimeAssetMetadata,
    coefficients: ObservableStateCoefficients,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_scales = build_observable_state_row_scales(
        assets=assets,
        gate=inputs.gate,
        coefficients=coefficients,
        eps=eps,
    )
    scaled_factor = inputs.cov_factor * row_scales.unsqueeze(-1)
    expanded_diag = inputs.cov_diag.unsqueeze(0).expand(row_scales.shape[0], -1)
    return scaled_factor, expanded_diag


def build_observable_state_row_scales(
    *,
    assets: RuntimeAssetMetadata,
    gate: torch.Tensor,
    coefficients: ObservableStateCoefficients,
    eps: float,
) -> torch.Tensor:
    device = gate.device
    dtype = gate.dtype
    safe_gate = gate.clamp(0.0, 1.0)
    row_scales = torch.ones(
        (int(safe_gate.shape[0]), len(assets.asset_names)),
        device=device,
        dtype=dtype,
    )
    index_mask = (assets.class_ids == INDEX_CLASS_ID).to(device=device)
    if not bool(index_mask.any()):
        return row_scales
    broad_scale = torch.exp(coefficients.broad_strength.to(device=device, dtype=dtype) * safe_gate)
    row_scales[:, index_mask] = broad_scale.unsqueeze(-1)
    us_mask = _build_region_mask(assets, _US_INDEX_NAMES, device)
    europe_mask = _build_region_mask(assets, _EUROPE_INDEX_NAMES, device)
    if bool(us_mask.any()):
        row_scales[:, us_mask] *= torch.exp(
            coefficients.us_strength.to(device=device, dtype=dtype)
            * safe_gate
        ).unsqueeze(-1)
    if bool(europe_mask.any()):
        row_scales[:, europe_mask] *= torch.exp(
            coefficients.europe_strength.to(device=device, dtype=dtype)
            * safe_gate
        ).unsqueeze(-1)
    return row_scales.clamp_min(float(eps))


def _global_feature_rms(*, X_global: torch.Tensor, eps: float) -> torch.Tensor:
    if int(X_global.shape[-1]) < 1:
        return torch.zeros((int(X_global.shape[0]),), device=X_global.device, dtype=X_global.dtype)
    values = torch.nan_to_num(X_global)
    return values.square().mean(dim=-1).clamp_min(float(eps)).sqrt()


def _index_feature_rms(
    *,
    X_asset: torch.Tensor,
    assets: RuntimeAssetMetadata,
    eps: float,
) -> torch.Tensor:
    index_mask = assets.index_mask.to(device=X_asset.device)
    if not bool(index_mask.any()):
        return torch.zeros((int(X_asset.shape[0]),), device=X_asset.device, dtype=X_asset.dtype)
    index_values = torch.nan_to_num(X_asset[:, index_mask, :])
    return index_values.square().mean(dim=(1, 2)).clamp_min(float(eps)).sqrt()


def _build_region_mask(
    assets: RuntimeAssetMetadata,
    region_names: frozenset[str],
    device: torch.device,
) -> torch.Tensor:
    mask = [asset_name in region_names for asset_name in assets.asset_names]
    return torch.tensor(mask, device=device, dtype=torch.bool)


__all__ = [
    "ObservableStateCoefficients",
    "ObservableStateDependenceConfig",
    "ObservableStateGateConfig",
    "ObservableStateOverlayInputs",
    "ObservableStatePriorScaleConfig",
    "apply_observable_state_dependence_overlay",
    "build_observable_state_gate_series",
    "build_observable_state_row_scales",
]
