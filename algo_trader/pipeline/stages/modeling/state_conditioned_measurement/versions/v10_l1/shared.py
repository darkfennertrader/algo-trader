from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.shared import (
    HybridMeasurementConfig,
    HybridMeasurementPosteriorMeans,
    HybridMeasurementPriorScaleConfig,
    HybridMeasurementStructure,
    LoadingPriorScaleConfig,
    ResidualPriorScaleConfig,
    apply_hybrid_measurement_residual_scale,
    build_hybrid_measurement_factor_block,
    build_hybrid_measurement_structure,
    build_nonindex_cov_factor_path,
    build_nonindex_cov_factor_step,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)

_COMPOSITE_NAMES = frozenset({"IBUS500", "IBEU50"})


@dataclass(frozen=True)
class StateConditionedMeasurementGateConfig:
    center: float = 0.50
    scale: float = 0.75


@dataclass(frozen=True)
class StateConditionedMeasurementConditionPriorScaleConfig:
    bias: float = 1.0
    global_weight: float = 0.50
    index_weight: float = 0.50
    contrast_strength: float = 0.12
    composite_residual_strength: float = 0.10


@dataclass(frozen=True)
class StateConditionedMeasurementConfig:
    enabled: bool = True
    state_df: float = 20.0
    prior_scales: HybridMeasurementPriorScaleConfig = (
        HybridMeasurementPriorScaleConfig()
    )
    gate: StateConditionedMeasurementGateConfig = (
        StateConditionedMeasurementGateConfig()
    )
    condition_prior_scales: StateConditionedMeasurementConditionPriorScaleConfig = (
        StateConditionedMeasurementConditionPriorScaleConfig()
    )
    eps: float = 1e-6


@dataclass(frozen=True)
class StateConditionedMeasurementCoefficients:
    bias: torch.Tensor
    global_weight: torch.Tensor
    index_weight: torch.Tensor
    contrast_strength: torch.Tensor
    composite_residual_strength: torch.Tensor

    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "StateConditionedMeasurementCoefficients":
        return StateConditionedMeasurementCoefficients(
            bias=self.bias.to(device=device, dtype=dtype),
            global_weight=self.global_weight.to(device=device, dtype=dtype),
            index_weight=self.index_weight.to(device=device, dtype=dtype),
            contrast_strength=self.contrast_strength.to(device=device, dtype=dtype),
            composite_residual_strength=self.composite_residual_strength.to(
                device=device,
                dtype=dtype,
            ),
        )


def build_state_conditioned_measurement_config(
    raw: object,
) -> StateConditionedMeasurementConfig:
    values = _coerce_mapping(raw, label="model.params.state_conditioned_measurement")
    if not values:
        return StateConditionedMeasurementConfig()
    base = StateConditionedMeasurementConfig()
    scale_values = _coerce_mapping(
        values.get("prior_scales"),
        label="model.params.state_conditioned_measurement.prior_scales",
    )
    gate_values = _coerce_mapping(
        values.get("gate"),
        label="model.params.state_conditioned_measurement.gate",
    )
    condition_values = _coerce_mapping(
        values.get("condition_prior_scales"),
        label="model.params.state_conditioned_measurement.condition_prior_scales",
    )
    return StateConditionedMeasurementConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        state_df=float(values.get("state_df", base.state_df)),
        prior_scales=HybridMeasurementPriorScaleConfig(
            state_scale=float(
                scale_values.get("state_scale", base.prior_scales.state_scale)
            ),
            correlation_concentration=float(
                scale_values.get(
                    "correlation_concentration",
                    base.prior_scales.correlation_concentration,
                )
            ),
            loading=LoadingPriorScaleConfig(
                primitive=float(
                    scale_values.get(
                        "loading_deviation",
                        base.prior_scales.loading.primitive,
                    )
                ),
                composite=float(
                    scale_values.get(
                        "composite_loading_deviation",
                        base.prior_scales.loading.composite,
                    )
                ),
            ),
            residual=ResidualPriorScaleConfig(
                primitive_scale=float(
                    scale_values.get(
                        "primitive_residual_scale",
                        base.prior_scales.residual.primitive_scale,
                    )
                ),
                composite_scale=float(
                    scale_values.get(
                        "composite_residual_scale",
                        base.prior_scales.residual.composite_scale,
                    )
                ),
                primitive_log_scale=float(
                    scale_values.get(
                        "residual_log_scale",
                        base.prior_scales.residual.primitive_log_scale,
                    )
                ),
                composite_log_scale=float(
                    scale_values.get(
                        "composite_residual_log_scale",
                        base.prior_scales.residual.composite_log_scale,
                    )
                ),
            ),
        ),
        gate=StateConditionedMeasurementGateConfig(
            center=float(gate_values.get("center", base.gate.center)),
            scale=float(gate_values.get("scale", base.gate.scale)),
        ),
        condition_prior_scales=StateConditionedMeasurementConditionPriorScaleConfig(
            bias=float(
                condition_values.get("bias", base.condition_prior_scales.bias)
            ),
            global_weight=float(
                condition_values.get(
                    "global_weight",
                    base.condition_prior_scales.global_weight,
                )
            ),
            index_weight=float(
                condition_values.get(
                    "index_weight",
                    base.condition_prior_scales.index_weight,
                )
            ),
            contrast_strength=float(
                condition_values.get(
                    "contrast_strength",
                    base.condition_prior_scales.contrast_strength,
                )
            ),
            composite_residual_strength=float(
                condition_values.get(
                    "composite_residual_strength",
                    base.condition_prior_scales.composite_residual_strength,
                )
            ),
        ),
        eps=float(values.get("eps", base.eps)),
    )


def build_state_conditioned_measurement_gate_series(
    *,
    X_asset: torch.Tensor,
    X_global: torch.Tensor,
    assets: RuntimeAssetMetadata,
    coefficients: StateConditionedMeasurementCoefficients,
    config: StateConditionedMeasurementConfig,
) -> torch.Tensor:
    global_signal = _global_feature_rms(X_global=X_global, eps=config.eps)
    index_signal = _index_feature_rms(X_asset=X_asset, assets=assets, eps=config.eps)
    safe_scale = max(config.gate.scale, config.eps)
    raw = (
        coefficients.bias
        + coefficients.global_weight * global_signal
        + coefficients.index_weight * index_signal
    )
    return torch.sigmoid((raw - config.gate.center) / safe_scale)


def build_base_hybrid_measurement_config(
    config: StateConditionedMeasurementConfig,
) -> HybridMeasurementConfig:
    return HybridMeasurementConfig(
        enabled=config.enabled,
        state_df=config.state_df,
        prior_scales=config.prior_scales,
        eps=config.eps,
    )


def build_state_conditioned_contrast_scale(
    *,
    gate: torch.Tensor,
    coefficients: StateConditionedMeasurementCoefficients,
    eps: float,
) -> torch.Tensor:
    centered_gate = gate.clamp(0.0, 1.0) - 0.5
    return torch.exp(coefficients.contrast_strength * centered_gate).clamp_min(
        float(eps)
    )


def build_state_conditioned_residual_scale(
    *,
    residual_scale: torch.Tensor,
    assets: RuntimeAssetMetadata,
    gate: torch.Tensor,
    coefficients: StateConditionedMeasurementCoefficients,
    eps: float,
) -> torch.Tensor:
    centered_gate = gate.clamp(0.0, 1.0) - 0.5
    composite_multiplier = torch.exp(
        coefficients.composite_residual_strength * centered_gate
    ).clamp_min(float(eps))
    row_scale = torch.ones(
        (int(gate.shape[0]), len(assets.asset_names)),
        device=gate.device,
        dtype=gate.dtype,
    )
    composite_mask = _build_composite_mask(assets, gate.device)
    if bool(composite_mask.any()):
        row_scale[:, composite_mask] = composite_multiplier.unsqueeze(-1)
    return residual_scale.unsqueeze(0) * row_scale


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


def _build_composite_mask(
    assets: RuntimeAssetMetadata,
    device: torch.device,
) -> torch.Tensor:
    mask = [asset_name in _COMPOSITE_NAMES for asset_name in assets.asset_names]
    return torch.tensor(mask, device=device, dtype=torch.bool)


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = [
    "HybridMeasurementPosteriorMeans",
    "HybridMeasurementStructure",
    "StateConditionedMeasurementCoefficients",
    "StateConditionedMeasurementConditionPriorScaleConfig",
    "StateConditionedMeasurementConfig",
    "StateConditionedMeasurementGateConfig",
    "apply_hybrid_measurement_residual_scale",
    "build_hybrid_measurement_factor_block",
    "build_hybrid_measurement_structure",
    "build_base_hybrid_measurement_config",
    "build_nonindex_cov_factor_path",
    "build_nonindex_cov_factor_step",
    "build_state_conditioned_contrast_scale",
    "build_state_conditioned_measurement_config",
    "build_state_conditioned_measurement_gate_series",
    "build_state_conditioned_residual_scale",
]
