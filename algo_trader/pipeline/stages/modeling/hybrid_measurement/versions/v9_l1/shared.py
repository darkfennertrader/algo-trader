from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.shared import (
    build_nonindex_cov_factor_path,
    build_nonindex_cov_factor_step,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)

_STATE_COUNT = 5
_STATE_ORDER = (
    "q_us_broad",
    "q_us_style",
    "q_euro_core",
    "q_iberia",
    "q_uk_ch",
)
_COMPOSITE_NAMES = frozenset({"IBUS500", "IBEU50"})
_INDEX_ROWS = {
    "IBUS30": (1.0, -0.7, 0.0, 0.0, 0.0),
    "IBUST100": (1.0, 0.8, 0.0, 0.0, 0.0),
    "IBUS500": (1.0, 0.1, 0.0, 0.0, 0.0),
    "IBDE40": (0.0, 0.0, 1.0, 0.0, 0.0),
    "IBFR40": (0.0, 0.0, 1.0, 0.0, 0.0),
    "IBNL25": (0.0, 0.0, 1.0, 0.0, 0.0),
    "IBES35": (0.0, 0.0, 0.75, 0.45, 0.0),
    "IBEU50": (0.0, 0.0, 1.0, 0.2, 0.0),
    "IBGB100": (0.0, 0.0, 0.0, 0.0, 1.0),
    "IBCH20": (0.0, 0.0, 0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class LoadingPriorScaleConfig:
    primitive: float = 0.08
    composite: float = 0.025


@dataclass(frozen=True)
class ResidualPriorScaleConfig:
    primitive_scale: float = 0.85
    composite_scale: float = 0.30
    primitive_log_scale: float = 0.20
    composite_log_scale: float = 0.08


@dataclass(frozen=True)
class HybridMeasurementPriorScaleConfig:
    state_scale: float = 0.10
    correlation_concentration: float = 15.0
    loading: LoadingPriorScaleConfig = LoadingPriorScaleConfig()
    residual: ResidualPriorScaleConfig = ResidualPriorScaleConfig()


@dataclass(frozen=True)
class HybridMeasurementConfig:
    enabled: bool = True
    state_df: float = 6.0
    prior_scales: HybridMeasurementPriorScaleConfig = HybridMeasurementPriorScaleConfig()
    eps: float = 1e-6


@dataclass(frozen=True)
class HybridMeasurementPosteriorMeans:
    state_scale: torch.Tensor
    state_corr_cholesky: torch.Tensor
    loading_delta: torch.Tensor
    residual_scale: torch.Tensor

    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "HybridMeasurementPosteriorMeans":
        return HybridMeasurementPosteriorMeans(
            state_scale=self.state_scale.to(device=device, dtype=dtype),
            state_corr_cholesky=self.state_corr_cholesky.to(
                device=device,
                dtype=dtype,
            ),
            loading_delta=self.loading_delta.to(device=device, dtype=dtype),
            residual_scale=self.residual_scale.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class HybridMeasurementStructure:
    anchor_loadings: torch.Tensor
    loading_deviation_scale: torch.Tensor
    residual_anchor: torch.Tensor
    residual_prior_scale: torch.Tensor


@dataclass(frozen=True)
class HybridMeasurementFactorState:
    state_scale: torch.Tensor
    state_corr_cholesky: torch.Tensor
    loading_delta: torch.Tensor
    regime_scale: torch.Tensor
    mix: torch.Tensor
    eps: float


@dataclass(frozen=True)
class _TensorSpec:
    device: torch.device
    dtype: torch.dtype


def build_hybrid_measurement_config(raw: object) -> HybridMeasurementConfig:
    values = _coerce_mapping(raw, label="model.params.hybrid_measurement")
    if not values:
        return HybridMeasurementConfig()
    base = HybridMeasurementConfig()
    scale_values = _coerce_mapping(
        values.get("prior_scales"),
        label="model.params.hybrid_measurement.prior_scales",
    )
    return HybridMeasurementConfig(
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
        eps=float(values.get("eps", base.eps)),
    )


def build_hybrid_measurement_structure(
    *,
    assets: RuntimeAssetMetadata,
    config: HybridMeasurementConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> HybridMeasurementStructure:
    tensor_spec = _TensorSpec(device=device, dtype=dtype)
    _validate_index_names(asset_names=assets.asset_names)
    anchor_rows = []
    delta_rows = []
    residual_anchor = []
    residual_prior_scale = []
    for asset_name in assets.asset_names:
        anchor_rows.append(_anchor_row(asset_name=asset_name, tensor_spec=tensor_spec))
        delta_rows.append(
            _loading_deviation_row(
                asset_name=asset_name,
                config=config,
                tensor_spec=tensor_spec,
            )
        )
        residual_anchor.append(_residual_anchor(asset_name=asset_name, config=config))
        residual_prior_scale.append(
            _residual_prior_scale(asset_name=asset_name, config=config)
        )
    return HybridMeasurementStructure(
        anchor_loadings=torch.stack(anchor_rows, dim=0),
        loading_deviation_scale=torch.stack(delta_rows, dim=0),
        residual_anchor=torch.tensor(
            residual_anchor,
            device=device,
            dtype=dtype,
        ),
        residual_prior_scale=torch.tensor(
            residual_prior_scale,
            device=device,
            dtype=dtype,
        ),
    )


def build_hybrid_measurement_factor_block(
    *,
    structure: HybridMeasurementStructure,
    state: HybridMeasurementFactorState,
) -> torch.Tensor:
    safe_mix = state.mix.clamp_min(float(state.eps)).rsqrt().view(-1, 1, 1)
    safe_regime = state.regime_scale.clamp_min(float(state.eps)).view(-1, 1, 1)
    loading_matrix = structure.anchor_loadings + state.loading_delta
    state_scale_tril = torch.diag(state.state_scale) @ state.state_corr_cholesky
    factor_block = loading_matrix @ state_scale_tril
    return factor_block.unsqueeze(0) * safe_regime * safe_mix


def apply_hybrid_measurement_residual_scale(
    *,
    cov_diag: torch.Tensor,
    residual_scale: torch.Tensor,
) -> torch.Tensor:
    return cov_diag * residual_scale.square()


def _validate_index_names(*, asset_names: tuple[str, ...]) -> None:
    unknown = sorted(
        asset_name
        for asset_name in asset_names
        if asset_name.startswith("IB") and asset_name not in _INDEX_ROWS
    )
    if unknown:
        raise ConfigError(
            "Hybrid-measurement block encountered unsupported index names",
            context={"assets": ", ".join(unknown)},
        )


def _anchor_row(*, asset_name: str, tensor_spec: _TensorSpec) -> torch.Tensor:
    row = _INDEX_ROWS.get(asset_name, (0.0, 0.0, 0.0, 0.0, 0.0))
    return torch.tensor(
        row,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )


def _loading_deviation_row(
    *,
    asset_name: str,
    config: HybridMeasurementConfig,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if asset_name not in _INDEX_ROWS:
        value = 1e-6
    elif asset_name in _COMPOSITE_NAMES:
        value = config.prior_scales.loading.composite
    else:
        value = config.prior_scales.loading.primitive
    return torch.full(
        (_STATE_COUNT,),
        float(value),
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )


def _residual_anchor(
    *,
    asset_name: str,
    config: HybridMeasurementConfig,
) -> float:
    if asset_name not in _INDEX_ROWS:
        return 1.0
    if asset_name in _COMPOSITE_NAMES:
        return config.prior_scales.residual.composite_scale
    return config.prior_scales.residual.primitive_scale


def _residual_prior_scale(
    *,
    asset_name: str,
    config: HybridMeasurementConfig,
) -> float:
    if asset_name not in _INDEX_ROWS:
        return 1e-6
    if asset_name in _COMPOSITE_NAMES:
        return config.prior_scales.residual.composite_log_scale
    return config.prior_scales.residual.primitive_log_scale


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = [
    "HybridMeasurementConfig",
    "HybridMeasurementFactorState",
    "HybridMeasurementPosteriorMeans",
    "HybridMeasurementPriorScaleConfig",
    "LoadingPriorScaleConfig",
    "ResidualPriorScaleConfig",
    "HybridMeasurementStructure",
    "apply_hybrid_measurement_residual_scale",
    "build_hybrid_measurement_config",
    "build_hybrid_measurement_factor_block",
    "build_hybrid_measurement_structure",
    "build_nonindex_cov_factor_path",
    "build_nonindex_cov_factor_step",
]
