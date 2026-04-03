from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    COMMODITY_CLASS_ID,
    FX_CLASS_ID,
    RuntimeAssetMetadata,
    asset_class_mask,
)

_US = ("IBUS30", "IBUS500", "IBUST100")
_EURO_CORE = ("IBDE40", "IBEU50", "IBFR40", "IBNL25")
_EUROPE = ("IBDE40", "IBES35", "IBEU50", "IBFR40", "IBGB100", "IBNL25")
_UK_CH = ("IBGB100", "IBCH20")
_SPREAD_COUNT = 4


@dataclass(frozen=True)
class IndexBasisPriorScaleConfig:
    global_scale: float = 0.10
    spread_scale: float = 0.06
    correlation_concentration: float = 12.0


@dataclass(frozen=True)
class IndexBasisConfig:
    enabled: bool = True
    global_df: float = 8.0
    spread_df: float = 6.0
    prior_scales: IndexBasisPriorScaleConfig = IndexBasisPriorScaleConfig()
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexBasisPosteriorMeans:
    global_scale: torch.Tensor
    spread_scale: torch.Tensor
    spread_corr_cholesky: torch.Tensor

    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "IndexBasisPosteriorMeans":
        return IndexBasisPosteriorMeans(
            global_scale=self.global_scale.to(device=device, dtype=dtype),
            spread_scale=self.spread_scale.to(device=device, dtype=dtype),
            spread_corr_cholesky=self.spread_corr_cholesky.to(
                device=device,
                dtype=dtype,
            ),
        )


@dataclass(frozen=True)
class IndexBasisCoordinates:
    global_vector: torch.Tensor
    spread_matrix: torch.Tensor


@dataclass(frozen=True)
class IndexBasisFactorState:
    global_scale: torch.Tensor
    spread_scale: torch.Tensor
    spread_corr_cholesky: torch.Tensor
    regime_scale: torch.Tensor
    global_mix: torch.Tensor
    spread_mix: torch.Tensor
    eps: float


@dataclass(frozen=True)
class _TensorSpec:
    device: torch.device
    dtype: torch.dtype


def build_index_basis_config(raw: object) -> IndexBasisConfig:
    values = _coerce_mapping(raw, label="model.params.index_basis")
    if not values:
        return IndexBasisConfig()
    base = IndexBasisConfig()
    scale_values = _coerce_mapping(
        values.get("prior_scales"),
        label="model.params.index_basis.prior_scales",
    )
    return IndexBasisConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        global_df=float(values.get("global_df", base.global_df)),
        spread_df=float(values.get("spread_df", base.spread_df)),
        prior_scales=IndexBasisPriorScaleConfig(
            global_scale=float(
                scale_values.get(
                    "global_scale",
                    base.prior_scales.global_scale,
                )
            ),
            spread_scale=float(
                scale_values.get(
                    "spread_scale",
                    base.prior_scales.spread_scale,
                )
            ),
            correlation_concentration=float(
                scale_values.get(
                    "correlation_concentration",
                    base.prior_scales.correlation_concentration,
                )
            ),
        ),
        eps=float(values.get("eps", base.eps)),
    )


def build_index_basis_coordinates(
    *,
    assets: RuntimeAssetMetadata,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexBasisCoordinates:
    tensor_spec = _TensorSpec(device=device, dtype=dtype)
    names = assets.asset_names
    global_vector = _equal_weight_vector(
        asset_names=names,
        requested=tuple(name for name in names if name.startswith("IB")),
        tensor_spec=tensor_spec,
        label="global_level",
    )
    spread_matrix = torch.stack(
        [
            _zero_sum_vector(
                asset_names=names,
                positive=_US,
                negative=_EUROPE,
                tensor_spec=tensor_spec,
                label="us_minus_europe",
            ),
            _manual_weight_vector(
                asset_names=names,
                weights_by_asset={
                    "IBUST100": 1.0,
                    "IBUS30": -0.5,
                    "IBUS500": -0.5,
                },
                tensor_spec=tensor_spec,
                label="us_internal_style",
            ),
            _zero_sum_vector(
                asset_names=names,
                positive=_EURO_CORE,
                negative=_UK_CH,
                tensor_spec=tensor_spec,
                label="euro_core_vs_uk_ch",
            ),
            _zero_sum_vector(
                asset_names=names,
                positive=("IBES35",),
                negative=_EURO_CORE,
                tensor_spec=tensor_spec,
                label="spain_vs_euro_core",
            ),
        ],
        dim=-1,
    )
    return IndexBasisCoordinates(
        global_vector=global_vector,
        spread_matrix=spread_matrix,
    )


def build_nonindex_cov_factor_path(
    *,
    loadings: Any,
    class_ids: torch.LongTensor,
    regime_path: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    fx_mask = asset_class_mask(
        class_ids,
        class_id=FX_CLASS_ID,
        dtype=dtype,
    ).unsqueeze(-1)
    commodity_mask = asset_class_mask(
        class_ids,
        class_id=COMMODITY_CLASS_ID,
        dtype=dtype,
    ).unsqueeze(-1)
    return torch.cat(
        [
            loadings.B_global.unsqueeze(0).expand(regime_path.shape[0], -1, -1),
            loadings.B_fx_broad.unsqueeze(0)
            * fx_mask.unsqueeze(0)
            * torch.exp(0.5 * regime_path[:, 0]).view(-1, 1, 1),
            loadings.B_fx_cross.unsqueeze(0)
            * fx_mask.unsqueeze(0)
            * torch.exp(0.5 * regime_path[:, 1]).view(-1, 1, 1),
            loadings.B_commodity.unsqueeze(0)
            * commodity_mask.unsqueeze(0)
            * torch.exp(0.5 * regime_path[:, 3]).view(-1, 1, 1),
        ],
        dim=-1,
    )


def build_nonindex_cov_factor_step(
    *,
    structural: Any,
    assets: RuntimeAssetMetadata,
    regime: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    fx_mask = asset_class_mask(
        assets.class_ids,
        class_id=FX_CLASS_ID,
        dtype=dtype,
    ).unsqueeze(0).unsqueeze(-1)
    commodity_mask = asset_class_mask(
        assets.class_ids,
        class_id=COMMODITY_CLASS_ID,
        dtype=dtype,
    ).unsqueeze(0).unsqueeze(-1)
    return torch.cat(
        [
            structural.B_global.unsqueeze(0).expand(regime.shape[0], -1, -1),
            structural.B_fx_broad.unsqueeze(0)
            * fx_mask
            * torch.exp(0.5 * regime[:, 0]).view(-1, 1, 1),
            structural.B_fx_cross.unsqueeze(0)
            * fx_mask
            * torch.exp(0.5 * regime[:, 1]).view(-1, 1, 1),
            structural.B_commodity.unsqueeze(0)
            * commodity_mask
            * torch.exp(0.5 * regime[:, 3]).view(-1, 1, 1),
        ],
        dim=-1,
    )


def build_index_basis_factor_block(
    *,
    coordinates: IndexBasisCoordinates,
    state: IndexBasisFactorState,
) -> torch.Tensor:
    safe_global_mix = state.global_mix.clamp_min(float(state.eps)).rsqrt().view(
        -1, 1, 1
    )
    safe_spread_mix = state.spread_mix.clamp_min(float(state.eps)).rsqrt().view(
        -1, 1, 1
    )
    safe_regime = state.regime_scale.clamp_min(float(state.eps)).view(-1, 1, 1)
    global_block = (
        coordinates.global_vector.view(1, -1, 1)
        * state.global_scale.view(1, 1, 1)
        * safe_regime
        * safe_global_mix
    )
    spread_scale_tril = torch.diag(state.spread_scale) @ state.spread_corr_cholesky
    spread_basis = coordinates.spread_matrix @ spread_scale_tril
    spread_block = spread_basis.unsqueeze(0) * safe_regime * safe_spread_mix
    return torch.cat([global_block, spread_block], dim=-1)


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _equal_weight_vector(
    *,
    asset_names: tuple[str, ...],
    requested: tuple[str, ...],
    tensor_spec: _TensorSpec,
    label: str,
) -> torch.Tensor:
    present = [asset for asset in requested if asset in asset_names]
    if not present:
        raise ConfigError(
            "Index-basis coordinate has no present assets",
            context={"coordinate": label},
    )
    weight = 1.0 / float(len(present))
    values = [weight if asset in present else 0.0 for asset in asset_names]
    return torch.tensor(
        values,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )


def _zero_sum_vector(
    *,
    asset_names: tuple[str, ...],
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    tensor_spec: _TensorSpec,
    label: str,
) -> torch.Tensor:
    positive_present = [asset for asset in positive if asset in asset_names]
    negative_present = [asset for asset in negative if asset in asset_names]
    if not positive_present or not negative_present:
        raise ConfigError(
            "Index-basis coordinate is missing required assets",
            context={"coordinate": label},
        )
    positive_weight = 1.0 / float(len(positive_present))
    negative_weight = -1.0 / float(len(negative_present))
    values = []
    for asset in asset_names:
        if asset in positive_present:
            values.append(positive_weight)
        elif asset in negative_present:
            values.append(negative_weight)
        else:
            values.append(0.0)
    return torch.tensor(
        values,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )


def _manual_weight_vector(
    *,
    asset_names: tuple[str, ...],
    weights_by_asset: dict[str, float],
    tensor_spec: _TensorSpec,
    label: str,
) -> torch.Tensor:
    missing = [asset for asset in weights_by_asset if asset not in asset_names]
    if missing:
        raise ConfigError(
            "Index-basis coordinate is missing required assets",
            context={"coordinate": label, "missing": ", ".join(sorted(missing))},
        )
    values = [float(weights_by_asset.get(asset, 0.0)) for asset in asset_names]
    return torch.tensor(
        values,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )


__all__ = [
    "IndexBasisConfig",
    "IndexBasisCoordinates",
    "IndexBasisFactorState",
    "IndexBasisPosteriorMeans",
    "IndexBasisPriorScaleConfig",
    "build_index_basis_config",
    "build_index_basis_coordinates",
    "build_index_basis_factor_block",
    "build_nonindex_cov_factor_path",
    "build_nonindex_cov_factor_step",
]
