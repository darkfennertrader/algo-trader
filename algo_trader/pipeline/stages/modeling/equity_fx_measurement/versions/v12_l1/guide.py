from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    _sample_index_t_copula_mix_sites,
)
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import FilteringState
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    V3L1UnifiedGuideConfig,
    _build_context as _build_base_context,
    _build_guide_config as _build_base_guide_config,
    _sample_local_state_sites as _sample_base_local_state_sites,
    _sample_regime_scale_sites as _sample_base_regime_scale_sites,
    _sample_structural_sites as _sample_base_structural_sites,
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v12_l1, merge_nested_params
from .shared import (
    EquityFXMeasurementPosteriorMeans,
    build_equity_fx_measurement_config,
    build_equity_fx_measurement_structure,
)

_STATE_COUNT = 6


@dataclass(frozen=True)
class _PositiveSiteInit:
    loc: float
    scale: float
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _GammaSiteInit:
    concentration: float
    rate: float
    shape: tuple[int, ...]


@dataclass(frozen=True)
class V12L1GuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    index_t_copula_enabled: bool = True
    equity_fx_measurement_enabled: bool = True


@dataclass
class EquityFXMeasurementGuideV12L1OnlineFiltering(PyroGuide):
    config: V12L1GuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch)
        _sample_base_structural_sites(context=context, config=self.config.base)
        _sample_base_regime_scale_sites(context)
        _sample_base_local_state_sites(context)
        _sample_index_t_copula_mix_sites(
            time_count=int(context.shape.T),
            device=context.device,
            dtype=context.dtype,
            enabled=self.config.index_t_copula_enabled,
        )
        _sample_equity_fx_measurement_sites(
            enabled=self.config.equity_fx_measurement_enabled,
            batch=runtime_batch,
            time_count=int(context.shape.T),
            device=context.device,
            dtype=context.dtype,
        )

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def equity_fx_measurement_posterior_means(
        self,
        *,
        batch: Any,
    ) -> EquityFXMeasurementPosteriorMeans:
        runtime_batch = (
            batch
            if hasattr(batch, "assets") and hasattr(batch, "X_asset")
            else build_v3_l1_unified_runtime_batch(batch)
        )
        structure = build_equity_fx_measurement_structure(
            assets=runtime_batch.assets,
            config=build_equity_fx_measurement_config({"enabled": True}),
            device=runtime_batch.X_asset.device,
            dtype=runtime_batch.X_asset.dtype,
        )
        if not self.config.equity_fx_measurement_enabled:
            zeros = torch.zeros((_STATE_COUNT,), dtype=runtime_batch.X_asset.dtype)
            return EquityFXMeasurementPosteriorMeans(
                state_scale=zeros,
                state_corr_cholesky=torch.eye(
                    _STATE_COUNT,
                    dtype=runtime_batch.X_asset.dtype,
                ),
                loading_delta=torch.zeros_like(structure.anchor_loadings),
                residual_scale=torch.ones_like(structure.residual_anchor),
            )
        return EquityFXMeasurementPosteriorMeans(
            state_scale=_lognormal_mean(
                pyro.param("equity_fx_measurement_state_scale_loc"),
                pyro.param("equity_fx_measurement_state_scale_scale"),
            ),
            state_corr_cholesky=pyro.param(
                "equity_fx_measurement_state_corr_cholesky_loc"
            ).detach().clone(),
            loading_delta=pyro.param(
                "equity_fx_measurement_loading_delta_loc"
            ).detach().clone(),
            residual_scale=_lognormal_mean(
                pyro.param("equity_fx_measurement_residual_scale_loc"),
                pyro.param("equity_fx_measurement_residual_scale_scale"),
            ),
        )

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(config=self.config.base)


@register_guide("equity_fx_measurement_guide_v12_l1_online_filtering")
def build_equity_fx_measurement_guide_v12_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v12_l1(), params)
    return EquityFXMeasurementGuideV12L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V12L1GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V12L1GuideConfig()
    extra = set(values) - {
        "global_factor_count",
        "fx_broad_factor_count",
        "fx_cross_factor_count",
        "index_factor_count",
        "index_static_factor_count",
        "commodity_factor_count",
        "phi_fx_broad",
        "phi_fx_cross",
        "phi_index",
        "phi_commodity",
        "index_group_enabled",
        "index_t_copula_enabled",
        "equity_fx_measurement_enabled",
    }
    if extra:
        raise ConfigError(
            "Unknown equity_fx_measurement_guide_v12_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value
        for key, value in values.items()
        if key not in {"index_t_copula_enabled", "equity_fx_measurement_enabled"}
    }
    return V12L1GuideConfig(
        base=_build_base_guide_config(base_payload),
        index_t_copula_enabled=bool(values.get("index_t_copula_enabled", True)),
        equity_fx_measurement_enabled=bool(
            values.get("equity_fx_measurement_enabled", True)
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_equity_fx_measurement_sites(
    *,
    enabled: bool,
    batch: Any,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not enabled:
        return
    config = build_equity_fx_measurement_config({"enabled": True})
    structure = build_equity_fx_measurement_structure(
        assets=batch.assets,
        config=config,
        device=device,
        dtype=dtype,
    )
    _sample_positive_site(
        name="equity_fx_measurement_state_scale",
        init=_PositiveSiteInit(loc=-2.95, scale=0.16, shape=(_STATE_COUNT,)),
        device=device,
        dtype=dtype,
    )
    cholesky = pyro.param(
        "equity_fx_measurement_state_corr_cholesky_loc",
        torch.eye(_STATE_COUNT, device=device, dtype=dtype),
        constraint=constraints.corr_cholesky,
    )
    pyro.sample(
        "equity_fx_measurement_state_corr_cholesky",
        dist.Delta(cholesky, event_dim=2),
    )
    _sample_matrix_site(
        name="equity_fx_measurement_loading_delta",
        scale=structure.loading_deviation_scale,
        device=device,
        dtype=dtype,
    )
    _sample_lognormal_site(
        name="equity_fx_measurement_residual_scale",
        anchor=structure.residual_anchor,
        log_scale=structure.residual_prior_scale,
    )
    _sample_gamma_path_site(
        name="equity_fx_measurement_mix",
        init=_GammaSiteInit(concentration=6.0, rate=6.0, shape=(time_count,)),
        device=device,
        dtype=dtype,
    )


def _sample_positive_site(
    *,
    name: str,
    init: _PositiveSiteInit,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    loc = pyro.param(
        f"{name}_loc",
        torch.full(init.shape, init.loc, device=device, dtype=dtype),
    )
    scale = pyro.param(
        f"{name}_scale",
        torch.full(init.shape, init.scale, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.LogNormal(loc, scale).to_event(1))


def _sample_matrix_site(
    *,
    name: str,
    scale: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    loc = pyro.param(
        f"{name}_loc",
        torch.zeros_like(scale, device=device, dtype=dtype),
    )
    scale_param = pyro.param(
        f"{name}_scale",
        scale.to(device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.Normal(loc, scale_param).to_event(2))


def _sample_lognormal_site(
    *,
    name: str,
    anchor: torch.Tensor,
    log_scale: torch.Tensor,
) -> None:
    loc = pyro.param(
        f"{name}_loc",
        torch.log(anchor.clamp_min(1.0e-6)),
    )
    scale = pyro.param(
        f"{name}_scale",
        log_scale,
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.LogNormal(loc, scale).to_event(1))


def _sample_gamma_path_site(
    *,
    name: str,
    init: _GammaSiteInit,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    concentration = pyro.param(
        f"{name}_concentration",
        torch.full(init.shape, init.concentration, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    rate = pyro.param(
        f"{name}_rate",
        torch.full(init.shape, init.rate, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.Gamma(concentration, rate).to_event(1))


def _lognormal_mean(loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(loc + 0.5 * scale.square()).detach().clone()


__all__ = [
    "EquityFXMeasurementGuideV12L1OnlineFiltering",
    "V12L1GuideConfig",
    "build_equity_fx_measurement_guide_v12_l1_online_filtering",
]
