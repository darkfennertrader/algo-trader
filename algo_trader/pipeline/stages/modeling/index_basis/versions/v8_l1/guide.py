from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.domain import ConfigError
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

from .defaults import guide_default_params_v8_l1, merge_nested_params
from .shared import IndexBasisPosteriorMeans

_SPREAD_DIM = 4


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
class V8L1GuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    index_basis_enabled: bool = True


@dataclass
class IndexBasisGuideV8L1OnlineFiltering(PyroGuide):
    config: V8L1GuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_base_context(build_v3_l1_unified_runtime_batch(batch))
        _sample_base_structural_sites(context=context, config=self.config.base)
        _sample_base_regime_scale_sites(context)
        _sample_base_local_state_sites(context)
        _sample_index_basis_sites(
            enabled=self.config.index_basis_enabled,
            time_count=context.shape.T,
            device=context.device,
            dtype=context.dtype,
        )

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def index_basis_posterior_means(self) -> IndexBasisPosteriorMeans:
        if not self.config.index_basis_enabled:
            zeros = torch.zeros(())
            return IndexBasisPosteriorMeans(
                global_scale=zeros,
                spread_scale=torch.zeros((_SPREAD_DIM,)),
                spread_corr_cholesky=torch.eye(_SPREAD_DIM),
            )
        return IndexBasisPosteriorMeans(
            global_scale=_lognormal_mean(
                pyro.param("index_basis_global_scale_loc"),
                pyro.param("index_basis_global_scale_scale"),
            ),
            spread_scale=_lognormal_mean(
                pyro.param("index_basis_spread_scale_loc"),
                pyro.param("index_basis_spread_scale_scale"),
            ),
            spread_corr_cholesky=pyro.param(
                "index_basis_spread_corr_cholesky_loc"
            ).detach().clone(),
        )

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(config=self.config.base)


@register_guide("index_basis_guide_v8_l1_online_filtering")
def build_index_basis_guide_v8_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v8_l1(), params)
    return IndexBasisGuideV8L1OnlineFiltering(config=_build_guide_config(merged_params))


def _build_guide_config(params: Mapping[str, Any]) -> V8L1GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V8L1GuideConfig()
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
        "index_basis_enabled",
    }
    if extra:
        raise ConfigError(
            "Unknown index_basis_guide_v8_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value for key, value in values.items() if key != "index_basis_enabled"
    }
    return V8L1GuideConfig(
        base=_build_base_guide_config(base_payload),
        index_basis_enabled=bool(values.get("index_basis_enabled", True)),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_index_basis_sites(
    *,
    enabled: bool,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not enabled:
        return
    _sample_positive_site(
        name="index_basis_global_scale",
        init=_PositiveSiteInit(loc=-2.3, scale=0.20, shape=()),
        device=device,
        dtype=dtype,
    )
    _sample_positive_site(
        name="index_basis_spread_scale",
        init=_PositiveSiteInit(loc=-2.8, scale=0.20, shape=(_SPREAD_DIM,)),
        device=device,
        dtype=dtype,
    )
    cholesky = pyro.param(
        "index_basis_spread_corr_cholesky_loc",
        torch.eye(_SPREAD_DIM, device=device, dtype=dtype),
        constraint=constraints.corr_cholesky,
    )
    pyro.sample(
        "index_basis_spread_corr_cholesky",
        dist.Delta(cholesky, event_dim=2),
    )
    gamma_init = _GammaSiteInit(
        concentration=2.0,
        rate=2.0,
        shape=(time_count,),
    )
    _sample_gamma_path_site(
        name="index_basis_global_mix",
        init=gamma_init,
        device=device,
        dtype=dtype,
    )
    _sample_gamma_path_site(
        name="index_basis_spread_mix",
        init=gamma_init,
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
    event_dim = 0 if not init.shape else 1
    pyro.sample(name, dist.LogNormal(loc, scale).to_event(event_dim))


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
    "IndexBasisGuideV8L1OnlineFiltering",
    "V8L1GuideConfig",
    "build_index_basis_guide_v8_l1_online_filtering",
]
