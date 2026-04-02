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

from .defaults import guide_default_params_v6_l1, merge_nested_params

_CALM_DF_INIT = 6.0
_STRESS_DF_INIT = 4.0
_US_STRESS_DF_INIT = 10.0
_EUROPE_STRESS_DF_INIT = 6.0
_MIX_ALPHA_INIT = 2.0
_MIX_BETA_INIT = 8.0


@dataclass(frozen=True)
class V6L1GuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    index_t_copula_enabled: bool = True


@dataclass
class MixtureCopulaGuideV6L1OnlineFiltering(PyroGuide):
    config: V6L1GuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_base_context(build_v3_l1_unified_runtime_batch(batch))
        _sample_base_structural_sites(context=context, config=self.config.base)
        _sample_base_regime_scale_sites(context)
        _sample_base_local_state_sites(context)
        _sample_index_t_copula_sites(
            time_count=int(context.shape.T),
            device=context.device,
            dtype=context.dtype,
            enabled=self.config.index_t_copula_enabled,
        )

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(config=self.config.base)


@register_guide("mixture_copula_guide_v6_l1_online_filtering")
def build_mixture_copula_guide_v6_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v6_l1(), params)
    return MixtureCopulaGuideV6L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V6L1GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V6L1GuideConfig()
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
    }
    if extra:
        raise ConfigError(
            "Unknown mixture_copula_guide_v6_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value for key, value in values.items() if key != "index_t_copula_enabled"
    }
    return V6L1GuideConfig(
        base=_build_base_guide_config(base_payload),
        index_t_copula_enabled=bool(values.get("index_t_copula_enabled", True)),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_index_t_copula_sites(
    *,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    if not enabled:
        return
    _sample_gamma_mix_site(
        name="index_t_copula_calm_mix",
        time_count=time_count,
        init_value=_CALM_DF_INIT / 2.0,
        device=device,
        dtype=dtype,
    )
    _sample_beta_site(
        name="index_t_copula_mixture_weight",
        time_count=time_count,
        init_prior=(_MIX_ALPHA_INIT, _MIX_BETA_INIT),
        device=device,
        dtype=dtype,
    )
    for name, init_value in (
        ("index_t_copula_stress_mix", _STRESS_DF_INIT / 2.0),
        ("index_t_copula_us_stress_mix", _US_STRESS_DF_INIT / 2.0),
        ("index_t_copula_europe_stress_mix", _EUROPE_STRESS_DF_INIT / 2.0),
    ):
        _sample_gamma_mix_site(
            name=name,
            time_count=time_count,
            init_value=init_value,
            device=device,
            dtype=dtype,
        )


def _sample_beta_site(
    *,
    name: str,
    time_count: int,
    init_prior: tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    alpha_init, beta_init = init_prior
    alpha = pyro.param(
        f"{name}_alpha",
        torch.full((time_count,), alpha_init, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    beta = pyro.param(
        f"{name}_beta",
        torch.full((time_count,), beta_init, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.Beta(alpha, beta).to_event(1))


def _sample_gamma_mix_site(
    *,
    name: str,
    time_count: int,
    init_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    concentration = pyro.param(
        f"{name}_concentration",
        torch.full((time_count,), init_value, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    rate = pyro.param(
        f"{name}_rate",
        torch.full((time_count,), init_value, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.Gamma(concentration, rate).to_event(1))


__all__ = [
    "MixtureCopulaGuideV6L1OnlineFiltering",
    "V6L1GuideConfig",
    "build_mixture_copula_guide_v6_l1_online_filtering",
]
