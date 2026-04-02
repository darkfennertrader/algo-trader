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

from .defaults import guide_default_params_v7_l1, merge_nested_params
from .shared import ObservableStateCoefficients


@dataclass(frozen=True)
class V7L1GuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    observable_state_dependence_enabled: bool = True


@dataclass
class ObservableStateDependenceGuideV7L1OnlineFiltering(PyroGuide):
    config: V7L1GuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_base_context(build_v3_l1_unified_runtime_batch(batch))
        _sample_base_structural_sites(context=context, config=self.config.base)
        _sample_base_regime_scale_sites(context)
        _sample_base_local_state_sites(context)
        _sample_observable_state_sites(
            enabled=self.config.observable_state_dependence_enabled,
            device=context.device,
            dtype=context.dtype,
        )

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def observable_state_posterior_means(self) -> ObservableStateCoefficients:
        if not self.config.observable_state_dependence_enabled:
            zeros = torch.zeros(())
            return ObservableStateCoefficients(
                bias=zeros,
                global_weight=zeros,
                index_weight=zeros,
                broad_strength=zeros,
                us_strength=zeros,
                europe_strength=zeros,
            )
        return ObservableStateCoefficients(
            bias=pyro.param("obs_state_bias_loc").detach().clone(),
            global_weight=_lognormal_mean(
                pyro.param("obs_state_global_weight_loc"),
                pyro.param("obs_state_global_weight_scale"),
            ),
            index_weight=_lognormal_mean(
                pyro.param("obs_state_index_weight_loc"),
                pyro.param("obs_state_index_weight_scale"),
            ),
            broad_strength=_lognormal_mean(
                pyro.param("obs_state_broad_strength_loc"),
                pyro.param("obs_state_broad_strength_scale"),
            ),
            us_strength=_lognormal_mean(
                pyro.param("obs_state_us_strength_loc"),
                pyro.param("obs_state_us_strength_scale"),
            ),
            europe_strength=_lognormal_mean(
                pyro.param("obs_state_europe_strength_loc"),
                pyro.param("obs_state_europe_strength_scale"),
            ),
        )

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(config=self.config.base)


@register_guide("observable_state_dependence_guide_v7_l1_online_filtering")
def build_observable_state_dependence_guide_v7_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v7_l1(), params)
    return ObservableStateDependenceGuideV7L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V7L1GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V7L1GuideConfig()
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
        "observable_state_dependence_enabled",
    }
    if extra:
        raise ConfigError(
            "Unknown observable_state_dependence_guide_v7_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value
        for key, value in values.items()
        if key != "observable_state_dependence_enabled"
    }
    return V7L1GuideConfig(
        base=_build_base_guide_config(base_payload),
        observable_state_dependence_enabled=bool(
            values.get("observable_state_dependence_enabled", True)
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_observable_state_sites(
    *,
    enabled: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not enabled:
        return
    _sample_normal_site(
        name="obs_state_bias",
        init_loc=0.0,
        init_scale=0.20,
        device=device,
        dtype=dtype,
    )
    for name, init_loc in (
        ("obs_state_global_weight", -1.2),
        ("obs_state_index_weight", -1.2),
        ("obs_state_broad_strength", -1.6),
        ("obs_state_us_strength", -2.1),
        ("obs_state_europe_strength", -2.1),
    ):
        _sample_positive_site(
            name=name,
            init_loc=init_loc,
            init_scale=0.20,
            device=device,
            dtype=dtype,
        )


def _sample_normal_site(
    *,
    name: str,
    init_loc: float,
    init_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    loc = pyro.param(
        f"{name}_loc",
        torch.tensor(init_loc, device=device, dtype=dtype),
    )
    scale = pyro.param(
        f"{name}_scale",
        torch.tensor(init_scale, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.Normal(loc, scale))


def _sample_positive_site(
    *,
    name: str,
    init_loc: float,
    init_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    loc = pyro.param(
        f"{name}_loc",
        torch.tensor(init_loc, device=device, dtype=dtype),
    )
    scale = pyro.param(
        f"{name}_scale",
        torch.tensor(init_scale, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample(name, dist.LogNormal(loc, scale))


def _lognormal_mean(loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(loc + 0.5 * scale.square()).detach().clone()


__all__ = [
    "ObservableStateDependenceGuideV7L1OnlineFiltering",
    "V7L1GuideConfig",
    "build_observable_state_dependence_guide_v7_l1_online_filtering",
]
