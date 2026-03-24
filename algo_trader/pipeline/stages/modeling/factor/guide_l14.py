from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_l12 import (
    _GuideContext,
    _GlobalGuideSites,
    _build_context,
    _INIT_LOGSCALE,
    _sample_B,
    _sample_alpha,
    _sample_beta,
    _sample_beta_hyperpriors,
    _sample_c,
    _sample_factor_loading_scale,
    _sample_feature_sites,
    _sample_lambda,
    _sample_regime_sites,
    _sample_s_u,
    _sample_scale_sites,
    _sample_tau0,
)
from .guide_l13 import (
    _GuideCallPlan,
    _run_guide_call,
    FactorGuideL13OnlineFiltering,
    Level13GuideConfig as Level14GuideConfig,
    build_level13_runtime_batch as build_level14_runtime_batch,
    _build_guide_config as _build_level14_guide_config,
)

_SIGMA_INIT_LOC_FX = math.log(0.03)
_SIGMA_INIT_LOC_INDEX = math.log(0.05)
_SIGMA_INIT_LOC_COMMODITY = math.log(0.06)


@dataclass
class FactorGuideL14OnlineFiltering(FactorGuideL13OnlineFiltering):
    config: Level14GuideConfig

    def _guide_call_plan(self) -> _GuideCallPlan:
        return _GuideCallPlan(
            config=self.config,
            require_encoder=self._require_encoder,
            module_name="factor_l14_online_filtering_encoder",
            build_runtime_batch=build_level14_runtime_batch,
            sample_global_sites=_sample_global_sites,
        )


@register_guide("factor_guide_l14_online_filtering")
def build_factor_guide_l14_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FactorGuideL14OnlineFiltering(config=_build_level14_guide_config(params))


def _sample_global_sites(
    *, context: _GuideContext, factor_count: int
) -> _GlobalGuideSites:
    _sample_tau0(context)
    _sample_lambda(context)
    _sample_c(context)
    _sample_beta_hyperpriors(context)
    _sample_factor_loading_scale(context, factor_count)
    _sample_asset_sites(context, factor_count)
    return _GlobalGuideSites(s_u_mean=_sample_s_u(context))


def _sample_asset_sites(context: _GuideContext, factor_count: int) -> None:
    with pyro.plate("asset", context.A, dim=-2):
        _sample_alpha(context)
        _sample_sigma(context)
        _sample_feature_sites(context)
        _sample_beta(context)
        _sample_B(context, factor_count)


def _sample_sigma(context: _GuideContext) -> None:
    shape = (context.A, 1)
    loc = pyro.param(
        "sigma_loc",
        _sigma_loc_init(context).reshape(shape),
    )
    scale = pyro.param(
        "sigma_scale",
        torch.full(shape, _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("sigma_idio", dist.LogNormal(loc, scale))


def _sigma_loc_init(context: _GuideContext) -> torch.Tensor:
    class_locs = torch.tensor(
        [
            _SIGMA_INIT_LOC_FX,
            _SIGMA_INIT_LOC_INDEX,
            _SIGMA_INIT_LOC_COMMODITY,
        ],
        device=context.device,
        dtype=context.dtype,
    )
    return class_locs.index_select(dim=0, index=context.batch.asset_class_ids)
