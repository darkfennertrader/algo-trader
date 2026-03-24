from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro

from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .guide_l15 import build_level15_runtime_batch
from .model_l14 import (
    _StageHooks,
    _build_structural_site_plan,
    _sample_shared_structural_sites,
)
from .model_l11 import (
    Level11ModelPriors as Level15ModelPriors,
    _build_context,
    _build_model_priors as _build_level15_model_priors,
    _build_observation_distribution,
    _log_asset_sites as _log_asset_sites_fx,
    _log_global_loadings as _log_global_loadings_fx,
    _log_inputs as _log_inputs_fx,
    _log_observation_distribution,
    _log_regime_and_scale,
    _log_shrinkage as _log_shrinkage_fx,
    _sample_asset_sites as _sample_asset_sites_fx,
    _sample_global_loadings as _sample_global_loadings_fx,
    _sample_observations,
    _sample_regime_path,
    _sample_shrinkage as _sample_shrinkage_fx,
    _sample_total_scale,
    half_student_t,
)
from .model_l13 import (
    FactorModelL13OnlineFiltering,
    _build_registered_model,
    _predict_from_request,
)
from .predict_l15 import predict_factor_l15


@dataclass(frozen=True)
class FactorModelL15OnlineFiltering(FactorModelL13OnlineFiltering):
    priors: Any = field(default_factory=Level15ModelPriors)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_level15_runtime_batch(batch)
        context = _build_context(runtime_batch, self.priors)
        asset_sites = cast(
            Any,
            _sample_shared_structural_sites(
                batch,
                cast(Any, context),
                plan=_build_structural_site_plan(
                    log_inputs=_log_inputs_fx,
                    shrinkage=_StageHooks(
                        sample=_sample_shrinkage_fx,
                        log=_log_shrinkage_fx,
                    ),
                    loadings=_StageHooks(
                        sample=_sample_global_loadings_fx,
                        log=_log_global_loadings_fx,
                    ),
                    assets=_StageHooks(
                        sample=_sample_asset_sites_fx,
                        log=_log_asset_sites_fx,
                    ),
                ),
            ),
        )
        s_u = pyro.sample(
            "s_u",
            half_student_t(
                df=context.priors.regime.s_u_df,
                scale=context.priors.regime.s_u_scale,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        h = _sample_regime_path(context, s_u)
        lambda_h = getattr(asset_sites, "lambda_h")
        u = _sample_total_scale(context, h, s_u, lambda_h)
        _log_regime_and_scale(batch, s_u, h, u)
        obs_dist = _build_observation_distribution(
            context,
            cast(Any, asset_sites),
            u,
        )
        _log_observation_distribution(batch, obs_dist)
        _sample_observations(context, obs_dist)

    def _posterior_predictor(
        self, request: Any
    ) -> Mapping[str, Any] | None:
        return _predict_from_request(
            request=request,
            model=self,
            predictor=predict_factor_l15,
        )


@register_model("factor_model_l15_online_filtering")
def build_factor_model_l15_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return _build_registered_model(
        params=params,
        model_type=FactorModelL15OnlineFiltering,
        prior_builder=_build_level15_model_priors,
    )
