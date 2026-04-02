from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified import (
    V3L1UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _cov_factor_path as _base_cov_factor_path,
    _mean_path as _base_mean_path,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide, PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.runtime_support import sample_time_observations

from .defaults import merge_nested_params, model_default_params_v5_l2
from .guide import ResidualCopulaGuideV5L2OnlineFiltering
from .predict import predict_residual_copula_v5_l2
from .shared import (
    IndexConditionalAsymmetricTCopulaMixSamples,
    IndexConditionalAsymmetricTCopulaOverlayConfig,
    RegionalTailConfig,
    StressAsymmetricRegionalOverlayConfig,
    StressWeightPriorConfig,
    apply_index_t_copula_overlay,
    _OverlayInputs,
)


@dataclass(frozen=True)
class V5L2ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_t_copula: IndexConditionalAsymmetricTCopulaOverlayConfig = field(
        default_factory=IndexConditionalAsymmetricTCopulaOverlayConfig
    )


@dataclass
class ResidualCopulaModelV5L2OnlineFiltering(PyroModel):
    priors: V5L2ModelPriors = field(default_factory=V5L2ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        mixes = _sample_index_t_copula_sites(context, self.priors.index_t_copula)
        obs_dist = _build_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            mixes=mixes,
            overlay=self.priors.index_t_copula,
        )
        sample_time_observations(
            time_count=context.shape.T,
            obs_dist=obs_dist,
            y_obs=runtime_batch.observations.y_obs,
            time_mask=runtime_batch.observations.time_mask,
            obs_scale=runtime_batch.observations.obs_scale,
        )

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        summaries = getattr(guide, "structural_predictive_summaries", None)
        if not callable(summaries):
            summaries = getattr(guide, "structural_posterior_means", None)
        if not callable(summaries):
            return None
        return predict_residual_copula_v5_l2(
            model=self,
            guide=cast(ResidualCopulaGuideV5L2OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("residual_copula_model_v5_l2_online_filtering")
def build_residual_copula_model_v5_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v5_l2(), params)
    return ResidualCopulaModelV5L2OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V5L2ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V5L2ModelPriors()
    extra = set(values) - {"mean", "factors", "regime", "index_t_copula"}
    if extra:
        raise ConfigError(
            "Unknown residual_copula_model_v5_l2_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V5L2ModelPriors(
        base=_build_base_model_priors(base_payload),
        index_t_copula=_build_index_t_copula_config(values.get("index_t_copula")),
    )


def _build_index_t_copula_config(
    raw: object,
) -> IndexConditionalAsymmetricTCopulaOverlayConfig:
    values = _coerce_mapping(raw, label="model.params.index_t_copula")
    if not values:
        return IndexConditionalAsymmetricTCopulaOverlayConfig()
    base = IndexConditionalAsymmetricTCopulaOverlayConfig()
    prior_values = _coerce_mapping(
        values.get("stress_prior"),
        label="model.params.index_t_copula.stress_prior",
    )
    stress_values = _coerce_mapping(
        values.get("stress"),
        label="model.params.index_t_copula.stress",
    )
    return IndexConditionalAsymmetricTCopulaOverlayConfig(
        calm_df=float(values.get("calm_df", base.calm_df)),
        enabled=bool(values.get("enabled", base.enabled)),
        stress_prior=StressWeightPriorConfig(
            alpha=float(prior_values.get("alpha", base.stress_prior.alpha)),
            beta=float(prior_values.get("beta", base.stress_prior.beta)),
        ),
        stress=StressAsymmetricRegionalOverlayConfig(
            broad_df=float(stress_values.get("broad_df", base.stress.broad_df)),
            broad_strength=float(
                stress_values.get("broad_strength", base.stress.broad_strength)
            ),
            gate_scale=float(
                stress_values.get("gate_scale", base.stress.gate_scale)
            ),
            us=_build_region_tail_config(
                values=stress_values,
                prefix="us",
                base=base.stress.us,
            ),
            europe=_build_region_tail_config(
                values=stress_values,
                prefix="europe",
                base=base.stress.europe,
            ),
        ),
        eps=float(values.get("eps", base.eps)),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _build_region_tail_config(
    *,
    values: Mapping[str, Any],
    prefix: str,
    base: RegionalTailConfig,
) -> RegionalTailConfig:
    return RegionalTailConfig(
        upper_df=float(values.get(f"{prefix}_upper_df", base.upper_df)),
        lower_df=float(values.get(f"{prefix}_lower_df", base.lower_df)),
        upper_strength=float(
            values.get(f"{prefix}_upper_strength", base.upper_strength)
        ),
        lower_strength=float(
            values.get(f"{prefix}_lower_strength", base.lower_strength)
        ),
    )


def _sample_index_t_copula_sites(
    context: Any,
    overlay: IndexConditionalAsymmetricTCopulaOverlayConfig,
) -> IndexConditionalAsymmetricTCopulaMixSamples:
    if not overlay.enabled:
        ones = torch.ones((context.shape.T,), device=context.device, dtype=context.dtype)
        zeros = torch.zeros(
            (context.shape.T,), device=context.device, dtype=context.dtype
        )
        return IndexConditionalAsymmetricTCopulaMixSamples(
            calm=ones,
            stress_weight=zeros,
            stress=ones,
            us_upper=ones,
            us_lower=ones,
            europe_upper=ones,
            europe_lower=ones,
        )
    return IndexConditionalAsymmetricTCopulaMixSamples(
        calm=_sample_gamma_mix(
            name="index_t_copula_calm_mix",
            time_count=int(context.shape.T),
            df=overlay.calm_df,
            device=context.device,
            dtype=context.dtype,
        ),
        stress_weight=_sample_stress_weight(
            time_count=int(context.shape.T),
            overlay=overlay,
            device=context.device,
            dtype=context.dtype,
        ),
        stress=_sample_gamma_mix(
            name="index_t_copula_stress_mix",
            time_count=int(context.shape.T),
            df=overlay.stress.broad_df,
            device=context.device,
            dtype=context.dtype,
        ),
        us_upper=_sample_gamma_mix(
            name="index_t_copula_us_upper_mix",
            time_count=int(context.shape.T),
            df=overlay.stress.us.upper_df,
            device=context.device,
            dtype=context.dtype,
        ),
        us_lower=_sample_gamma_mix(
            name="index_t_copula_us_lower_mix",
            time_count=int(context.shape.T),
            df=overlay.stress.us.lower_df,
            device=context.device,
            dtype=context.dtype,
        ),
        europe_upper=_sample_gamma_mix(
            name="index_t_copula_europe_upper_mix",
            time_count=int(context.shape.T),
            df=overlay.stress.europe.upper_df,
            device=context.device,
            dtype=context.dtype,
        ),
        europe_lower=_sample_gamma_mix(
            name="index_t_copula_europe_lower_mix",
            time_count=int(context.shape.T),
            df=overlay.stress.europe.lower_df,
            device=context.device,
            dtype=context.dtype,
        ),
    )


def _sample_gamma_mix(
    *,
    name: str,
    time_count: int,
    df: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    concentration = torch.full((time_count,), df / 2.0, device=device, dtype=dtype)
    return pyro.sample(name, dist.Gamma(concentration, concentration).to_event(1))


def _sample_stress_weight(
    *,
    time_count: int,
    overlay: IndexConditionalAsymmetricTCopulaOverlayConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    alpha = torch.full(
        (time_count,),
        overlay.stress_prior.alpha,
        device=device,
        dtype=dtype,
    )
    beta = torch.full(
        (time_count,),
        overlay.stress_prior.beta,
        device=device,
        dtype=dtype,
    )
    return pyro.sample(
        "index_t_copula_stress_weight", dist.Beta(alpha, beta).to_event(1)
    )


def _build_observation_distribution(
    *,
    context: Any,
    structural: Any,
    regime_path: torch.Tensor,
    mixes: IndexConditionalAsymmetricTCopulaMixSamples,
    overlay: IndexConditionalAsymmetricTCopulaOverlayConfig,
) -> dist.LowRankMultivariateNormal:
    loc = _base_mean_path(context, structural)
    cov_factor = _base_cov_factor_path(context, structural, regime_path)
    base_cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
    if not overlay.enabled:
        return dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
        )
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        inputs=_OverlayInputs(
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
            index_signal=regime_path[:, 2],
        ),
        assets=context.batch.assets,
        mixes=mixes,
        overlay=overlay,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    )


__all__ = [
    "ResidualCopulaModelV5L2OnlineFiltering",
    "V5L2ModelPriors",
    "build_residual_copula_model_v5_l2_online_filtering",
]
