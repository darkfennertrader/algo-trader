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

from .defaults import merge_nested_params, model_default_params_v7_l1
from .guide import ObservableStateDependenceGuideV7L1OnlineFiltering
from .predict import predict_observable_state_dependence_v7_l1
from .shared import (
    ObservableStateCoefficients,
    ObservableStateDependenceConfig,
    ObservableStateGateConfig,
    ObservableStateOverlayInputs,
    ObservableStatePriorScaleConfig,
    apply_observable_state_dependence_overlay,
    build_observable_state_gate_series,
)


@dataclass(frozen=True)
class V7L1ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    observable_state_dependence: ObservableStateDependenceConfig = field(
        default_factory=ObservableStateDependenceConfig
    )


@dataclass
class ObservableStateDependenceModelV7L1OnlineFiltering(PyroModel):
    priors: V7L1ModelPriors = field(default_factory=V7L1ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        coefficients = _sample_observable_state_coefficients(
            context=context,
            overlay=self.priors.observable_state_dependence,
        )
        obs_dist = _build_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            coefficients=coefficients,
            overlay=self.priors.observable_state_dependence,
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
        return predict_observable_state_dependence_v7_l1(
            model=self,
            guide=cast(ObservableStateDependenceGuideV7L1OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("observable_state_dependence_model_v7_l1_online_filtering")
def build_observable_state_dependence_model_v7_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v7_l1(), params)
    return ObservableStateDependenceModelV7L1OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V7L1ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V7L1ModelPriors()
    extra = set(values) - {"mean", "factors", "regime", "observable_state_dependence"}
    if extra:
        raise ConfigError(
            "Unknown observable_state_dependence_model_v7_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V7L1ModelPriors(
        base=_build_base_model_priors(base_payload),
        observable_state_dependence=_build_overlay_config(
            values.get("observable_state_dependence")
        ),
    )


def _build_overlay_config(raw: object) -> ObservableStateDependenceConfig:
    values = _coerce_mapping(raw, label="model.params.observable_state_dependence")
    if not values:
        return ObservableStateDependenceConfig()
    base = ObservableStateDependenceConfig()
    gate_values = _coerce_mapping(
        values.get("gate"),
        label="model.params.observable_state_dependence.gate",
    )
    scale_values = _coerce_mapping(
        values.get("prior_scales"),
        label="model.params.observable_state_dependence.prior_scales",
    )
    return ObservableStateDependenceConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        gate=ObservableStateGateConfig(
            center=float(gate_values.get("center", base.gate.center)),
            scale=float(gate_values.get("scale", base.gate.scale)),
        ),
        prior_scales=ObservableStatePriorScaleConfig(
            bias=float(scale_values.get("bias", base.prior_scales.bias)),
            global_weight=float(
                scale_values.get(
                    "global_weight", base.prior_scales.global_weight
                )
            ),
            index_weight=float(
                scale_values.get("index_weight", base.prior_scales.index_weight)
            ),
            broad_strength=float(
                scale_values.get(
                    "broad_strength", base.prior_scales.broad_strength
                )
            ),
            regional_strength=float(
                scale_values.get(
                    "regional_strength", base.prior_scales.regional_strength
                )
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


def _sample_observable_state_coefficients(
    context: Any,
    overlay: ObservableStateDependenceConfig,
) -> ObservableStateCoefficients:
    if not overlay.enabled:
        zeros = torch.zeros((), device=context.device, dtype=context.dtype)
        return ObservableStateCoefficients(
            bias=zeros,
            global_weight=zeros,
            index_weight=zeros,
            broad_strength=zeros,
            us_strength=zeros,
            europe_strength=zeros,
        )
    return ObservableStateCoefficients(
        bias=pyro.sample(
            "obs_state_bias",
            dist.Normal(
                torch.zeros((), device=context.device, dtype=context.dtype),
                torch.tensor(
                    overlay.prior_scales.bias,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        ),
        global_weight=_sample_positive_site(
            name="obs_state_global_weight",
            scale=overlay.prior_scales.global_weight,
            device=context.device,
            dtype=context.dtype,
        ),
        index_weight=_sample_positive_site(
            name="obs_state_index_weight",
            scale=overlay.prior_scales.index_weight,
            device=context.device,
            dtype=context.dtype,
        ),
        broad_strength=_sample_positive_site(
            name="obs_state_broad_strength",
            scale=overlay.prior_scales.broad_strength,
            device=context.device,
            dtype=context.dtype,
        ),
        us_strength=_sample_positive_site(
            name="obs_state_us_strength",
            scale=overlay.prior_scales.regional_strength,
            device=context.device,
            dtype=context.dtype,
        ),
        europe_strength=_sample_positive_site(
            name="obs_state_europe_strength",
            scale=overlay.prior_scales.regional_strength,
            device=context.device,
            dtype=context.dtype,
        ),
    )


def _sample_positive_site(
    *,
    name: str,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return pyro.sample(
        name,
        dist.HalfNormal(torch.tensor(scale, device=device, dtype=dtype)),
    )


def _build_observation_distribution(
    *,
    context: Any,
    structural: Any,
    regime_path: torch.Tensor,
    coefficients: ObservableStateCoefficients,
    overlay: ObservableStateDependenceConfig,
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
    gate = build_observable_state_gate_series(
        X_asset=context.batch.X_asset,
        X_global=context.batch.X_global,
        assets=context.batch.assets,
        coefficients=coefficients,
        overlay=overlay,
    )
    scaled_factor, scaled_diag = apply_observable_state_dependence_overlay(
        inputs=ObservableStateOverlayInputs(
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
            gate=gate,
        ),
        assets=context.batch.assets,
        coefficients=coefficients,
        eps=overlay.eps,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    )


__all__ = [
    "ObservableStateDependenceModelV7L1OnlineFiltering",
    "V7L1ModelPriors",
    "build_observable_state_dependence_model_v7_l1_online_filtering",
]
