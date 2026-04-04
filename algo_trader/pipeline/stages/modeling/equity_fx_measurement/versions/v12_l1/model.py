from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model import (
    _sample_index_t_copula_mix,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.shared import (
    IndexTCopulaOverlayConfig,
    apply_index_t_copula_overlay,
)
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

from .defaults import merge_nested_params, model_default_params_v12_l1
from .guide import EquityFXMeasurementGuideV12L1OnlineFiltering
from .predict import predict_equity_fx_measurement_v12_l1
from .shared import (
    EquityFXMeasurementConfig,
    EquityFXMeasurementFactorState,
    EquityFXMeasurementPosteriorMeans,
    apply_equity_fx_measurement_residual_scale,
    build_equity_fx_measurement_config,
    build_equity_fx_measurement_factor_block,
    build_equity_fx_measurement_structure,
)

_STATE_COUNT = 6


@dataclass(frozen=True)
class V12L1ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_t_copula: IndexTCopulaOverlayConfig = field(
        default_factory=IndexTCopulaOverlayConfig
    )
    equity_fx_measurement: EquityFXMeasurementConfig = field(
        default_factory=EquityFXMeasurementConfig
    )


@dataclass(frozen=True)
class _ObservationDistributionInputs:
    context: Any
    structural: Any
    regime_path: torch.Tensor
    index_t_mix: torch.Tensor
    measurement_params: EquityFXMeasurementPosteriorMeans
    index_t_overlay: IndexTCopulaOverlayConfig
    measurement_overlay: EquityFXMeasurementConfig


@dataclass
class EquityFXMeasurementModelV12L1OnlineFiltering(PyroModel):
    priors: V12L1ModelPriors = field(default_factory=V12L1ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        mix = _sample_index_t_copula_mix(context, self.priors.index_t_copula)
        measurement_params = _sample_equity_fx_measurement_sites(
            assets=context.batch.assets,
            overlay=self.priors.equity_fx_measurement,
            device=context.device,
            dtype=context.dtype,
        )
        obs_dist = _build_observation_distribution(
            _ObservationDistributionInputs(
                context=context,
                structural=structural,
                regime_path=regime_path,
                index_t_mix=mix,
                measurement_params=measurement_params,
                index_t_overlay=self.priors.index_t_copula,
                measurement_overlay=self.priors.equity_fx_measurement,
            )
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
        return predict_equity_fx_measurement_v12_l1(
            model=self,
            guide=cast(EquityFXMeasurementGuideV12L1OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("equity_fx_measurement_model_v12_l1_online_filtering")
def build_equity_fx_measurement_model_v12_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v12_l1(), params)
    return EquityFXMeasurementModelV12L1OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V12L1ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V12L1ModelPriors()
    extra = set(values) - {
        "mean",
        "factors",
        "regime",
        "index_t_copula",
        "equity_fx_measurement",
    }
    if extra:
        raise ConfigError(
            "Unknown equity_fx_measurement_model_v12_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V12L1ModelPriors(
        base=_build_base_model_priors(base_payload),
        index_t_copula=_build_index_t_copula_config(values.get("index_t_copula")),
        equity_fx_measurement=build_equity_fx_measurement_config(
            values.get("equity_fx_measurement")
        ),
    )


def _build_index_t_copula_config(raw: object) -> IndexTCopulaOverlayConfig:
    values = _coerce_mapping(raw, label="model.params.index_t_copula")
    if not values:
        return IndexTCopulaOverlayConfig()
    base = IndexTCopulaOverlayConfig()
    return IndexTCopulaOverlayConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        df=float(values.get("df", base.df)),
        eps=float(values.get("eps", base.eps)),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_equity_fx_measurement_sites(
    *,
    assets: Any,
    overlay: EquityFXMeasurementConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> EquityFXMeasurementPosteriorMeans:
    structure = build_equity_fx_measurement_structure(
        assets=assets,
        config=overlay,
        device=device,
        dtype=dtype,
    )
    if not overlay.enabled:
        return EquityFXMeasurementPosteriorMeans(
            state_scale=torch.zeros((_STATE_COUNT,), device=device, dtype=dtype),
            state_corr_cholesky=torch.eye(_STATE_COUNT, device=device, dtype=dtype),
            loading_delta=torch.zeros_like(structure.anchor_loadings),
            residual_scale=torch.ones_like(structure.residual_anchor),
        )
    state_scale = pyro.sample(
        "equity_fx_measurement_state_scale",
        dist.HalfNormal(
            torch.full(
                (_STATE_COUNT,),
                overlay.prior_scales.state_scale,
                device=device,
                dtype=dtype,
            )
        ).to_event(1),
    )
    state_corr_cholesky = pyro.sample(
        "equity_fx_measurement_state_corr_cholesky",
        dist.LKJCholesky(
            dim=_STATE_COUNT,
            concentration=torch.tensor(
                overlay.prior_scales.correlation_concentration,
                device=device,
                dtype=dtype,
            ),
        ),
    )
    loading_delta = pyro.sample(
        "equity_fx_measurement_loading_delta",
        dist.Normal(
            torch.zeros_like(structure.anchor_loadings),
            structure.loading_deviation_scale,
        ).to_event(2),
    )
    residual_scale = pyro.sample(
        "equity_fx_measurement_residual_scale",
        dist.LogNormal(
            torch.log(structure.residual_anchor.clamp_min(float(overlay.eps))),
            structure.residual_prior_scale,
        ).to_event(1),
    )
    return EquityFXMeasurementPosteriorMeans(
        state_scale=state_scale,
        state_corr_cholesky=state_corr_cholesky,
        loading_delta=loading_delta,
        residual_scale=residual_scale,
    )


def _sample_equity_fx_measurement_mix(
    *,
    df_value: float,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    concentration = torch.full(
        (time_count,),
        df_value / 2.0,
        device=device,
        dtype=dtype,
    )
    return pyro.sample(
        "equity_fx_measurement_mix",
        dist.Gamma(concentration, concentration).to_event(1),
    )


def _build_observation_distribution(
    inputs: _ObservationDistributionInputs,
) -> dist.LowRankMultivariateNormal:
    loc = _base_mean_path(inputs.context, inputs.structural)
    base_cov_diag = (
        inputs.structural.mean.sigma_idio.pow(2) + inputs.context.priors.regime.eps
    )
    base_cov_factor = _base_cov_factor_path(
        inputs.context,
        inputs.structural,
        inputs.regime_path,
    )
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        cov_factor=base_cov_factor,
        cov_diag=base_cov_diag,
        assets=inputs.context.batch.assets,
        mix=inputs.index_t_mix,
        eps=inputs.index_t_overlay.eps,
    )
    if not inputs.measurement_overlay.enabled:
        return dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=scaled_factor,
            cov_diag=scaled_diag,
        )
    measurement_block = _build_measurement_block(inputs)
    cov_diag = apply_equity_fx_measurement_residual_scale(
        cov_diag=scaled_diag,
        residual_scale=inputs.measurement_params.residual_scale,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([scaled_factor, measurement_block], dim=-1),
        cov_diag=cov_diag,
    )


def _build_measurement_block(inputs: _ObservationDistributionInputs) -> torch.Tensor:
    structure = build_equity_fx_measurement_structure(
        assets=inputs.context.batch.assets,
        config=inputs.measurement_overlay,
        device=inputs.context.device,
        dtype=inputs.context.dtype,
    )
    measurement_mix = _sample_equity_fx_measurement_mix(
        df_value=inputs.measurement_overlay.state_df,
        time_count=inputs.context.shape.T,
        device=inputs.context.device,
        dtype=inputs.context.dtype,
    )
    state = EquityFXMeasurementFactorState(
        state_scale=inputs.measurement_params.state_scale,
        state_corr_cholesky=inputs.measurement_params.state_corr_cholesky,
        loading_delta=inputs.measurement_params.loading_delta,
        local_regime_scale=torch.exp(0.5 * inputs.regime_path[:, 2]),
        translation_regime_scale=torch.exp(0.5 * inputs.regime_path[:, 1]),
        mix=measurement_mix,
        eps=inputs.measurement_overlay.eps,
    )
    return build_equity_fx_measurement_factor_block(structure=structure, state=state)


__all__ = [
    "EquityFXMeasurementModelV12L1OnlineFiltering",
    "V12L1ModelPriors",
    "build_equity_fx_measurement_model_v12_l1_online_filtering",
]
