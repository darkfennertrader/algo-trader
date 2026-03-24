from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, cast

import pyro
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_l12 import (
    FilteringState,
    Level12RuntimeBatch as Level13RuntimeBatch,
    RegimeEncoder,
    StructuralPosteriorMeans,
    _GuideContext,
    _LocalGuideSites,
    _build_context,
    _coerce_mapping,
    _encoder_features,
    _initial_prior_message,
    _lognormal_mean,
    _lognormal_median,
    _next_steps_seen,
    _propagated_message,
    _sample_global_sites,
    _sample_regime_sites,
    _sample_scale_sites,
    build_level12_runtime_batch,
)

_MIN_SCALE = 1e-4
_MAX_H_GAIN = 0.35
_MAX_H_LOG_SCALE_STEP = 0.40
_MAX_H_SCALE = 0.25


build_level13_runtime_batch = build_level12_runtime_batch


@dataclass(frozen=True)
class Level13GuideConfig:
    factor_count: int = 3
    phi: float = 0.97
    eps: float = 1e-12
    hidden_dim: int = 64


@dataclass(frozen=True)
class _GuideCallPlan:
    config: Level13GuideConfig
    require_encoder: Callable[[_GuideContext], RegimeEncoder]
    module_name: str
    build_runtime_batch: Callable[[ModelBatch], Level13RuntimeBatch]
    sample_global_sites: Callable[..., _LocalGlobalSitesLike]


@dataclass(frozen=True)
class _GainLocalEncodingInputs:
    context: _GuideContext
    encoder: RegimeEncoder
    s_u_mean: torch.Tensor
    phi: float
    eps: float


@dataclass(frozen=True)
class _GainLocalEncodingPlan:
    decode_step: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    initial_prior_message: Callable[
        [_GuideContext, torch.Tensor, float, float],
        tuple[torch.Tensor, torch.Tensor],
    ]
    encoder_features: Callable[..., torch.Tensor]
    propagated_message: Callable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class _GainStepSlices:
    gain_raw: torch.Tensor
    innovation_raw: torch.Tensor
    h_scale_raw: torch.Tensor
    v_loc_raw: torch.Tensor
    v_scale_raw: torch.Tensor


@dataclass
class FactorGuideL13OnlineFiltering(PyroGuide):
    config: Level13GuideConfig
    _encoder: RegimeEncoder | None = None
    _encoder_input_dim: int | None = None

    def __call__(self, batch: ModelBatch) -> None:
        _run_guide_call(batch=batch, plan=self._guide_call_plan())

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch = build_level13_runtime_batch(batch)
        context = _build_context(runtime_batch)
        encoder = self._require_encoder(context)
        structural = self.structural_posterior_means()
        local = _encode_local_sites(
            context=context,
            encoder=encoder,
            s_u_mean=structural.s_u_mean,
            phi=self.config.phi,
            eps=self.config.eps,
        )
        return FilteringState(
            h_loc=local.h_loc[-1].detach(),
            h_scale=local.h_scale[-1].detach(),
            steps_seen=_next_steps_seen(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=_lognormal_mean(
                store.get_param("sigma_loc"),
                store.get_param("sigma_scale"),
            )
            .squeeze(-1)
            .detach(),
            w=store.get_param("w_loc").detach(),
            beta=store.get_param("beta_loc").detach(),
            B=store.get_param("B_loc").detach(),
            s_u_mean=_lognormal_mean(
                store.get_param("s_u_loc"),
                store.get_param("s_u_scale"),
            ).detach(),
        )

    def structural_predictive_summaries(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=_lognormal_median(store.get_param("sigma_loc"))
            .squeeze(-1)
            .detach(),
            w=store.get_param("w_loc").detach(),
            beta=store.get_param("beta_loc").detach(),
            B=store.get_param("B_loc").detach(),
            s_u_mean=_lognormal_median(store.get_param("s_u_loc")).detach(),
        )

    def _require_encoder(self, context: _GuideContext) -> RegimeEncoder:
        if self._encoder is None:
            self._encoder = RegimeEncoder(
                input_dim=context.encoder_input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=5 * context.C,
            ).to(device=context.device, dtype=context.dtype)
            self._encoder_input_dim = context.encoder_input_dim
        if self._encoder_input_dim != context.encoder_input_dim:
            raise ConfigError(
                "Level 13 guide encoder input dimension changed across calls"
            )
        return cast(RegimeEncoder, self._encoder)

    def _guide_call_plan(self) -> _GuideCallPlan:
        return _GuideCallPlan(
            config=self.config,
            require_encoder=self._require_encoder,
            module_name="factor_l13_online_filtering_encoder",
            build_runtime_batch=build_level13_runtime_batch,
            sample_global_sites=_sample_global_sites,
        )


@register_guide("factor_guide_l13_online_filtering")
def build_factor_guide_l13_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FactorGuideL13OnlineFiltering(config=_build_guide_config(params))


def _build_guide_config(params: Mapping[str, Any]) -> Level13GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return Level13GuideConfig()
    extra = set(values) - {"factor_count", "phi", "eps", "hidden_dim"}
    if extra:
        raise ConfigError(
            "Unknown factor_guide_l13_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = Level13GuideConfig()
    try:
        updated = Level13GuideConfig(
            factor_count=int(values.get("factor_count", base.factor_count)),
            phi=float(values.get("phi", base.phi)),
            eps=float(values.get("eps", base.eps)),
            hidden_dim=int(values.get("hidden_dim", base.hidden_dim)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid factor_guide_l13_online_filtering params",
            context={"params": str(dict(values))},
        ) from exc
    if updated.factor_count <= 0:
        raise ConfigError("factor_count must be positive")
    if updated.hidden_dim <= 0:
        raise ConfigError("hidden_dim must be positive")
    if not 0.0 < updated.phi < 1.0:
        raise ConfigError("phi must be in (0, 1)")
    return updated


def _encode_local_sites(
    *,
    context: _GuideContext,
    encoder: RegimeEncoder,
    s_u_mean: torch.Tensor,
    phi: float,
    eps: float,
) -> _LocalGuideSites:
    return _build_gain_local_sites(
        inputs=_build_gain_inputs(
            context=context,
            encoder=encoder,
            s_u_mean=s_u_mean,
            phi=phi,
            eps=eps,
        ),
        plan=_build_gain_plan(
            decode_step=lambda encoded, prior_loc, prior_scale: _decode_local_parameters(
                encoded=encoded,
                class_count=context.C,
                prior_loc=prior_loc,
                prior_scale=prior_scale,
            ),
        ),
    )


def _decode_local_parameters(
    *,
    encoded: torch.Tensor,
    class_count: int,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _decode_gain_components(
        slices=_split_gain_slices(encoded, class_count=class_count),
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _build_gain_inputs(
    *,
    context: _GuideContext,
    encoder: RegimeEncoder,
    s_u_mean: torch.Tensor,
    phi: float,
    eps: float,
) -> _GainLocalEncodingInputs:
    return _GainLocalEncodingInputs(
        context=context,
        encoder=encoder,
        s_u_mean=s_u_mean,
        phi=phi,
        eps=eps,
    )


def _build_gain_plan(
    *,
    decode_step: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
) -> _GainLocalEncodingPlan:
    return _GainLocalEncodingPlan(
        decode_step=decode_step,
        initial_prior_message=_initial_prior_message,
        encoder_features=_encoder_features,
        propagated_message=_propagated_message,
    )


def _build_gain_local_sites(
    *,
    inputs: _GainLocalEncodingInputs,
    plan: _GainLocalEncodingPlan,
) -> _LocalGuideSites:
    return _encode_gain_local_sites(inputs=inputs, plan=plan)


def _encode_gain_local_sites(
    *,
    inputs: _GainLocalEncodingInputs,
    plan: _GainLocalEncodingPlan,
) -> _LocalGuideSites:
    h_loc: list[torch.Tensor] = []
    h_scale: list[torch.Tensor] = []
    v_loc: list[torch.Tensor] = []
    v_scale: list[torch.Tensor] = []
    prior_loc, prior_scale = plan.initial_prior_message(
        inputs.context,
        inputs.s_u_mean,
        inputs.phi,
        inputs.eps,
    )
    for time_index in range(inputs.context.T):
        current = _encode_gain_single_step(
            inputs=inputs,
            plan=plan,
            time_index=time_index,
            prior_loc=prior_loc,
            prior_scale=prior_scale,
        )
        h_loc.append(current[0])
        h_scale.append(current[1])
        v_loc.append(current[2])
        v_scale.append(current[3])
        prior_loc, prior_scale = plan.propagated_message(
            current_loc=current[0],
            current_scale=current[1],
            s_u_mean=inputs.s_u_mean,
            phi=inputs.phi,
            eps=inputs.eps,
        )
    return _LocalGuideSites(
        h_loc=torch.stack(h_loc),
        h_scale=torch.stack(h_scale),
        v_loc=torch.stack(v_loc),
        v_scale=torch.stack(v_scale),
    )


def _encode_gain_single_step(
    *,
    inputs: _GainLocalEncodingInputs,
    plan: _GainLocalEncodingPlan,
    time_index: int,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = inputs.encoder(
        plan.encoder_features(
            context=inputs.context,
            time_index=time_index,
            prior_loc=prior_loc,
            prior_scale=prior_scale,
        )
    )
    return plan.decode_step(encoded, prior_loc, prior_scale)


def _decode_gain_parameters(
    *,
    slices: _GainStepSlices,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gain = _MAX_H_GAIN * prior_scale * torch.sigmoid(slices.gain_raw)
    innovation = torch.tanh(slices.innovation_raw)
    current_h_loc = prior_loc + gain * innovation
    scale_step = torch.exp(
        _MAX_H_LOG_SCALE_STEP * torch.tanh(slices.h_scale_raw)
    )
    current_h_scale = torch.clamp(
        prior_scale * scale_step,
        min=_MIN_SCALE,
        max=_MAX_H_SCALE,
    )
    current_v_scale = (
        torch.nn.functional.softplus(slices.v_scale_raw) + _MIN_SCALE
    )
    return current_h_loc, current_h_scale, slices.v_loc_raw, current_v_scale


def _decode_gain_components(
    *,
    slices: _GainStepSlices,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _decode_gain_parameters(
        slices=slices,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _split_gain_slices(
    encoded: torch.Tensor,
    *,
    class_count: int,
) -> _GainStepSlices:
    gain_raw, innovation_raw, h_scale_raw, v_loc_raw, v_scale_raw = (
        torch.split(encoded, class_count)
    )
    return _GainStepSlices(
        gain_raw=gain_raw,
        innovation_raw=innovation_raw,
        h_scale_raw=h_scale_raw,
        v_loc_raw=v_loc_raw,
        v_scale_raw=v_scale_raw,
    )


def _unbound_gain_slices(encoded: torch.Tensor) -> _GainStepSlices:
    gain_raw, innovation_raw, h_scale_raw, v_loc_raw, v_scale_raw = torch.unbind(
        encoded
    )
    return _GainStepSlices(
        gain_raw=gain_raw,
        innovation_raw=innovation_raw,
        h_scale_raw=h_scale_raw,
        v_loc_raw=v_loc_raw,
        v_scale_raw=v_scale_raw,
    )


def _run_guide_call(
    *,
    batch: ModelBatch,
    plan: _GuideCallPlan,
) -> None:
    runtime_batch = plan.build_runtime_batch(batch)
    context = _build_context(runtime_batch)
    encoder = plan.require_encoder(context)
    pyro.module(plan.module_name, encoder)
    structural = plan.sample_global_sites(
        context=context,
        factor_count=plan.config.factor_count,
    )
    local = _encode_local_sites(
        context=context,
        encoder=encoder,
        s_u_mean=structural.s_u_mean,
        phi=plan.config.phi,
        eps=plan.config.eps,
    )
    _sample_regime_sites(local)
    _sample_scale_sites(local)


class _LocalGlobalSitesLike(Protocol):
    @property
    def s_u_mean(self) -> torch.Tensor: ...
