from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PyroGuide,
    PyroModel,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .model_l12 import FactorModelL12OnlineFiltering, Level12ModelPriors, _build_model_priors
from .predict_l13 import predict_factor_l13


@dataclass(frozen=True)
class FactorModelL13OnlineFiltering(FactorModelL12OnlineFiltering):
    priors: Level12ModelPriors = field(default_factory=Level12ModelPriors)

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        return _posterior_predict_with_explicit_predictor(
            request=_PosteriorPredictRequest(
                guide=guide,
                batch=batch,
                num_samples=num_samples,
                state=state,
            ),
            predictor=self._posterior_predictor,
        )

    def _posterior_predictor(
        self, request: _PosteriorPredictRequest
    ) -> Mapping[str, Any] | None:
        return _predict_from_request(
            request=request,
            model=self,
            predictor=predict_factor_l13,
        )


@register_model("factor_model_l13_online_filtering")
def build_factor_model_l13_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return _build_registered_model(
        params=params,
        model_type=FactorModelL13OnlineFiltering,
        prior_builder=_build_model_priors,
    )


def _posterior_predict_with_explicit_predictor(
    *,
    request: _PosteriorPredictRequest,
    predictor: Callable[[_PosteriorPredictRequest], Mapping[str, Any] | None],
) -> Mapping[str, Any] | None:
    structural_summaries = getattr(
        request.guide, "structural_predictive_summaries", None
    )
    if not callable(structural_summaries):
        structural_summaries = getattr(
            request.guide, "structural_posterior_means", None
        )
    if not callable(structural_summaries):
        return None
    return predictor(request)


def _build_registered_model(
    *,
    params: Mapping[str, Any],
    model_type: Callable[..., PyroModel],
    prior_builder: Callable[[Mapping[str, Any]], Any],
) -> PyroModel:
    return model_type(priors=prior_builder(params))


@dataclass(frozen=True)
class _PosteriorPredictRequest:
    guide: PyroGuide
    batch: ModelBatch
    num_samples: int
    state: Mapping[str, Any] | None


def _predict_from_request(
    *,
    request: _PosteriorPredictRequest,
    model: Any,
    predictor: Callable[..., Mapping[str, Any] | None],
) -> Mapping[str, Any] | None:
    return predictor(
        model=model,
        guide=request.guide,  # type: ignore[arg-type]
        batch=request.batch,
        num_samples=request.num_samples,
        state=request.state,
    )
