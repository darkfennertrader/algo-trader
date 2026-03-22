from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
        structural_summaries = getattr(
            guide, "structural_predictive_summaries", None
        )
        if not callable(structural_summaries):
            structural_summaries = getattr(
                guide, "structural_posterior_means", None
            )
        if not callable(structural_summaries):
            return None
        return predict_factor_l13(
            model=self,
            guide=guide,  # type: ignore[arg-type]
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("factor_model_l13_online_filtering")
def build_factor_model_l13_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return FactorModelL13OnlineFiltering(priors=_build_model_priors(params))
