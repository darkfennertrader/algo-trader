from __future__ import annotations

from typing import Any, Mapping, TypeVar, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    DependenceLayerGuideV4L1OnlineFiltering,
    _build_guide_config,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model import (
    DependenceLayerModelV4L1OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.predict import (
    predict_dependence_layer_v4_l1,
)
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PredictiveRequest,
    PyroGuide,
)

GuideType = TypeVar("GuideType", bound=DependenceLayerGuideV4L1OnlineFiltering)


def build_dependence_followup_guide(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    guide_type: type[GuideType],
) -> PyroGuide:
    merged = dict(defaults)
    merged.update(params)
    return guide_type(config=_build_guide_config(merged))


class DependenceFollowupPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_dependence_layer_v4_l1(
            model=cast(DependenceLayerModelV4L1OnlineFiltering, request.model),
            guide=cast(DependenceLayerGuideV4L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def build_dependence_followup_predictor(
    *,
    params: Mapping[str, object],
    label: str,
) -> DependenceFollowupPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(f"Unknown {label} params: {unknown}")
    return DependenceFollowupPredictor()


def predict_dependence_followup_runtime(
    *,
    model: Any,
    guide: Any,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    return predict_dependence_layer_v4_l1(
        model=cast(DependenceLayerModelV4L1OnlineFiltering, model),
        guide=cast(DependenceLayerGuideV4L1OnlineFiltering, guide),
        batch=batch,
        num_samples=num_samples,
        state=state,
    )


__all__ = [
    "DependenceFollowupPredictor",
    "build_dependence_followup_guide",
    "build_dependence_followup_predictor",
    "predict_dependence_followup_runtime",
]
