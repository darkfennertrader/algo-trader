from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.defaults import (
    guide_default_params_v4_l1,
    merge_nested_params,
    model_default_params_v4_l1,
)
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
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.model_runtime import (
    _RuntimeObservationInputs,
    IndexRelativeMeasurementModelRuntime,
    build_index_relative_measurement_model_priors,
)
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PredictiveRequest,
    PyroGuide,
    PyroModel,
)

GuideType = TypeVar("GuideType", bound=DependenceLayerGuideV4L1OnlineFiltering)
ConfigBuilder = Callable[[object], Any]


def guide_default_params_dependence_followup() -> dict[str, Any]:
    return guide_default_params_v4_l1()


def make_dependence_followup_model_defaults(
    *,
    param_key: str,
    overrides: Mapping[str, Any],
) -> Callable[[], dict[str, Any]]:
    def builder() -> dict[str, Any]:
        return merge_nested_params(
            model_default_params_v4_l1(),
            {param_key: dict(overrides)},
        )

    return builder


class RawPlusAuxiliaryIndexRelativeRuntime(IndexRelativeMeasurementModelRuntime):
    def _sample_observations(
        self,
        *,
        inputs: _RuntimeObservationInputs,
    ) -> None:
        self._sample_full_raw_head(inputs)
        if (
            not self.priors.index_relative_measurement.enabled
            or inputs.coordinates.coordinate_count == 0
        ):
            return
        self._sample_auxiliary_observations(inputs)


@dataclass(frozen=True)
class IndexRelativeFollowupModelBuildSpec:
    defaults: Callable[[], dict[str, Any]]
    runtime_type: type[RawPlusAuxiliaryIndexRelativeRuntime]
    config_builder: ConfigBuilder
    label: str
    param_key: str


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


def build_index_relative_followup_model(
    *,
    params: Mapping[str, Any],
    spec: IndexRelativeFollowupModelBuildSpec,
) -> PyroModel:
    merged_params = merge_nested_params(spec.defaults(), params)
    return spec.runtime_type(
        priors=build_index_relative_measurement_model_priors(
            params=merged_params,
            config_builder=spec.config_builder,
            label=spec.label,
            param_key=spec.param_key,
        )
    )


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
    "guide_default_params_dependence_followup",
    "build_dependence_followup_guide",
    "build_index_relative_followup_model",
    "build_dependence_followup_predictor",
    "IndexRelativeFollowupModelBuildSpec",
    "make_dependence_followup_model_defaults",
    "predict_dependence_followup_runtime",
    "RawPlusAuxiliaryIndexRelativeRuntime",
]
