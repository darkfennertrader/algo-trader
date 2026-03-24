from __future__ import annotations

from typing import Any, Mapping, cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PredictiveRequest,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide_l15 import FactorGuideL15OnlineFiltering, build_level15_runtime_batch
from .predict_l11 import (
    _prediction_mapping_from_rollout,
    _resolve_structural_means,
)


class _Level15Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_factor_l15(
            model=cast(Any, request.model),
            guide=cast(FactorGuideL15OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_factor_l15(
    *,
    model: Any,
    guide: FactorGuideL15OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    runtime_batch = build_level15_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    structural = _resolve_structural_means(
        state=state,
        guide=cast(Any, guide),
    )
    return _prediction_mapping_from_rollout(
        model=cast(Any, model),
        structural=structural,
        batch=cast(Any, runtime_batch),
        num_samples=num_samples,
    )


@register_predictor("factor_predict_l15_online_filtering")
def build_factor_predict_l15_online_filtering(
    params: Mapping[str, Any],
) -> _Level15Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown factor_predict_l15_online_filtering params: "
            f"{unknown}"
        )
    return _Level15Predictor()
