from __future__ import annotations

from typing import Any, Mapping, cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide_l14 import FactorGuideL14OnlineFiltering
from .predict_l13 import predict_factor_l13


class _Level14Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_factor_l14(
            model=cast(Any, request.model),
            guide=cast(FactorGuideL14OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_factor_l14(
    *,
    model: Any,
    guide: FactorGuideL14OnlineFiltering,
    batch: Any,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    return cast(
        Mapping[str, torch.Tensor] | None,
        predict_factor_l13(
            model=model,
            guide=cast(Any, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        ),
    )


@register_predictor("factor_predict_l14_online_filtering")
def build_factor_predict_l14_online_filtering(
    params: Mapping[str, Any],
) -> _Level14Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown factor_predict_l14_online_filtering params: "
            f"{unknown}"
        )
    return _Level14Predictor()
