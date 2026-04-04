from __future__ import annotations
# pylint: disable=duplicate-code

from typing import TYPE_CHECKING, Any, Mapping, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide import IndexSubspaceConsistencyGuideV11L2OnlineFiltering
from .model import predict_index_subspace_consistency_v11_l2

if TYPE_CHECKING:
    from .model import IndexSubspaceConsistencyModelV11L2OnlineFiltering


class _V11L2Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_index_subspace_consistency_v11_l2(
            model=cast(Any, request.model),
            guide=cast(IndexSubspaceConsistencyGuideV11L2OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("index_subspace_consistency_predict_v11_l2_online_filtering")
def build_index_subspace_consistency_predict_v11_l2_online_filtering(
    params: Mapping[str, Any],
) -> _V11L2Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown index_subspace_consistency_predict_v11_l2_online_filtering "
            f"params: {unknown}"
        )
    return _V11L2Predictor()


__all__ = [
    "build_index_subspace_consistency_predict_v11_l2_online_filtering",
    "predict_index_subspace_consistency_v11_l2",
]
