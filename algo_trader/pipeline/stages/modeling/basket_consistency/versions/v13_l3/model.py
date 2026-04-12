from __future__ import annotations

from typing import Any, Mapping

import torch

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
    BasketConsistencyModelV13L1OnlineFiltering,
    V13L1ModelPriors,
    _build_model_priors,
    predict_basket_consistency_v13_l1,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .defaults import merge_nested_params, model_default_params_v13_l3
from .shared import (
    BasketConsistencyCoordinates,
    BasketObservationGroup,
    build_relative_basket_consistency_coordinates,
    build_relative_basket_observation_groups,
)


class BasketConsistencyModelV13L3OnlineFiltering(
    BasketConsistencyModelV13L1OnlineFiltering
):
    def _build_basket_coordinates(
        self,
        *,
        assets: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BasketConsistencyCoordinates:
        return build_relative_basket_consistency_coordinates(
            assets=assets,
            device=device,
            dtype=dtype,
        )

    def _build_basket_observation_groups(
        self,
        *,
        basket_names: tuple[str, ...],
        device: torch.device,
    ) -> tuple[BasketObservationGroup, ...]:
        return build_relative_basket_observation_groups(
            config=self.priors.basket_consistency,
            basket_names=basket_names,
            device=device,
        )


@register_model("basket_consistency_model_v13_l3_online_filtering")
def build_basket_consistency_model_v13_l3_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v13_l3(), params)
    return BasketConsistencyModelV13L3OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


__all__ = [
    "BasketConsistencyModelV13L3OnlineFiltering",
    "V13L1ModelPriors",
    "build_basket_consistency_model_v13_l3_online_filtering",
    "predict_basket_consistency_v13_l1",
]
