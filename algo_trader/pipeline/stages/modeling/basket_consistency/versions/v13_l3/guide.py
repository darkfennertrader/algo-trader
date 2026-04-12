from __future__ import annotations

from typing import Any, Mapping

import torch

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide import (
    BasketConsistencyGuideV13L1OnlineFiltering,
    V13L1GuideConfig,
)
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.shared import (
    BasketConsistencyCoordinates,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v13_l3, merge_nested_params
from .shared import build_relative_basket_consistency_coordinates
from ..runtime_helpers import build_followup_guide


class BasketConsistencyGuideV13L3OnlineFiltering(
    BasketConsistencyGuideV13L1OnlineFiltering
):
    def _build_basket_coordinates(
        self,
        *,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BasketConsistencyCoordinates:
        return build_relative_basket_consistency_coordinates(
            assets=batch.assets,
            device=device,
            dtype=dtype,
        )


@register_guide("basket_consistency_guide_v13_l3_online_filtering")
def build_basket_consistency_guide_v13_l3_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return build_followup_guide(
        params=params,
        guide_defaults=guide_default_params_v13_l3,
        guide_factory=BasketConsistencyGuideV13L3OnlineFiltering,
        merge_params=merge_nested_params,
    )


__all__ = [
    "BasketConsistencyGuideV13L3OnlineFiltering",
    "V13L1GuideConfig",
    "build_basket_consistency_guide_v13_l3_online_filtering",
]
