from __future__ import annotations

from typing import Callable, Generic, Sequence, TypeVar

import pandas as pd

from .protocols import FeatureGroup, FeatureInputs, FeatureOutput
from .utils import compute_weekly_group_features

ConfigT = TypeVar("ConfigT")


class WeeklyFeatureGroup(FeatureGroup, Generic[ConfigT]):
    supported_features: Sequence[str]
    error_message: str
    compute_asset: Callable[[pd.DataFrame, ConfigT, set[str]], pd.DataFrame]

    def __init__(self, config: ConfigT) -> None:
        self._config = config

    def compute(self, inputs: FeatureInputs) -> FeatureOutput:
        compute_asset = type(self).compute_asset
        return compute_weekly_group_features(
            inputs,
            config=self._config,
            supported_features=type(self).supported_features,
            error_message=type(self).error_message,
            compute_asset=compute_asset,
        )
