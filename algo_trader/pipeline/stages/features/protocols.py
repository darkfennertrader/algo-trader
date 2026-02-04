from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

import pandas as pd

FeatureFrequency = Literal["hourly", "daily", "weekly"]


@dataclass(frozen=True)
class FeatureInputs:
    frames: Mapping[str, pd.DataFrame]
    frequency: FeatureFrequency


@dataclass(frozen=True)
class FeatureOutput:
    frame: pd.DataFrame
    feature_names: Sequence[str]


class FeatureGroup(Protocol):
    name: str

    def compute(self, inputs: FeatureInputs) -> FeatureOutput: ...
