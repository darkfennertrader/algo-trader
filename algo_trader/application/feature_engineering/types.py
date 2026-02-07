from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from algo_trader.infrastructure.data import ReturnType


@dataclass(frozen=True)
class FeatureSettings:
    return_type: ReturnType
    horizon_days: Sequence[int] | None
    eps: float


@dataclass(frozen=True)
class FeatureSelection:
    groups: Sequence[str]
    features: Sequence[str] | None


@dataclass(frozen=True)
class FeaturePaths:
    data_lake: Path
    feature_store: Path


@dataclass(frozen=True)
class FeatureInputSources:
    weekly_path: Path
    daily_path: Path | None
    hourly_path: Path | None


@dataclass(frozen=True)
class RunConfig:
    settings: FeatureSettings
    selection: FeatureSelection
    paths: FeaturePaths
