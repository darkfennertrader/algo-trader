from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from ..group_base import WeeklyFeatureGroup
from ..utils import require_ohlc_columns

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (20, 130)
SUPPORTED_FEATURES: tuple[str, ...] = ("brk_up", "brk_dn")
BREAKOUT_WEEKS: tuple[int, ...] = (4, 26)


@dataclass(frozen=True)
class BreakoutConfig:
    horizons: Sequence[HorizonSpec]
    features: Sequence[str] | None = None


def _compute_asset_features(
    asset_frame: pd.DataFrame,
    config: BreakoutConfig,
    feature_set: set[str],
) -> pd.DataFrame:
    require_ohlc_columns(asset_frame)
    high = asset_frame["High"]
    low = asset_frame["Low"]
    close = asset_frame["Close"]

    high_shifted = high.shift(1)
    low_shifted = low.shift(1)

    feature_data: dict[str, np.ndarray] = {}
    specs = [spec for spec in config.horizons if spec.weeks in BREAKOUT_WEEKS]
    if not specs and feature_set:
        raise ConfigError(
            "No horizons available for breakout features",
            context={"required_weeks": ",".join(map(str, BREAKOUT_WEEKS))},
        )
    for spec in specs:
        if "brk_up" in feature_set:
            prev_high = high_shifted.rolling(
                window=spec.weeks, min_periods=spec.weeks
            ).max()
            feature_data[_breakout_up_name(spec.weeks)] = (
                close > prev_high
            ).astype(int).to_numpy(dtype=float)
        if "brk_dn" in feature_set:
            prev_low = low_shifted.rolling(
                window=spec.weeks, min_periods=spec.weeks
            ).min()
            feature_data[_breakout_down_name(spec.weeks)] = (
                close < prev_low
            ).astype(int).to_numpy(dtype=float)
    return pd.DataFrame(feature_data, index=asset_frame.index)


def _breakout_up_name(weeks: int) -> str:
    return f"brk_up_{weeks}w"


def _breakout_down_name(weeks: int) -> str:
    return f"brk_dn_{weeks}w"


class BreakoutFeatureGroup(WeeklyFeatureGroup[BreakoutConfig]):
    name = "breakout"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown breakout features requested"
    compute_asset = _compute_asset_features
