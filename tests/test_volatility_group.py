from __future__ import annotations

import numpy as np
import pandas as pd

from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.protocols import FeatureInputs
from algo_trader.pipeline.stages.features.volatility import (
    VolatilityConfig,
    VolatilityFeatureGroup,
)


def _weekly_index_from_daily(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    week_start = normalized - offsets
    week_end_by_start = pd.Series(index, index=week_start).groupby(level=0).max()
    return pd.DatetimeIndex(week_end_by_start)


def test_volatility_goodness_ratio_weekly() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    columns = pd.MultiIndex.from_product(
        [["ASSET"], ["Open", "High", "Low", "Close"]]
    )
    values = np.tile(
        np.linspace(100.0, 109.0, len(dates)).reshape(-1, 1), (1, 4)
    )
    daily_ohlc = pd.DataFrame(values, index=dates, columns=columns)
    daily_ohlc.iloc[7] = np.nan

    weekly_index = _weekly_index_from_daily(dates)
    weekly_ohlc = pd.DataFrame(1.0, index=weekly_index, columns=columns)

    config = VolatilityConfig(
        horizons=[HorizonSpec(days=5, weeks=1)],
        features=["vol_cc_d"],
    )
    group = VolatilityFeatureGroup(config)
    output = group.compute(
        FeatureInputs(
            frames={
                "daily_ohlc": daily_ohlc,
                "weekly_ohlc": weekly_ohlc,
            },
            frequency="weekly",
        )
    )

    assert "vol_cc_d_1w" in output.feature_names
    assert output.frame.index.equals(weekly_index)

    goodness = group.goodness
    assert goodness is not None
    ratios = goodness.ratios_by_feature["vol_cc_d_1w"]["ASSET"]
    second_week = weekly_index[1]
    assert ratios[second_week.isoformat()] == 0.8
