from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.protocols import FeatureInputs
from algo_trader.pipeline.stages.features.seasonal import (
    SeasonalConfig,
    SeasonalFeatureGroup,
)


def _weekly_index_from_daily(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    week_start = normalized - offsets
    week_end_by_start = pd.Series(index, index=week_start).groupby(level=0).max()
    return pd.DatetimeIndex(week_end_by_start)


def test_seasonal_weekday_means_and_spread() -> None:
    dates = pd.date_range("2023-12-29", periods=16, freq="B", tz="UTC")
    returns_by_weekday = {
        0: 0.01,
        1: 0.02,
        2: 0.03,
        3: 0.04,
        4: 0.05,
    }
    closes = [100.0]
    for stamp in dates[1:]:
        r_d = returns_by_weekday[stamp.dayofweek]
        closes.append(closes[-1] * float(np.exp(r_d)))
    values = np.tile(np.array(closes).reshape(-1, 1), (1, 4))
    columns = pd.MultiIndex.from_product(
        [["ASSET"], ["Open", "High", "Low", "Close"]]
    )
    daily_ohlc = pd.DataFrame(values, index=dates, columns=columns)

    weekly_index = _weekly_index_from_daily(dates)
    weekly_ohlc = pd.DataFrame(1.0, index=weekly_index, columns=columns)

    config = SeasonalConfig(
        horizons=[HorizonSpec(days=10, weeks=2)],
        features=["dow_alpha", "dow_spread"],
    )
    group = SeasonalFeatureGroup(config)
    output = group.compute(
        FeatureInputs(
            frames={
                "daily_ohlc": daily_ohlc,
                "weekly_ohlc": weekly_ohlc,
            },
            frequency="weekly",
        )
    )

    assert output.frame.index.equals(weekly_index)

    target_week = weekly_index[-1]
    for weekday, expected in returns_by_weekday.items():
        day_name = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}[weekday]
        feature = f"dow_alpha_{day_name}_2w"
        value = output.frame.loc[target_week, ("ASSET", feature)]
        assert value == pytest.approx(expected)

    spread = output.frame.loc[target_week, ("ASSET", "dow_spread_2w")]
    assert spread == pytest.approx(0.04)


def test_seasonal_missing_weekdays_payload() -> None:
    dates = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-04",
            "2024-01-05",
        ],
        tz="UTC",
    )
    values = np.tile(np.linspace(100.0, 103.0, len(dates)).reshape(-1, 1), (1, 4))
    columns = pd.MultiIndex.from_product(
        [["ASSET"], ["Open", "High", "Low", "Close"]]
    )
    daily_ohlc = pd.DataFrame(values, index=dates, columns=columns)
    daily_ohlc.loc[pd.Timestamp("2024-01-04", tz="UTC")] = np.nan

    weekly_index = _weekly_index_from_daily(dates)
    weekly_ohlc = pd.DataFrame(1.0, index=weekly_index, columns=columns)

    config = SeasonalConfig(
        horizons=[HorizonSpec(days=5, weeks=1)],
        features=["dow_alpha"],
    )
    group = SeasonalFeatureGroup(config)
    _ = group.compute(
        FeatureInputs(
            frames={
                "daily_ohlc": daily_ohlc,
                "weekly_ohlc": weekly_ohlc,
            },
            frequency="weekly",
        )
    )

    missing = group.missing_weekdays
    assert missing is not None
    asset_missing = missing["ASSET"]
    assert asset_missing["count"] == 2
    assert set(asset_missing["dates"]) == {"2024-01-03", "2024-01-04"}
