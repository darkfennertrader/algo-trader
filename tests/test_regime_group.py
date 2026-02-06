from __future__ import annotations

import numpy as np
import pandas as pd

from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.protocols import FeatureInputs
from algo_trader.pipeline.stages.features.regime import (
    RegimeConfig,
    RegimeFeatureGroup,
)


def _weekly_index_from_daily(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    week_start = normalized - offsets
    week_end_by_start = pd.Series(index, index=week_start).groupby(level=0).max()
    return pd.DatetimeIndex(week_end_by_start)


def _weekly_ohlc_frame(
    weekly_index: pd.DatetimeIndex,
    closes_by_asset: dict[str, list[float]],
) -> pd.DataFrame:
    fields = ["Open", "High", "Low", "Close"]
    columns = pd.MultiIndex.from_product(
        [list(closes_by_asset.keys()), fields]
    )
    values = np.empty((len(weekly_index), len(columns)), dtype=float)
    for asset_idx, (asset, closes) in enumerate(closes_by_asset.items()):
        close = np.array(closes, dtype=float)
        asset_block = np.column_stack([close, close, close, close])
        start = asset_idx * len(fields)
        values[:, start : start + len(fields)] = asset_block
    return pd.DataFrame(values, index=weekly_index, columns=columns)


def _daily_ohlc_frame(
    daily_index: pd.DatetimeIndex, assets: list[str]
) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([assets, ["Open", "High", "Low", "Close"]])
    values = np.tile(
        np.linspace(100.0, 100.0 + len(daily_index) - 1, len(daily_index)).reshape(-1, 1),
        (1, len(columns)),
    )
    return pd.DataFrame(values, index=daily_index, columns=columns)


def test_regime_dispersion_mad_broadcast() -> None:
    weekly_index = pd.to_datetime(
        ["2024-01-05", "2024-01-12", "2024-01-19"], utc=True
    )
    weekly_ohlc = _weekly_ohlc_frame(
        weekly_index,
        {
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 105.0, 110.25],
            "C": [100.0, 90.0, 81.0],
        },
    )
    daily_index = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    daily_ohlc = _daily_ohlc_frame(daily_index, ["A", "B", "C"])

    config = RegimeConfig(
        horizons=[HorizonSpec(days=5, weeks=1)],
        features=["glob_disp_ret"],
    )
    group = RegimeFeatureGroup(config)
    output = group.compute(
        FeatureInputs(
            frames={
                "weekly_ohlc": weekly_ohlc,
                "daily_ohlc": daily_ohlc,
            },
            frequency="weekly",
        )
    )

    assert output.feature_names == ["glob_disp_ret_1w"]

    disp = output.frame.xs("glob_disp_ret_1w", axis=1, level=1, drop_level=True)
    assert disp.columns.tolist() == ["A", "B", "C"]
    assert disp["A"].equals(disp["B"])
    assert disp["A"].equals(disp["C"])

    log_close = np.log(
        np.array(
            [
                [100.0, 110.0, 121.0],
                [100.0, 105.0, 110.25],
                [100.0, 90.0, 81.0],
            ]
        ).T
    )
    returns = np.diff(log_close, axis=0)
    median = np.median(returns, axis=1)
    deviations = np.abs(returns - median[:, None])
    mad = np.median(deviations, axis=1) * 1.4826

    assert np.isnan(disp.iloc[0, 0])
    assert np.isclose(disp.iloc[1, 0], mad[0])
    assert np.isclose(disp.iloc[2, 0], mad[1])


def test_regime_goodness_ratio_daily() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    daily_ohlc = _daily_ohlc_frame(dates, ["ASSET"])
    daily_ohlc.iloc[7] = np.nan

    weekly_index = _weekly_index_from_daily(dates)
    weekly_ohlc = _weekly_ohlc_frame(weekly_index, {"ASSET": [1.0] * len(weekly_index)})

    config = RegimeConfig(
        horizons=[HorizonSpec(days=5, weeks=1)],
        features=["glob_disp_ret"],
    )
    group = RegimeFeatureGroup(config)
    group.compute(
        FeatureInputs(
            frames={
                "daily_ohlc": daily_ohlc,
                "weekly_ohlc": weekly_ohlc,
            },
            frequency="weekly",
        )
    )

    goodness = group.goodness
    assert goodness is not None
    ratios = goodness.ratios_by_feature["glob_disp_ret_1w"]["ASSET"]
    second_week = weekly_index[1]
    assert ratios[second_week.isoformat()] == 0.8
