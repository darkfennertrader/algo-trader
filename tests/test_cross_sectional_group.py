from __future__ import annotations

import numpy as np
import pandas as pd

from algo_trader.pipeline.stages.features.cross_sectional import (
    CrossSectionalConfig,
    CrossSectionalFeatureGroup,
)
from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.protocols import FeatureInputs


def _build_ohlc_frame(
    index: pd.DatetimeIndex,
    assets: list[str],
    closes: dict[str, np.ndarray],
) -> pd.DataFrame:
    data: dict[tuple[str, str], np.ndarray] = {}
    for asset in assets:
        series = closes[asset]
        data[(asset, "Open")] = series
        data[(asset, "High")] = series
        data[(asset, "Low")] = series
        data[(asset, "Close")] = series
    frame = pd.DataFrame(data, index=index)
    frame.columns = pd.MultiIndex.from_tuples(
        frame.columns, names=["asset", "field"]
    )
    return frame


def test_cross_sectional_rank_momentum() -> None:
    weekly_index = pd.date_range(
        "2024-01-05", periods=30, freq="W-FRI", tz="UTC"
    )
    assets = ["ASSET_A", "ASSET_B"]
    weekly_closes = {
        "ASSET_A": np.linspace(100.0, 160.0, len(weekly_index)),
        "ASSET_B": np.linspace(200.0, 140.0, len(weekly_index)),
    }
    weekly_ohlc = _build_ohlc_frame(weekly_index, assets, weekly_closes)

    config = CrossSectionalConfig(
        horizons=[
            HorizonSpec(days=5, weeks=1),
            HorizonSpec(days=20, weeks=4),
            HorizonSpec(days=60, weeks=12),
            HorizonSpec(days=130, weeks=26),
        ],
        features=["cs_rank"],
    )
    group = CrossSectionalFeatureGroup(config)
    output = group.compute(
        FeatureInputs(
            frames={"weekly_ohlc": weekly_ohlc},
            frequency="weekly",
        )
    )

    assert "cs_rank_z_mom_4w" in output.feature_names

    last_stamp = weekly_index[-1]
    close_a = pd.Series(weekly_closes["ASSET_A"], index=weekly_index)
    close_b = pd.Series(weekly_closes["ASSET_B"], index=weekly_index)
    returns_a = np.log(close_a / close_a.shift(1))
    returns_b = np.log(close_b / close_b.shift(1))
    sigma_a = returns_a.rolling(window=26, min_periods=26).std(ddof=1)
    sigma_b = returns_b.rolling(window=26, min_periods=26).std(ddof=1)
    mom_a = np.log(close_a / close_a.shift(4))
    mom_b = np.log(close_b / close_b.shift(4))
    z_mom_a = mom_a / (sigma_a * np.sqrt(4.0))
    z_mom_b = mom_b / (sigma_b * np.sqrt(4.0))
    last_a = float(z_mom_a.loc[last_stamp])
    last_b = float(z_mom_b.loc[last_stamp])
    assert not np.isnan(last_a)
    assert not np.isnan(last_b)

    expected_rank_a = 0.75 if last_a > last_b else 0.25
    expected_rank_b = 0.25 if last_a > last_b else 0.75
    rank_a = output.frame.loc[last_stamp, ("ASSET_A", "cs_rank_z_mom_4w")]
    rank_b = output.frame.loc[last_stamp, ("ASSET_B", "cs_rank_z_mom_4w")]
    assert np.isclose(rank_a, expected_rank_a)
    assert np.isclose(rank_b, expected_rank_b)
