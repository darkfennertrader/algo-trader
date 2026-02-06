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


def test_cross_sectional_center_and_rank_momentum() -> None:
    weekly_index = pd.date_range(
        "2024-01-05", periods=6, freq="W-FRI", tz="UTC"
    )
    assets = ["ASSET_A", "ASSET_B"]
    weekly_closes = {
        "ASSET_A": np.array([100, 102, 104, 106, 108, 110], dtype=float),
        "ASSET_B": np.array([200, 198, 196, 194, 192, 190], dtype=float),
    }
    weekly_ohlc = _build_ohlc_frame(weekly_index, assets, weekly_closes)

    daily_index = pd.date_range(
        "2023-12-11", periods=40, freq="B", tz="UTC"
    )
    daily_closes = {
        "ASSET_A": np.linspace(100.0, 120.0, len(daily_index)),
        "ASSET_B": np.linspace(200.0, 180.0, len(daily_index)),
    }
    daily_ohlc = _build_ohlc_frame(daily_index, assets, daily_closes)

    config = CrossSectionalConfig(
        horizons=[
            HorizonSpec(days=5, weeks=1),
            HorizonSpec(days=20, weeks=4),
            HorizonSpec(days=60, weeks=12),
            HorizonSpec(days=130, weeks=26),
        ],
        features=["cs_centered", "cs_rank"],
    )
    group = CrossSectionalFeatureGroup(config)
    output = group.compute(
        FeatureInputs(
            frames={"weekly_ohlc": weekly_ohlc, "daily_ohlc": daily_ohlc},
            frequency="weekly",
        )
    )

    assert "cs_centered_mom_4w" in output.feature_names
    assert "cs_rank_mom_4w" in output.feature_names

    last_stamp = weekly_index[-1]
    mom_a = np.log(weekly_closes["ASSET_A"][-1] / weekly_closes["ASSET_A"][-5])
    mom_b = np.log(weekly_closes["ASSET_B"][-1] / weekly_closes["ASSET_B"][-5])
    mean = 0.5 * (mom_a + mom_b)
    expected_center_a = mom_a - mean
    expected_center_b = mom_b - mean

    centered_a = output.frame.loc[
        last_stamp, ("ASSET_A", "cs_centered_mom_4w")
    ]
    centered_b = output.frame.loc[
        last_stamp, ("ASSET_B", "cs_centered_mom_4w")
    ]
    assert np.isclose(centered_a, expected_center_a)
    assert np.isclose(centered_b, expected_center_b)

    expected_rank_a = 0.75
    expected_rank_b = 0.25
    rank_a = output.frame.loc[last_stamp, ("ASSET_A", "cs_rank_mom_4w")]
    rank_b = output.frame.loc[last_stamp, ("ASSET_B", "cs_rank_mom_4w")]
    assert np.isclose(rank_a, expected_rank_a)
    assert np.isclose(rank_b, expected_rank_b)
