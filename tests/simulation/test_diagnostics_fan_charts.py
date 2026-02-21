import numpy as np
import pandas as pd
import pytest

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import FanChartsConfig
from algo_trader.application.simulation.diagnostics import (
    FanChartData,
    _coverage_series,
    _resolve_assets,
)


def test_resolve_assets_raises_on_missing() -> None:
    config = FanChartsConfig(enable=True, assets_mode="list", assets=("A",))
    with pytest.raises(SimulationError):
        _resolve_assets(["B"], config)


def test_coverage_series_computes_full_coverage() -> None:
    timestamps = [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-08", tz="UTC"),
    ]
    z_true = np.array([[0.0, 0.0], [2.0, 2.0]])
    quantiles = {
        0.05: np.array([[-1.0, -1.0], [1.0, 1.0]]),
        0.95: np.array([[1.0, 1.0], [3.0, 3.0]]),
    }
    data = FanChartData(
        timestamps=timestamps,
        asset_names=["A", "B"],
        z_true=z_true,
        quantiles=quantiles,
    )
    coverage, band = _coverage_series(data, 0.9)
    assert coverage.shape == (2,)
    assert np.allclose(coverage, [1.0, 1.0], equal_nan=True)
    assert np.allclose(band[0], coverage, equal_nan=True)
    assert np.allclose(band[1], coverage, equal_nan=True)
