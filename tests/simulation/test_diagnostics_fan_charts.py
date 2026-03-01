import numpy as np
import pandas as pd
import pytest

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import FanChartsConfig
from algo_trader.application.simulation.diagnostics import (
    CalibrationLevelSeries,
    FanChartData,
    _build_calibration_level_frame,
    _build_calibration_level_series,
    _resolve_assets,
)
from algo_trader.application.simulation.svi_loss_diagnostics import (
    _aggregate_svi_loss_curves,
    _extract_svi_loss_curve,
)


def test_resolve_assets_raises_on_missing() -> None:
    config = FanChartsConfig(enable=True, assets_mode="list", assets=("A",))
    with pytest.raises(SimulationError):
        _resolve_assets(["B"], config)


def test_build_calibration_level_series_computes_full_coverage() -> None:
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
    series = _build_calibration_level_series(
        data=data, level=0.9, rolling_windows=()
    )
    assert series.weekly_coverage.shape == (2,)
    assert np.allclose(series.weekly_coverage, [1.0, 1.0], equal_nan=True)
    assert np.allclose(
        series.weekly_band[0], series.weekly_coverage, equal_nan=True
    )
    assert np.allclose(
        series.weekly_band[1], series.weekly_coverage, equal_nan=True
    )


def test_extract_svi_loss_curve_handles_missing_payload() -> None:
    assert _extract_svi_loss_curve({}) is None
    assert _extract_svi_loss_curve({"training": {}}) is None


def test_aggregate_svi_loss_curves_variable_lengths() -> None:
    summary = _aggregate_svi_loss_curves(
        [
            np.asarray([10.0, 9.0, 8.0], dtype=float),
            np.asarray([12.0, 10.0], dtype=float),
        ]
    )
    assert summary.steps.tolist() == [1, 2, 3]
    assert summary.counts.tolist() == [2, 2, 1]
    assert np.allclose(summary.median, [11.0, 9.5, 8.0], equal_nan=True)


def test_build_calibration_level_series_adds_cumulative_and_rolling() -> None:
    timestamps = [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-08", tz="UTC"),
        pd.Timestamp("2024-01-15", tz="UTC"),
    ]
    z_true = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    quantiles = {
        0.05: np.array(
            [
                [-1.0, -1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
            ]
        ),
        0.95: np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [3.0, 3.0],
            ]
        ),
    }
    data = FanChartData(
        timestamps=timestamps,
        asset_names=["A", "B"],
        z_true=z_true,
        quantiles=quantiles,
    )
    series = _build_calibration_level_series(
        data=data,
        level=0.9,
        rolling_windows=(2,),
    )
    assert np.allclose(series.weekly_coverage, [1.0, 0.5, 1.0], equal_nan=True)
    assert np.allclose(
        series.cumulative_coverage,
        [1.0, 0.75, 5.0 / 6.0],
        equal_nan=True,
    )
    assert 2 in series.rolling_coverage
    assert np.allclose(
        series.rolling_coverage[2], [np.nan, 0.75, 0.75], equal_nan=True
    )


def test_build_calibration_level_frame_contains_expected_columns() -> None:
    timestamps = [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-08", tz="UTC"),
    ]
    data = FanChartData(
        timestamps=timestamps,
        asset_names=["A"],
        z_true=np.asarray([[0.0], [1.0]], dtype=float),
        quantiles={
            0.05: np.asarray([[-1.0], [0.0]], dtype=float),
            0.95: np.asarray([[1.0], [2.0]], dtype=float),
        },
    )
    series = CalibrationLevelSeries(
        level=0.9,
        weekly_coverage=np.asarray([1.0, 1.0], dtype=float),
        weekly_band=(
            np.asarray([1.0, 1.0], dtype=float),
            np.asarray([1.0, 1.0], dtype=float),
        ),
        cumulative_coverage=np.asarray([1.0, 1.0], dtype=float),
        rolling_coverage={13: np.asarray([np.nan, 1.0], dtype=float)},
    )
    frame = _build_calibration_level_frame(data, series)
    assert frame.columns.tolist() == [
        "timestamp",
        "nominal_level",
        "weekly_coverage",
        "weekly_band_low",
        "weekly_band_high",
        "cumulative_coverage",
        "rolling_mean_13w",
    ]
