from pathlib import Path

import numpy as np
import pytest

from algo_trader.application.simulation.calibration_summary_diagnostics import (
    CalibrationSummary,
    aggregate_calibration_summaries,
    build_calibration_summary,
    pit_uniform_rmse,
    read_calibration_summary,
    write_calibration_summary,
)
from algo_trader.application.simulation.diagnostics import (
    FanChartData,
    TimeSampleRecord,
    _compute_pit,
    _coverage_curve_from_raw,
    _pool_equal_split_samples,
    _resolve_true_row,
)
from algo_trader.domain import SimulationError


def test_compute_pit_basic() -> None:
    z_samples = np.array(
        [
            [[0.0], [1.0]],
            [[1.0], [2.0]],
        ]
    )
    z_true = np.array([[0.5], [1.5]])
    pit = _compute_pit(z_samples, z_true)
    assert pit.shape == (2, 1)
    assert np.allclose(pit[:, 0], [0.5, 0.5], equal_nan=True)


def test_coverage_curve_from_raw() -> None:
    data = FanChartData(
        timestamps=[],
        asset_names=["A", "B"],
        z_true=np.array([[0.0, 2.0], [0.0, 0.0]], dtype=float),
        quantiles={
            0.05: np.array([[-1.0, 1.0], [-1.0, -1.0]], dtype=float),
            0.95: np.array([[1.0, 3.0], [1.0, 1.0]], dtype=float),
        },
    )
    curve = _coverage_curve_from_raw(data=data, coverage_levels=(0.9,))
    assert np.isclose(curve[0.9], 1.0)


def test_pit_uniform_rmse_zero_for_uniform_bins() -> None:
    values = np.linspace(0.025, 0.975, num=20, dtype=float)
    score, sample_count, bin_count = pit_uniform_rmse(values, bin_count=20)
    assert np.isclose(score, 0.0)
    assert sample_count == 20
    assert bin_count == 20


def test_build_calibration_summary_aggregates_coverage_and_pit() -> None:
    summary = build_calibration_summary(
        curve={0.5: 0.6, 0.9: 0.85},
        pit_values=np.linspace(0.025, 0.975, num=20, dtype=float),
    )
    assert np.isclose(summary.abs_error_by_level[0.5], 0.1)
    assert np.isclose(summary.abs_error_by_level[0.9], 0.05)
    assert np.isclose(summary.mean_abs_coverage_error, 0.075)
    assert np.isclose(summary.max_abs_coverage_error, 0.1)
    assert np.isclose(summary.pit_uniform_rmse, 0.0)
    assert summary.pit_sample_count == 20


def test_aggregate_calibration_summaries_uses_fold_medians() -> None:
    summary = aggregate_calibration_summaries(
        [
            CalibrationSummary(
                coverage_by_level={0.5: 0.58, 0.9: 0.92},
                abs_error_by_level={0.5: 0.08, 0.9: 0.02},
                mean_abs_coverage_error=0.05,
                max_abs_coverage_error=0.08,
                pit_uniform_rmse=0.01,
                pit_sample_count=100,
                pit_bin_count=20,
            ),
            CalibrationSummary(
                coverage_by_level={0.5: 0.62, 0.9: 0.94},
                abs_error_by_level={0.5: 0.12, 0.9: 0.04},
                mean_abs_coverage_error=0.08,
                max_abs_coverage_error=0.12,
                pit_uniform_rmse=0.03,
                pit_sample_count=120,
                pit_bin_count=20,
            ),
        ]
    )
    assert np.isclose(summary.coverage_by_level[0.5], 0.6)
    assert np.isclose(summary.coverage_by_level[0.9], 0.93)
    assert np.isclose(summary.abs_error_by_level[0.5], 0.1)
    assert np.isclose(summary.abs_error_by_level[0.9], 0.03)
    assert np.isclose(summary.mean_abs_coverage_error, 0.065)
    assert np.isclose(summary.max_abs_coverage_error, 0.1)
    assert np.isclose(summary.pit_uniform_rmse, 0.02)
    assert summary.pit_sample_count == 110
    assert summary.pit_bin_count == 20


def test_read_calibration_summary_round_trip(tmp_path: Path) -> None:
    output_dir = tmp_path / "calibration"
    output_dir.mkdir()
    summary = CalibrationSummary(
        coverage_by_level={0.5: 0.6, 0.9: 0.93},
        abs_error_by_level={0.5: 0.1, 0.9: 0.03},
        mean_abs_coverage_error=0.065,
        max_abs_coverage_error=0.1,
        pit_uniform_rmse=0.02,
        pit_sample_count=110,
        pit_bin_count=20,
    )
    write_calibration_summary(summary=summary, output_dir=output_dir)
    loaded = read_calibration_summary(output_dir / "calibration_summary.json")
    assert loaded == summary


def test_pool_equal_split_samples_uses_equal_count_per_split() -> None:
    records = [
        TimeSampleRecord(
            source="split_1",
            z_true=np.array([0.0], dtype=float),
            z_samples=np.array([[0.0], [10.0]], dtype=float),
        ),
        TimeSampleRecord(
            source="split_2",
            z_true=np.array([0.0], dtype=float),
            z_samples=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float),
        ),
    ]
    pooled = _pool_equal_split_samples(records=records, asset_idx=0)
    assert np.allclose(pooled, [0.0, 10.0, 0.0, 2.0])


def test_resolve_true_row_raises_on_inconsistent_duplicates() -> None:
    records = [
        TimeSampleRecord(
            source="split_1",
            z_true=np.array([1.0], dtype=float),
            z_samples=np.array([[0.0]], dtype=float),
        ),
        TimeSampleRecord(
            source="split_2",
            z_true=np.array([1.1], dtype=float),
            z_samples=np.array([[0.0]], dtype=float),
        ),
    ]
    with pytest.raises(SimulationError, match="Inconsistent z_true"):
        _resolve_true_row(records=records, time_idx=10, asset_count=1)
