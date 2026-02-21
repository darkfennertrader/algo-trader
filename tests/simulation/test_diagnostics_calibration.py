import numpy as np

from algo_trader.application.simulation.diagnostics import (
    _compute_pit,
    _coverage_curve_from_groups,
)


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


def test_coverage_curve_from_groups() -> None:
    coverage_groups = {
        0.5: np.array([[0.2, 0.4], [0.6, 0.8]]),
        0.9: np.array([[0.9, 0.7], [0.8, 0.6]]),
    }
    curve = _coverage_curve_from_groups(coverage_groups)
    assert np.isclose(curve[0.5], 0.5)
    assert np.isclose(curve[0.9], 0.75)
