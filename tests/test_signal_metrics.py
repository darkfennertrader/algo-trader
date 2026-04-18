from __future__ import annotations

import math

import numpy as np

from algo_trader.application.signal_metrics import spearman_correlation


def test_spearman_correlation_returns_nan_for_constant_input() -> None:
    lhs = np.array([1.0, 1.0, 1.0], dtype=float)
    rhs = np.array([0.1, 0.2, 0.3], dtype=float)

    result = spearman_correlation(lhs, rhs)

    assert math.isnan(result)
