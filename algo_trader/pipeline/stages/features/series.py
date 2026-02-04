from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriceSeries:
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
