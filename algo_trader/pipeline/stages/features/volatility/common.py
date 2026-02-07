from __future__ import annotations

import numpy as np
import pandas as pd

from algo_trader.pipeline.stages.features.utils import to_weekly


def log_ratio(
    short_series: pd.Series, long_series: pd.Series, *, eps: float
) -> pd.Series:
    short_aligned, long_aligned = short_series.align(long_series)
    short_clip = short_aligned.clip(lower=eps)
    long_clip = long_aligned.clip(lower=eps)
    ratio = (short_clip + eps) / (long_clip + eps)
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=ratio.index)
