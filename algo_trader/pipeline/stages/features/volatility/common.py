from __future__ import annotations

import numpy as np
import pandas as pd

from algo_trader.domain import DataProcessingError


def require_datetime_index(
    index: pd.Index, *, label: str
) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            f"{label} index must be datetime",
            context={"index_type": type(index).__name__},
        )
    return index


def week_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    return normalized - offsets


def week_end_by_start(index: pd.DatetimeIndex) -> pd.Series:
    if index.empty:
        return pd.Series(dtype="datetime64[ns]")
    week_start = week_start_index(index)
    return pd.Series(index, index=week_start).groupby(level=0).max()


def to_weekly(
    series: pd.Series,
    week_end_by_week_start: pd.Series,
    weekly_index: pd.DatetimeIndex,
) -> pd.Series:
    if series.empty:
        return pd.Series(index=weekly_index, dtype=float)
    series = series.sort_index()
    series_index = require_datetime_index(series.index, label="daily_ohlc")
    week_start = week_start_index(series_index)
    weekly = series.groupby(week_start, sort=False).last()
    week_end = week_end_by_week_start.reindex(weekly.index)
    mask = week_end.notna()
    weekly = weekly[mask]
    week_end = week_end[mask]
    weekly.index = pd.DatetimeIndex(week_end)
    return weekly.reindex(weekly_index)


def log_ratio(
    short_series: pd.Series, long_series: pd.Series, *, eps: float
) -> pd.Series:
    short_aligned, long_aligned = short_series.align(long_series)
    short_clip = short_aligned.clip(lower=eps)
    long_clip = long_aligned.clip(lower=eps)
    ratio = (short_clip + eps) / (long_clip + eps)
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=ratio.index)
