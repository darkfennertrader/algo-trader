from __future__ import annotations

import pandas as pd

from algo_trader.domain import DataProcessingError


def require_datetime_index(
    index: pd.Index, *, label: str
) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    raise DataProcessingError(
        f"{label} index must be datetime",
        context={"index_type": type(index).__name__},
    )


def combine_hourly_indexes(
    hourly_ohlc_by_asset: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    indexes = [
        require_datetime_index(frame.index, label="ohlc")
        for frame in hourly_ohlc_by_asset.values()
        if frame is not None and not frame.empty
    ]
    if not indexes:
        return pd.DatetimeIndex([])
    combined = indexes[0]
    if len(indexes) > 1:
        combined = combined.append(indexes[1:])
    return pd.DatetimeIndex(combined)


def weekday_only_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.empty:
        return index
    mask = index.dayofweek <= 4
    return pd.DatetimeIndex(index[mask])
