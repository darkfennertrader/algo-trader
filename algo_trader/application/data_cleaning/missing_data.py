from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from algo_trader.domain import DataProcessingError
from algo_trader.infrastructure.data import (
    combine_hourly_indexes,
    require_datetime_index,
    weekday_only_index,
)


@dataclass(frozen=True)
class MissingDataSummary:
    missing_by_asset: dict[str, list[pd.Timestamp]]
    missing_counts_by_month: dict[str, dict[str, int]]


def build_missing_data_summary(
    hourly_ohlc_by_asset: dict[str, pd.DataFrame],
    *,
    assets: list[str],
) -> MissingDataSummary:
    presence_by_asset = _build_daily_presence_by_asset(
        hourly_ohlc_by_asset, assets
    )
    presence_frame = pd.DataFrame(presence_by_asset).sort_index()
    if presence_frame.empty:
        return MissingDataSummary(
            missing_by_asset={asset: [] for asset in assets},
            missing_counts_by_month={asset: {} for asset in assets},
        )
    trimmed = _trim_to_first_complete_row(presence_frame)
    if trimmed.empty:
        raise DataProcessingError(
            "No aligned hourly data available for missing data summary",
            context={"assets": ",".join(assets)},
        )
    combined_index = weekday_only_index(
        combine_hourly_indexes(hourly_ohlc_by_asset)
    )
    expected_days = _expected_daily_index(combined_index)
    expected_days = _trim_expected_days(
        expected_days, trimmed.index.min()
    )
    months = _expected_months(expected_days)
    day_end_by_day = _day_end_by_start(combined_index)
    missing_by_asset: dict[str, list[pd.Timestamp]] = {}
    missing_counts_by_month: dict[str, dict[str, int]] = {}
    for asset in assets:
        asset_presence = trimmed.get(asset, pd.Series(dtype=bool))
        if asset_presence.empty:
            asset_presence = pd.Series(
                False, index=trimmed.index, name=asset
            )
        asset_presence = asset_presence.reindex(
            expected_days, fill_value=False
        )
        missing_days = expected_days[~asset_presence.to_numpy()]
        missing_by_asset[asset] = _missing_day_timestamps(
            missing_days, day_end_by_day
        )
        missing_counts_by_month[asset] = _missing_counts_by_month(
            missing_days, months
        )
    return MissingDataSummary(
        missing_by_asset=missing_by_asset,
        missing_counts_by_month=missing_counts_by_month,
    )


def _build_daily_presence_by_asset(
    hourly_ohlc_by_asset: dict[str, pd.DataFrame],
    assets: list[str],
) -> dict[str, pd.Series]:
    presence_by_asset: dict[str, pd.Series] = {}
    for asset in assets:
        frame = hourly_ohlc_by_asset.get(asset)
        if frame is None or frame.empty:
            presence_by_asset[asset] = pd.Series(dtype=bool, name=asset)
            continue
        index = require_datetime_index(
            frame.index, label=f"hourly ohlc {asset}"
        )
        index = weekday_only_index(index)
        if index.empty:
            presence_by_asset[asset] = pd.Series(dtype=bool, name=asset)
            continue
        day_start = _day_start_index(index)
        grouped = pd.Series(index, index=day_start).groupby(level=0)
        day_keys: list[pd.Timestamp] = []
        day_presence: list[bool] = []
        for day, values in grouped:
            day_keys.append(cast(pd.Timestamp, day))
            day_presence.append(bool(len(values)))
        presence_by_asset[asset] = pd.Series(
            day_presence, index=pd.Index(day_keys), name=asset
        )
    return presence_by_asset


def _trim_to_first_complete_row(
    presence_frame: pd.DataFrame,
) -> pd.DataFrame:
    if presence_frame.empty:
        return presence_frame
    presence = presence_frame.copy()
    presence = presence.where(presence.notna(), False)
    presence = presence.astype(bool, copy=False)
    presence = presence.loc[presence.any(axis=1)]
    if presence.empty:
        return presence
    full_rows = presence.all(axis=1)
    if not full_rows.any():
        return pd.DataFrame()
    first_full_index = full_rows[full_rows].index[0]
    return presence.loc[presence.index >= first_full_index]


def _expected_daily_index(
    index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if index.empty:
        return index
    day_start = index.normalize()
    expected = pd.DatetimeIndex(day_start.unique())
    return expected.sort_values()


def _trim_expected_days(
    expected_days: pd.DatetimeIndex, start: pd.Timestamp | None
) -> pd.DatetimeIndex:
    if expected_days.empty or start is None:
        return expected_days
    start_day = start.normalize()
    return expected_days[expected_days >= start_day]


def _expected_months(index: pd.DatetimeIndex) -> list[str]:
    if index.empty:
        return []
    values = index.tz_convert("UTC").strftime("%Y-%m")
    return list(dict.fromkeys(str(value) for value in values))


def _missing_counts_by_month(
    missing_index: pd.DatetimeIndex, months: list[str]
) -> dict[str, int]:
    if not months:
        return {}
    missing_months = (
        [
            str(value)
            for value in missing_index.tz_convert("UTC").strftime("%Y-%m")
        ]
        if not missing_index.empty
        else []
    )
    return {month: missing_months.count(month) for month in months}


def _week_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    return normalized - offsets


def _day_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.normalize()


def _day_end_by_start(index: pd.DatetimeIndex) -> pd.Series:
    if index.empty:
        return pd.Series(dtype="datetime64[ns]")
    day_start = index.normalize()
    return pd.Series(index, index=day_start).groupby(level=0).max()


def _missing_day_timestamps(
    missing_days: pd.DatetimeIndex, day_end_by_day: pd.Series
) -> list[pd.Timestamp]:
    if missing_days.empty:
        return []
    day_end = day_end_by_day.reindex(missing_days)
    if day_end.isna().any():
        missing = day_end.isna()
        raise DataProcessingError(
            "Missing day end timestamps",
            context={"missing_days": str(list(missing_days[missing]))},
        )
    return [pd.Timestamp(value) for value in day_end]
