from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import tzinfo as TzInfo
from pathlib import Path
from typing import Iterable, Literal, cast

import numpy as np
import pandas as pd

from algo_trader.domain import DataProcessingError, DataSourceError

ReturnType = Literal["log", "simple"]
ReturnFrequency = Literal["weekly"]
YearMonth = tuple[int, int]

logger = logging.getLogger(__name__)

_FILE_MONTH_PATTERN = re.compile(
    r"^hist_data_(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])\.csv$"
)

TimeZone = str | TzInfo


@dataclass(frozen=True)
class PriceColumns:
    time_col: str = "Datetime"
    price_col: str = "Close"
    open_col: str = "Open"
    high_col: str = "High"
    low_col: str = "Low"
    close_col: str = "Close"


@dataclass(frozen=True)
class ReturnsSourceConfig:
    base_dir: Path
    assets: Iterable[str]
    return_type: ReturnType
    start: YearMonth | None = None
    end: YearMonth | None = None
    columns: PriceColumns = field(default_factory=PriceColumns)


class ReturnsSource:
    def __init__(self, config: ReturnsSourceConfig) -> None:
        self._config = config
        self._daily_asset_data: dict[str, pd.Series] | None = None
        self._hourly_asset_data: dict[str, pd.Series] | None = None
        self._hourly_ohlc_data: dict[str, pd.DataFrame] | None = None

    def get_returns_frame(self) -> pd.DataFrame:
        returns = self._get_weekly_returns()
        if returns.index.name is None:
            returns.index.name = self._config.columns.time_col
        return returns

    def get_daily_price_series(self) -> dict[str, pd.Series]:
        return self._get_asset_data(resample_daily=True)

    def get_hourly_price_series(self) -> dict[str, pd.Series]:
        return self._get_asset_data(resample_daily=False)

    def get_weekly_ohlc_bundle(
        self,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        asset_data = self._get_hourly_ohlc_data()
        if not asset_data:
            raise DataProcessingError(
                "No OHLC data available",
                context={"assets": ",".join(self._config.assets)},
            )
        expected_weeks = _expected_week_starts(
            asset_data,
            start=self._config.start,
            end=self._config.end,
        )
        if expected_weeks.empty:
            raise DataProcessingError(
                "No OHLC data available after alignment",
                context={"assets": ",".join(asset_data.keys())},
            )
        weekly_ohlc, time_meta = _compute_weekly_ohlc(
            asset_data,
            columns=self._config.columns,
            expected_weeks=expected_weeks,
        )
        combined_index = _combine_hourly_indexes(asset_data)
        combined_index = _weekday_only_index(combined_index)
        week_end_by_week = _week_end_by_start(combined_index)
        weekly_ohlc = _apply_week_end_index(
            weekly_ohlc, week_end_by_week
        )
        time_meta = _apply_week_end_index_by_asset(
            time_meta, week_end_by_week
        )
        if weekly_ohlc.index.name is None:
            weekly_ohlc.index.name = self._config.columns.time_col
        return weekly_ohlc, time_meta

    def _get_asset_data(
        self, *, resample_daily: bool
    ) -> dict[str, pd.Series]:
        cache = (
            self._daily_asset_data
            if resample_daily
            else self._hourly_asset_data
        )
        if cache is not None:
            return cache
        asset_data: dict[str, pd.Series] = {}
        for asset in self._config.assets:
            series = self._load_asset_series(
                asset, resample_daily=resample_daily
            )
            if series is None:
                logger.warning(
                    "No data found for asset=%s base_dir=%s",
                    asset,
                    self._config.base_dir,
                )
                series = pd.Series(dtype=float, name=asset)
            asset_data[asset] = series
        if resample_daily:
            self._daily_asset_data = asset_data
        else:
            self._hourly_asset_data = asset_data
        return asset_data

    def _get_hourly_ohlc_data(self) -> dict[str, pd.DataFrame]:
        if self._hourly_ohlc_data is not None:
            return self._hourly_ohlc_data
        asset_data: dict[str, pd.DataFrame] = {}
        columns = self._config.columns
        ohlc_columns = [
            columns.open_col,
            columns.high_col,
            columns.low_col,
            columns.close_col,
        ]
        for asset in self._config.assets:
            frame = self._load_asset_frame(
                asset, columns=ohlc_columns, resample_daily=False
            )
            if frame is None:
                logger.warning(
                    "No OHLC data found for asset=%s base_dir=%s",
                    asset,
                    self._config.base_dir,
                )
                frame = pd.DataFrame(columns=ohlc_columns)
            asset_data[asset] = frame
        self._hourly_ohlc_data = asset_data
        return asset_data

    def _load_asset_series(
        self, asset: str, *, resample_daily: bool
    ) -> pd.Series | None:
        columns = self._config.columns
        frame = self._load_asset_frame(
            asset, columns=[columns.price_col], resample_daily=resample_daily
        )
        if frame is None or frame.empty:
            return None
        series = frame[columns.price_col].rename(asset)
        return series

    def _load_asset_frame(
        self,
        asset: str,
        *,
        columns: list[str],
        resample_daily: bool,
    ) -> pd.DataFrame | None:
        asset_dir = self._config.base_dir / asset
        if asset_dir.exists() and not asset_dir.is_dir():
            raise DataSourceError(
                "Asset path must be a directory",
                context={"path": str(asset_dir), "asset": asset},
            )
        if not asset_dir.exists():
            return None
        csv_paths = sorted(asset_dir.rglob("*.csv"))
        if not csv_paths:
            return None

        frames: list[pd.DataFrame] = []
        for path in csv_paths:
            frame = self._read_csv(path, columns=columns)
            frame = _filter_frame_to_month(
                frame, path, self._config.columns.time_col, asset
            )
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return None

        time_col = self._config.columns.time_col
        combined = combined.dropna(subset=[time_col])
        combined = combined.set_index(time_col)
        _ensure_datetime_index(combined, time_col)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        combined = self._filter_by_month_range(combined)
        if resample_daily:
            combined = self._resample_daily(combined)
        return combined

    def _read_csv(self, path: Path, *, columns: list[str]) -> pd.DataFrame:
        time_col = self._config.columns.time_col
        usecols = [time_col, *columns]
        unique_cols = list(dict.fromkeys(usecols))
        try:
            frame = pd.read_csv(
                path,
                usecols=unique_cols,
                parse_dates=[time_col],
            )
        except ValueError as exc:
            raise DataSourceError(
                "CSV missing required columns",
                context={"path": str(path), "columns": ",".join(unique_cols)},
            ) from exc
        except Exception as exc:
            raise DataSourceError(
                "Failed to read CSV",
                context={"path": str(path)},
            ) from exc
        return frame

    def _filter_by_month_range(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._config.start is None and self._config.end is None:
            return frame

        index = _ensure_datetime_index(frame, self._config.columns.time_col)
        tzinfo = cast(TimeZone | None, index.tz)
        start_ts = (
            _month_start(self._config.start, tzinfo)
            if self._config.start
            else None
        )
        end_ts = (
            _month_end_exclusive(self._config.end, tzinfo)
            if self._config.end
            else None
        )
        if start_ts is not None:
            frame = frame[frame.index >= start_ts]
        if end_ts is not None:
            frame = frame[frame.index < end_ts]
        return frame

    def _resample_daily(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.sort_index()
        daily = frame.resample("1D", label="left", closed="left").last()
        daily_index = _ensure_datetime_index(
            daily, self._config.columns.time_col
        )
        weekday_mask = (daily_index.dayofweek >= 0) & (
            daily_index.dayofweek <= 4
        )
        daily = daily[weekday_mask]
        daily_index = _ensure_datetime_index(
            daily, self._config.columns.time_col
        )
        if daily_index.tz is None:
            frame_index = _ensure_datetime_index(
                frame, self._config.columns.time_col
            )
            tz = cast(TimeZone | None, frame_index.tz)
            if tz:
                daily_index = daily_index.tz_localize(tz)
        daily.index = daily_index
        if daily.index.name is None:
            daily.index.name = self._config.columns.time_col
        return daily

    def _align_and_trim_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.dropna(axis=0, how="all")
        if frame.empty:
            return frame
        valid_mask = ~frame.isna().any(axis=1)
        if valid_mask.any():
            first_valid = valid_mask.idxmax()
            frame = frame.loc[first_valid:]
        return frame

    def _get_weekly_returns(self) -> pd.DataFrame:
        asset_data = self._get_hourly_ohlc_data()
        if not asset_data:
            raise DataProcessingError(
                "No price data available after alignment",
                context={"assets": ",".join(asset_data.keys())},
            )
        weekly_returns = _weekly_returns_from_ohlc(
            asset_data,
            columns=self._config.columns,
            return_type=self._config.return_type,
        )
        weekly_returns = self._align_and_trim_frame(weekly_returns)
        combined_index = _combine_hourly_indexes(asset_data)
        combined_index = _weekday_only_index(combined_index)
        week_end_by_week = _week_end_by_start(combined_index)
        weekly_returns = _apply_week_end_index(
            weekly_returns, week_end_by_week
        )
        return weekly_returns


def _month_start(month: YearMonth, tzinfo: TimeZone | None) -> pd.Timestamp:
    year, month_value = month
    start = pd.Timestamp(year=year, month=month_value, day=1)
    if tzinfo and start.tzinfo is None:
        start = start.tz_localize(tzinfo)
    return start


def _month_end_exclusive(
    month: YearMonth, tzinfo: TimeZone | None
) -> pd.Timestamp:
    year, month_value = month
    start = pd.Timestamp(year=year, month=month_value, day=1)
    end = (start + pd.offsets.MonthBegin(1)).normalize()
    if tzinfo and end.tzinfo is None:
        end = end.tz_localize(tzinfo)
    return end


def _ensure_datetime_index(
    frame: pd.DataFrame, time_col: str
) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    try:
        index = pd.DatetimeIndex(frame.index)
    except Exception as exc:
        raise DataProcessingError(
            "Index must be datetime",
            context={"column": time_col},
        ) from exc
    frame.index = index
    return index


def _extract_month_from_path(path: Path) -> YearMonth:
    match = _FILE_MONTH_PATTERN.match(path.name)
    if not match:
        raise DataSourceError(
            "Historical CSV filename must match hist_data_YYYY-MM.csv",
            context={"path": str(path)},
        )
    return int(match.group("year")), int(match.group("month"))


def _filter_frame_to_month(
    frame: pd.DataFrame, path: Path, time_col: str, asset: str
) -> pd.DataFrame:
    year, month = _extract_month_from_path(path)
    if time_col not in frame.columns:
        raise DataSourceError(
            "CSV missing time column",
            context={"path": str(path), "column": time_col},
        )
    if frame.empty:
        return frame
    month_mask = (frame[time_col].dt.year == year) & (
        frame[time_col].dt.month == month
    )
    if not month_mask.all():
        dropped = len(frame) - int(month_mask.sum())
        if dropped:
            logger.debug(
                "Found out-of-month rows asset=%s path=%s count=%s",
                asset,
                path,
                dropped,
            )
    return frame[month_mask]


def _weekly_returns_from_ohlc(
    asset_data: dict[str, pd.DataFrame],
    *,
    columns: PriceColumns,
    return_type: ReturnType,
) -> pd.DataFrame:
    returns_by_asset: dict[str, pd.Series] = {}
    for asset, frame in asset_data.items():
        returns_by_asset[asset] = _weekly_return_series_from_ohlc(
            frame,
            columns=columns,
            return_type=return_type,
            name=asset,
        )
    return pd.DataFrame(returns_by_asset)


def _weekly_return_series_from_ohlc(
    frame: pd.DataFrame,
    *,
    columns: PriceColumns,
    return_type: ReturnType,
    name: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    frame = frame.sort_index()
    index = _require_datetime_index(frame.index, label="ohlc returns")
    frame = frame.loc[_weekday_only_index(index)]
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    index = _require_datetime_index(frame.index, label="ohlc returns")
    week_start = _week_start_index(index)
    grouped = frame.groupby(week_start, sort=False)
    week_keys: list[pd.Timestamp] = []
    values: list[float] = []
    for week, group in grouped:
        open_value, _ = _weekly_open_value(
            group, open_col=columns.open_col
        )
        close_value, _ = _weekly_close_value(
            group, close_col=columns.close_col
        )
        week_keys.append(week)
        if np.isnan(open_value) or np.isnan(close_value):
            values.append(float("nan"))
            continue
        ratio = close_value / open_value
        if return_type == "log":
            values.append(float(np.log(ratio)))
        else:
            values.append(float(ratio - 1))
    return pd.Series(values, index=pd.Index(week_keys), name=name)


def _compute_weekly_ohlc(
    asset_data: dict[str, pd.DataFrame],
    *,
    columns: PriceColumns,
    expected_weeks: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    weekly_by_asset: dict[str, pd.DataFrame] = {}
    time_by_asset: dict[str, pd.DataFrame] = {}
    missing_by_asset: dict[str, list[str]] = {}
    for asset, frame in asset_data.items():
        weekly, time_meta = _weekly_ohlc_frame(frame, columns=columns)
        weekly = weekly.reindex(expected_weeks)
        time_meta = time_meta.reindex(expected_weeks)
        missing = weekly.index[weekly.isna().all(axis=1)]
        if not missing.empty:
            missing_by_asset[asset] = [
                stamp.isoformat() for stamp in missing
            ]
        weekly_by_asset[asset] = weekly
        time_by_asset[asset] = time_meta
    if missing_by_asset:
        raise DataProcessingError(
            "Weekly OHLC missing complete weeks",
            context={"missing_weeks_by_asset": str(missing_by_asset)},
        )
    if not weekly_by_asset:
        return pd.DataFrame(), {}
    return pd.concat(weekly_by_asset, axis=1), time_by_asset


def _weekly_ohlc_frame(
    frame: pd.DataFrame, *, columns: PriceColumns
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return _empty_weekly_frames(columns)
    frame = frame.sort_index()
    index = _require_datetime_index(frame.index, label="ohlc")
    frame = frame.loc[_weekday_only_index(index)]
    if frame.empty:
        return _empty_weekly_frames(columns)
    index = _require_datetime_index(frame.index, label="ohlc")
    week_start = _week_start_index(index)
    grouped = frame.groupby(week_start, sort=False)
    week_keys: list[pd.Timestamp] = []
    ohlc_rows: list[dict[str, float]] = []
    time_rows: list[dict[str, pd.Timestamp | None]] = []
    for week, group in grouped:
        week_keys.append(week)
        ohlc_row, time_row = _weekly_ohlc_rows(group, columns)
        ohlc_rows.append(ohlc_row)
        time_rows.append(time_row)
    index = pd.Index(week_keys)
    weekly = pd.DataFrame(ohlc_rows, index=index)
    time_meta = pd.DataFrame(time_rows, index=index)
    return weekly, time_meta


def _weekly_open_value(
    group: pd.DataFrame, *, open_col: str
) -> tuple[float, pd.Timestamp | None]:
    index = _require_datetime_index(group.index, label="weekly ohlc")
    monday_mask = index.dayofweek == 0
    monday_values = group.loc[monday_mask, open_col].dropna()
    if not monday_values.empty:
        time = cast(pd.Timestamp, monday_values.index[0])
        return float(monday_values.iloc[0]), time
    non_null = group[open_col].dropna()
    if non_null.empty:
        return float("nan"), None
    time = cast(pd.Timestamp, non_null.index[0])
    return float(non_null.iloc[0]), time


def _weekly_ohlc_rows(
    group: pd.DataFrame, columns: PriceColumns
) -> tuple[dict[str, float], dict[str, pd.Timestamp | None]]:
    open_value, open_time = _weekly_open_value(
        group, open_col=columns.open_col
    )
    high_value, high_time = _weekly_high_value(
        group, high_col=columns.high_col
    )
    low_value, low_time = _weekly_low_value(
        group, low_col=columns.low_col
    )
    close_value, close_time = _weekly_close_value(
        group, close_col=columns.close_col
    )
    ohlc_row = {
        columns.open_col: open_value,
        columns.high_col: high_value,
        columns.low_col: low_value,
        columns.close_col: close_value,
    }
    time_row = {
        "open_time": open_time,
        "high_time": high_time,
        "low_time": low_time,
        "close_time": close_time,
    }
    return ohlc_row, time_row


def _weekly_close_value(
    group: pd.DataFrame, *, close_col: str
) -> tuple[float, pd.Timestamp | None]:
    index = _require_datetime_index(group.index, label="weekly ohlc")
    friday_mask = index.dayofweek == 4
    friday_values = group.loc[friday_mask, close_col].dropna()
    if not friday_values.empty:
        time = cast(pd.Timestamp, friday_values.index[-1])
        return float(friday_values.iloc[-1]), time
    non_null = group[close_col].dropna()
    if non_null.empty:
        return float("nan"), None
    time = cast(pd.Timestamp, non_null.index[-1])
    return float(non_null.iloc[-1]), time


def _weekly_high_value(
    group: pd.DataFrame, *, high_col: str
) -> tuple[float, pd.Timestamp | None]:
    values = group[high_col].dropna()
    if values.empty:
        return float("nan"), None
    max_value = values.max()
    time = cast(pd.Timestamp, values.idxmax())
    return float(max_value), time


def _weekly_low_value(
    group: pd.DataFrame, *, low_col: str
) -> tuple[float, pd.Timestamp | None]:
    values = group[low_col].dropna()
    if values.empty:
        return float("nan"), None
    min_value = values.min()
    time = cast(pd.Timestamp, values.idxmin())
    return float(min_value), time


def _empty_weekly_frames(
    columns: PriceColumns,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = pd.DataFrame(
        columns=[
            columns.open_col,
            columns.high_col,
            columns.low_col,
            columns.close_col,
        ]
    )
    time_meta = pd.DataFrame(
        columns=["open_time", "high_time", "low_time", "close_time"]
    )
    return weekly, time_meta


def _week_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    return normalized - offsets


def _expected_week_starts(
    asset_data: dict[str, pd.DataFrame],
    *,
    start: YearMonth | None,
    end: YearMonth | None,
) -> pd.DatetimeIndex:
    range_start, range_end = _resolve_week_range(
        asset_data, start=start, end=end
    )
    if range_start is None or range_end is None:
        return pd.DatetimeIndex([])
    if range_start >= range_end:
        return pd.DatetimeIndex([])
    range_end = range_end - pd.Timedelta(1, "ns")
    start_week = _week_start_index(pd.DatetimeIndex([range_start]))[0]
    end_week = _week_start_index(pd.DatetimeIndex([range_end]))[0]
    return pd.date_range(
        start=start_week,
        end=end_week,
        freq="W-MON",
        tz=start_week.tz,
    )


def _resolve_week_range(
    asset_data: dict[str, pd.DataFrame],
    *,
    start: YearMonth | None,
    end: YearMonth | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    indexes = [
        _require_datetime_index(frame.index, label="ohlc")
        for frame in asset_data.values()
        if not frame.empty
    ]
    if not indexes:
        return None, None
    min_ts = min(index.min() for index in indexes)
    max_ts = max(index.max() for index in indexes)
    tzinfo = cast(TimeZone | None, min_ts.tzinfo)
    range_start = _month_start(start, tzinfo) if start else min_ts
    range_end = (
        _month_end_exclusive(end, tzinfo)
        if end
        else max_ts + pd.Timedelta(hours=1)
    )
    return range_start, range_end


def _combine_hourly_indexes(
    asset_data: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    indexes = [
        _require_datetime_index(frame.index, label="ohlc")
        for frame in asset_data.values()
        if not frame.empty
    ]
    if not indexes:
        return pd.DatetimeIndex([])
    combined = indexes[0]
    if len(indexes) > 1:
        combined = combined.append(indexes[1:])
    return pd.DatetimeIndex(combined)


def _weekday_only_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.empty:
        return index
    mask = index.dayofweek <= 4
    return pd.DatetimeIndex(index[mask])


def _week_end_by_start(index: pd.DatetimeIndex) -> pd.Series:
    if index.empty:
        return pd.Series(dtype="datetime64[ns]")
    week_start = _week_start_index(index)
    return pd.Series(index, index=week_start).groupby(level=0).max()


def _apply_week_end_index(
    weekly_returns: pd.DataFrame, week_end_by_week: pd.Series
) -> pd.DataFrame:
    if weekly_returns.empty:
        return weekly_returns
    week_end = week_end_by_week.reindex(weekly_returns.index)
    if week_end.isna().any():
        missing = week_end.isna()
        raise DataProcessingError(
            "Weekly timestamps missing week end values",
            context={"missing_weeks": str(list(weekly_returns.index[missing]))},
        )
    adjusted = weekly_returns.copy()
    adjusted.index = pd.DatetimeIndex(week_end)
    return adjusted


def _apply_week_end_index_by_asset(
    time_meta_by_asset: dict[str, pd.DataFrame],
    week_end_by_week: pd.Series,
) -> dict[str, pd.DataFrame]:
    if not time_meta_by_asset:
        return time_meta_by_asset
    adjusted: dict[str, pd.DataFrame] = {}
    for asset, frame in time_meta_by_asset.items():
        if frame.empty:
            adjusted[asset] = frame
            continue
        week_end = week_end_by_week.reindex(frame.index)
        if week_end.isna().any():
            missing = week_end.isna()
            raise DataProcessingError(
                "Weekly timestamps missing week end values",
                context={
                    "missing_weeks": str(list(frame.index[missing]))
                },
            )
        updated = frame.copy()
        updated["week_start"] = pd.DatetimeIndex(updated.index)
        updated["week_end"] = pd.DatetimeIndex(week_end)
        updated.index = pd.DatetimeIndex(week_end)
        adjusted[asset] = updated
    return adjusted


def _require_datetime_index(
    index: pd.Index, *, label: str
) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    raise DataProcessingError(
        f"{label} index must be datetime",
        context={"index_type": type(index).__name__},
    )
