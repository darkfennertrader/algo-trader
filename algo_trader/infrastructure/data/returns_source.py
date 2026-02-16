from __future__ import annotations

import logging
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import tzinfo as TzInfo
from pathlib import Path
from typing import Iterable, Literal, cast

import numpy as np
import pandas as pd

from algo_trader.domain import DataProcessingError
from .asset_loading import (
    AssetLoadRequest,
    load_asset_frame,
    load_asset_frame_job,
    month_end_exclusive,
    month_start,
)
from .indexing import (
    combine_hourly_indexes,
    require_datetime_index,
    weekday_only_index,
)

ReturnType = Literal["log", "simple"]
ReturnFrequency = Literal["weekly"]
YearMonth = tuple[int, int]

logger = logging.getLogger(__name__)

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
    workers: int | None = None


class ReturnsSource:
    def __init__(self, config: ReturnsSourceConfig) -> None:
        self._config = config
        self._assets = list(config.assets)
        self._worker_count = _resolve_worker_count(
            len(self._assets), config.workers
        )
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
                context={"assets": ",".join(self._assets)},
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
        combined_index = combine_hourly_indexes(asset_data)
        combined_index = weekday_only_index(combined_index)
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

    def get_daily_ohlc_frame(self) -> pd.DataFrame:
        asset_data = self._get_hourly_ohlc_data()
        if not asset_data:
            raise DataProcessingError(
                "No OHLC data available",
                context={"assets": ",".join(self._assets)},
            )
        daily_by_asset: dict[str, pd.DataFrame] = {}
        for asset, frame in asset_data.items():
            daily_by_asset[asset] = _daily_ohlc_frame(
                frame, columns=self._config.columns
            )
        if not daily_by_asset:
            return pd.DataFrame()
        daily_ohlc = pd.concat(daily_by_asset, axis=1)
        combined_index = combine_hourly_indexes(asset_data)
        combined_index = weekday_only_index(combined_index)
        day_end_by_day = _day_end_by_start(combined_index)
        daily_ohlc = _apply_day_end_index(daily_ohlc, day_end_by_day)
        if daily_ohlc.index.name is None:
            daily_ohlc.index.name = self._config.columns.time_col
        return daily_ohlc

    def get_hourly_ohlc_data(self) -> dict[str, pd.DataFrame]:
        return self._get_hourly_ohlc_data()

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
        if self._worker_count > 1 and len(self._assets) > 1:
            asset_data = self._load_asset_series_parallel(
                resample_daily=resample_daily
            )
        else:
            asset_data = self._load_asset_series_sequential(
                resample_daily=resample_daily
            )
        if resample_daily:
            self._daily_asset_data = asset_data
        else:
            self._hourly_asset_data = asset_data
        return asset_data

    def _get_hourly_ohlc_data(self) -> dict[str, pd.DataFrame]:
        if self._hourly_ohlc_data is not None:
            return self._hourly_ohlc_data
        columns = self._config.columns
        ohlc_columns = [
            columns.open_col,
            columns.high_col,
            columns.low_col,
            columns.close_col,
        ]
        if self._worker_count > 1 and len(self._assets) > 1:
            asset_data = self._load_ohlc_parallel(
                columns=ohlc_columns
            )
        else:
            asset_data = self._load_ohlc_sequential(
                columns=ohlc_columns
            )
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
        return load_asset_frame(
            AssetLoadRequest(
                base_dir=self._config.base_dir,
                asset=asset,
                columns=columns,
                resample_daily=resample_daily,
                time_col=self._config.columns.time_col,
                start=self._config.start,
                end=self._config.end,
            )
        )

    def _load_asset_series_sequential(
        self, *, resample_daily: bool
    ) -> dict[str, pd.Series]:
        asset_data: dict[str, pd.Series] = {}
        for asset in self._assets:
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
        return asset_data

    def _load_asset_series_parallel(
        self, *, resample_daily: bool
    ) -> dict[str, pd.Series]:
        price_col = self._config.columns.price_col
        futures_by_asset: dict[str, Future[pd.DataFrame | None]] = {}
        with ProcessPoolExecutor(
            max_workers=self._worker_count
        ) as executor:
            for asset in self._assets:
                request = AssetLoadRequest(
                    base_dir=self._config.base_dir,
                    asset=asset,
                    columns=[price_col],
                    resample_daily=resample_daily,
                    time_col=self._config.columns.time_col,
                    start=self._config.start,
                    end=self._config.end,
                )
                futures_by_asset[asset] = executor.submit(
                    load_asset_frame_job,
                    request,
                )
            results = self._collect_parallel_results(
                futures_by_asset,
                error_message="Failed to load asset data",
            )
        asset_data: dict[str, pd.Series] = {}
        for asset in self._assets:
            frame = results.get(asset)
            if frame is None or frame.empty:
                logger.warning(
                    "No data found for asset=%s base_dir=%s",
                    asset,
                    self._config.base_dir,
                )
                series = pd.Series(dtype=float, name=asset)
            else:
                series = frame[price_col].rename(asset)
            asset_data[asset] = series
        return asset_data

    def _load_ohlc_sequential(
        self, *, columns: list[str]
    ) -> dict[str, pd.DataFrame]:
        asset_data: dict[str, pd.DataFrame] = {}
        for asset in self._assets:
            frame = self._load_asset_frame(
                asset, columns=columns, resample_daily=False
            )
            if frame is None:
                logger.warning(
                    "No OHLC data found for asset=%s base_dir=%s",
                    asset,
                    self._config.base_dir,
                )
                frame = pd.DataFrame(columns=columns)
            asset_data[asset] = frame
        return asset_data

    def _load_ohlc_parallel(
        self, *, columns: list[str]
    ) -> dict[str, pd.DataFrame]:
        futures_by_asset: dict[str, Future[pd.DataFrame | None]] = {}
        with ProcessPoolExecutor(
            max_workers=self._worker_count
        ) as executor:
            for asset in self._assets:
                request = AssetLoadRequest(
                    base_dir=self._config.base_dir,
                    asset=asset,
                    columns=columns,
                    resample_daily=False,
                    time_col=self._config.columns.time_col,
                    start=self._config.start,
                    end=self._config.end,
                )
                futures_by_asset[asset] = executor.submit(
                    load_asset_frame_job,
                    request,
                )
            results = self._collect_parallel_results(
                futures_by_asset,
                error_message="Failed to load OHLC data",
            )
        asset_data: dict[str, pd.DataFrame] = {}
        for asset in self._assets:
            frame = results.get(asset)
            if frame is None:
                logger.warning(
                    "No OHLC data found for asset=%s base_dir=%s",
                    asset,
                    self._config.base_dir,
                )
                frame = pd.DataFrame(columns=columns)
            asset_data[asset] = frame
        return asset_data

    def _collect_parallel_results(
        self,
        futures_by_asset: dict[str, Future[pd.DataFrame | None]],
        *,
        error_message: str,
    ) -> dict[str, pd.DataFrame | None]:
        results: dict[str, pd.DataFrame | None] = {}
        for asset in self._assets:
            future = futures_by_asset[asset]
            try:
                results[asset] = future.result()
            except Exception as exc:
                for pending in futures_by_asset.values():
                    pending.cancel()
                raise DataProcessingError(
                    error_message,
                    context={"asset": asset},
                ) from exc
        return results

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
        expected_weeks = _expected_week_starts(
            asset_data,
            start=self._config.start,
            end=self._config.end,
        )
        if not expected_weeks.empty and not weekly_returns.empty:
            before_rows = len(weekly_returns)
            mask = weekly_returns.index.isin(expected_weeks)
            weekly_returns = weekly_returns.loc[mask]
            after_rows = len(weekly_returns)
            if after_rows != before_rows:
                logger.info(
                    "Trimmed weekly returns to expected weeks rows_before=%s rows_after=%s",
                    before_rows,
                    after_rows,
                )
        weekly_returns = self._align_and_trim_frame(weekly_returns)
        combined_index = combine_hourly_indexes(asset_data)
        combined_index = weekday_only_index(combined_index)
        week_end_by_week = _week_end_by_start(combined_index)
        weekly_returns = _apply_week_end_index(
            weekly_returns, week_end_by_week
        )
        return weekly_returns


def _resolve_worker_count(asset_count: int, requested: int | None) -> int:
    if asset_count <= 1:
        return 1
    if requested is None:
        return 1
    return max(1, min(asset_count, requested))


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
    index = require_datetime_index(frame.index, label="ohlc returns")
    frame = frame.loc[weekday_only_index(index)]
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    index = require_datetime_index(frame.index, label="ohlc returns")
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
                _format_week_label(stamp) for stamp in missing
            ]
        weekly_by_asset[asset] = weekly
        time_by_asset[asset] = time_meta
    if missing_by_asset:
        raise DataProcessingError(
            "Weekly OHLC missing complete weeks",
            context={
                "missing_weeks_by_asset (start week)": str(missing_by_asset)
            },
        )
    if not weekly_by_asset:
        return pd.DataFrame(), {}
    return pd.concat(weekly_by_asset, axis=1), time_by_asset


def _format_week_label(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d")


def _weekly_ohlc_frame(
    frame: pd.DataFrame, *, columns: PriceColumns
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return _empty_weekly_frames(columns)
    frame = frame.sort_index()
    index = require_datetime_index(frame.index, label="ohlc")
    frame = frame.loc[weekday_only_index(index)]
    if frame.empty:
        return _empty_weekly_frames(columns)
    index = require_datetime_index(frame.index, label="ohlc")
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


def _daily_ohlc_frame(
    frame: pd.DataFrame, *, columns: PriceColumns
) -> pd.DataFrame:
    if frame.empty:
        return _empty_daily_frame(columns)
    frame = frame.sort_index()
    index = require_datetime_index(frame.index, label="daily ohlc")
    frame = frame.loc[weekday_only_index(index)]
    if frame.empty:
        return _empty_daily_frame(columns)
    index = require_datetime_index(frame.index, label="daily ohlc")
    day_start = _day_start_index(index)
    grouped = frame.groupby(day_start, sort=False)
    day_keys: list[pd.Timestamp] = []
    ohlc_rows: list[dict[str, float]] = []
    for day, group in grouped:
        day_keys.append(day)
        ohlc_rows.append(_daily_ohlc_row(group, columns))
    return pd.DataFrame(ohlc_rows, index=pd.Index(day_keys))


def _daily_ohlc_row(
    group: pd.DataFrame, columns: PriceColumns
) -> dict[str, float]:
    open_value = _daily_open_value(group, open_col=columns.open_col)
    close_value = _daily_close_value(group, close_col=columns.close_col)
    high_value = _daily_high_value(group, high_col=columns.high_col)
    low_value = _daily_low_value(group, low_col=columns.low_col)
    return {
        columns.open_col: open_value,
        columns.high_col: high_value,
        columns.low_col: low_value,
        columns.close_col: close_value,
    }


def _daily_open_value(group: pd.DataFrame, *, open_col: str) -> float:
    values = group[open_col].dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def _daily_close_value(group: pd.DataFrame, *, close_col: str) -> float:
    values = group[close_col].dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _daily_high_value(group: pd.DataFrame, *, high_col: str) -> float:
    values = group[high_col].dropna()
    if values.empty:
        return float("nan")
    return float(values.max())


def _daily_low_value(group: pd.DataFrame, *, low_col: str) -> float:
    values = group[low_col].dropna()
    if values.empty:
        return float("nan")
    return float(values.min())


def _weekly_open_value(
    group: pd.DataFrame, *, open_col: str
) -> tuple[float, pd.Timestamp | None]:
    index = require_datetime_index(group.index, label="weekly ohlc")
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
    index = require_datetime_index(group.index, label="weekly ohlc")
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


def _empty_daily_frame(columns: PriceColumns) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            columns.open_col,
            columns.high_col,
            columns.low_col,
            columns.close_col,
        ]
    )


def _week_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = index.normalize()
    offsets = pd.to_timedelta(normalized.dayofweek, unit="D")
    return normalized - offsets


def _day_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.normalize()


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
    start_week = _start_week_for_range(range_start, skip_partial=True)
    end_week = _week_start_index(pd.DatetimeIndex([range_end]))[0]
    if start_week > end_week:
        return pd.DatetimeIndex([])
    return pd.date_range(
        start=start_week,
        end=end_week,
        freq="W-MON",
        tz=start_week.tz,
    )


def _start_week_for_range(
    range_start: pd.Timestamp, *, skip_partial: bool
) -> pd.Timestamp:
    if not skip_partial:
        return _week_start_index(pd.DatetimeIndex([range_start]))[0]
    normalized = range_start.normalize()
    if normalized.dayofweek == 0:
        return normalized
    return normalized + pd.Timedelta(days=7 - normalized.dayofweek)


def _resolve_week_range(
    asset_data: dict[str, pd.DataFrame],
    *,
    start: YearMonth | None,
    end: YearMonth | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    indexes = [
        require_datetime_index(frame.index, label="ohlc")
        for frame in asset_data.values()
        if not frame.empty
    ]
    if not indexes:
        return None, None
    min_ts = min(index.min() for index in indexes)
    max_ts = max(index.max() for index in indexes)
    tzinfo = cast(TimeZone | None, min_ts.tzinfo)
    range_start = month_start(start, tzinfo) if start else min_ts
    range_end = (
        month_end_exclusive(end, tzinfo)
        if end
        else max_ts + pd.Timedelta(hours=1)
    )
    return range_start, range_end


def _week_end_by_start(index: pd.DatetimeIndex) -> pd.Series:
    if index.empty:
        return pd.Series(dtype="datetime64[ns]")
    week_start = _week_start_index(index)
    return pd.Series(index, index=week_start).groupby(level=0).max()


def _day_end_by_start(index: pd.DatetimeIndex) -> pd.Series:
    if index.empty:
        return pd.Series(dtype="datetime64[ns]")
    day_start = _day_start_index(index)
    return pd.Series(index, index=day_start).groupby(level=0).max()


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

def _apply_day_end_index(
    daily_ohlc: pd.DataFrame, day_end_by_day: pd.Series
) -> pd.DataFrame:
    if daily_ohlc.empty:
        return daily_ohlc
    day_end = day_end_by_day.reindex(daily_ohlc.index)
    if day_end.isna().any():
        missing = day_end.isna()
        raise DataProcessingError(
            "Daily timestamps missing day end values",
            context={"missing_days": str(list(daily_ohlc.index[missing]))},
        )
    adjusted = daily_ohlc.copy()
    adjusted.index = pd.DatetimeIndex(day_end)
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
