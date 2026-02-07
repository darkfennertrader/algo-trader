from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import tzinfo as TzInfo
from pathlib import Path
from typing import cast

import pandas as pd

from algo_trader.domain import DataProcessingError, DataSourceError

YearMonth = tuple[int, int]
TimeZone = str | TzInfo

logger = logging.getLogger(__name__)

_FILE_MONTH_PATTERN = re.compile(
    r"^hist_data_(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])\.csv$"
)


@dataclass(frozen=True)
class AssetLoadRequest:
    base_dir: Path
    asset: str
    columns: list[str]
    resample_daily: bool
    time_col: str
    start: YearMonth | None
    end: YearMonth | None


def load_asset_frame_job(
    request: AssetLoadRequest,
) -> pd.DataFrame | None:
    return load_asset_frame(request)


def load_asset_frame(
    request: AssetLoadRequest,
) -> pd.DataFrame | None:
    asset_dir = request.base_dir / request.asset
    if asset_dir.exists() and not asset_dir.is_dir():
        raise DataSourceError(
            "Asset path must be a directory",
            context={"path": str(asset_dir), "asset": request.asset},
        )
    if not asset_dir.exists():
        return None
    csv_paths = sorted(asset_dir.rglob("*.csv"))
    if not csv_paths:
        return None

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frame = _read_csv_file(
            path, columns=request.columns, time_col=request.time_col
        )
        frame = _filter_frame_to_month(
            frame, path, request.time_col, request.asset
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return None

    combined = combined.dropna(subset=[request.time_col])
    combined = combined.set_index(request.time_col)
    _ensure_datetime_index(combined, request.time_col)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined = _filter_by_month_range(
        combined,
        time_col=request.time_col,
        start=request.start,
        end=request.end,
    )
    if request.resample_daily:
        combined = _resample_daily_frame(
            combined, time_col=request.time_col
        )
    return combined


def month_start(
    month: YearMonth, tzinfo: TimeZone | None
) -> pd.Timestamp:
    year, month_value = month
    start = pd.Timestamp(year=year, month=month_value, day=1)
    if tzinfo and start.tzinfo is None:
        start = start.tz_localize(tzinfo)
    return start


def month_end_exclusive(
    month: YearMonth, tzinfo: TimeZone | None
) -> pd.Timestamp:
    year, month_value = month
    start = pd.Timestamp(year=year, month=month_value, day=1)
    end = (start + pd.offsets.MonthBegin(1)).normalize()
    if tzinfo and end.tzinfo is None:
        end = end.tz_localize(tzinfo)
    return end


def _read_csv_file(
    path: Path, *, columns: list[str], time_col: str
) -> pd.DataFrame:
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


def _filter_by_month_range(
    frame: pd.DataFrame,
    *,
    time_col: str,
    start: YearMonth | None,
    end: YearMonth | None,
) -> pd.DataFrame:
    if start is None and end is None:
        return frame

    index = _ensure_datetime_index(frame, time_col)
    tzinfo = cast(TimeZone | None, index.tz)
    start_ts = month_start(start, tzinfo) if start else None
    end_ts = month_end_exclusive(end, tzinfo) if end else None
    if start_ts is not None:
        frame = frame[frame.index >= start_ts]
    if end_ts is not None:
        frame = frame[frame.index < end_ts]
    return frame


def _resample_daily_frame(
    frame: pd.DataFrame, *, time_col: str
) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_index()
    daily = frame.resample("1D", label="left", closed="left").last()
    daily_index = _ensure_datetime_index(daily, time_col)
    weekday_mask = (daily_index.dayofweek >= 0) & (
        daily_index.dayofweek <= 4
    )
    daily = daily[weekday_mask]
    daily_index = _ensure_datetime_index(daily, time_col)
    if daily_index.tz is None:
        frame_index = _ensure_datetime_index(frame, time_col)
        tz = cast(TimeZone | None, frame_index.tz)
        if tz:
            daily_index = daily_index.tz_localize(tz)
    daily.index = daily_index
    if daily.index.name is None:
        daily.index.name = time_col
    return daily


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
