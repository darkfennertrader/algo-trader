from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import tzinfo as TzInfo
from pathlib import Path
from typing import Iterable, Literal, cast

import numpy as np
import pandas as pd

from algo_trader.domain import DataProcessingError, DataSourceError

ReturnType = Literal["log", "simple"]
YearMonth = tuple[int, int]

logger = logging.getLogger(__name__)


TimeZone = str | TzInfo


@dataclass(frozen=True)
class ReturnsSourceConfig:
    base_dir: Path
    assets: Iterable[str]
    return_type: ReturnType
    start: YearMonth | None = None
    end: YearMonth | None = None
    time_col: str = "Datetime"
    price_col: str = "Close"


class ReturnsSource:
    def __init__(self, config: ReturnsSourceConfig) -> None:
        self._config = config

    def get_returns_frame(self) -> pd.DataFrame:
        asset_data: dict[str, pd.Series] = {}
        for asset in self._config.assets:
            series = self._load_asset_series(asset)
            if series is None:
                logger.warning(
                    "No data found for asset=%s base_dir=%s",
                    asset,
                    self._config.base_dir,
                )
                series = pd.Series(dtype=float, name=asset)
            asset_data[asset] = series

        price_matrix = self._align_and_trim(asset_data)
        if price_matrix.empty:
            raise DataProcessingError(
                "No price data available after alignment",
                context={"assets": ",".join(asset_data.keys())},
            )
        returns = self._compute_returns(price_matrix)
        returns = returns.iloc[1:]
        if returns.index.name is None:
            returns.index.name = self._config.time_col
        return returns

    def _load_asset_series(self, asset: str) -> pd.Series | None:
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
            frames.append(self._read_csv(path))

        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return None

        time_col = self._config.time_col
        price_col = self._config.price_col
        combined = combined.dropna(subset=[time_col])
        combined = combined.set_index(time_col)
        _ensure_datetime_index(combined, time_col)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        combined = self._filter_by_month_range(combined)
        combined = self._resample_daily(combined)

        series = combined[price_col].rename(asset)
        return series

    def _read_csv(self, path: Path) -> pd.DataFrame:
        time_col = self._config.time_col
        price_col = self._config.price_col
        try:
            frame = pd.read_csv(
                path,
                usecols=[time_col, price_col],
                parse_dates=[time_col],
            )
        except ValueError as exc:
            raise DataSourceError(
                "CSV missing required columns",
                context={"path": str(path)},
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

        index = _ensure_datetime_index(frame, self._config.time_col)
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
        daily_index = _ensure_datetime_index(daily, self._config.time_col)
        weekday_mask = (daily_index.dayofweek >= 0) & (
            daily_index.dayofweek <= 4
        )
        daily = daily[weekday_mask]
        daily_index = _ensure_datetime_index(daily, self._config.time_col)
        if daily_index.tz is None:
            frame_index = _ensure_datetime_index(frame, self._config.time_col)
            tz = cast(TimeZone | None, frame_index.tz)
            if tz:
                daily_index = daily_index.tz_localize(tz)
        daily.index = daily_index
        if daily.index.name is None:
            daily.index.name = self._config.time_col
        return daily

    def _align_and_trim(self, asset_data: dict[str, pd.Series]) -> pd.DataFrame:
        price_matrix = pd.DataFrame(asset_data)
        price_matrix = price_matrix.dropna(axis=0, how="all")
        if price_matrix.empty:
            return price_matrix
        valid_mask = ~price_matrix.isna().any(axis=1)
        if valid_mask.any():
            first_valid = valid_mask.idxmax()
            price_matrix = price_matrix.loc[first_valid:]
        return price_matrix

    def _compute_returns(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        prev_price = price_matrix.ffill().shift(1)
        if self._config.return_type == "log":
            returns = _log_frame(price_matrix) - _log_frame(prev_price)
        elif self._config.return_type == "simple":
            returns = (price_matrix / prev_price) - 1
        else:
            raise DataProcessingError(
                "Unknown return type",
                context={"return_type": self._config.return_type},
            )
        returns[price_matrix.isna() | prev_price.isna()] = np.nan
        return returns


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


def _log_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = np.log(frame.to_numpy(dtype=float))
    return pd.DataFrame(data, index=frame.index, columns=frame.columns)
