from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import DataProcessingError
from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.protocols import (
    FeatureGroup,
    FeatureInputs,
    FeatureOutput,
)
from ..utils import (
    asset_frame,
    load_asset_daily,
    prepare_weekly_daily_inputs,
    require_datetime_index,
    require_no_missing,
    require_weekly_index,
    week_end_by_start,
    week_start_index,
)

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (130,)
SUPPORTED_FEATURES: tuple[str, ...] = (
    "dow_alpha",
    "dow_spread",
)
_WEEKDAY_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4)
_DOW_ALPHA_WEEKDAYS: tuple[int, ...] = (0, 4)
_WEEKDAY_NAMES: dict[int, str] = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
}


@dataclass(frozen=True)
class SeasonalConfig:
    horizons: Sequence[HorizonSpec]
    features: Sequence[str] | None = None


class SeasonalFeatureGroup(FeatureGroup):
    name = "seasonal"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown seasonal features requested"

    def __init__(self, config: SeasonalConfig) -> None:
        self._config = config
        self._missing_weekdays: Mapping[str, Mapping[str, object]] | None = None

    @property
    def missing_weekdays(self) -> Mapping[str, Mapping[str, object]] | None:
        return self._missing_weekdays

    def compute(self, inputs: FeatureInputs) -> FeatureOutput:
        feature_set, weekly_ohlc, daily_ohlc, assets = (
            prepare_weekly_daily_inputs(
                inputs,
                features=self._config.features,
                supported_features=type(self).supported_features,
                error_message=type(self).error_message,
            )
        )
        if not assets:
            return FeatureOutput(frame=pd.DataFrame(), feature_names=[])
        require_no_missing(weekly_ohlc, assets)
        context = _build_context(
            weekly_ohlc, feature_set=feature_set, config=self._config
        )
        self._missing_weekdays = _missing_weekdays_by_asset(
            daily_ohlc, assets
        )
        features_by_asset = _compute_assets(daily_ohlc, assets, context)
        return _build_feature_output(
            features_by_asset, assets, context.weekly_index
        )


@dataclass(frozen=True)
class _SeasonalContext:
    weekly_index: pd.DatetimeIndex
    weekly_starts: pd.DatetimeIndex
    week_end_by_week_start: pd.Series
    horizons: Sequence[HorizonSpec]
    feature_set: set[str]


def _build_context(
    weekly_ohlc: pd.DataFrame,
    *,
    feature_set: set[str],
    config: SeasonalConfig,
) -> _SeasonalContext:
    weekly_index = require_weekly_index(weekly_ohlc)
    weekly_starts = week_start_index(weekly_index)
    week_end_by_week_start = week_end_by_start(weekly_index)
    return _SeasonalContext(
        weekly_index=weekly_index,
        weekly_starts=weekly_starts,
        week_end_by_week_start=week_end_by_week_start,
        horizons=config.horizons,
        feature_set=feature_set,
    )


def _compute_assets(
    daily_ohlc: pd.DataFrame,
    assets: Sequence[str],
    context: _SeasonalContext,
) -> dict[str, pd.DataFrame]:
    features_by_asset: dict[str, pd.DataFrame] = {}
    for asset in assets:
        asset_daily = load_asset_daily(daily_ohlc, asset)
        features_by_asset[asset] = _compute_asset_features(
            asset_daily,
            context=context,
        )
    return features_by_asset


def _build_feature_output(
    features_by_asset: dict[str, pd.DataFrame],
    assets: Sequence[str],
    weekly_index: pd.DatetimeIndex,
) -> FeatureOutput:
    if not features_by_asset:
        return FeatureOutput(frame=pd.DataFrame(index=weekly_index), feature_names=[])
    combined = pd.concat(features_by_asset, axis=1)
    combined.columns = combined.columns.set_names(["asset", "feature"])
    feature_names = list(features_by_asset[assets[0]].columns)
    return FeatureOutput(frame=combined, feature_names=feature_names)


def _compute_asset_features(
    asset_daily: pd.DataFrame,
    *,
    context: _SeasonalContext,
) -> pd.DataFrame:
    weekday_returns = _weekday_returns(asset_daily, context.weekly_starts)
    feature_data: dict[str, np.ndarray] = {}
    for spec in context.horizons:
        if spec.weeks != 26:
            continue
        rolling = _rolling_weekday_means(
            weekday_returns,
            weekly_starts=context.weekly_starts,
            weeks=spec.weeks,
        )
        rolling = _align_to_weekly_index(
            rolling,
            weekly_index=context.weekly_index,
            week_end_by_week_start=context.week_end_by_week_start,
        )
        feature_data.update(
            _horizon_feature_data(
                rolling,
                weeks=spec.weeks,
                weekly_len=len(context.weekly_index),
                feature_set=context.feature_set,
            )
        )
    return pd.DataFrame(feature_data, index=context.weekly_index)


def _require_positive_close(close: pd.Series) -> pd.Series:
    close_values = close.to_numpy(dtype=float)
    if (close_values <= 0).any():
        raise DataProcessingError(
            "daily_ohlc contains non-positive Close values",
            context={"invalid_close": "true"},
        )
    return close.astype(float)


def _log_returns(close: pd.Series) -> pd.Series:
    close = close.sort_index()
    log_close = np.log(close.to_numpy(dtype=float))
    series = pd.Series(log_close, index=close.index)
    return series.diff()


def _weekday_returns(
    asset_daily: pd.DataFrame, weekly_starts: pd.DatetimeIndex
) -> pd.DataFrame:
    if asset_daily.empty:
        return pd.DataFrame(index=weekly_starts)
    close = _require_positive_close(asset_daily["Close"])
    daily_returns = _log_returns(close)
    weekday_returns = _weekday_return_pivot(daily_returns)
    return weekday_returns.reindex(weekly_starts)


def _horizon_feature_data(
    rolling: pd.DataFrame,
    *,
    weeks: int,
    weekly_len: int,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    feature_data: dict[str, np.ndarray] = {}
    if "dow_alpha" in feature_set:
        feature_data.update(
            _dow_alpha_values(rolling, weeks, weekly_len)
        )
    if "dow_spread" in feature_set:
        feature_data[_dow_spread_name(weeks)] = _dow_spread_values(
            rolling, weekly_len
        )
    return feature_data


def _dow_alpha_values(
    rolling: pd.DataFrame, weeks: int, weekly_len: int
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for weekday in _DOW_ALPHA_WEEKDAYS:
        name = _dow_alpha_name(_WEEKDAY_NAMES[weekday], weeks)
        series = rolling.get(weekday)
        if series is None:
            values[name] = np.full(weekly_len, np.nan)
        else:
            values[name] = series.to_numpy(dtype=float)
    return values


def _dow_spread_values(
    rolling: pd.DataFrame, weekly_len: int
) -> np.ndarray:
    if rolling.empty:
        return np.full(weekly_len, np.nan)
    spread = _dow_spread(rolling)
    return spread.to_numpy(dtype=float)


def _weekday_return_pivot(daily_returns: pd.Series) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame()
    index = require_datetime_index(
        daily_returns.index, label="daily_returns"
    )
    weekdays = index.dayofweek
    week_start = week_start_index(index)
    frame = pd.DataFrame(
        {
            "return": daily_returns,
            "weekday": weekdays,
            "week_start": week_start,
        }
    )
    frame = frame[frame["weekday"].isin(_WEEKDAY_ORDER)]
    frame = frame.dropna(subset=["return"])
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index="week_start",
        columns="weekday",
        values="return",
        aggfunc="mean",
    )
    return pivot.reindex(columns=list(_WEEKDAY_ORDER))


def _rolling_weekday_means(
    weekday_returns: pd.DataFrame,
    *,
    weekly_starts: pd.DatetimeIndex,
    weeks: int,
) -> pd.DataFrame:
    if weekday_returns.empty:
        return pd.DataFrame(index=weekly_starts)
    rolling = weekday_returns.rolling(window=weeks, min_periods=1).mean()
    window_counts = (
        pd.Series(1, index=weekly_starts)
        .rolling(window=weeks, min_periods=1)
        .sum()
    )
    return rolling.where(window_counts >= weeks)


def _align_to_weekly_index(
    frame: pd.DataFrame,
    *,
    weekly_index: pd.DatetimeIndex,
    week_end_by_week_start: pd.Series,
) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(weekly_index)
    week_end = week_end_by_week_start.reindex(frame.index)
    if week_end.isna().any():
        raise DataProcessingError(
            "Missing weekly index mapping for seasonal features",
            context={"missing_weeks": str(int(week_end.isna().sum()))},
        )
    aligned = frame.copy()
    aligned.index = pd.DatetimeIndex(week_end.to_numpy())
    return aligned.reindex(weekly_index)


def _dow_spread(rolling: pd.DataFrame) -> pd.Series:
    if rolling.empty:
        return pd.Series(dtype=float)
    counts = rolling.notna().sum(axis=1)
    max_vals = rolling.max(axis=1, skipna=True)
    min_vals = rolling.min(axis=1, skipna=True)
    spread = max_vals - min_vals
    return spread.where(counts >= 2)


def _dow_alpha_name(day: str, weeks: int) -> str:
    return f"dow_alpha_{day}_{weeks}w"


def _dow_spread_name(weeks: int) -> str:
    return f"dow_spread_{weeks}w"


def _missing_weekdays_by_asset(
    daily_ohlc: pd.DataFrame, assets: Sequence[str]
) -> dict[str, dict[str, object]]:
    index = require_datetime_index(daily_ohlc.index, label="daily_ohlc")
    missing_all = _expected_missing_weekdays(index)
    missing_by_asset: dict[str, dict[str, object]] = {}
    for asset in assets:
        asset_missing = _asset_missing_weekdays(daily_ohlc, asset)
        combined = missing_all.union(asset_missing).sort_values()
        missing_by_asset[asset] = _serialize_missing_dates(combined)
    return missing_by_asset


def _expected_missing_weekdays(
    index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if index.empty:
        return pd.DatetimeIndex([], tz=index.tz)
    normalized = _normalize_index(index)
    weekdays = _weekday_only(normalized)
    if weekdays.empty:
        return pd.DatetimeIndex([], tz=index.tz)
    expected = pd.date_range(
        weekdays.min(), weekdays.max(), freq="B", tz=weekdays.tz
    )
    present = pd.DatetimeIndex(weekdays.unique()).sort_values()
    return expected.difference(present)


def _asset_missing_weekdays(
    daily_ohlc: pd.DataFrame, asset: str
) -> pd.DatetimeIndex:
    index = require_datetime_index(daily_ohlc.index, label="daily_ohlc")
    asset_data = asset_frame(daily_ohlc, asset)
    missing_mask = asset_data.isna().all(axis=1)
    if not missing_mask.any():
        return pd.DatetimeIndex([], tz=index.tz)
    asset_missing = pd.DatetimeIndex(asset_data.index[missing_mask])
    return _weekday_only(_normalize_index(asset_missing))


def _normalize_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index.date)
    if index.tz is not None:
        dates = dates.tz_localize(index.tz)
    return dates


def _weekday_only(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index[index.dayofweek < 5]


def _serialize_missing_dates(index: pd.DatetimeIndex) -> dict[str, object]:
    dates = [stamp.date().isoformat() for stamp in index]
    return {"count": len(dates), "dates": dates}
