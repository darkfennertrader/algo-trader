from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar

import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError
from .protocols import FeatureInputs, FeatureOutput


def ordered_assets(frame: pd.DataFrame) -> list[str]:
    if not isinstance(frame.columns, pd.MultiIndex):
        return []
    assets: list[str] = []
    for asset in frame.columns.get_level_values(0):
        if asset not in assets:
            assets.append(str(asset))
    return assets


def reindex_asset_features(
    frame: pd.DataFrame,
    assets: Sequence[str],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    return frame.reindex(
        columns=pd.MultiIndex.from_product(
            [
                pd.Index(assets, name="asset"),
                pd.Index(feature_names, name="feature"),
            ],
            names=["asset", "feature"],
        )
    )


def asset_frame(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    asset_slice = frame.xs(asset, axis=1, level=0, drop_level=True)
    if isinstance(asset_slice, pd.Series):
        return asset_slice.to_frame()
    return asset_slice


def require_weekly_ohlc(inputs: FeatureInputs) -> pd.DataFrame:
    frame = inputs.frames.get("weekly_ohlc")
    if frame is None:
        raise DataProcessingError(
            "weekly_ohlc input is required",
            context={"inputs": ",".join(inputs.frames.keys())},
        )
    if not isinstance(frame.columns, pd.MultiIndex):
        raise DataProcessingError(
            "weekly_ohlc must have multi-index columns",
            context={"columns_type": type(frame.columns).__name__},
        )
    if frame.columns.nlevels < 2:
        raise DataProcessingError(
            "weekly_ohlc must include asset and field columns",
            context={"column_levels": str(frame.columns.nlevels)},
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataProcessingError(
            "weekly_ohlc index must be datetime",
            context={"index_type": type(frame.index).__name__},
        )
    return frame


def require_daily_ohlc(inputs: FeatureInputs) -> pd.DataFrame:
    frame = inputs.frames.get("daily_ohlc")
    if frame is None:
        raise DataProcessingError(
            "daily_ohlc input is required",
            context={"inputs": ",".join(inputs.frames.keys())},
        )
    if not isinstance(frame.columns, pd.MultiIndex):
        raise DataProcessingError(
            "daily_ohlc must have multi-index columns",
            context={"columns_type": type(frame.columns).__name__},
        )
    if frame.columns.nlevels < 2:
        raise DataProcessingError(
            "daily_ohlc must include asset and field columns",
            context={"column_levels": str(frame.columns.nlevels)},
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataProcessingError(
            "daily_ohlc index must be datetime",
            context={"index_type": type(frame.index).__name__},
        )
    return frame


def require_no_missing(frame: pd.DataFrame, assets: Iterable[str]) -> None:
    missing_counts: dict[str, int] = {}
    for asset in assets:
        asset_data = asset_frame(frame, asset)
        missing = int(asset_data.isna().sum().sum())
        if missing:
            missing_counts[asset] = missing
    if missing_counts:
        raise DataProcessingError(
            "Weekly OHLC contains missing values",
            context={"missing_by_asset": str(missing_counts)},
        )


def require_ohlc_columns(frame: pd.DataFrame, *, label: str = "weekly_ohlc") -> None:
    required = ["Open", "High", "Low", "Close"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise DataProcessingError(
            f"{label} missing required columns",
            context={"columns": ",".join(missing)},
        )


def require_datetime_index(
    index: pd.Index, *, label: str
) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            f"{label} index must be datetime",
            context={"index_type": type(index).__name__},
        )
    return index


def require_weekly_index(
    frame: pd.DataFrame, *, label: str = "weekly_ohlc"
) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataProcessingError(
            f"{label} index must be datetime",
            context={"index_type": type(frame.index).__name__},
        )
    return frame.index


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
    week_end_by_week_start: pd.Series | None = None,
    weekly_index: pd.DatetimeIndex | None = None,
    *,
    aggregator: str = "last",
    label: str = "daily_ohlc",
) -> pd.Series:
    if weekly_index is None:
        raise DataProcessingError(
            "weekly_index is required for weekly aggregation",
            context={"label": label},
        )
    if series.empty:
        return pd.Series(index=weekly_index, dtype=float)
    series = series.sort_index()
    series_index = require_datetime_index(series.index, label=label)
    week_start = week_start_index(series_index)
    grouped = series.groupby(week_start, sort=False)
    if aggregator == "last":
        weekly = grouped.last()
    elif aggregator == "mean":
        weekly = grouped.mean()
    else:
        raise DataProcessingError(
            "Unknown weekly aggregation",
            context={"aggregator": aggregator},
        )
    if week_end_by_week_start is None:
        week_end_by_week_start = week_end_by_start(weekly_index)
    week_end = week_end_by_week_start.reindex(weekly.index)
    mask = week_end.notna()
    weekly = weekly[mask]
    week_end = week_end[mask]
    weekly.index = pd.DatetimeIndex(week_end)
    return weekly.reindex(weekly_index)


def daily_series_to_weekly_mean(
    series: pd.Series, *, weekly_index: pd.DatetimeIndex
) -> pd.Series:
    return to_weekly(
        series,
        weekly_index=weekly_index,
        aggregator="mean",
    )


def weekly_missing_fraction_from_daily(
    daily_ohlc: pd.DataFrame,
    *,
    asset: str,
    weekly_index: pd.DatetimeIndex,
) -> pd.Series:
    asset_daily = load_asset_daily(daily_ohlc, asset)
    valid_mask = ~asset_daily.isna().any(axis=1)
    missing_by_day = 1.0 - valid_mask.astype(float)
    return daily_series_to_weekly_mean(
        missing_by_day, weekly_index=weekly_index
    )


def load_asset_daily(
    daily_ohlc: pd.DataFrame,
    asset: str,
    *,
    label: str = "daily_ohlc",
) -> pd.DataFrame:
    if asset not in daily_ohlc.columns.get_level_values(0):
        raise DataProcessingError(
            f"{label} missing asset",
            context={"asset": asset},
        )
    asset_daily = asset_frame(daily_ohlc, asset)
    require_ohlc_columns(asset_daily, label=label)
    return asset_daily


def normalize_feature_set(
    features: Sequence[str] | None,
    supported: Sequence[str],
    *,
    error_message: str,
) -> set[str]:
    if not features:
        return set(supported)
    normalized = {item.strip().lower() for item in features}
    unknown = sorted(normalized.difference(supported))
    if unknown:
        raise ConfigError(
            error_message,
            context={"features": ",".join(unknown)},
        )
    return normalized


def prepare_weekly_daily_inputs(
    inputs: FeatureInputs,
    *,
    features: Sequence[str] | None,
    supported_features: Sequence[str],
    error_message: str,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame, Sequence[str]]:
    feature_set = normalize_feature_set(
        features,
        supported_features,
        error_message=error_message,
    )
    weekly_ohlc = require_weekly_ohlc(inputs)
    daily_ohlc = require_daily_ohlc(inputs)
    assets = ordered_assets(weekly_ohlc)
    return feature_set, weekly_ohlc, daily_ohlc, assets


ConfigT = TypeVar("ConfigT")


def compute_weekly_features(
    inputs: FeatureInputs,
    *,
    compute_asset: Callable[[pd.DataFrame], pd.DataFrame],
) -> FeatureOutput:
    weekly = require_weekly_ohlc(inputs)
    assets = ordered_assets(weekly)
    if not assets:
        return FeatureOutput(frame=pd.DataFrame(), feature_names=[])
    require_no_missing(weekly, assets)
    features_by_asset: dict[str, pd.DataFrame] = {}
    for asset in assets:
        asset_data = asset_frame(weekly, asset)
        features_by_asset[asset] = compute_asset(asset_data)
    combined = pd.concat(features_by_asset, axis=1)
    combined.columns = combined.columns.set_names(["asset", "feature"])
    feature_names = list(features_by_asset[assets[0]].columns)
    return FeatureOutput(frame=combined, feature_names=feature_names)


def compute_weekly_group_features(
    inputs: FeatureInputs,
    *,
    config: ConfigT,
    supported_features: Sequence[str],
    error_message: str,
    compute_asset: Callable[[pd.DataFrame, ConfigT, set[str]], pd.DataFrame],
) -> FeatureOutput:
    feature_set = normalize_feature_set(
        getattr(config, "features", None),
        supported_features,
        error_message=error_message,
    )
    return compute_weekly_features(
        inputs,
        compute_asset=lambda asset_data: compute_asset(
            asset_data, config, feature_set
        ),
    )


def serialize_series(series: pd.Series) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    index = require_datetime_index(series.index, label="feature_series")
    values = series.to_numpy(dtype=float)
    for stamp, value in zip(index, values, strict=False):
        key = stamp.isoformat(timespec="seconds").replace("T", "_")
        if pd.isna(value):
            payload[key] = None
        else:
            payload[key] = float(value)
    return payload


def serialize_series_positive(series: pd.Series) -> dict[str, float]:
    payload: dict[str, float] = {}
    index = require_datetime_index(series.index, label="feature_series")
    values = series.to_numpy(dtype=float)
    for stamp, value in zip(index, values, strict=False):
        if pd.isna(value) or value <= 0:
            continue
        key = stamp.isoformat(timespec="seconds").replace("T", "_")
        payload[key] = float(value)
    return payload
