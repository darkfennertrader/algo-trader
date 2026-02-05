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
