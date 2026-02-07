from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from algo_trader.pipeline.stages.features import (
    asset_frame,
    ordered_assets,
    require_weekly_index,
    serialize_series_positive,
    weekly_missing_fraction_from_daily,
)


def weekly_goodness_ratios(
    weekly_frame: pd.DataFrame,
    *,
    horizon_days_by_feature: Mapping[str, int],
    trading_days_per_week: int,
) -> dict[str, dict[str, dict[str, str]]]:
    ratios_by_feature: dict[str, dict[str, dict[str, str]]] = {}
    if not horizon_days_by_feature:
        return ratios_by_feature
    weeks_by_feature = _weeks_by_feature(
        horizon_days_by_feature, trading_days_per_week
    )
    ratios_by_feature = {name: {} for name in weeks_by_feature}
    horizons = sorted(set(weeks_by_feature.values()))
    assets = ordered_assets(weekly_frame)
    for asset in assets:
        asset_data = asset_frame(weekly_frame, asset)
        ratios_by_horizon = _weekly_ratios_by_horizon(asset_data, horizons)
        for feature_name, weeks in weeks_by_feature.items():
            ratios_by_feature[feature_name][asset] = serialize_series_positive(
                ratios_by_horizon[weeks]
            )
    return ratios_by_feature


def daily_goodness_ratios(
    daily_frame: pd.DataFrame,
    weekly_frame: pd.DataFrame,
    *,
    horizon_days_by_feature: Mapping[str, int],
    trading_days_per_week: int,
) -> dict[str, dict[str, dict[str, str]]]:
    ratios_by_feature: dict[str, dict[str, dict[str, str]]] = {}
    if not horizon_days_by_feature:
        return ratios_by_feature
    weeks_by_feature = _weeks_by_feature(
        horizon_days_by_feature, trading_days_per_week
    )
    ratios_by_feature = {name: {} for name in weeks_by_feature}
    weekly_index = require_weekly_index(weekly_frame)
    assets = ordered_assets(weekly_frame)
    for asset in assets:
        weekly_missing = weekly_missing_fraction_from_daily(
            daily_frame, asset=asset, weekly_index=weekly_index
        )
        for feature_name, weeks in weeks_by_feature.items():
            ratios_by_feature[feature_name][asset] = serialize_series_positive(
                weekly_missing.rolling(window=weeks, min_periods=weeks).mean()
            )
    return ratios_by_feature


def _weeks_by_feature(
    horizon_days_by_feature: Mapping[str, int],
    trading_days_per_week: int,
) -> dict[str, int]:
    return {
        name: max(1, days // trading_days_per_week)
        for name, days in horizon_days_by_feature.items()
    }


def _weekly_ratios_by_horizon(
    asset_data: pd.DataFrame,
    horizons: Sequence[int],
) -> dict[int, pd.Series]:
    ratios_by_horizon: dict[int, pd.Series] = {}
    valid_mask = ~asset_data.isna().any(axis=1)
    for weeks in horizons:
        counts = valid_mask.astype(float).rolling(
            window=weeks, min_periods=weeks
        ).sum()
        ratios_by_horizon[weeks] = (float(weeks) - counts) / float(weeks)
    return ratios_by_horizon
