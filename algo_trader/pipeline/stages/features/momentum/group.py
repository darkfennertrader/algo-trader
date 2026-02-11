from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import talib as ta

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.series import PriceSeries
from ..group_base import WeeklyFeatureGroup
from ..utils import require_ohlc_columns

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (5, 20, 60, 130)
DEFAULT_EPSILON = 1e-8
Z_MOM_REF_WEEKS = 26
SUPPORTED_FEATURES: tuple[str, ...] = (
    "vol_scaled_momentum",
    "ema_spread",
)
Z_MOM_WEEKS: tuple[int, ...] = (4, 12, 26)


@dataclass(frozen=True)
class MomentumConfig:
    horizons: Sequence[HorizonSpec]
    eps: float = DEFAULT_EPSILON
    features: Sequence[str] | None = None


def _compute_asset_features(
    asset_frame: pd.DataFrame,
    config: MomentumConfig,
    feature_set: set[str],
) -> pd.DataFrame:
    require_ohlc_columns(asset_frame)
    price_series = PriceSeries(
        close=asset_frame["Close"].to_numpy(dtype=float),
        high=asset_frame["High"].to_numpy(dtype=float),
        low=asset_frame["Low"].to_numpy(dtype=float),
    )
    horizon_specs = list(config.horizons)
    feature_data: dict[str, np.ndarray] = {}
    feature_data.update(
        _momentum_features(price_series.close, horizon_specs, config, feature_set)
    )
    feature_data.update(
        _ema_spread_features(
            price_series, horizon_specs, config, feature_set
        )
    )
    return pd.DataFrame(feature_data, index=asset_frame.index)


def _weekly_returns(close: np.ndarray) -> np.ndarray:
    if close.size < 2:
        return np.full_like(close, np.nan, dtype=float)
    ratio = close[1:] / close[:-1]
    values = np.log(ratio)
    return np.concatenate((np.array([np.nan]), values))


def _cumulative_return(
    close: np.ndarray,
    weeks: int,
) -> np.ndarray:
    if weeks <= 0 or close.size <= weeks:
        return np.full_like(close, np.nan, dtype=float)
    ratio = close[weeks:] / close[:-weeks]
    values = np.log(ratio)
    padding = np.full(weeks, np.nan)
    return np.concatenate((padding, values))


def _momentum_features(
    close: np.ndarray,
    horizons: Sequence[HorizonSpec],
    config: MomentumConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    needs_momentum = "momentum" in feature_set
    needs_scaled = "vol_scaled_momentum" in feature_set
    if not needs_momentum and not needs_scaled:
        return {}
    weekly_returns = _weekly_returns(close) if needs_scaled else None
    sigma_ref = (
        _rolling_std_sample(weekly_returns, Z_MOM_REF_WEEKS)
        if weekly_returns is not None
        else None
    )
    feature_data: dict[str, np.ndarray] = {}
    for spec in horizons:
        if spec.weeks not in Z_MOM_WEEKS:
            continue
        momentum = _cumulative_return(close, spec.weeks)
        if needs_momentum:
            feature_data[_momentum_name(spec.days, spec.weeks)] = momentum
        if needs_scaled and weekly_returns is not None and sigma_ref is not None:
            scale = sigma_ref * np.sqrt(float(spec.weeks))
            feature_data[_z_momentum_name(spec.days, spec.weeks)] = (
                momentum / (scale + config.eps)
            )
    return feature_data


def _slope_features(
    close: np.ndarray,
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "slope" not in feature_set:
        return {}
    log_close = np.log(close)
    feature_data: dict[str, np.ndarray] = {}
    for spec in _slope_specs(horizons):
        slope = ta.LINEARREG_SLOPE(log_close, timeperiod=spec.weeks)
        feature_data[_slope_name(spec.days, spec.weeks)] = slope
    return feature_data


def _ema_spread_features(
    prices: PriceSeries,
    horizons: Sequence[HorizonSpec],
    config: MomentumConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "ema_spread" not in feature_set:
        return {}
    short, mid, long = _ema_specs(horizons)
    ema_short = ta.EMA(prices.close, timeperiod=short.weeks)
    ema_mid = ta.EMA(prices.close, timeperiod=mid.weeks)
    ema_long = ta.EMA(prices.close, timeperiod=long.weeks)
    atr_long = ta.ATR(
        prices.high, prices.low, prices.close, timeperiod=long.weeks
    )
    denom = atr_long + config.eps
    return {
        _ema_spread_name(short.weeks, long.weeks): (
            ema_short - ema_long
        )
        / denom,
        _ema_spread_name(mid.weeks, long.weeks): (
            ema_mid - ema_long
        )
        / denom,
    }


def _rolling_std_sample(values: np.ndarray, weeks: int) -> np.ndarray:
    if weeks <= 1:
        return np.full_like(values, np.nan, dtype=float)
    series = pd.Series(values, dtype=float)
    return (
        series.rolling(window=weeks, min_periods=weeks)
        .std(ddof=1)
        .to_numpy(dtype=float)
    )


def _slope_specs(horizons: Sequence[HorizonSpec]) -> list[HorizonSpec]:
    return [spec for spec in horizons if spec.weeks > 1]


def _ema_specs(
    horizons: Sequence[HorizonSpec],
) -> tuple[HorizonSpec, HorizonSpec, HorizonSpec]:
    candidates = [spec for spec in horizons if spec.weeks > 1]
    if len(candidates) < 3:
        raise ConfigError(
            "At least three horizons are required for EMA spreads",
            context={"horizons": str([spec.days for spec in horizons])},
        )
    candidates = sorted(candidates, key=lambda item: item.weeks)
    return candidates[-3], candidates[-2], candidates[-1]


def _momentum_name(days: int, weeks: int) -> str:
    return f"mom_{weeks}w"


def _z_momentum_name(days: int, weeks: int) -> str:
    return f"z_mom_{weeks}w"


def _slope_name(days: int, weeks: int) -> str:
    return f"slope_{weeks}w"


def _ema_spread_name(short_weeks: int, long_weeks: int) -> str:
    return f"ema_spread_{short_weeks}w_{long_weeks}w"


class MomentumFeatureGroup(WeeklyFeatureGroup[MomentumConfig]):
    name = "momentum"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown momentum features requested"
    compute_asset = _compute_asset_features
