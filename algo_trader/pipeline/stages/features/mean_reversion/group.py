from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError
from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.series import PriceSeries
from ..group_base import WeeklyFeatureGroup
from ..utils import require_ohlc_columns

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (5, 20, 60, 130)
DEFAULT_EPSILON = 1e-8
SUPPORTED_FEATURES: tuple[str, ...] = (
    "z_price_ema",
    "z_price_med",
    "donch_pos",
    "rsi_centered",
    "rev",
    "shock",
    "range_pos",
    "range_z",
)
Z_PRICE_EMA_WEEKS: tuple[int, ...] = (12, 26)
Z_PRICE_MED_WEEKS: tuple[int, ...] = (26,)
DONCH_POS_WEEKS: tuple[int, ...] = (4,)
RSI_CENTERED_WEEKS: tuple[int, ...] = (4,)
REV_WEEKS: tuple[int, ...] = (1,)
RANGE_Z_WEEKS: tuple[int, ...] = (12,)
SHOCK_WEEKS = 4


@dataclass(frozen=True)
class MeanReversionConfig:
    horizons: Sequence[HorizonSpec]
    eps: float = DEFAULT_EPSILON
    features: Sequence[str] | None = None


def _require_positive_close(close: np.ndarray) -> None:
    if np.any(close <= 0):
        raise DataProcessingError(
            "weekly_ohlc contains non-positive Close values",
            context={"invalid_close": "true"},
        )


def _compute_asset_features(
    asset_frame: pd.DataFrame,
    config: MeanReversionConfig,
    feature_set: set[str],
) -> pd.DataFrame:
    require_ohlc_columns(asset_frame)
    close = asset_frame["Close"].to_numpy(dtype=float)
    high = asset_frame["High"].to_numpy(dtype=float)
    low = asset_frame["Low"].to_numpy(dtype=float)
    prices = PriceSeries(close=close, high=high, low=low)
    _require_positive_close(close)
    log_close = np.log(close)
    log_series = pd.Series(log_close, index=asset_frame.index)
    feature_data: dict[str, np.ndarray] = {}
    horizon_specs = list(config.horizons)
    feature_data.update(
        _z_price_ema_features(log_series, horizon_specs, config, feature_set)
    )
    feature_data.update(
        _z_price_med_features(log_series, horizon_specs, config, feature_set)
    )
    feature_data.update(
        _donch_pos_features(prices, horizon_specs, config, feature_set)
    )
    feature_data.update(
        _rsi_centered_features(close, horizon_specs, feature_set)
    )
    feature_data.update(
        _return_reversal_features(log_series, horizon_specs, feature_set)
    )
    feature_data.update(
        _shock_features(log_series, horizon_specs, config, feature_set)
    )
    feature_data.update(
        _range_pos_features(prices, config, feature_set)
    )
    feature_data.update(
        _range_z_features(prices, horizon_specs, config, feature_set)
    )
    return pd.DataFrame(feature_data, index=asset_frame.index)


def _z_price_ema_features(
    log_series: pd.Series,
    horizons: Sequence[HorizonSpec],
    config: MeanReversionConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "z_price_ema" not in feature_set:
        return {}
    specs = _select_weeks(horizons, Z_PRICE_EMA_WEEKS, "z_price_ema")
    feature_data: dict[str, np.ndarray] = {}
    for spec in specs:
        ema = _ewm_mean(log_series, spec.weeks)
        vol = _ewm_std(log_series, spec.weeks)
        z_score = (log_series - ema) / (vol + config.eps)
        feature_data[_z_price_ema_name(spec.weeks)] = z_score.to_numpy(
            dtype=float
        )
    return feature_data


def _z_price_med_features(
    log_series: pd.Series,
    horizons: Sequence[HorizonSpec],
    config: MeanReversionConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "z_price_med" not in feature_set:
        return {}
    specs = _select_weeks(horizons, Z_PRICE_MED_WEEKS, "z_price_med")
    feature_data: dict[str, np.ndarray] = {}
    for spec in specs:
        median = log_series.rolling(
            window=spec.weeks, min_periods=spec.weeks
        ).median()
        deviation = (log_series - median).abs()
        mad = deviation.rolling(
            window=spec.weeks, min_periods=spec.weeks
        ).median()
        scale = 1.4826 * mad
        z_score = (log_series - median) / (scale + config.eps)
        feature_data[_z_price_med_name(spec.weeks)] = z_score.to_numpy(
            dtype=float
        )
    return feature_data


def _donch_pos_features(
    prices: PriceSeries,
    horizons: Sequence[HorizonSpec],
    config: MeanReversionConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "donch_pos" not in feature_set:
        return {}
    specs = _select_weeks(horizons, DONCH_POS_WEEKS, "donch_pos")
    high_series = pd.Series(prices.high)
    low_series = pd.Series(prices.low)
    feature_data: dict[str, np.ndarray] = {}
    for spec in specs:
        high_roll = high_series.rolling(
            window=spec.weeks, min_periods=spec.weeks
        ).max()
        low_roll = low_series.rolling(
            window=spec.weeks, min_periods=spec.weeks
        ).min()
        denom = high_roll - low_roll
        pos = (prices.close - low_roll.to_numpy(dtype=float)) / (
            denom.to_numpy(dtype=float) + config.eps
        )
        pos = np.where(denom.to_numpy(dtype=float) > 0, pos, np.nan)
        feature_data[_donch_pos_name(spec.weeks)] = pos
    return feature_data


def _rsi_centered_features(
    close: np.ndarray,
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "rsi_centered" not in feature_set:
        return {}
    specs = _select_weeks(horizons, RSI_CENTERED_WEEKS, "rsi_centered")
    feature_data: dict[str, np.ndarray] = {}
    for spec in specs:
        rsi = _rsi(close, spec.weeks)
        feature_data[_rsi_centered_name(spec.weeks)] = rsi - 50.0
    return feature_data


def _return_reversal_features(
    log_series: pd.Series,
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "rev" not in feature_set:
        return {}
    specs = _select_weeks(horizons, REV_WEEKS, "rev")
    feature_data: dict[str, np.ndarray] = {}
    for spec in specs:
        cumulative = log_series.diff(spec.weeks)
        reversal = -cumulative.shift(1)
        feature_data[_rev_name(spec.weeks)] = reversal.to_numpy(dtype=float)
    return feature_data


def _shock_features(
    log_series: pd.Series,
    horizons: Sequence[HorizonSpec],
    config: MeanReversionConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "shock" not in feature_set:
        return {}
    if not any(spec.weeks == SHOCK_WEEKS for spec in horizons):
        raise ConfigError(
            "shock feature requires a 4-week horizon",
            context={"required_weeks": str(SHOCK_WEEKS)},
        )
    returns = log_series.diff()
    vol = returns.rolling(window=SHOCK_WEEKS, min_periods=SHOCK_WEEKS).std(
        ddof=0
    )
    shock = returns.shift(1) / (vol + config.eps)
    return {_shock_name(SHOCK_WEEKS): shock.to_numpy(dtype=float)}


def _range_pos_features(
    prices: PriceSeries,
    config: MeanReversionConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "range_pos" not in feature_set:
        return {}
    mid = (prices.high + prices.low) / 2.0
    half_range = (prices.high - prices.low) / 2.0
    pos = (prices.close - mid) / (half_range + config.eps)
    pos = np.where(half_range > 0, pos, np.nan)
    return {_range_pos_name(): pos}


def _range_z_features(
    prices: PriceSeries,
    horizons: Sequence[HorizonSpec],
    config: MeanReversionConfig,
    feature_set: set[str],
) -> dict[str, np.ndarray]:
    if "range_z" not in feature_set:
        return {}
    specs = _select_weeks(horizons, RANGE_Z_WEEKS, "range_z")
    range_series = pd.Series(prices.high - prices.low)
    feature_data: dict[str, np.ndarray] = {}
    for spec in specs:
        mean_range = range_series.rolling(
            window=spec.weeks, min_periods=spec.weeks
        ).mean()
        std_range = range_series.rolling(
            window=spec.weeks, min_periods=spec.weeks
        ).std(ddof=0)
        z_score = (range_series - mean_range) / (
            std_range + config.eps
        )
        feature_data[_range_z_name(spec.weeks)] = z_score.to_numpy(
            dtype=float
        )
    return feature_data


def _ewm_mean(series: pd.Series, weeks: int) -> pd.Series:
    return series.ewm(
        halflife=weeks, adjust=False, min_periods=weeks
    ).mean()


def _ewm_std(series: pd.Series, weeks: int) -> pd.Series:
    return series.ewm(
        halflife=weeks, adjust=False, min_periods=weeks
    ).std(bias=False)


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    series = pd.Series(close, dtype=float)
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    avg_loss = losses.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.to_numpy(dtype=float)


def _z_price_ema_name(weeks: int) -> str:
    return f"z_price_ema_{weeks}w"


def _z_price_med_name(weeks: int) -> str:
    return f"z_price_med_{weeks}w"


def _donch_pos_name(weeks: int) -> str:
    return f"donch_pos_{weeks}w"


def _rsi_centered_name(weeks: int) -> str:
    return f"rsi_centered_{weeks}w"


def _rev_name(weeks: int) -> str:
    return f"rev_{weeks}w"


def _shock_name(weeks: int) -> str:
    return f"shock_{weeks}w"


def _range_pos_name() -> str:
    return "range_pos_1w"


def _range_z_name(weeks: int) -> str:
    return f"range_z_{weeks}w"


def _select_weeks(
    horizons: Sequence[HorizonSpec],
    weeks: Sequence[int],
    feature_name: str,
) -> list[HorizonSpec]:
    specs = [spec for spec in horizons if spec.weeks in weeks]
    if not specs:
        raise ConfigError(
            f"No horizons available for {feature_name} features",
            context={"required_weeks": ",".join(map(str, weeks))},
        )
    return specs


class MeanReversionFeatureGroup(WeeklyFeatureGroup[MeanReversionConfig]):
    name = "mean_reversion"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown mean-reversion features requested"
    compute_asset = _compute_asset_features
