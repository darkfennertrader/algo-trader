from __future__ import annotations

# pylint: disable=duplicate-code
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from .common import log_ratio, to_weekly


@dataclass(frozen=True)
class LevelSpecs:
    vol_cc_specs: Sequence[HorizonSpec]
    atrp_specs: Sequence[HorizonSpec]
    parkinson_specs: Sequence[HorizonSpec]
    vov_specs: Sequence[HorizonSpec]
    regime_spec: HorizonSpec | None


@dataclass(frozen=True)
class LevelContext:
    weekly_index: pd.DatetimeIndex
    week_end_by_week_start: pd.Series
    eps: float
    regime_eps: float
    horizon_days_by_feature: Mapping[str, int]
    specs: LevelSpecs


@dataclass(frozen=True)
class LevelNames:
    vol_cc: Callable[[int], str]
    atrp: Callable[[int], str]
    parkinson: Callable[[int], str]
    vov: Callable[[int], str]
    regime: Callable[[int], str]


@dataclass(frozen=True)
class LevelOutputs:
    feature_data: dict[str, pd.Series]
    vol_cc_weekly: dict[str, pd.Series]
    atrp_weekly: dict[str, pd.Series]


@dataclass(frozen=True)
class PriceInputs:
    returns: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series


def compute_features(
    inputs: PriceInputs,
    *,
    context: LevelContext,
    names: LevelNames,
) -> LevelOutputs:
    vol_cc_weekly = _vol_cc_weekly(
        inputs.returns,
        specs=context.specs.vol_cc_specs,
        weekly_index=context.weekly_index,
        week_end_by_week_start=context.week_end_by_week_start,
        vol_cc_name=names.vol_cc,
    )
    atrp_weekly = _atrp_weekly(
        inputs.high,
        inputs.low,
        inputs.close,
        context=context,
        atrp_name=names.atrp,
    )
    feature_data: dict[str, pd.Series] = {}
    feature_data.update(
        _vol_cc_feature_data(
            vol_cc_weekly,
            specs=context.specs.vol_cc_specs,
            horizon_days_by_feature=context.horizon_days_by_feature,
            vol_cc_name=names.vol_cc,
        )
    )
    feature_data.update(atrp_weekly)
    feature_data.update(
        _parkinson_feature_data(
            inputs.high,
            inputs.low,
            context=context,
            parkinson_name=names.parkinson,
        )
    )
    feature_data.update(
        _vov_feature_data(
            inputs.returns,
            context=context,
            vov_name=names.vov,
        )
    )
    feature_data.update(
        _regime_feature_data(
            vol_cc_weekly,
            context=context,
            regime_name=names.regime,
            vol_cc_name=names.vol_cc,
        )
    )
    return LevelOutputs(
        feature_data=feature_data,
        vol_cc_weekly=vol_cc_weekly,
        atrp_weekly=atrp_weekly,
    )


def _vol_cc_feature_data(
    vol_cc_weekly: Mapping[str, pd.Series],
    *,
    specs: Sequence[HorizonSpec],
    horizon_days_by_feature: Mapping[str, int],
    vol_cc_name: Callable[[int], str],
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    for spec in specs:
        name = vol_cc_name(spec.weeks)
        series = vol_cc_weekly.get(name)
        if series is not None and name in horizon_days_by_feature:
            feature_data[name] = series
    return feature_data


def _vol_cc_weekly(
    returns: pd.Series,
    *,
    specs: Sequence[HorizonSpec],
    weekly_index: pd.DatetimeIndex,
    week_end_by_week_start: pd.Series,
    vol_cc_name: Callable[[int], str],
) -> dict[str, pd.Series]:
    weekly: dict[str, pd.Series] = {}
    for spec in specs:
        vol = _rolling_std(returns, window=spec.days)
        weekly[vol_cc_name(spec.weeks)] = to_weekly(
            vol,
            week_end_by_week_start,
            weekly_index,
        )
    return weekly


def _atrp_weekly(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    context: LevelContext,
    atrp_name: Callable[[int], str],
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    for spec in context.specs.atrp_specs:
        atrp = _atr_percent(
            high,
            low,
            close,
            window=spec.days,
            eps=context.eps,
        )
        feature_data[atrp_name(spec.weeks)] = to_weekly(
            atrp,
            context.week_end_by_week_start,
            context.weekly_index,
        )
    return feature_data


def _parkinson_feature_data(
    high: pd.Series,
    low: pd.Series,
    *,
    context: LevelContext,
    parkinson_name: Callable[[int], str],
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    for spec in context.specs.parkinson_specs:
        parkinson = _parkinson_vol(high, low, window=spec.days)
        feature_data[parkinson_name(spec.weeks)] = to_weekly(
            parkinson,
            context.week_end_by_week_start,
            context.weekly_index,
        )
    return feature_data


def _vov_feature_data(
    returns: pd.Series,
    *,
    context: LevelContext,
    vov_name: Callable[[int], str],
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    for spec in context.specs.vov_specs:
        feature_data[vov_name(spec.weeks)] = _vov_norm_weekly(
            returns,
            window_days=spec.days,
            weekly_index=context.weekly_index,
            week_end_by_week_start=context.week_end_by_week_start,
            eps=context.eps,
        )
    return feature_data


def _regime_feature_data(
    vol_cc_weekly: Mapping[str, pd.Series],
    *,
    context: LevelContext,
    regime_name: Callable[[int], str],
    vol_cc_name: Callable[[int], str],
) -> dict[str, pd.Series]:
    if context.specs.regime_spec is None:
        return {}
    return {
        regime_name(context.specs.regime_spec.weeks): _vol_regime(
            vol_cc_weekly,
            eps=context.regime_eps,
            weekly_index=context.weekly_index,
            vol_cc_name=vol_cc_name,
        )
    }


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def _atr_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int,
    eps: float,
) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr / (close + eps)


def _parkinson_vol(high: pd.Series, low: pd.Series, *, window: int) -> pd.Series:
    ratio = (high / low).to_numpy(dtype=float)
    log_hl = pd.Series(np.log(ratio), index=high.index, dtype=float)
    mean_sq = (log_hl ** 2).rolling(
        window=window, min_periods=window
    ).mean()
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * mean_sq)


def _vov_norm_weekly(
    returns: pd.Series,
    *,
    window_days: int,
    weekly_index: pd.DatetimeIndex,
    week_end_by_week_start: pd.Series,
    eps: float,
) -> pd.Series:
    abs_returns = returns.abs()
    vov_daily = abs_returns.rolling(
        window=window_days,
        min_periods=window_days,
    ).std(ddof=0)
    vov_weekly = to_weekly(
        vov_daily,
        week_end_by_week_start,
        weekly_index,
    )
    baseline = vov_weekly.shift(1).rolling(
        window=26,
        min_periods=26,
    ).median()
    return log_ratio(vov_weekly, baseline, eps=eps)


def _vol_regime(
    vol_cc_weekly: Mapping[str, pd.Series],
    *,
    eps: float,
    weekly_index: pd.DatetimeIndex,
    vol_cc_name: Callable[[int], str],
) -> pd.Series:
    vol_4w = vol_cc_weekly.get(vol_cc_name(4))
    if vol_4w is None:
        return pd.Series(index=weekly_index, dtype=float)
    baseline = vol_4w.shift(1).rolling(
        window=26,
        min_periods=26,
    ).median()
    return log_ratio(vol_4w, baseline, eps=eps)
