from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

LogRatio = Callable[[pd.Series, pd.Series], pd.Series]


@dataclass(frozen=True)
class AsymmetryWindows:
    window_4w: int
    window_12w: int
    min_sign_count: int


@dataclass(frozen=True)
class AsymmetryClips:
    skew: tuple[float, float]
    kurt: tuple[float, float]


@dataclass(frozen=True)
class AsymmetryConfig:
    eps: float
    var_eps: float
    tail_sigma_scale: float
    windows: AsymmetryWindows
    clips: AsymmetryClips


@dataclass(frozen=True)
class _SigmaBundle:
    sigma_4w: pd.Series
    sigma_12w: pd.Series


def build_daily_features(
    returns: pd.Series,
    *,
    config: AsymmetryConfig,
    log_ratio: LogRatio,
) -> dict[str, pd.Series]:
    sigma = _sigma_bundle(returns, config)
    downside, upside = _down_up_vol(returns, sigma, config)
    down_up_ratio = log_ratio(downside, upside)
    skew = _realized_skew(
        returns,
        window=config.windows.window_12w,
        var_eps=config.var_eps,
        clip_range=config.clips.skew,
    )
    kurt = _realized_kurt(
        returns,
        window=config.windows.window_12w,
        var_eps=config.var_eps,
        clip_range=config.clips.kurt,
    )
    tail_ratio = _tail_sigma_ratio(
        returns,
        sigma_12w=sigma.sigma_12w,
        config=config,
        log_ratio=log_ratio,
    )
    jump_freq = _jump_freq(
        returns,
        sigma_12w=sigma.sigma_12w,
        window=config.windows.window_4w,
    )
    return {
        "downside_vol_d_4w": downside,
        "upside_vol_d_4w": upside,
        "down_up_vol_ratio_4w": down_up_ratio,
        "realized_skew_d_12w": skew,
        "realized_kurt_d_12w": kurt,
        "tail_5p_sigma_ratio_12w": tail_ratio,
        "jump_freq_4w": jump_freq,
    }


def _sigma_bundle(
    returns: pd.Series, config: AsymmetryConfig
) -> _SigmaBundle:
    sigma_4w = _rolling_std_sample(returns, window=config.windows.window_4w)
    sigma_12w = _rolling_std_sample(returns, window=config.windows.window_12w)
    return _SigmaBundle(sigma_4w=sigma_4w, sigma_12w=sigma_12w)


def _down_up_vol(
    returns: pd.Series,
    sigma: _SigmaBundle,
    config: AsymmetryConfig,
) -> tuple[pd.Series, pd.Series]:
    downside = _conditional_semidev(
        returns,
        window=config.windows.window_4w,
        sign=-1,
        min_sign_count=config.windows.min_sign_count,
    )
    upside = _conditional_semidev(
        returns,
        window=config.windows.window_4w,
        sign=1,
        min_sign_count=config.windows.min_sign_count,
    )
    downside = _fallback_series(downside, sigma.sigma_4w)
    upside = _fallback_series(upside, sigma.sigma_4w)
    return downside, upside


def _rolling_std_sample(series: pd.Series, *, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def _conditional_semidev(
    series: pd.Series,
    *,
    window: int,
    sign: int,
    min_sign_count: int,
) -> pd.Series:
    def _semidev(values: np.ndarray) -> float:
        selected = values[values * sign > 0]
        if selected.size >= min_sign_count:
            return float(np.sqrt(np.mean(selected ** 2)))
        return float("nan")

    return series.rolling(window=window, min_periods=window).apply(
        _semidev, raw=True
    )


def _fallback_series(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    primary_aligned, fallback_aligned = primary.align(fallback)
    return primary_aligned.fillna(fallback_aligned)


def _realized_skew(
    series: pd.Series,
    *,
    window: int,
    var_eps: float,
    clip_range: tuple[float, float],
) -> pd.Series:
    def _skew(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        diffs = values - mean
        s2 = float(np.sum(diffs ** 2) / (len(values) - 1))
        scale = np.sqrt(s2 + var_eps)
        skew = float(np.mean(diffs ** 3) / (scale ** 3 + var_eps))
        return float(np.clip(skew, *clip_range))

    return series.rolling(window=window, min_periods=window).apply(
        _skew, raw=True
    )


def _realized_kurt(
    series: pd.Series,
    *,
    window: int,
    var_eps: float,
    clip_range: tuple[float, float],
) -> pd.Series:
    def _kurt(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        diffs = values - mean
        s2 = float(np.sum(diffs ** 2) / (len(values) - 1))
        scale = np.sqrt(s2 + var_eps)
        kurt = float(np.mean(diffs ** 4) / (scale ** 4 + var_eps) - 3.0)
        return float(np.clip(kurt, *clip_range))

    return series.rolling(window=window, min_periods=window).apply(
        _kurt, raw=True
    )


def _tail_sigma_ratio(
    returns: pd.Series,
    *,
    sigma_12w: pd.Series,
    config: AsymmetryConfig,
    log_ratio: LogRatio,
) -> pd.Series:
    def _ratio(values: np.ndarray) -> float:
        q05 = float(np.quantile(values, 0.05))
        return max(-q05, config.eps)

    window = config.windows.window_12w
    numerator = returns.rolling(window=window, min_periods=window).apply(
        _ratio, raw=True
    )
    sigma_clip = sigma_12w.clip(lower=config.eps)
    denom = config.tail_sigma_scale * sigma_clip
    return log_ratio(numerator, denom)


def _jump_freq(
    returns: pd.Series,
    *,
    sigma_12w: pd.Series,
    window: int,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    values = returns.to_numpy(dtype=float)
    sigma = sigma_12w.to_numpy(dtype=float)
    result: list[float] = [float("nan")] * len(values)
    for idx, value in enumerate(values):
        if idx < window - 1 or np.isnan(sigma[idx]):
            continue
        window_values = values[idx - window + 1 : idx + 1]
        threshold = 2.0 * sigma[idx]
        result[idx] = float(np.sum(np.abs(window_values) > threshold)) / float(
            window
        )
    return pd.Series(result, index=returns.index)
