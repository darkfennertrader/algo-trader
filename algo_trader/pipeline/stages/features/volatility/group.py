from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError
from algo_trader.pipeline.stages.features.horizons import HorizonSpec
from algo_trader.pipeline.stages.features.protocols import (
    FeatureGroup,
    FeatureInputs,
    FeatureOutput,
)
from .asymmetry import (
    AsymmetryClips,
    AsymmetryConfig,
    AsymmetryWindows,
    build_daily_features,
)
from .common import (
    log_ratio,
    to_weekly,
)
from .level import (
    LevelContext,
    LevelNames,
    LevelSpecs,
    PriceInputs,
    compute_features as compute_level,
)
from .term_structure import (
    TermStructureContext,
    TermStructurePair,
    build_atr_pairs,
    build_cc_pairs,
    build_features,
)
from ..utils import (
    asset_frame,
    load_asset_daily,
    normalize_feature_set,
    ordered_assets,
    require_daily_ohlc,
    require_no_missing,
    require_weekly_index,
    require_weekly_ohlc,
    serialize_series_positive,
    weekly_missing_fraction_from_daily,
    week_end_by_start,
)

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (5, 20, 60, 130)
DEFAULT_EPSILON = 1e-8
SUPPORTED_FEATURES: tuple[str, ...] = (
    "vol_cc_d",
    "atrp_d",
    "vol_regime_cc",
    "vov_norm",
    "vol_ts_cc_1w_4w",
    "vol_ts_cc_4w_26w",
    "down_up_vol_ratio_4w",
    "realized_skew_d_12w",
    "tail_5p_sigma_ratio_12w",
    "jump_freq_4w",
)
VOL_CC_WEEKS: tuple[int, ...] = (4, 26)
ATR_WEEKS: tuple[int, ...] = (4,)
PARKINSON_WEEKS = 4
REGIME_FAST_WEEKS = 4
REGIME_BASELINE_WEEKS = 26
VOV_WEEKS = 12
TERM_STRUCTURE_EPS = 1e-6
VOL_REGIME_EPS = 1e-6
ASYM_EPS = 1e-6
ASYM_VAR_EPS = 1e-12
ASYM_4W_WEEKS = 4
ASYM_12W_WEEKS = 12
ASYM_4W_WINDOW = 20
ASYM_12W_WINDOW = 60
ASYM_MIN_SIGN_COUNT = 3
TAIL_SIGMA_SCALE = 1.645
SKEW_CLIP = (-5.0, 5.0)
KURT_CLIP = (-2.0, 10.0)
ASYM_4W_FEATURES: tuple[str, ...] = (
    "down_up_vol_ratio_4w",
    "jump_freq_4w",
)
ASYM_12W_FEATURES: tuple[str, ...] = (
    "realized_skew_d_12w",
    "tail_5p_sigma_ratio_12w",
)
ASYM_FEATURES: tuple[str, ...] = ASYM_4W_FEATURES + ASYM_12W_FEATURES


@dataclass(frozen=True)
class VolatilityConfig:
    horizons: Sequence[HorizonSpec]
    eps: float = DEFAULT_EPSILON
    features: Sequence[str] | None = None


@dataclass(frozen=True)
class VolatilityGoodness:
    ratios_by_feature: Mapping[str, Mapping[str, Mapping[str, str]]]
    horizon_days_by_feature: Mapping[str, int]


class VolatilityFeatureGroup(FeatureGroup):
    name = "volatility"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown volatility features requested"

    def __init__(self, config: VolatilityConfig) -> None:
        self._config = config
        self._goodness: VolatilityGoodness | None = None

    @property
    def goodness(self) -> VolatilityGoodness | None:
        return self._goodness

    def compute(self, inputs: FeatureInputs) -> FeatureOutput:
        feature_set = normalize_feature_set(
            self._config.features,
            type(self).supported_features,
            error_message=type(self).error_message,
        )
        weekly_ohlc = require_weekly_ohlc(inputs)
        daily_ohlc = require_daily_ohlc(inputs)
        features, feature_names, goodness = _compute_features(
            daily_ohlc,
            weekly_ohlc,
            config=self._config,
            feature_set=feature_set,
        )
        self._goodness = goodness
        return FeatureOutput(frame=features, feature_names=feature_names)


@dataclass(frozen=True)
class _VolatilityContext:
    weekly_index: pd.DatetimeIndex
    week_end_by_week_start: pd.Series
    plan: "_FeaturePlan"
    eps: float


def _compute_features(
    daily_ohlc: pd.DataFrame,
    weekly_ohlc: pd.DataFrame,
    *,
    config: VolatilityConfig,
    feature_set: set[str],
) -> tuple[pd.DataFrame, list[str], VolatilityGoodness | None]:
    assets = ordered_assets(weekly_ohlc)
    if not assets:
        return pd.DataFrame(), [], None
    context = _build_context(weekly_ohlc, config, feature_set, assets)
    ratios_by_feature = _compute_goodness_ratios(
        daily_ohlc,
        assets=assets,
        weekly_index=context.weekly_index,
        horizon_days_by_feature=context.plan.horizon_days_by_feature,
    )
    features_by_asset = _compute_assets(daily_ohlc, assets, context)
    combined = pd.concat(features_by_asset, axis=1)
    combined.columns = combined.columns.set_names(["asset", "feature"])
    feature_names = list(features_by_asset[assets[0]].columns)
    goodness = VolatilityGoodness(
        ratios_by_feature=ratios_by_feature,
        horizon_days_by_feature=context.plan.horizon_days_by_feature,
    )
    return combined, feature_names, goodness


def _build_context(
    weekly_ohlc: pd.DataFrame,
    config: VolatilityConfig,
    feature_set: set[str],
    assets: Sequence[str],
) -> _VolatilityContext:
    require_no_missing(weekly_ohlc, assets)
    weekly_index = require_weekly_index(weekly_ohlc)
    week_end_by_week_start = week_end_by_start(weekly_index)
    plan = _build_feature_plan(config.horizons, feature_set)
    return _VolatilityContext(
        weekly_index=weekly_index,
        week_end_by_week_start=week_end_by_week_start,
        plan=plan,
        eps=config.eps,
    )


def _compute_assets(
    daily_ohlc: pd.DataFrame,
    assets: Sequence[str],
    context: _VolatilityContext,
) -> dict[str, pd.DataFrame]:
    features_by_asset: dict[str, pd.DataFrame] = {}
    for asset in assets:
        asset_daily = load_asset_daily(daily_ohlc, asset)
        feature_frame = _compute_asset_features(asset_daily, context=context)
        features_by_asset[asset] = feature_frame
    return features_by_asset


@dataclass(frozen=True)
class _FeaturePlan:
    specs: "_FeatureSpecSet"
    ts_cc_pairs: Sequence[TermStructurePair]
    ts_atr_pairs: Sequence[TermStructurePair]
    asym_specs: Sequence[HorizonSpec]
    horizon_days_by_feature: Mapping[str, int]


@dataclass(frozen=True)
class _FeatureSpecSet:
    vol_cc_specs: Sequence[HorizonSpec]
    atrp_specs: Sequence[HorizonSpec]
    parkinson_specs: Sequence[HorizonSpec]
    vov_specs: Sequence[HorizonSpec]
    regime_spec: HorizonSpec | None


def _build_feature_plan(
    horizons: Sequence[HorizonSpec], feature_set: set[str]
) -> _FeaturePlan:
    include_regime = "vol_regime_cc" in feature_set
    regime_spec = _resolve_regime_spec(horizons, include_regime)
    ts_cc_pairs = build_cc_pairs(
        horizons,
        feature_set,
        require_weeks=_require_weeks,
    )
    ts_atr_pairs = build_atr_pairs(
        horizons,
        feature_set,
        require_weeks=_require_weeks,
    )
    asym_specs = _asym_specs(horizons, feature_set)
    required_cc_weeks = {pair.short_spec.weeks for pair in ts_cc_pairs}
    required_cc_weeks.update(pair.long_spec.weeks for pair in ts_cc_pairs)
    required_atr_weeks = {pair.short_spec.weeks for pair in ts_atr_pairs}
    required_atr_weeks.update(pair.long_spec.weeks for pair in ts_atr_pairs)
    vol_cc_specs = _vol_cc_specs(
        horizons, feature_set, regime_spec, required_cc_weeks
    )
    atrp_specs = _atrp_specs(
        horizons, feature_set, required_atr_weeks
    )
    parkinson_specs = _feature_specs(
        horizons,
        feature_set,
        "vol_range_parkinson_d",
        (PARKINSON_WEEKS,),
    )
    vov_specs = _feature_specs(horizons, feature_set, "vov_norm", (VOV_WEEKS,))
    spec_set = _FeatureSpecSet(
        vol_cc_specs=vol_cc_specs,
        atrp_specs=atrp_specs,
        parkinson_specs=parkinson_specs,
        vov_specs=vov_specs,
        regime_spec=regime_spec,
    )
    horizon_days_by_feature = _build_horizon_days_by_feature(
        feature_set,
        spec_set,
        ts_cc_pairs=ts_cc_pairs,
        ts_atr_pairs=ts_atr_pairs,
        asym_specs=asym_specs,
    )
    return _FeaturePlan(
        specs=spec_set,
        ts_cc_pairs=ts_cc_pairs,
        ts_atr_pairs=ts_atr_pairs,
        asym_specs=asym_specs,
        horizon_days_by_feature=horizon_days_by_feature,
    )


def _resolve_regime_spec(
    horizons: Sequence[HorizonSpec], include_regime: bool
) -> HorizonSpec | None:
    if not include_regime:
        return None
    spec = _find_weeks(horizons, REGIME_FAST_WEEKS)
    if spec is None:
        raise ConfigError(
            "vol_regime_cc requires a 4-week horizon",
            context={"required_weeks": str(REGIME_FAST_WEEKS)},
        )
    return spec


def _vol_cc_specs(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
    regime_spec: HorizonSpec | None,
    required_weeks: set[int],
) -> list[HorizonSpec]:
    required = set(required_weeks)
    if "vol_cc_d" in feature_set:
        required.update(VOL_CC_WEEKS)
    if regime_spec is not None:
        required.add(REGIME_FAST_WEEKS)
    if not required:
        return []
    if "vol_cc_d" in feature_set:
        feature_name = "vol_cc_d"
    elif required_weeks:
        feature_name = "vol_ts_cc"
    else:
        feature_name = "vol_regime_cc"
    return _require_weeks(horizons, sorted(required), feature_name)


def _feature_specs(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
    feature_name: str,
    weeks: Sequence[int],
) -> list[HorizonSpec]:
    if feature_name not in feature_set:
        return []
    return _require_weeks(horizons, weeks, feature_name)


def _asym_specs(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
) -> list[HorizonSpec]:
    required: set[int] = set()
    if any(name in feature_set for name in ASYM_4W_FEATURES):
        required.add(ASYM_4W_WEEKS)
    if any(name in feature_set for name in ASYM_12W_FEATURES):
        required.add(ASYM_12W_WEEKS)
    if not required:
        return []
    return _require_weeks(
        horizons,
        sorted(required),
        "vol_asymmetry_tails",
    )


def _atrp_specs(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
    required_weeks: set[int],
) -> list[HorizonSpec]:
    if "atrp_d" in feature_set:
        return _require_weeks(horizons, ATR_WEEKS, "atrp_d")
    if required_weeks:
        return _require_weeks(horizons, sorted(required_weeks), "vol_ts_atr")
    return []


def _asym_horizon_days(
    feature_set: set[str],
    asym_specs: Sequence[HorizonSpec],
) -> dict[str, int]:
    weeks_to_days = {spec.weeks: spec.days for spec in asym_specs}
    horizon_days_by_feature: dict[str, int] = {}
    if ASYM_4W_WEEKS in weeks_to_days:
        days_4w = weeks_to_days[ASYM_4W_WEEKS]
        if "down_up_vol_ratio_4w" in feature_set:
            horizon_days_by_feature["down_up_vol_ratio_4w"] = days_4w
        if "jump_freq_4w" in feature_set:
            horizon_days_by_feature["jump_freq_4w"] = days_4w
    if ASYM_12W_WEEKS in weeks_to_days:
        days_12w = weeks_to_days[ASYM_12W_WEEKS]
        if "realized_skew_d_12w" in feature_set:
            horizon_days_by_feature["realized_skew_d_12w"] = days_12w
        if "tail_5p_sigma_ratio_12w" in feature_set:
            horizon_days_by_feature["tail_5p_sigma_ratio_12w"] = days_12w
    return horizon_days_by_feature



def _build_horizon_days_by_feature(
    feature_set: set[str],
    spec_set: _FeatureSpecSet,
    *,
    ts_cc_pairs: Sequence[TermStructurePair],
    ts_atr_pairs: Sequence[TermStructurePair],
    asym_specs: Sequence[HorizonSpec],
) -> dict[str, int]:
    horizon_days_by_feature: dict[str, int] = {}
    for feature_name, specs, name_for, allowed_weeks in (
        ("vol_cc_d", spec_set.vol_cc_specs, _vol_cc_name, VOL_CC_WEEKS),
        ("atrp_d", spec_set.atrp_specs, _atrp_name, ATR_WEEKS),
        (
            "vol_range_parkinson_d",
            spec_set.parkinson_specs,
            _parkinson_name,
            None,
        ),
        ("vov_norm", spec_set.vov_specs, _vov_name, None),
    ):
        if feature_name not in feature_set:
            continue
        for spec in specs:
            if allowed_weeks is not None and spec.weeks not in allowed_weeks:
                continue
            horizon_days_by_feature[name_for(spec.weeks)] = spec.days
    if spec_set.regime_spec is not None:
        horizon_days_by_feature[
            _regime_name(REGIME_FAST_WEEKS, REGIME_BASELINE_WEEKS)
        ] = spec_set.regime_spec.days
    for pair in ts_cc_pairs:
        horizon_days_by_feature[pair.name] = pair.long_spec.days
    for pair in ts_atr_pairs:
        horizon_days_by_feature[pair.name] = pair.long_spec.days
    horizon_days_by_feature.update(_asym_horizon_days(feature_set, asym_specs))
    return horizon_days_by_feature

def _compute_asset_features(
    asset_daily: pd.DataFrame,
    *,
    context: _VolatilityContext,
) -> pd.DataFrame:
    trimmed = _drop_missing_rows(asset_daily)
    feature_frame = _compute_asset_feature_frame(trimmed, context)
    return feature_frame


def _drop_missing_rows(asset_daily: pd.DataFrame) -> pd.DataFrame:
    asset_daily = asset_daily.copy()
    valid_mask = ~asset_daily.isna().any(axis=1)
    return asset_daily[valid_mask]


def _compute_asset_feature_frame(
    asset_daily: pd.DataFrame,
    context: _VolatilityContext,
) -> pd.DataFrame:
    if asset_daily.empty:
        return _empty_feature_frame(
            context.weekly_index,
            context.plan.horizon_days_by_feature,
        )
    close, high, low, returns = _prepare_asset_inputs(asset_daily)
    level_outputs = compute_level(
        PriceInputs(
            returns=returns,
            high=high,
            low=low,
            close=close,
        ),
        context=_level_context(context),
        names=_level_names(),
    )
    feature_data: dict[str, pd.Series] = dict(level_outputs.feature_data)
    feature_data.update(
        _ts_feature_data(
            level_outputs.vol_cc_weekly,
            level_outputs.atrp_weekly,
            context,
        )
    )
    feature_data.update(
        _asym_tail_feature_data(
            returns,
            context,
        )
    )
    return _finalize_feature_frame(
        feature_data,
        context.weekly_index,
        context.plan.horizon_days_by_feature,
    )


def _empty_feature_frame(
    weekly_index: pd.DatetimeIndex,
    horizon_days_by_feature: Mapping[str, int],
) -> pd.DataFrame:
    feature_frame = pd.DataFrame(index=weekly_index)
    if horizon_days_by_feature:
        feature_frame = feature_frame.reindex(
            columns=list(horizon_days_by_feature.keys())
        )
    return feature_frame


def _prepare_asset_inputs(
    asset_daily: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = asset_daily["Close"].astype(float)
    high = asset_daily["High"].astype(float)
    low = asset_daily["Low"].astype(float)
    _require_positive_close(close)
    returns = _close_to_close_returns(close)
    return close, high, low, returns


def _level_context(context: _VolatilityContext) -> LevelContext:
    specs = context.plan.specs
    level_specs = LevelSpecs(
        vol_cc_specs=specs.vol_cc_specs,
        atrp_specs=specs.atrp_specs,
        parkinson_specs=specs.parkinson_specs,
        vov_specs=specs.vov_specs,
        regime_spec=specs.regime_spec,
    )
    return LevelContext(
        weekly_index=context.weekly_index,
        week_end_by_week_start=context.week_end_by_week_start,
        eps=context.eps,
        regime_eps=VOL_REGIME_EPS,
        horizon_days_by_feature=context.plan.horizon_days_by_feature,
        specs=level_specs,
    )


def _level_names() -> LevelNames:
    return LevelNames(
        vol_cc=_vol_cc_name,
        atrp=_atrp_name,
        parkinson=_parkinson_name,
        vov=_vov_name,
        regime=lambda weeks: _regime_name(
            REGIME_FAST_WEEKS, REGIME_BASELINE_WEEKS
        ),
    )


def _ts_feature_data(
    vol_cc_weekly: Mapping[str, pd.Series],
    atrp_weekly: Mapping[str, pd.Series],
    context: _VolatilityContext,
) -> dict[str, pd.Series]:
    ts_context = TermStructureContext(
        cc_pairs=context.plan.ts_cc_pairs,
        atr_pairs=context.plan.ts_atr_pairs,
        log_ratio=lambda short, long: log_ratio(
            short, long, eps=TERM_STRUCTURE_EPS
        ),
        vol_cc_name=_vol_cc_name,
        atrp_name=_atrp_name,
    )
    return build_features(vol_cc_weekly, atrp_weekly, ts_context)


def _asym_tail_feature_data(
    returns: pd.Series,
    context: _VolatilityContext,
) -> dict[str, pd.Series]:
    if not context.plan.asym_specs:
        return {}
    allowed_features = [
        name
        for name in ASYM_FEATURES
        if name in context.plan.horizon_days_by_feature
    ]
    if not allowed_features:
        return {}
    config = AsymmetryConfig(
        eps=ASYM_EPS,
        var_eps=ASYM_VAR_EPS,
        tail_sigma_scale=TAIL_SIGMA_SCALE,
        windows=AsymmetryWindows(
            window_4w=ASYM_4W_WINDOW,
            window_12w=ASYM_12W_WINDOW,
            min_sign_count=ASYM_MIN_SIGN_COUNT,
        ),
        clips=AsymmetryClips(
            skew=SKEW_CLIP,
            kurt=KURT_CLIP,
        ),
    )
    asym_daily = build_daily_features(
        returns,
        config=config,
        log_ratio=lambda short, long: log_ratio(short, long, eps=ASYM_EPS),
        allowed_features=allowed_features,
    )
    return _asymmetry_weekly_features(asym_daily, context)


def _asymmetry_weekly_features(
    daily_features: Mapping[str, pd.Series],
    context: _VolatilityContext,
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    for name, series in daily_features.items():
        feature_data[name] = to_weekly(
            series,
            context.week_end_by_week_start,
            context.weekly_index,
        )
    return feature_data


def _finalize_feature_frame(
    feature_data: Mapping[str, pd.Series],
    weekly_index: pd.DatetimeIndex,
    horizon_days_by_feature: Mapping[str, int],
) -> pd.DataFrame:
    feature_frame = pd.DataFrame(feature_data, index=weekly_index)
    if horizon_days_by_feature:
        feature_frame = feature_frame.reindex(
            columns=list(horizon_days_by_feature.keys())
        )
    return feature_frame


def _compute_goodness_ratios(
    daily_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    weekly_index: pd.DatetimeIndex,
    horizon_days_by_feature: Mapping[str, int],
) -> dict[str, dict[str, dict[str, str]]]:
    ratios_by_feature: dict[str, dict[str, dict[str, str]]] = {}
    if not horizon_days_by_feature:
        return ratios_by_feature
    weeks_by_feature = {
        name: max(1, days // 5) for name, days in horizon_days_by_feature.items()
    }
    ratios_by_feature = {name: {} for name in weeks_by_feature}
    for asset in assets:
        weekly_missing = weekly_missing_fraction_from_daily(
            daily_ohlc, asset=asset, weekly_index=weekly_index
        )
        for feature_name, weeks in weeks_by_feature.items():
            ratios_by_feature[feature_name][asset] = (
                serialize_series_positive(
                    weekly_missing.rolling(window=weeks, min_periods=weeks).mean()
                )
            )
    return ratios_by_feature


def _require_positive_close(close: pd.Series) -> None:
    if (close <= 0).any():
        raise DataProcessingError(
            "daily_ohlc contains non-positive Close values",
            context={"invalid_close": "true"},
        )


def _find_weeks(
    horizons: Sequence[HorizonSpec], weeks: int
) -> HorizonSpec | None:
    for spec in horizons:
        if spec.weeks == weeks:
            return spec
    return None


def _require_weeks(
    horizons: Sequence[HorizonSpec],
    weeks: Sequence[int],
    feature_name: str,
) -> list[HorizonSpec]:
    specs = [spec for spec in horizons if spec.weeks in weeks]
    if not specs:
        raise ConfigError(
            f"{feature_name} requires horizons of {', '.join(map(str, weeks))} weeks",
            context={"required_weeks": ",".join(map(str, weeks))},
        )
    return specs


def _n_eff_ratio(asset_daily: pd.DataFrame, horizon_days: int) -> pd.Series:
    if asset_daily.empty:
        return pd.Series(dtype=float)
    valid_mask = ~asset_daily.isna().any(axis=1)
    counts = valid_mask.astype(float).rolling(
        window=horizon_days, min_periods=horizon_days
    ).sum()
    return counts / float(horizon_days)


def _close_to_close_returns(close: pd.Series) -> pd.Series:
    if close.empty:
        return pd.Series(dtype=float)
    ratio = close / close.shift(1)
    values = np.log(ratio.to_numpy(dtype=float))
    return pd.Series(values, index=close.index, dtype=float)


def _vol_cc_name(weeks: int) -> str:
    return f"vol_cc_d_{weeks}w"


def _atrp_name(weeks: int) -> str:
    return f"atrp_d_{weeks}w"


def _parkinson_name(weeks: int) -> str:
    return f"vol_range_parkinson_d_{weeks}w"


def _regime_name(fast_weeks: int, slow_weeks: int) -> str:
    return f"vol_regime_cc_{fast_weeks}w_{slow_weeks}w"


def _vov_name(weeks: int) -> str:
    return f"vov_norm_{weeks}w"


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_HORIZON_DAYS",
    "SUPPORTED_FEATURES",
    "VolatilityConfig",
    "VolatilityFeatureGroup",
    "VolatilityGoodness",
]
