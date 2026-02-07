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
from algo_trader.pipeline.stages.features.volatility.common import to_weekly
from ..utils import (
    asset_frame,
    load_asset_daily,
    prepare_weekly_daily_inputs,
    require_no_missing,
    require_ohlc_columns,
    require_weekly_index,
    reindex_asset_features,
    serialize_series_positive,
    weekly_missing_fraction_from_daily,
    week_end_by_start,
)

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (5, 20, 60, 130)
DEFAULT_EPSILON = 1e-6
SUPPORTED_FEATURES: tuple[str, ...] = (
    "glob_vol_cc_d",
    "glob_vol_regime_cc",
    "glob_corr_mean",
    "glob_pc1_share",
    "glob_disp_ret",
    "glob_disp_mom",
    "glob_disp_vol",
)
VOL_WEEKS: tuple[int, ...] = (4, 12)
REGIME_FAST_WEEKS = 4
REGIME_BASELINE_WEEKS = 26
CORR_WEEKS = 12
PC1_WEEKS = 12
DISP_RET_WEEKS = 1
DISP_MOM_WEEKS = 12
DISP_VOL_WEEKS = 4
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class RegimeConfig:
    horizons: Sequence[HorizonSpec]
    eps: float = DEFAULT_EPSILON
    features: Sequence[str] | None = None


@dataclass(frozen=True)
class RegimeGoodness:
    ratios_by_feature: Mapping[str, Mapping[str, Mapping[str, float]]]
    horizon_days_by_feature: Mapping[str, int]


class RegimeFeatureGroup(FeatureGroup):
    name = "regime"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown regime features requested"

    def __init__(self, config: RegimeConfig) -> None:
        self._config = config
        self._goodness: RegimeGoodness | None = None

    @property
    def goodness(self) -> RegimeGoodness | None:
        return self._goodness

    def compute(self, inputs: FeatureInputs) -> FeatureOutput:
        selection = self._config.features
        feature_set, weekly_ohlc, daily_ohlc, assets = prepare_weekly_daily_inputs(
            inputs,
            features=selection,
            supported_features=type(self).supported_features,
            error_message=type(self).error_message,
        )
        output, goodness = _compute_with_assets(
            daily_ohlc,
            weekly_ohlc,
            assets=assets,
            feature_set=feature_set,
            config=self._config,
        )
        self._goodness = goodness
        return output


@dataclass(frozen=True)
class _RegimePlan:
    vol_specs: Sequence[HorizonSpec]
    corr_spec: HorizonSpec | None
    pc1_spec: HorizonSpec | None
    disp_ret_spec: HorizonSpec | None
    disp_mom_spec: HorizonSpec | None
    disp_vol_spec: HorizonSpec | None
    horizon_days_by_feature: Mapping[str, int]


@dataclass(frozen=True)
class _RegimeContext:
    weekly_index: pd.DatetimeIndex
    week_end_by_week_start: pd.Series
    plan: _RegimePlan
    eps: float


@dataclass(frozen=True)
class _BuildInputs:
    daily_ohlc: pd.DataFrame
    weekly_ohlc: pd.DataFrame
    assets: Sequence[str]
    vol_frames: Mapping[int, pd.DataFrame]
    context: _RegimeContext
    feature_set: set[str]


def _compute_with_assets(
    daily_ohlc: pd.DataFrame,
    weekly_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    feature_set: set[str],
    config: RegimeConfig,
) -> tuple[FeatureOutput, RegimeGoodness | None]:
    if not assets:
        return FeatureOutput(frame=pd.DataFrame(), feature_names=[]), None
    context = _build_context(
        weekly_ohlc, config=config, feature_set=feature_set
    )
    require_no_missing(weekly_ohlc, assets)
    features, feature_names = _compute_features(
        daily_ohlc,
        weekly_ohlc,
        assets=assets,
        context=context,
        feature_set=feature_set,
    )
    goodness = _compute_goodness(
        daily_ohlc,
        assets=assets,
        context=context,
    )
    return FeatureOutput(frame=features, feature_names=feature_names), goodness
def _build_context(
    weekly_ohlc: pd.DataFrame,
    *,
    config: RegimeConfig,
    feature_set: set[str],
) -> _RegimeContext:
    weekly_index = require_weekly_index(weekly_ohlc)
    week_end_by_week_start = week_end_by_start(weekly_index)
    plan = _build_feature_plan(config.horizons, feature_set)
    return _RegimeContext(
        weekly_index=weekly_index,
        week_end_by_week_start=week_end_by_week_start,
        plan=plan,
        eps=config.eps,
    )


def _build_feature_plan(
    horizons: Sequence[HorizonSpec], feature_set: set[str]
) -> _RegimePlan:
    spec_by_weeks = {spec.weeks: spec for spec in horizons}
    vol_specs: list[HorizonSpec] = []
    corr_spec = None
    pc1_spec = None
    disp_ret_spec = None
    disp_mom_spec = None
    disp_vol_spec = None

    if "glob_vol_cc_d" in feature_set:
        vol_specs.extend(
            _require_weeks(spec_by_weeks, VOL_WEEKS, "glob_vol_cc_d")
        )
    if "glob_vol_regime_cc" in feature_set:
        vol_specs.extend(
            _require_weeks(
                spec_by_weeks,
                (REGIME_FAST_WEEKS,),
                "glob_vol_regime_cc",
            )
        )
    if "glob_disp_vol" in feature_set:
        vol_specs.extend(
            _require_weeks(
                spec_by_weeks,
                (DISP_VOL_WEEKS,),
                "glob_disp_vol",
            )
        )
    if "glob_corr_mean" in feature_set:
        corr_spec = _require_week(spec_by_weeks, CORR_WEEKS, "glob_corr_mean")
    if "glob_pc1_share" in feature_set:
        pc1_spec = _require_week(spec_by_weeks, PC1_WEEKS, "glob_pc1_share")
    if "glob_disp_ret" in feature_set:
        disp_ret_spec = _require_week(
            spec_by_weeks, DISP_RET_WEEKS, "glob_disp_ret"
        )
    if "glob_disp_mom" in feature_set:
        disp_mom_spec = _require_week(
            spec_by_weeks, DISP_MOM_WEEKS, "glob_disp_mom"
        )
    if "glob_disp_vol" in feature_set:
        disp_vol_spec = _require_week(
            spec_by_weeks, DISP_VOL_WEEKS, "glob_disp_vol"
        )

    vol_specs = _unique_specs(vol_specs)
    horizon_days_by_feature = _build_horizon_days_by_feature(
        feature_set,
        spec_by_weeks=spec_by_weeks,
    )
    return _RegimePlan(
        vol_specs=vol_specs,
        corr_spec=corr_spec,
        pc1_spec=pc1_spec,
        disp_ret_spec=disp_ret_spec,
        disp_mom_spec=disp_mom_spec,
        disp_vol_spec=disp_vol_spec,
        horizon_days_by_feature=horizon_days_by_feature,
    )


def _unique_specs(specs: Sequence[HorizonSpec]) -> list[HorizonSpec]:
    by_weeks: dict[int, HorizonSpec] = {}
    for spec in specs:
        by_weeks[spec.weeks] = spec
    return [by_weeks[weeks] for weeks in sorted(by_weeks)]


def _require_week(
    spec_by_weeks: Mapping[int, HorizonSpec],
    weeks: int,
    feature_name: str,
) -> HorizonSpec:
    spec = spec_by_weeks.get(weeks)
    if spec is None:
        raise ConfigError(
            f"{feature_name} requires a {weeks}-week horizon",
            context={"required_weeks": str(weeks)},
        )
    return spec


def _require_weeks(
    spec_by_weeks: Mapping[int, HorizonSpec],
    weeks: Sequence[int],
    feature_name: str,
) -> list[HorizonSpec]:
    specs: list[HorizonSpec] = []
    missing = [str(week) for week in weeks if week not in spec_by_weeks]
    if missing:
        required = ", ".join(str(week) for week in weeks)
        raise ConfigError(
            f"{feature_name} requires horizons of {required} weeks",
            context={"required_weeks": required},
        )
    for week in weeks:
        specs.append(spec_by_weeks[week])
    return specs


def _build_horizon_days_by_feature(
    feature_set: set[str],
    *,
    spec_by_weeks: Mapping[int, HorizonSpec],
) -> dict[str, int]:
    horizon_days_by_feature: dict[str, int] = {}
    if "glob_vol_cc_d" in feature_set:
        for weeks in VOL_WEEKS:
            name = _glob_vol_name(weeks)
            horizon_days_by_feature[name] = spec_by_weeks[weeks].days
    if "glob_vol_regime_cc" in feature_set:
        name = _glob_vol_regime_name()
        horizon_days_by_feature[name] = spec_by_weeks[REGIME_FAST_WEEKS].days
    if "glob_corr_mean" in feature_set:
        name = _glob_corr_name(CORR_WEEKS)
        horizon_days_by_feature[name] = spec_by_weeks[CORR_WEEKS].days
    if "glob_pc1_share" in feature_set:
        name = _glob_pc1_name(PC1_WEEKS)
        horizon_days_by_feature[name] = spec_by_weeks[PC1_WEEKS].days
    if "glob_disp_ret" in feature_set:
        name = _glob_disp_ret_name(DISP_RET_WEEKS)
        horizon_days_by_feature[name] = spec_by_weeks[DISP_RET_WEEKS].days
    if "glob_disp_mom" in feature_set:
        name = _glob_disp_mom_name(DISP_MOM_WEEKS)
        horizon_days_by_feature[name] = spec_by_weeks[DISP_MOM_WEEKS].days
    if "glob_disp_vol" in feature_set:
        name = _glob_disp_vol_name(DISP_VOL_WEEKS)
        horizon_days_by_feature[name] = spec_by_weeks[DISP_VOL_WEEKS].days
    return horizon_days_by_feature


def _compute_features(
    daily_ohlc: pd.DataFrame,
    weekly_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    context: _RegimeContext,
    feature_set: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    feature_names: list[str] = []
    feature_data: dict[str, pd.Series] = {}
    vol_frames = _compute_vol_frames(
        daily_ohlc,
        assets=assets,
        context=context,
        vol_specs=context.plan.vol_specs,
    )
    build_inputs = _BuildInputs(
        daily_ohlc=daily_ohlc,
        weekly_ohlc=weekly_ohlc,
        assets=assets,
        vol_frames=vol_frames,
        context=context,
        feature_set=feature_set,
    )
    _add_vol_features(
        feature_data,
        feature_names=feature_names,
        inputs=build_inputs,
    )
    _add_corr_features(
        feature_data,
        feature_names=feature_names,
        inputs=build_inputs,
    )
    _add_dispersion_features(
        feature_data,
        feature_names=feature_names,
        inputs=build_inputs,
    )
    return _assemble_output(
        feature_data,
        feature_names=feature_names,
        assets=assets,
        weekly_index=context.weekly_index,
    )


def _add_vol_features(
    feature_data: dict[str, pd.Series],
    *,
    feature_names: list[str],
    inputs: _BuildInputs,
) -> None:
    if "glob_vol_cc_d" in inputs.feature_set:
        for weeks in VOL_WEEKS:
            name = _glob_vol_name(weeks)
            feature_names.append(name)
            feature_data[name] = _global_mean(inputs.vol_frames[weeks])
    if "glob_vol_regime_cc" in inputs.feature_set:
        name = _glob_vol_regime_name()
        feature_names.append(name)
        glob_vol_4w = _global_mean(inputs.vol_frames[REGIME_FAST_WEEKS])
        feature_data[name] = _vol_regime(
            glob_vol_4w, eps=inputs.context.eps
        )


def _add_corr_features(
    feature_data: dict[str, pd.Series],
    *,
    feature_names: list[str],
    inputs: _BuildInputs,
) -> None:
    if "glob_corr_mean" in inputs.feature_set:
        name = _glob_corr_name(CORR_WEEKS)
        feature_names.append(name)
        window_days = (
            inputs.context.plan.corr_spec.days
            if inputs.context.plan.corr_spec is not None
            else CORR_WEEKS * 5
        )
        feature_data[name] = _compute_corr_mean(
            inputs.daily_ohlc,
            assets=inputs.assets,
            context=inputs.context,
            window_days=window_days,
        )
    if "glob_pc1_share" in inputs.feature_set:
        name = _glob_pc1_name(PC1_WEEKS)
        feature_names.append(name)
        window_days = (
            inputs.context.plan.pc1_spec.days
            if inputs.context.plan.pc1_spec is not None
            else PC1_WEEKS * 5
        )
        feature_data[name] = _compute_pc1_share(
            inputs.daily_ohlc,
            assets=inputs.assets,
            context=inputs.context,
            window_days=window_days,
        )


def _add_dispersion_features(
    feature_data: dict[str, pd.Series],
    *,
    feature_names: list[str],
    inputs: _BuildInputs,
) -> None:
    if not _needs_dispersion(inputs.feature_set):
        return
    weekly_log_close = _weekly_log_close_frame(
        inputs.weekly_ohlc, assets=inputs.assets
    )
    if "glob_disp_ret" in inputs.feature_set:
        name = _glob_disp_ret_name(DISP_RET_WEEKS)
        feature_names.append(name)
        weekly_returns = weekly_log_close.diff()
        feature_data[name] = _cross_sectional_mad(weekly_returns)
    if "glob_disp_mom" in inputs.feature_set:
        name = _glob_disp_mom_name(DISP_MOM_WEEKS)
        feature_names.append(name)
        momentum = weekly_log_close - weekly_log_close.shift(DISP_MOM_WEEKS)
        feature_data[name] = _cross_sectional_mad(momentum)
    if "glob_disp_vol" in inputs.feature_set:
        name = _glob_disp_vol_name(DISP_VOL_WEEKS)
        feature_names.append(name)
        feature_data[name] = _cross_sectional_mad(
            inputs.vol_frames[DISP_VOL_WEEKS]
        )


def _needs_dispersion(feature_set: set[str]) -> bool:
    return any(
        name in feature_set
        for name in (
            "glob_disp_ret",
            "glob_disp_mom",
            "glob_disp_vol",
        )
    )


def _compute_goodness(
    daily_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    context: _RegimeContext,
) -> RegimeGoodness | None:
    horizon_days_by_feature = context.plan.horizon_days_by_feature
    if not horizon_days_by_feature:
        return None
    weeks_by_feature = {
        name: max(1, days // 5) for name, days in horizon_days_by_feature.items()
    }
    ratios_by_feature: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in weeks_by_feature
    }
    for asset in assets:
        weekly_missing = weekly_missing_fraction_from_daily(
            daily_ohlc, asset=asset, weekly_index=context.weekly_index
        )
        for feature_name, weeks in weeks_by_feature.items():
            ratios_by_feature[feature_name][asset] = (
                serialize_series_positive(
                    weekly_missing.rolling(window=weeks, min_periods=weeks).mean()
                )
            )
    return RegimeGoodness(
        ratios_by_feature=ratios_by_feature,
        horizon_days_by_feature=horizon_days_by_feature,
    )


def _compute_vol_frames(
    daily_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    context: _RegimeContext,
    vol_specs: Sequence[HorizonSpec],
) -> Mapping[int, pd.DataFrame]:
    if not vol_specs:
        return {}
    frames_by_weeks: dict[int, dict[str, pd.Series]] = {
        spec.weeks: {} for spec in vol_specs
    }
    for asset in assets:
        _update_vol_frames_for_asset(
            daily_ohlc,
            asset=asset,
            context=context,
            vol_specs=vol_specs,
            frames_by_weeks=frames_by_weeks,
        )
    return _finalize_vol_frames(
        frames_by_weeks, assets=assets, weekly_index=context.weekly_index
    )


def _update_vol_frames_for_asset(
    daily_ohlc: pd.DataFrame,
    *,
    asset: str,
    context: _RegimeContext,
    vol_specs: Sequence[HorizonSpec],
    frames_by_weeks: dict[int, dict[str, pd.Series]],
) -> None:
    asset_daily = load_asset_daily(daily_ohlc, asset)
    trimmed = _drop_missing_rows(asset_daily)
    returns = _close_to_close_returns(trimmed["Close"])
    for spec in vol_specs:
        vol = returns.rolling(window=spec.days, min_periods=spec.days).std(
            ddof=1
        )
        weekly = to_weekly(
            vol,
            context.week_end_by_week_start,
            context.weekly_index,
        )
        frames_by_weeks[spec.weeks][asset] = weekly


def _finalize_vol_frames(
    frames_by_weeks: Mapping[int, Mapping[str, pd.Series]],
    *,
    assets: Sequence[str],
    weekly_index: pd.DatetimeIndex,
) -> dict[int, pd.DataFrame]:
    output: dict[int, pd.DataFrame] = {}
    for weeks, series_by_asset in frames_by_weeks.items():
        frame = pd.DataFrame(series_by_asset, index=weekly_index)
        output[weeks] = frame.reindex(columns=list(assets))
    return output


def _compute_corr_mean(
    daily_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    context: _RegimeContext,
    window_days: int,
) -> pd.Series:
    daily_returns = _daily_returns_frame(daily_ohlc, assets=assets)
    values: list[float] = []
    for week_end in context.weekly_index:
        window = _windowed_returns(
            daily_returns, end_time=week_end, window_days=window_days
        )
        if window is None:
            values.append(np.nan)
        else:
            values.append(_corr_mean_from_window(window))
    return pd.Series(values, index=context.weekly_index, dtype=float)


def _compute_pc1_share(
    daily_ohlc: pd.DataFrame,
    *,
    assets: Sequence[str],
    context: _RegimeContext,
    window_days: int,
) -> pd.Series:
    daily_returns = _daily_returns_frame(daily_ohlc, assets=assets)
    values: list[float] = []
    for week_end in context.weekly_index:
        window = _windowed_returns(
            daily_returns, end_time=week_end, window_days=window_days
        )
        if window is None:
            values.append(np.nan)
        else:
            values.append(_pc1_share_from_window(window))
    return pd.Series(values, index=context.weekly_index, dtype=float)


def _daily_returns_frame(
    daily_ohlc: pd.DataFrame, *, assets: Sequence[str]
) -> pd.DataFrame:
    returns_by_asset: dict[str, pd.Series] = {}
    for asset in assets:
        asset_daily = load_asset_daily(daily_ohlc, asset)
        trimmed = _drop_missing_rows(asset_daily)
        returns_by_asset[asset] = _close_to_close_returns(trimmed["Close"])
    frame = pd.DataFrame(returns_by_asset)
    frame = frame.sort_index()
    return frame.dropna(axis=0, how="all")


def _windowed_returns(
    returns_frame: pd.DataFrame,
    *,
    end_time: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame | None:
    if returns_frame.empty:
        return None
    subset = returns_frame.loc[:end_time]
    if subset.empty:
        return None
    window = subset.tail(window_days)
    if len(window) < window_days:
        return None
    window = window.dropna(axis=1, how="any")
    if window.empty:
        return None
    return window


def _corr_mean_from_window(window: pd.DataFrame) -> float:
    if window.shape[1] < 2:
        return np.nan
    corr = window.corr()
    values = corr.to_numpy(dtype=float)
    upper = values[np.triu_indices(values.shape[0], k=1)]
    if upper.size == 0:
        return np.nan
    return float(np.nanmean(upper))


def _pc1_share_from_window(window: pd.DataFrame) -> float:
    if window.shape[1] < 2:
        return np.nan
    cov = window.cov(ddof=1)
    if cov.isna().any().any():
        return np.nan
    matrix = cov.to_numpy(dtype=float)
    trace = float(np.trace(matrix))
    if trace <= 0.0:
        return np.nan
    eigenvalues = np.linalg.eigvalsh(matrix)
    max_eigen = float(np.max(np.asarray(eigenvalues, dtype=float)))
    return max_eigen / trace


def _weekly_log_close_frame(
    weekly_ohlc: pd.DataFrame, *, assets: Sequence[str]
) -> pd.DataFrame:
    close_by_asset: dict[str, pd.Series] = {}
    for asset in assets:
        asset_weekly = asset_frame(weekly_ohlc, asset)
        require_ohlc_columns(asset_weekly)
        close = asset_weekly["Close"].astype(float)
        _require_positive_close(close, label="weekly_ohlc")
        log_close = np.log(close.to_numpy(dtype=float))
        close_by_asset[asset] = pd.Series(
            log_close, index=close.index, dtype=float
        )
    frame = pd.DataFrame(close_by_asset, index=weekly_ohlc.index)
    return frame.reindex(columns=list(assets))


def _cross_sectional_mad(values: pd.DataFrame) -> pd.Series:
    if values.empty:
        return pd.Series(index=values.index, dtype=float)
    row_has_value = values.notna().any(axis=1)
    if not row_has_value.any():
        return pd.Series(np.nan, index=values.index, dtype=float)
    subset = values.loc[row_has_value]
    median = subset.median(axis=1, skipna=True)
    deviations = subset.sub(median, axis=0).abs()
    mad = deviations.median(axis=1, skipna=True)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[row_has_value] = MAD_SCALE * mad
    return result


def _assemble_output(
    feature_data: Mapping[str, pd.Series],
    *,
    feature_names: Sequence[str],
    assets: Sequence[str],
    weekly_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[str]]:
    if not feature_names:
        return pd.DataFrame(index=weekly_index), []
    frames = [
        _broadcast_feature(feature_data[name], assets, name)
        for name in feature_names
    ]
    combined = pd.concat(frames, axis=1)
    combined.columns = combined.columns.set_names(["asset", "feature"])
    combined = reindex_asset_features(combined, assets, feature_names)
    return combined, list(feature_names)


def _broadcast_feature(
    series: pd.Series, assets: Sequence[str], feature_name: str
) -> pd.DataFrame:
    values = series.to_numpy(dtype=float).reshape(-1, 1)
    tiled = np.tile(values, (1, len(assets)))
    columns = pd.MultiIndex.from_product(
        [pd.Index(assets, name="asset"), [feature_name]],
        names=["asset", "feature"],
    )
    return pd.DataFrame(tiled, index=series.index, columns=columns)


def _drop_missing_rows(asset_daily: pd.DataFrame) -> pd.DataFrame:
    asset_daily = asset_daily.copy()
    valid_mask = ~asset_daily.isna().any(axis=1)
    return asset_daily[valid_mask]


def _close_to_close_returns(close: pd.Series) -> pd.Series:
    if close.empty:
        return pd.Series(dtype=float)
    _require_positive_close(close, label="daily_ohlc")
    ratio = close / close.shift(1)
    values = np.log(ratio.to_numpy(dtype=float))
    return pd.Series(values, index=close.index, dtype=float)


def _require_positive_close(close: pd.Series, *, label: str) -> None:
    if (close.dropna() <= 0).any():
        raise DataProcessingError(
            f"{label} contains non-positive Close values",
            context={"invalid_close": "true"},
        )


def _vol_regime(series: pd.Series, *, eps: float) -> pd.Series:
    baseline = series.shift(1).rolling(
        window=REGIME_BASELINE_WEEKS, min_periods=REGIME_BASELINE_WEEKS
    ).median()
    ratio = (series + eps) / (baseline + eps)
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=series.index)


def _global_mean(frame: pd.DataFrame) -> pd.Series:
    return frame.mean(axis=1, skipna=True)


def _glob_vol_name(weeks: int) -> str:
    return f"glob_vol_cc_d_{weeks}w"


def _glob_vol_regime_name() -> str:
    return (
        f"glob_vol_regime_cc_{REGIME_FAST_WEEKS}w_{REGIME_BASELINE_WEEKS}w"
    )


def _glob_corr_name(weeks: int) -> str:
    return f"glob_corr_mean_{weeks}w"


def _glob_pc1_name(weeks: int) -> str:
    return f"glob_pc1_share_{weeks}w"


def _glob_disp_ret_name(weeks: int) -> str:
    return f"glob_disp_ret_{weeks}w"


def _glob_disp_mom_name(weeks: int) -> str:
    return f"glob_disp_mom_{weeks}w"


def _glob_disp_vol_name(weeks: int) -> str:
    return f"glob_disp_vol_{weeks}w"
