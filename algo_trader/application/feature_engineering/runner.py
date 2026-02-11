from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, TypeVar

import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.infrastructure import log_boundary, resolve_latest_week_dir
from algo_trader.infrastructure.data import require_utc_hourly_index
from algo_trader.pipeline.stages.features import (
    FeatureGroup,
    FeatureInputs,
    HorizonSpec,
    ordered_assets,
)
from algo_trader.pipeline.stages.features.cross_sectional import (
    DEFAULT_HORIZON_DAYS as DEFAULT_CROSS_SECTIONAL_HORIZON_DAYS,
    CrossSectionalConfig,
    CrossSectionalFeatureGroup,
)
from algo_trader.pipeline.stages.features.breakout import (
    DEFAULT_HORIZON_DAYS as DEFAULT_BREAKOUT_DAYS,
    BreakoutConfig,
    BreakoutFeatureGroup,
)
from algo_trader.pipeline.stages.features.mean_reversion import (
    DEFAULT_HORIZON_DAYS as DEFAULT_MEAN_REV_HORIZON_DAYS,
    MeanReversionConfig,
    MeanReversionFeatureGroup,
)
from algo_trader.pipeline.stages.features.momentum import (
    DEFAULT_EPSILON,
    DEFAULT_HORIZON_DAYS,
    MomentumConfig,
    MomentumFeatureGroup,
)
from algo_trader.pipeline.stages.features.volatility import (
    DEFAULT_HORIZON_DAYS as DEFAULT_VOLATILITY_HORIZON_DAYS,
    VolatilityConfig,
    VolatilityFeatureGroup,
)
from algo_trader.pipeline.stages.features.seasonal import (
    DEFAULT_HORIZON_DAYS as DEFAULT_SEASONAL_HORIZON_DAYS,
    SeasonalConfig,
    SeasonalFeatureGroup,
)
from algo_trader.pipeline.stages.features.regime import (
    DEFAULT_EPSILON as DEFAULT_REGIME_EPSILON,
    DEFAULT_HORIZON_DAYS as DEFAULT_REGIME_HORIZON_DAYS,
    RegimeConfig,
    RegimeFeatureGroup,
)
from algo_trader.pipeline.stages.features.registry import default_registry
from .constants import (
    _ALL_GROUP,
    _CROSS_SECTIONAL_GROUP,
    _DAILY_OHLC_NAME,
    _FREQUENCY,
    _TENSOR_TIMEZONE,
    _TRADING_DAYS_PER_WEEK,
    _WEEKLY_OHLC_NAME,
)
from .outputs import (
    FeatureDataContext,
    GoodnessContext,
    MetadataContext,
    _build_goodness_payload,
    _extra_metadata,
    _prepare_output_paths,
    _write_goodness_payload,
    _write_outputs,
)
from .types import (
    FeatureInputSources,
    FeaturePaths,
    FeatureSelection,
    FeatureSettings,
    RunConfig,
)
from ..data_sources import resolve_data_lake, resolve_feature_store

logger = logging.getLogger(__name__)
_DAILY_REQUIRED_GROUPS = {"volatility", "seasonal", "regime"}


@dataclass(frozen=True)
class RunRequest:
    horizons: str | None = None
    groups: Sequence[str] | None = None
    features: Sequence[str] | None = None


class HorizonLike(Protocol):
    @property
    def days(self) -> int: ...

    @property
    def weeks(self) -> int: ...


HorizonSpecT = TypeVar("HorizonSpecT", bound=HorizonLike)


def _run_context(request: RunRequest) -> dict[str, str]:
    return {
        "horizons": request.horizons or "",
        "groups": ",".join(request.groups or []),
        "features": ",".join(request.features or []),
    }


@log_boundary("feature_engineering.run", context=_run_context)
def run(*, request: RunRequest) -> list[Path]:
    start_time = time.perf_counter()
    use_parallel_all = _uses_parallel_all(request.groups)
    config = _resolve_run_config(request)
    if use_parallel_all:
        output_paths = _run_parallel_all(config)
        _log_run_duration(config.selection.groups, start_time)
        return output_paths
    needs_daily = bool(
        set(config.selection.groups).intersection(
            _DAILY_REQUIRED_GROUPS
        )
    )
    inputs, sources, version_label = _load_feature_inputs(
        config.paths.data_lake, needs_daily=needs_daily
    )
    output_paths: list[Path] = []
    for group_name in config.selection.groups:
        output_paths.append(
            _run_group_with_inputs(
                group_name=group_name,
                config=config,
                inputs=inputs,
                sources=sources,
                version_label=version_label,
            )
        )
    _log_run_duration(config.selection.groups, start_time)
    return output_paths


def _run_parallel_all(config: RunConfig) -> list[Path]:
    latest_dir = _resolve_latest_directory(config.paths.data_lake)
    worker_count = _resolve_parallel_workers()
    parallel_groups = [
        group
        for group in config.selection.groups
        if group != _CROSS_SECTIONAL_GROUP
    ]
    output_paths: list[Path] = []
    if parallel_groups:
        logger.info(
            "Running feature groups in parallel workers=%s groups=%s",
            worker_count,
            ",".join(parallel_groups),
        )
        output_paths.extend(
            _run_parallel_groups(
                groups=parallel_groups,
                config=config,
                latest_dir=latest_dir,
                worker_count=worker_count,
            )
        )
    if _CROSS_SECTIONAL_GROUP in config.selection.groups:
        output_paths.append(
            _run_group_job(_CROSS_SECTIONAL_GROUP, config, latest_dir)
        )
    return output_paths


def _resolve_parallel_workers() -> int:
    count = os.cpu_count()
    if count is None:
        return 1
    return max(1, count - 1)


def _run_parallel_groups(
    *,
    groups: Sequence[str],
    config: RunConfig,
    latest_dir: Path,
    worker_count: int,
) -> list[Path]:
    output_paths: list[Path] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures_by_group = {
            group: executor.submit(_run_group_job, group, config, latest_dir)
            for group in groups
        }
        for group in groups:
            future = futures_by_group[group]
            try:
                output_paths.append(future.result())
            except Exception as exc:
                for pending in futures_by_group.values():
                    pending.cancel()
                raise DataProcessingError(
                    "Feature group computation failed",
                    context={"group": group},
                ) from exc
    return output_paths


def _run_group_job(
    group_name: str, config: RunConfig, latest_dir: Path
) -> Path:
    inputs, sources, version_label = _load_feature_inputs_from_dir(
        latest_dir, needs_daily=_group_requires_daily(group_name)
    )
    return _run_group_with_inputs(
        group_name=group_name,
        config=config,
        inputs=inputs,
        sources=sources,
        version_label=version_label,
    )


def _run_group_with_inputs(
    *,
    group_name: str,
    config: RunConfig,
    inputs: FeatureInputs,
    sources: FeatureInputSources,
    version_label: str,
) -> Path:
    start_time = time.perf_counter()
    group_horizons = _resolve_group_horizons(
        group_name, config.settings.horizon_days
    )
    group = _build_group(group_name, config, group_horizons)
    output = group.compute(inputs)
    paths = _prepare_output_paths(
        config.paths.feature_store, version_label, group_name
    )
    extra_metadata = _extra_metadata(group)
    _write_outputs(
        output.frame,
        output.feature_names,
        paths,
        metadata=MetadataContext(
            group=group_name,
            paths=paths,
            data=FeatureDataContext(
                input_path=sources.weekly_path,
                frame=output.frame,
                assets=ordered_assets(output.frame),
                features=output.feature_names,
            ),
            settings=config.settings,
            horizons=group_horizons,
            sources=sources,
        ),
        extra_metadata=extra_metadata,
    )
    goodness_payload = _build_goodness_payload(
        GoodnessContext(
            group=group,
            group_name=group_name,
            output=output,
            inputs=inputs,
            paths=paths,
            sources=sources,
        )
    )
    if goodness_payload is not None:
        _write_goodness_payload(goodness_payload, paths=paths)
    duration = time.perf_counter() - start_time
    logger.info(
        "Saved %s features path=%s rows=%s assets=%s duration=%.2fs",
        group_name,
        paths.output_path,
        len(output.frame),
        len(output.frame.columns),
        duration,
    )
    return paths.output_path


def _log_run_duration(groups: Sequence[str], start_time: float) -> None:
    duration = time.perf_counter() - start_time
    duration_minutes = duration / 60.0
    logger.info("----- TOTAL -----")
    logger.info(
        "Feature engineering completed groups=%s duration=%.2fm",
        ",".join(groups),
        duration_minutes,
    )


def _resolve_run_config(request: RunRequest) -> RunConfig:
    horizon_days = _normalize_horizon_days(request.horizons)
    data_lake = resolve_data_lake()
    feature_store = resolve_feature_store()
    groups = _resolve_groups(request.groups)
    return RunConfig(
        settings=FeatureSettings(
            return_type="log",
            horizon_days=horizon_days,
            eps=DEFAULT_EPSILON,
        ),
        selection=FeatureSelection(
            groups=groups,
            features=request.features,
        ),
        paths=FeaturePaths(
            data_lake=data_lake,
            feature_store=feature_store,
        ),
    )


def _resolve_groups(groups: Sequence[str] | None) -> list[str]:
    registry = default_registry()
    available = registry.list_names()
    if not groups:
        return available
    normalized = [_normalize_group_name(name) for name in groups]
    if _ALL_GROUP in normalized:
        if set(normalized) != {_ALL_GROUP}:
            raise ConfigError(
                "group=all cannot be combined with other groups",
                context={"groups": ",".join(normalized)},
            )
        return available
    unknown = sorted(set(normalized).difference(available))
    if unknown:
        raise ConfigError(
            "Unknown feature groups requested",
            context={"groups": ",".join(unknown)},
        )
    return normalized


def _normalize_group_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("feature group name must not be empty")
    return normalized


def _uses_parallel_all(groups: Sequence[str] | None) -> bool:
    if not groups:
        return False
    normalized = [_normalize_group_name(name) for name in groups]
    if _ALL_GROUP in normalized:
        if set(normalized) != {_ALL_GROUP}:
            raise ConfigError(
                "group=all cannot be combined with other groups",
                context={"groups": ",".join(normalized)},
            )
        return True
    return False


def _normalize_horizon_days(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    days: list[int] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            days.append(int(value))
        except ValueError as exc:
            raise ConfigError(
                "horizons must be comma-separated integers",
                context={"value": value},
            ) from exc
    return _validate_horizon_days(days)


def _validate_horizon_days(days: Sequence[int]) -> list[int]:
    validated: list[int] = []
    seen: set[int] = set()
    for day in days:
        if day <= 0:
            raise ConfigError(
                "horizons must be positive day counts",
                context={"value": str(day)},
            )
        if day in seen:
            continue
        seen.add(day)
        validated.append(day)
    if not validated:
        raise ConfigError("horizons must not be empty")
    return validated


def _build_horizon_specs(
    days: Sequence[int], factory: Callable[[int, int], HorizonSpecT]
) -> list[HorizonSpecT]:
    specs: list[HorizonSpecT] = []
    seen: set[int] = set()
    for day in days:
        if day <= 0:
            raise ConfigError(
                "horizons must be positive day counts",
                context={"value": str(day)},
            )
        if day in seen:
            continue
        seen.add(day)
        weeks = int(math.ceil(day / _TRADING_DAYS_PER_WEEK))
        if weeks <= 0:
            raise ConfigError(
                "horizon must map to at least one week",
                context={"value": str(day)},
            )
        specs.append(factory(day, weeks))
    if not specs:
        raise ConfigError("horizons must not be empty")
    return specs


def _resolve_group_horizons(
    group_name: str, horizon_days: Sequence[int] | None
) -> list[HorizonLike]:
    def _momentum_factory(day: int, weeks: int) -> HorizonSpec:
        return HorizonSpec(days=day, weeks=weeks)

    default_days_by_group = {
        "momentum": list(DEFAULT_HORIZON_DAYS),
        "mean_reversion": list(DEFAULT_MEAN_REV_HORIZON_DAYS),
        "breakout": list(DEFAULT_BREAKOUT_DAYS),
        "cross_sectional": list(DEFAULT_CROSS_SECTIONAL_HORIZON_DAYS),
        "volatility": list(DEFAULT_VOLATILITY_HORIZON_DAYS),
        "seasonal": list(DEFAULT_SEASONAL_HORIZON_DAYS),
        "regime": list(DEFAULT_REGIME_HORIZON_DAYS),
    }
    if horizon_days is None:
        days = default_days_by_group.get(group_name)
    else:
        days = list(horizon_days) if group_name in default_days_by_group else None
    if not days:
        raise ConfigError(
            "Unknown feature group for horizons",
            context={"group": group_name},
        )
    return _build_horizon_specs(days, _momentum_factory)


def _load_feature_inputs(
    data_lake: Path, *, needs_daily: bool
) -> tuple[FeatureInputs, FeatureInputSources, str]:
    latest_dir = _resolve_latest_directory(data_lake)
    return _load_feature_inputs_from_dir(latest_dir, needs_daily=needs_daily)


def _load_feature_inputs_from_dir(
    latest_dir: Path, *, needs_daily: bool
) -> tuple[FeatureInputs, FeatureInputSources, str]:
    weekly_path = latest_dir / _WEEKLY_OHLC_NAME
    weekly_ohlc = _read_weekly_ohlc(weekly_path)
    daily_ohlc: pd.DataFrame | None = None
    daily_path = latest_dir / _DAILY_OHLC_NAME
    if daily_path.exists():
        daily_ohlc = _read_daily_ohlc(daily_path)
    elif needs_daily:
        daily_ohlc = _read_daily_ohlc(daily_path)
    else:
        daily_path = None
    frames: dict[str, pd.DataFrame] = {"weekly_ohlc": weekly_ohlc}
    if daily_ohlc is not None:
        frames["daily_ohlc"] = daily_ohlc
    inputs = FeatureInputs(frames=frames, frequency=_FREQUENCY)
    sources = FeatureInputSources(
        weekly_path=weekly_path,
        daily_path=daily_path,
        hourly_path=None,
    )
    return inputs, sources, latest_dir.name


def _resolve_latest_directory(base_dir: Path) -> Path:
    return resolve_latest_week_dir(
        base_dir,
        error_type=DataSourceError,
        error_message="No YYYY-WW data directories found",
    )


def _group_requires_daily(group_name: str) -> bool:
    return group_name in _DAILY_REQUIRED_GROUPS


def _read_weekly_ohlc(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataSourceError(
            "weekly_ohlc.csv not found",
            context={"path": str(path)},
        )
    try:
        frame = pd.read_csv(
            path,
            header=[0, 1],
            index_col=0,
            parse_dates=[0],
        )
    except Exception as exc:
        raise DataSourceError(
            "Failed to read weekly_ohlc CSV",
            context={"path": str(path)},
        ) from exc
    require_utc_hourly_index(
        frame.index, label="weekly_ohlc", timezone=_TENSOR_TIMEZONE
    )
    return frame


def _read_daily_ohlc(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataSourceError(
            "daily_ohlc.csv not found",
            context={"path": str(path)},
        )
    try:
        frame = pd.read_csv(
            path,
            header=[0, 1],
            index_col=0,
            parse_dates=[0],
        )
    except Exception as exc:
        raise DataSourceError(
            "Failed to read daily_ohlc CSV",
            context={"path": str(path)},
        ) from exc
    require_utc_hourly_index(
        frame.index, label="daily_ohlc", timezone=_TENSOR_TIMEZONE
    )
    return frame


def _build_group(
    name: str, config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    builders: dict[str, Callable[[RunConfig, Sequence[HorizonLike]], FeatureGroup]] = {
        "momentum": _build_momentum_group,
        "mean_reversion": _build_mean_reversion_group,
        "breakout": _build_breakout_group,
        "cross_sectional": _build_cross_sectional_group,
        "volatility": _build_volatility_group,
        "seasonal": _build_seasonal_group,
        "regime": _build_regime_group,
    }
    builder = builders.get(name)
    if builder is None:
        raise ConfigError(
            "Feature group implementation unavailable",
            context={"group": name},
        )
    return builder(config, horizons)


def _build_horizon_specs_for_group(
    horizons: Sequence[HorizonLike],
) -> list[HorizonSpec]:
    return [HorizonSpec(days=spec.days, weeks=spec.weeks) for spec in horizons]


def _build_momentum_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    momentum_horizons = _build_horizon_specs_for_group(horizons)
    momentum_config = MomentumConfig(
        horizons=momentum_horizons,
        eps=config.settings.eps,
        features=config.selection.features,
    )
    return MomentumFeatureGroup(momentum_config)


def _build_mean_reversion_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    mean_rev_horizons = _build_horizon_specs_for_group(horizons)
    mean_rev_config = MeanReversionConfig(
        horizons=mean_rev_horizons,
        eps=config.settings.eps,
        features=config.selection.features,
    )
    return MeanReversionFeatureGroup(mean_rev_config)


def _build_breakout_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    breakout_horizons = _build_horizon_specs_for_group(horizons)
    breakout_config = BreakoutConfig(
        horizons=breakout_horizons,
        features=config.selection.features,
    )
    return BreakoutFeatureGroup(breakout_config)


def _build_cross_sectional_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    cross_sectional_horizons = _build_horizon_specs_for_group(horizons)
    cross_sectional_config = CrossSectionalConfig(
        horizons=cross_sectional_horizons,
        eps=config.settings.eps,
        features=config.selection.features,
    )
    return CrossSectionalFeatureGroup(cross_sectional_config)


def _build_volatility_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    volatility_horizons = _build_horizon_specs_for_group(horizons)
    volatility_config = VolatilityConfig(
        horizons=volatility_horizons,
        eps=config.settings.eps,
        features=config.selection.features,
    )
    return VolatilityFeatureGroup(volatility_config)


def _build_seasonal_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    seasonal_horizons = _build_horizon_specs_for_group(horizons)
    seasonal_config = SeasonalConfig(
        horizons=seasonal_horizons,
        features=config.selection.features,
    )
    return SeasonalFeatureGroup(seasonal_config)


def _build_regime_group(
    config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    regime_horizons = _build_horizon_specs_for_group(horizons)
    regime_config = RegimeConfig(
        horizons=regime_horizons,
        eps=DEFAULT_REGIME_EPSILON,
        features=config.selection.features,
    )
    return RegimeFeatureGroup(regime_config)
