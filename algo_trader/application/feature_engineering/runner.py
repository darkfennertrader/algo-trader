from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, TypeVar

import numpy as np
import pandas as pd
import torch

from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputNames,
    OutputWriter,
    ensure_directory,
    format_run_at,
    log_boundary,
    resolve_latest_week_dir,
)
from algo_trader.infrastructure.data import (
    ReturnType,
    require_utc_hourly_index,
    timestamps_to_epoch_hours,
    write_tensor_bundle,
)
from algo_trader.infrastructure.paths import format_tilde_path
from algo_trader.pipeline.stages.features import (
    FeatureGroup,
    FeatureInputs,
    HorizonSpec,
    asset_frame,
    ordered_assets,
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
from algo_trader.pipeline.stages.features.registry import default_registry
from ..data_sources import resolve_data_lake, resolve_feature_store

logger = logging.getLogger(__name__)

_WEEKLY_OHLC_NAME = "weekly_ohlc.csv"
_OUTPUT_NAMES = OutputNames(
    output_name="features.csv",
    metadata_name="metadata.json",
)
_TENSOR_NAME = "features_tensor.pt"
_FREQUENCY = "weekly"
_TRADING_DAYS_PER_WEEK = 5
_TENSOR_TIMESTAMP_UNIT = "epoch_hours"
_TENSOR_TIMEZONE = "UTC"
_TENSOR_VALUE_DTYPE = "float64"


@dataclass(frozen=True)
class RunRequest:
    return_type: str
    horizons: str | None = None
    groups: Sequence[str] | None = None
    features: Sequence[str] | None = None


@dataclass(frozen=True)
class FeatureSettings:
    return_type: ReturnType
    horizon_days: Sequence[int] | None
    eps: float


@dataclass(frozen=True)
class FeatureSelection:
    groups: Sequence[str]
    features: Sequence[str] | None


@dataclass(frozen=True)
class FeaturePaths:
    data_lake: Path
    feature_store: Path


@dataclass(frozen=True)
class RunConfig:
    settings: FeatureSettings
    selection: FeatureSelection
    paths: FeaturePaths


@dataclass(frozen=True)
class FeatureOutputPaths:
    output_dir: Path
    output_path: Path
    metadata_path: Path
    tensor_path: Path


@dataclass(frozen=True)
class FeatureTensorBundle:
    values: torch.Tensor
    timestamps: torch.Tensor
    missing_mask: torch.Tensor


@dataclass(frozen=True)
class FeatureDataContext:
    input_path: Path
    frame: pd.DataFrame
    assets: Sequence[str]
    features: Sequence[str]


@dataclass(frozen=True)
class MetadataContext:
    group: str
    paths: FeatureOutputPaths
    data: FeatureDataContext
    settings: FeatureSettings
    horizons: Sequence["HorizonLike"]


class HorizonLike(Protocol):
    @property
    def days(self) -> int: ...

    @property
    def weeks(self) -> int: ...


HorizonSpecT = TypeVar("HorizonSpecT", bound=HorizonLike)


def _run_context(request: RunRequest) -> dict[str, str]:
    return {
        "return_type": request.return_type,
        "horizons": request.horizons or "",
        "groups": ",".join(request.groups or []),
        "features": ",".join(request.features or []),
    }


@log_boundary("feature_engineering.run", context=_run_context)
def run(*, request: RunRequest) -> list[Path]:
    config = _resolve_run_config(request)
    weekly_ohlc, input_path, version_label = _load_weekly_ohlc(
        config.paths.data_lake
    )
    inputs = FeatureInputs(
        frames={"weekly_ohlc": weekly_ohlc}, frequency=_FREQUENCY
    )
    output_paths: list[Path] = []
    for group_name in config.selection.groups:
        group_horizons = _resolve_group_horizons(
            group_name, config.settings.horizon_days
        )
        group = _build_group(group_name, config, group_horizons)
        output = group.compute(inputs)
        paths = _prepare_output_paths(
            config.paths.feature_store, version_label, group_name
        )
        _write_outputs(
            output.frame,
            output.feature_names,
            paths,
            metadata=MetadataContext(
                group=group_name,
                paths=paths,
                data=FeatureDataContext(
                    input_path=input_path,
                    frame=output.frame,
                    assets=ordered_assets(output.frame),
                    features=output.feature_names,
                ),
                settings=config.settings,
                horizons=group_horizons,
            ),
        )
        logger.info(
            "Saved %s features path=%s rows=%s assets=%s",
            group_name,
            paths.output_path,
            len(output.frame),
            len(output.frame.columns),
        )
        output_paths.append(paths.output_path)
    return output_paths


def _resolve_run_config(request: RunRequest) -> RunConfig:
    return_type = _normalize_return_type(request.return_type)
    horizon_days = _normalize_horizon_days(request.horizons)
    data_lake = resolve_data_lake()
    feature_store = resolve_feature_store()
    groups = _resolve_groups(request.groups)
    return RunConfig(
        settings=FeatureSettings(
            return_type=return_type,
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


def _normalize_return_type(raw: str) -> ReturnType:
    normalized = raw.strip().lower()
    if normalized not in ("simple", "log"):
        raise ConfigError(
            "return-type must be simple or log",
            context={"return_type": raw},
        )
    return normalized  # type: ignore[return-value]


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

    if horizon_days is None:
        if group_name == "momentum":
            days = list(DEFAULT_HORIZON_DAYS)
            return _build_horizon_specs(days, _momentum_factory)
        if group_name == "mean_reversion":
            days = list(DEFAULT_MEAN_REV_HORIZON_DAYS)
            return _build_horizon_specs(days, _momentum_factory)
        if group_name == "breakout":
            days = list(DEFAULT_BREAKOUT_DAYS)
            return _build_horizon_specs(days, _momentum_factory)
        raise ConfigError(
            "Unknown feature group for horizons",
            context={"group": group_name},
        )
    if group_name == "momentum":
        return _build_horizon_specs(horizon_days, _momentum_factory)
    if group_name == "mean_reversion":
        return _build_horizon_specs(horizon_days, _momentum_factory)
    if group_name == "breakout":
        return _build_horizon_specs(horizon_days, _momentum_factory)
    raise ConfigError(
        "Unknown feature group for horizons",
        context={"group": group_name},
    )


def _load_weekly_ohlc(
    data_lake: Path,
) -> tuple[pd.DataFrame, Path, str]:
    latest_dir = _resolve_latest_directory(data_lake)
    input_path = latest_dir / _WEEKLY_OHLC_NAME
    frame = _read_weekly_ohlc(input_path)
    return frame, input_path, latest_dir.name


def _resolve_latest_directory(base_dir: Path) -> Path:
    return resolve_latest_week_dir(
        base_dir,
        error_type=DataSourceError,
        error_message="No YYYY-WW data directories found",
    )


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


def _build_group(
    name: str, config: RunConfig, horizons: Sequence[HorizonLike]
) -> FeatureGroup:
    if name == "momentum":
        momentum_horizons = [
            HorizonSpec(days=spec.days, weeks=spec.weeks)
            for spec in horizons
        ]
        momentum_config = MomentumConfig(
            horizons=momentum_horizons,
            return_type=config.settings.return_type,
            eps=config.settings.eps,
            features=config.selection.features,
        )
        return MomentumFeatureGroup(momentum_config)
    if name == "mean_reversion":
        mean_rev_horizons = [
            HorizonSpec(days=spec.days, weeks=spec.weeks)
            for spec in horizons
        ]
        mean_rev_config = MeanReversionConfig(
            horizons=mean_rev_horizons,
            eps=config.settings.eps,
            features=config.selection.features,
        )
        return MeanReversionFeatureGroup(mean_rev_config)
    if name == "breakout":
        breakout_horizons = [
            HorizonSpec(days=spec.days, weeks=spec.weeks)
            for spec in horizons
        ]
        breakout_config = BreakoutConfig(
            horizons=breakout_horizons,
            features=config.selection.features,
        )
        return BreakoutFeatureGroup(breakout_config)
    raise ConfigError(
        "Feature group implementation unavailable",
        context={"group": name},
    )


def _prepare_output_paths(
    feature_store: Path,
    version_label: str,
    group: str,
) -> FeatureOutputPaths:
    output_dir = feature_store / "features" / version_label / group
    ensure_directory(
        output_dir,
        error_type=DataProcessingError,
        invalid_message="Feature output path must be a directory",
        create_message="Failed to prepare feature output directory",
    )
    return FeatureOutputPaths(
        output_dir=output_dir,
        output_path=output_dir / _OUTPUT_NAMES.output_name,
        metadata_path=output_dir / _OUTPUT_NAMES.metadata_name,
        tensor_path=output_dir / _TENSOR_NAME,
    )


def _write_outputs(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    paths: FeatureOutputPaths,
    *,
    metadata: MetadataContext,
) -> None:
    writer = _build_output_writer()
    writer.write_frame(frame, paths.output_path)
    bundle = _build_tensor_bundle(frame, feature_names)
    write_tensor_bundle(
        paths.tensor_path,
        values=bundle.values,
        timestamps=bundle.timestamps,
        missing_mask=bundle.missing_mask,
        error_message="Failed to write feature tensor output",
    )
    payload = _build_metadata(metadata, feature_names)
    writer.write_metadata(payload, paths.metadata_path)


def _build_output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write feature CSV",
        ),
        metadata_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write feature metadata",
        ),
    )


def _build_tensor_bundle(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> FeatureTensorBundle:
    assets = ordered_assets(frame)
    values = _frame_to_tensor(frame, assets, feature_names)
    missing_mask = np.isnan(values)
    index = _require_datetime_index(frame.index)
    timestamps = timestamps_to_epoch_hours(index)
    return FeatureTensorBundle(
        values=torch.tensor(values, dtype=torch.float64),
        timestamps=torch.tensor(timestamps, dtype=torch.int64),
        missing_mask=torch.tensor(missing_mask, dtype=torch.bool),
    )


def _frame_to_tensor(
    frame: pd.DataFrame,
    assets: Sequence[str],
    feature_names: Sequence[str],
) -> np.ndarray:
    if not assets or not feature_names:
        return np.empty((len(frame), 0, 0), dtype=float)
    output = np.empty(
        (len(frame), len(assets), len(feature_names)), dtype=float
    )
    for asset_idx, asset in enumerate(assets):
        asset_data = asset_frame(frame, asset)
        asset_data = asset_data.reindex(columns=list(feature_names))
        output[:, asset_idx, :] = asset_data.to_numpy(dtype=float)
    return output


def _build_metadata(
    context: MetadataContext, feature_names: Sequence[str]
) -> dict[str, object]:
    missing_by_asset = _missing_by_asset(
        context.data.frame, context.data.assets
    )
    return {
        "group": context.group,
        "frequency": _FREQUENCY,
        "return_type": context.settings.return_type,
        "eps": context.settings.eps,
        "horizons": [
            {"days": spec.days, "weeks": spec.weeks}
            for spec in context.horizons
        ],
        "feature_name_units": "weeks",
        "days_to_weeks": {
            str(spec.days): spec.weeks for spec in context.horizons
        },
        "rows": len(context.data.frame),
        "assets": len(context.data.assets),
        "features": len(feature_names),
        "feature_names": list(feature_names),
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source": format_tilde_path(context.data.input_path),
        "destination": format_tilde_path(context.paths.output_path),
        "artifacts": {
            "tensor": format_tilde_path(context.paths.tensor_path),
            "tensor_timestamp_unit": _TENSOR_TIMESTAMP_UNIT,
            "tensor_timezone": _TENSOR_TIMEZONE,
            "tensor_value_dtype": _TENSOR_VALUE_DTYPE,
        },
        "missing_by_asset": missing_by_asset,
    }


def _missing_by_asset(
    frame: pd.DataFrame, assets: Iterable[str]
) -> dict[str, object]:
    missing: dict[str, object] = {}
    for asset in assets:
        asset_data = asset_frame(frame, asset)
        counts = asset_data.isna().sum()
        missing[asset] = {
            "total": int(counts.sum()),
            "by_feature": {
                feature: int(count)
                for feature, count in counts.items()
            },
        }
    return missing


def _require_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            "feature index must be datetime",
            context={"index_type": type(index).__name__},
        )
    return index
