from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import torch

from algo_trader.domain import ConfigError, DataProcessingError
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputWriter,
    ensure_directory,
    format_run_at,
)
from algo_trader.infrastructure.data import (
    timestamps_to_epoch_hours,
    write_tensor_bundle,
)
from algo_trader.infrastructure.paths import format_tilde_path
from algo_trader.pipeline.stages.features import (
    FeatureGroup,
    FeatureInputs,
    FeatureOutput,
    asset_frame,
    ordered_assets,
    require_daily_ohlc,
    require_weekly_ohlc,
)
from algo_trader.pipeline.stages.features.regime import (
    RegimeFeatureGroup,
    RegimeGoodness,
)
from algo_trader.pipeline.stages.features.seasonal import SeasonalFeatureGroup
from algo_trader.pipeline.stages.features.volatility import (
    VolatilityFeatureGroup,
    VolatilityGoodness,
)

from .constants import (
    _GOODNESS_NAME,
    _OUTPUT_NAMES,
    _TENSOR_NAME,
    _TENSOR_TIMESTAMP_UNIT,
    _TENSOR_TIMEZONE,
    _TENSOR_VALUE_DTYPE,
    _TRADING_DAYS_PER_WEEK,
)
from .goodness import daily_goodness_ratios, weekly_goodness_ratios
from .types import FeatureInputSources, FeatureSettings


class HorizonLike(Protocol):
    @property
    def days(self) -> int: ...

    @property
    def weeks(self) -> int: ...


@dataclass(frozen=True)
class FeatureOutputPaths:
    output_dir: Path
    output_path: Path
    metadata_path: Path
    tensor_path: Path
    goodness_path: Path


@dataclass(frozen=True)
class GoodnessContext:
    group: FeatureGroup
    group_name: str
    output: FeatureOutput
    inputs: FeatureInputs
    paths: FeatureOutputPaths
    sources: FeatureInputSources


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
    horizons: Sequence[HorizonLike]
    sources: FeatureInputSources


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
        goodness_path=output_dir / _GOODNESS_NAME,
    )


def _write_outputs(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    paths: FeatureOutputPaths,
    *,
    metadata: MetadataContext,
    extra_metadata: Mapping[str, object] | None = None,
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
    payload = _build_metadata(
        metadata, feature_names, extra_metadata=extra_metadata
    )
    writer.write_metadata(payload, paths.metadata_path)


def _build_goodness_payload(
    context: GoodnessContext,
) -> dict[str, object] | None:
    group = context.group
    sources = _build_source_paths(context.group_name, context.sources)
    if isinstance(group, VolatilityFeatureGroup):
        if group.goodness is None:
            return None
        return _build_volatility_goodness_payload(
            group.goodness,
            sources=sources,
            destination_path=context.paths.output_path,
        )
    if isinstance(group, RegimeFeatureGroup):
        if group.goodness is None:
            return None
        return _build_regime_goodness_payload(
            group.goodness,
            sources=sources,
            destination_path=context.paths.output_path,
        )
    weekly = require_weekly_ohlc(context.inputs)
    horizon_days_by_feature = _weekly_horizon_days_by_feature(
        context.group_name, context.output.feature_names
    )
    if _group_uses_daily_source(context.group_name):
        daily = require_daily_ohlc(context.inputs)
        ratios_by_feature = daily_goodness_ratios(
            daily,
            weekly,
            horizon_days_by_feature=horizon_days_by_feature,
            trading_days_per_week=_TRADING_DAYS_PER_WEEK,
        )
        ratio_definition = (
            "avg_missing_daily_ohlc_fraction_over_horizon_weeks"
        )
    else:
        ratios_by_feature = weekly_goodness_ratios(
            weekly,
            horizon_days_by_feature=horizon_days_by_feature,
            trading_days_per_week=_TRADING_DAYS_PER_WEEK,
        )
        ratio_definition = "missing_weekly_ohlc / horizon_weeks"
    return {
        "group": context.group_name,
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source": sources,
        "dest": format_tilde_path(context.paths.output_path),
        "ratio_definition": ratio_definition,
        "ratios_by_feature": ratios_by_feature,
    }


def _build_source_paths(
    group_name: str, sources: FeatureInputSources
) -> list[str]:
    return [
        _resolve_source_path(group_name, sources, frequency)
        for frequency in _group_source_frequencies(group_name)
    ]


def _resolve_source_path(
    group_name: str, sources: FeatureInputSources, frequency: str
) -> str:
    if frequency == "weekly":
        return format_tilde_path(sources.weekly_path)
    if frequency == "daily":
        if sources.daily_path is None:
            raise DataProcessingError(
                "daily_ohlc source path is required for metadata output",
                context={"group": group_name},
            )
        return format_tilde_path(sources.daily_path)
    if frequency == "hourly":
        if sources.hourly_path is None:
            raise DataProcessingError(
                "hourly_ohlc source path is required for metadata output",
                context={"group": group_name},
            )
        return format_tilde_path(sources.hourly_path)
    raise ConfigError(
        "Unknown feature frequency for metadata output",
        context={"group": group_name, "frequency": frequency},
    )


def _group_source_frequencies(group_name: str) -> tuple[str, ...]:
    frequencies_by_group = {
        "momentum": ("weekly",),
        "mean_reversion": ("weekly",),
        "breakout": ("weekly",),
        "cross_sectional": ("weekly", "daily"),
        "volatility": ("weekly", "daily"),
        "seasonal": ("weekly", "daily"),
        "regime": ("weekly", "daily"),
    }
    frequencies = frequencies_by_group.get(group_name)
    if frequencies is None:
        raise ConfigError(
            "Unknown feature group for goodness output",
            context={"group": group_name},
        )
    return frequencies


def _group_uses_daily_source(group_name: str) -> bool:
    return "daily" in _group_source_frequencies(group_name)


def _extra_metadata(group: FeatureGroup) -> Mapping[str, object] | None:
    if isinstance(group, SeasonalFeatureGroup):
        missing = group.missing_weekdays
        if missing is None:
            return None
        return {"missing_weekdays_by_asset": missing}
    return None


def _write_goodness_payload(
    payload: dict[str, object],
    *,
    paths: FeatureOutputPaths,
) -> None:
    writer = _build_output_writer()
    writer.write_metadata(payload, paths.goodness_path)


def _build_volatility_goodness_payload(
    goodness: VolatilityGoodness,
    *,
    sources: Sequence[str],
    destination_path: Path,
) -> dict[str, object]:
    return {
        "group": "volatility",
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source": sources,
        "dest": format_tilde_path(destination_path),
        "ratio_definition": "valid_daily_ohlc / horizon_days",
        "ratios_by_feature": goodness.ratios_by_feature,
    }


def _build_regime_goodness_payload(
    goodness: RegimeGoodness,
    *,
    sources: Sequence[str],
    destination_path: Path,
) -> dict[str, object]:
    return {
        "group": "regime",
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source": sources,
        "dest": format_tilde_path(destination_path),
        "ratio_definition": "valid_daily_ohlc / horizon_days",
        "ratios_by_feature": goodness.ratios_by_feature,
    }


def _weekly_horizon_days_by_feature(
    group_name: str, feature_names: Sequence[str]
) -> dict[str, int]:
    if not feature_names:
        return {}
    mapping: dict[str, int] = {}
    for name in feature_names:
        weeks = _feature_weeks(group_name, name)
        if weeks is None:
            continue
        mapping[name] = weeks * _TRADING_DAYS_PER_WEEK
    return mapping


def _feature_weeks(group_name: str, feature_name: str) -> int | None:
    handlers = {
        "momentum": _momentum_feature_weeks,
        "mean_reversion": _mean_reversion_feature_weeks,
        "breakout": _parse_weeks_suffix,
        "cross_sectional": _cross_sectional_feature_weeks,
        "seasonal": _parse_weeks_suffix,
        "regime": _parse_weeks_suffix,
    }
    handler = handlers.get(group_name)
    if handler is None:
        return None
    return handler(feature_name)


def _momentum_feature_weeks(feature_name: str) -> int | None:
    if feature_name.startswith("ema_spread_"):
        parts = feature_name.replace("ema_spread_", "").split("w_")
        if len(parts) == 2:
            long_weeks = max(int(parts[0]), int(parts[1].rstrip("w")))
            return long_weeks
    return _parse_weeks_suffix(feature_name)


def _mean_reversion_feature_weeks(feature_name: str) -> int | None:
    if feature_name == "range_pos_1w":
        return 1
    return _parse_weeks_suffix(feature_name)


def _cross_sectional_feature_weeks(feature_name: str) -> int | None:
    prefixes = ("cs_centered_", "cs_rank_")
    for prefix in prefixes:
        if feature_name.startswith(prefix):
            base = feature_name[len(prefix) :]
            return _parse_weeks_suffix(base)
    return None


def _parse_weeks_suffix(feature_name: str) -> int | None:
    if not feature_name.endswith("w"):
        return None
    parts = feature_name.rsplit("_", maxsplit=1)
    if len(parts) != 2:
        return None
    suffix = parts[1]
    if not suffix.endswith("w"):
        return None
    try:
        return int(suffix[:-1])
    except ValueError:
        return None


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
    context: MetadataContext,
    feature_names: Sequence[str],
    *,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    missing_by_asset = _missing_by_asset(
        context.data.frame, context.data.assets
    )
    payload: dict[str, object] = {
        "group": context.group,
        "frequency": list(_group_source_frequencies(context.group)),
        "return_type": context.settings.return_type,
        "eps": context.settings.eps,
        "days_to_weeks": {
            str(spec.days): spec.weeks for spec in context.horizons
        },
        "rows": len(context.data.frame),
        "assets": len(context.data.assets),
        "features": len(feature_names),
        "feature_names": list(feature_names),
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source": _build_source_paths(context.group, context.sources),
        "destination": format_tilde_path(context.paths.output_path),
        "artifacts": {
            "tensor": format_tilde_path(context.paths.tensor_path),
            "tensor_timestamp_unit": _TENSOR_TIMESTAMP_UNIT,
            "tensor_timezone": _TENSOR_TIMEZONE,
            "tensor_value_dtype": _TENSOR_VALUE_DTYPE,
        },
        "missing_by_asset": missing_by_asset,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def _missing_by_asset(
    frame: pd.DataFrame, assets: Iterable[str]
) -> dict[str, object]:
    missing: dict[str, object] = {}
    for asset in assets:
        asset_data = asset_frame(frame, asset)
        counts = asset_data.isna().sum()
        by_feature = {
            feature: int(count)
            for feature, count in counts.items()
            if int(count) > 0
        }
        missing[asset] = {
            "total": int(counts.sum()),
            "by_feature": by_feature,
        }
    return missing


def _require_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            "feature index must be datetime",
            context={"index_type": type(index).__name__},
        )
    return index
