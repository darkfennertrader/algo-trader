from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from algo_trader.application.data_io import read_indexed_csv
from algo_trader.application.data_sources import (
    resolve_data_lake,
    resolve_feature_store,
)
from algo_trader.application.exogenous.config import (
    FredFamilyConfig,
    FredRequestConfig,
    FredSeriesConfig,
)
from algo_trader.domain import ConfigError, DataProcessingError
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputPaths as BaseOutputPaths,
    OutputWriter,
    ensure_directory,
    format_run_at,
    format_tilde_path,
    log_boundary,
    resolve_latest_week_dir,
)
from algo_trader.infrastructure.data import (
    require_utc_hourly_index,
    timestamps_to_epoch_hours,
    write_tensor_bundle,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "fred_config.yml"
)
_FEATURE_NAMESPACE = "exogenous"
_ASSET_BLOCK = "asset"
_GLOBAL_BLOCK = "global"
_OUTPUT_NAME = "features.csv"
_METADATA_NAME = "metadata.json"
_TENSOR_NAME = "features_tensor.pt"
_TENSOR_TIMESTAMP_UNIT = "epoch_hours"
_TENSOR_TIMEZONE = "UTC"
_TENSOR_VALUE_DTYPE = "float64"
_CARRY_FEATURE_NAME = "carry_3m_diff"
_LOG_LEVEL_FAMILIES = {
    "equity_implied_vol",
    "broad_usd_factor",
    "credit_spreads_risk",
}


@dataclass(frozen=True)
class RunRequest:
    config_path: Path | None = None
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class RunSources:
    version_dir: Path
    returns_path: Path
    exogenous_path: Path


@dataclass(frozen=True)
class ExogenousOutputPaths:
    base: BaseOutputPaths
    tensor_path: Path


@dataclass(frozen=True)
class DerivedFeatureSpec:
    feature_name: str
    family_key: str
    transform: str
    source_series: tuple[str, ...]
    future_role: str
    applicability: str


@dataclass(frozen=True)
class AssetFeatureBlock:
    frame: pd.DataFrame
    feature_names: list[str]
    feature_specs: Sequence[DerivedFeatureSpec]
    assets: list[str]


@dataclass(frozen=True)
class GlobalFeatureBlock:
    frame: pd.DataFrame
    feature_names: list[str]
    feature_specs: Sequence[DerivedFeatureSpec]


def _run_context(request: RunRequest) -> dict[str, str]:
    return {
        "config_path": str(request.config_path or DEFAULT_CONFIG_PATH),
        "start_date": request.start_date or "",
        "end_date": request.end_date or "",
    }


@log_boundary("exogenous_feature_engineering.run", context=_run_context)
def run(*, request: RunRequest) -> list[Path]:
    config = FredRequestConfig.load(request.config_path or DEFAULT_CONFIG_PATH)
    sources = _resolve_sources(config)
    returns_frame = _load_indexed_frame(
        sources.returns_path,
        missing_message="returns.csv not found for exogenous_feature_engineering",
        read_message="Failed to read returns.csv for exogenous_feature_engineering",
    )
    exogenous_frame = _load_indexed_frame(
        sources.exogenous_path,
        missing_message=(
            "exogenous_cleaned.csv not found for exogenous_feature_engineering"
        ),
        read_message=(
            "Failed to read exogenous_cleaned.csv for exogenous_feature_engineering"
        ),
    )
    date_window = _resolve_date_window(request)
    common_index = _resolve_common_index(
        returns_frame.index,
        exogenous_frame.index,
        date_window=date_window,
    )
    aligned_returns = returns_frame.loc[common_index].copy()
    aligned_exogenous = exogenous_frame.loc[common_index].copy()
    asset_block = _build_asset_block(
        config=config,
        returns_frame=aligned_returns,
        exogenous_frame=aligned_exogenous,
    )
    global_block = _build_global_block(config=config, exogenous_frame=aligned_exogenous)
    feature_store = resolve_feature_store()
    asset_paths = _prepare_output_paths(
        feature_store, sources.version_dir.name, block_name=_ASSET_BLOCK
    )
    global_paths = _prepare_output_paths(
        feature_store, sources.version_dir.name, block_name=_GLOBAL_BLOCK
    )
    writer = _build_output_writer()
    _write_asset_outputs(
        block=asset_block,
        paths=asset_paths,
        writer=writer,
        sources=sources,
        request=request,
    )
    _write_global_outputs(
        block=global_block,
        paths=global_paths,
        writer=writer,
        sources=sources,
        request=request,
    )
    logger.info(
        "Saved exogenous feature outputs asset_path=%s global_path=%s rows=%s",
        asset_paths.base.output_path,
        global_paths.base.output_path,
        len(common_index),
    )
    return [asset_paths.base.output_path, global_paths.base.output_path]


def _resolve_sources(config: FredRequestConfig) -> RunSources:
    version_dir = resolve_latest_week_dir(
        resolve_data_lake(),
        error_type=DataProcessingError,
        error_message="No YYYY-WW data directories found",
    )
    returns_path = version_dir / config.cleaning.calendar_index_file
    exogenous_path = (
        version_dir
        / config.cleaning.output.subdir
        / config.cleaning.output.cleaned_csv
    )
    return RunSources(
        version_dir=version_dir,
        returns_path=returns_path,
        exogenous_path=exogenous_path,
    )


def _load_indexed_frame(
    path: Path, *, missing_message: str, read_message: str
) -> pd.DataFrame:
    frame = read_indexed_csv(
        path,
        missing_message=missing_message,
        read_message=read_message,
    )
    index = require_utc_hourly_index(
        frame.index,
        label=path.stem,
        timezone=_TENSOR_TIMEZONE,
    )
    frame.index = index
    return frame.sort_index()


def _resolve_common_index(
    returns_index: pd.Index,
    exogenous_index: pd.Index,
    *,
    date_window: tuple[date | None, date | None],
) -> pd.DatetimeIndex:
    common = returns_index.intersection(exogenous_index)
    common = _apply_date_window(common, date_window=date_window)
    if common.empty:
        raise DataProcessingError(
            "Returns and cleaned exogenous data do not share any timestamps"
        )
    return require_utc_hourly_index(
        pd.DatetimeIndex(common),
        label="exogenous_feature_engineering_common_index",
        timezone=_TENSOR_TIMEZONE,
    )


def _resolve_date_window(request: RunRequest) -> tuple[date | None, date | None]:
    start_date = _parse_optional_date(
        request.start_date, field_name="start_date"
    )
    end_date = _parse_optional_date(request.end_date, field_name="end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ConfigError(
            "start_date must be <= end_date",
            context={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
    return start_date, end_date


def _parse_optional_date(
    value: str | None, *, field_name: str
) -> date | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{field_name} must be in YYYY-MM-DD format",
            context={field_name: raw},
        ) from exc


def _apply_date_window(
    index: pd.Index, *, date_window: tuple[date | None, date | None]
) -> pd.DatetimeIndex:
    filtered = pd.DatetimeIndex(index)
    normalized = filtered.tz_convert(_TENSOR_TIMEZONE).normalize()
    start_date, end_date = date_window
    if start_date is not None:
        start_stamp = pd.Timestamp(start_date, tz=_TENSOR_TIMEZONE)
        filtered = filtered[normalized >= start_stamp]
        normalized = normalized[normalized >= start_stamp]
    if end_date is not None:
        end_stamp = pd.Timestamp(end_date, tz=_TENSOR_TIMEZONE)
        filtered = filtered[normalized <= end_stamp]
    if filtered.empty:
        context: dict[str, str] = {}
        if start_date is not None:
            context["start_date"] = start_date.isoformat()
        if end_date is not None:
            context["end_date"] = end_date.isoformat()
        raise DataProcessingError(
            "Requested exogenous feature date window is empty",
            context=context or None,
        )
    return filtered


def _build_asset_block(
    *,
    config: FredRequestConfig,
    returns_frame: pd.DataFrame,
    exogenous_frame: pd.DataFrame,
) -> AssetFeatureBlock:
    assets = [str(column) for column in returns_frame.columns]
    currency_columns = _carry_currency_columns(config)
    carry_by_asset = {
        asset: _carry_diff_for_asset(
            asset,
            exogenous_frame=exogenous_frame,
            currency_columns=currency_columns,
        )
        for asset in assets
    }
    observed = any(series.notna().any() for series in carry_by_asset.values())
    if not observed:
        raise DataProcessingError(
            "No applicable asset-level exogenous features were derived",
            context={"feature": _CARRY_FEATURE_NAME},
        )
    per_asset_frames = [
        pd.DataFrame({_CARRY_FEATURE_NAME: carry_by_asset[asset]}, index=exogenous_frame.index)
        for asset in assets
    ]
    frame = pd.concat(per_asset_frames, axis=1, keys=assets)
    specs = [
        DerivedFeatureSpec(
            feature_name=_CARRY_FEATURE_NAME,
            family_key="carry",
            transform="base_minus_quote_level",
            source_series=tuple(
                series.series_id
                for series in config.series
                if series.family_key == "carry"
            ),
            future_role="asset",
            applicability="fx_pairs_only",
        )
    ]
    return AssetFeatureBlock(
        frame=frame,
        feature_names=[_CARRY_FEATURE_NAME],
        feature_specs=specs,
        assets=assets,
    )


def _carry_currency_columns(config: FredRequestConfig) -> dict[str, str]:
    columns: dict[str, str] = {}
    for series in config.series:
        if series.family_key != "carry" or series.currency is None:
            continue
        columns[series.currency] = _cleaned_series_name(config.provider, series)
    if not columns:
        raise ConfigError(
            "No carry series are configured for exogenous feature engineering",
            context={"config_path": str(config.config_path)},
        )
    return columns


def _carry_diff_for_asset(
    asset: str,
    *,
    exogenous_frame: pd.DataFrame,
    currency_columns: dict[str, str],
) -> pd.Series:
    base, quote = _split_asset_currencies(asset)
    if base is None or quote is None:
        return _nan_series(exogenous_frame.index, name=asset)
    base_column = currency_columns.get(base)
    quote_column = currency_columns.get(quote)
    if base_column is None or quote_column is None:
        return _nan_series(exogenous_frame.index, name=asset)
    return (exogenous_frame[base_column] - exogenous_frame[quote_column]).rename(asset)


def _split_asset_currencies(asset: str) -> tuple[str | None, str | None]:
    if "." not in asset:
        return None, None
    base, quote = asset.split(".", 1)
    base_code = base.strip().upper() or None
    quote_code = quote.strip().upper() or None
    return base_code, quote_code


def _build_global_block(
    *, config: FredRequestConfig, exogenous_frame: pd.DataFrame
) -> GlobalFeatureBlock:
    series_by_family = _global_series_by_family(config)
    columns: dict[str, pd.Series] = {}
    specs: list[DerivedFeatureSpec] = []
    for family_key, series_list in series_by_family.items():
        family = _family_by_key(config.families, family_key)
        if family is None:
            raise ConfigError(
                "Configured global series requires a matching family definition",
                context={"family_key": family_key},
            )
        for series in series_list:
            column_name = _cleaned_series_name(config.provider, series)
            raw = exogenous_frame[column_name]
            feature_name, values, transform = _global_feature_from_series(
                series, raw=raw
            )
            columns[feature_name] = values
            specs.append(
                DerivedFeatureSpec(
                    feature_name=feature_name,
                    family_key=family.key,
                    transform=transform,
                    source_series=(series.series_id,),
                    future_role="global",
                    applicability="all_assets",
                )
            )
    if not columns:
        raise DataProcessingError(
            "No global exogenous features were derived from cleaned inputs"
        )
    frame = pd.DataFrame(columns, index=exogenous_frame.index)
    return GlobalFeatureBlock(
        frame=frame,
        feature_names=list(columns.keys()),
        feature_specs=specs,
    )


def _global_series_by_family(
    config: FredRequestConfig,
) -> dict[str, list[FredSeriesConfig]]:
    grouped: dict[str, list[FredSeriesConfig]] = {}
    for series in config.series:
        if series.future_role != "global":
            continue
        family_key = series.family_key
        if family_key is None:
            raise ConfigError(
                "Global exogenous series must define family_key",
                context={"series_id": series.series_id},
            )
        grouped.setdefault(family_key, []).append(series)
    return grouped


def _family_by_key(
    families: Sequence[FredFamilyConfig], key: str
) -> FredFamilyConfig | None:
    for family in families:
        if family.key == key:
            return family
    return None


def _global_feature_from_series(
    series: FredSeriesConfig, *, raw: pd.Series
) -> tuple[str, pd.Series, str]:
    alias = series.alias or series.series_id.lower()
    if (series.family_key or "") in _LOG_LEVEL_FAMILIES:
        return f"log_{alias}", _log_series(raw, alias), "log_level"
    return alias, raw.astype(float), "level"


def _log_series(raw: pd.Series, alias: str) -> pd.Series:
    values = raw.astype(float)
    invalid = values.le(0) & values.notna()
    if invalid.any():
        raise DataProcessingError(
            "Global exogenous feature cannot apply log transform to non-positive values",
            context={"feature": alias},
        )
    return pd.Series(
        np.log(values.to_numpy(dtype=float)),
        index=values.index,
        dtype=float,
        name=f"log_{alias}",
    )


def _cleaned_series_name(provider: str, series: FredSeriesConfig) -> str:
    family = series.family_key or series.dir_name.replace("/", "_")
    name = series.alias or series.series_id
    return f"{provider}__{family}__{name}"


def _prepare_output_paths(
    feature_store: Path, version_label: str, *, block_name: str
) -> ExogenousOutputPaths:
    output_dir = feature_store / version_label / _FEATURE_NAMESPACE / block_name
    ensure_directory(
        output_dir,
        error_type=DataProcessingError,
        invalid_message="Exogenous feature output path must be a directory",
        create_message="Failed to prepare exogenous feature output directory",
    )
    return ExogenousOutputPaths(
        base=BaseOutputPaths(
            output_dir=output_dir,
            output_path=output_dir / _OUTPUT_NAME,
            metadata_path=output_dir / _METADATA_NAME,
        ),
        tensor_path=output_dir / _TENSOR_NAME,
    )


def _build_output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write exogenous feature CSV",
        ),
        metadata_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write exogenous feature metadata",
        ),
    )


def _write_asset_outputs(
    *,
    block: AssetFeatureBlock,
    paths: ExogenousOutputPaths,
    writer: OutputWriter,
    sources: RunSources,
    request: RunRequest,
) -> None:
    writer.write_frame(block.frame, paths.base.output_path)
    _write_asset_tensor(block, paths.tensor_path)
    writer.write_metadata(
        _asset_metadata(
            block=block,
            paths=paths,
            sources=sources,
            request=request,
        ),
        paths.base.metadata_path,
    )


def _write_asset_tensor(block: AssetFeatureBlock, path: Path) -> None:
    values_by_asset = [
        _asset_frame(block.frame, asset)
        .reindex(columns=block.feature_names)
        .to_numpy(dtype=float)
        for asset in block.assets
    ]
    values = np.stack(values_by_asset, axis=1)
    missing_mask = np.isnan(values)
    timestamps = timestamps_to_epoch_hours(
        require_utc_hourly_index(
            block.frame.index,
            label="exogenous_asset",
            timezone=_TENSOR_TIMEZONE,
        )
    )
    write_tensor_bundle(
        path,
        values=torch.as_tensor(np.array(values, copy=True), dtype=torch.float64),
        timestamps=torch.as_tensor(timestamps, dtype=torch.int64),
        missing_mask=torch.as_tensor(missing_mask, dtype=torch.bool),
        error_message="Failed to write exogenous asset tensor",
    )


def _asset_metadata(
    *,
    block: AssetFeatureBlock,
    paths: ExogenousOutputPaths,
    sources: RunSources,
    request: RunRequest,
) -> dict[str, object]:
    return {
        "namespace": _FEATURE_NAMESPACE,
        "group": _ASSET_BLOCK,
        "feature_block": _ASSET_BLOCK,
        "version_label": sources.version_dir.name,
        "rows": len(block.frame),
        "assets": len(block.assets),
        "asset_names": block.assets,
        "features": len(block.feature_names),
        "feature_names": block.feature_names,
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "calendar_start": block.frame.index.min().isoformat(),
        "calendar_end": block.frame.index.max().isoformat(),
        "requested_start_date": request.start_date,
        "requested_end_date": request.end_date,
        "calendar_source": format_tilde_path(sources.returns_path),
        "source": [
            format_tilde_path(sources.returns_path),
            format_tilde_path(sources.exogenous_path),
        ],
        "destination": format_tilde_path(paths.base.output_path),
        "artifacts": _artifact_metadata(paths),
        "engineering_stage": "raw_engineered_unnormalized",
        "normalization_policy": _normalization_policy(),
        "derived_features": [_derived_feature_payload(spec) for spec in block.feature_specs],
        "missing_by_asset": _missing_by_asset(block.frame, block.assets),
    }


def _write_global_outputs(
    *,
    block: GlobalFeatureBlock,
    paths: ExogenousOutputPaths,
    writer: OutputWriter,
    sources: RunSources,
    request: RunRequest,
) -> None:
    writer.write_frame(block.frame, paths.base.output_path)
    _write_global_tensor(block.frame, paths.tensor_path)
    writer.write_metadata(
        _global_metadata(
            block=block,
            paths=paths,
            sources=sources,
            request=request,
        ),
        paths.base.metadata_path,
    )


def _write_global_tensor(frame: pd.DataFrame, path: Path) -> None:
    index = require_utc_hourly_index(
        frame.index,
        label="exogenous_global",
        timezone=_TENSOR_TIMEZONE,
    )
    values = np.array(frame.to_numpy(dtype=float), copy=True)
    timestamps = timestamps_to_epoch_hours(index)
    write_tensor_bundle(
        path,
        values=torch.as_tensor(values, dtype=torch.float64),
        timestamps=torch.as_tensor(timestamps, dtype=torch.int64),
        missing_mask=torch.as_tensor(np.isnan(values), dtype=torch.bool),
        error_message="Failed to write exogenous global tensor",
    )


def _global_metadata(
    *,
    block: GlobalFeatureBlock,
    paths: ExogenousOutputPaths,
    sources: RunSources,
    request: RunRequest,
) -> dict[str, object]:
    return {
        "namespace": _FEATURE_NAMESPACE,
        "group": _GLOBAL_BLOCK,
        "feature_block": _GLOBAL_BLOCK,
        "version_label": sources.version_dir.name,
        "rows": len(block.frame),
        "features": len(block.feature_names),
        "feature_names": block.feature_names,
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "calendar_start": block.frame.index.min().isoformat(),
        "calendar_end": block.frame.index.max().isoformat(),
        "requested_start_date": request.start_date,
        "requested_end_date": request.end_date,
        "calendar_source": format_tilde_path(sources.returns_path),
        "source": [
            format_tilde_path(sources.returns_path),
            format_tilde_path(sources.exogenous_path),
        ],
        "destination": format_tilde_path(paths.base.output_path),
        "artifacts": _artifact_metadata(paths),
        "engineering_stage": "raw_engineered_unnormalized",
        "normalization_policy": _normalization_policy(),
        "derived_features": [_derived_feature_payload(spec) for spec in block.feature_specs],
        "missing_by_feature": _missing_by_feature(block.frame),
    }


def _artifact_metadata(paths: ExogenousOutputPaths) -> dict[str, str]:
    return {
        "tensor": format_tilde_path(paths.tensor_path),
        "tensor_timestamp_unit": _TENSOR_TIMESTAMP_UNIT,
        "tensor_timezone": _TENSOR_TIMEZONE,
        "tensor_value_dtype": _TENSOR_VALUE_DTYPE,
    }


def _normalization_policy() -> dict[str, object]:
    return {
        "fit_scope": "train_only",
        "method": "robust",
        "winsorization": "training_quantiles",
        "clip": [-10.0, 10.0],
    }


def _derived_feature_payload(spec: DerivedFeatureSpec) -> dict[str, object]:
    return {
        "feature_name": spec.feature_name,
        "family_key": spec.family_key,
        "transform": spec.transform,
        "source_series": list(spec.source_series),
        "future_role": spec.future_role,
        "applicability": spec.applicability,
    }


def _missing_by_asset(
    frame: pd.DataFrame, assets: Sequence[str]
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for asset in assets:
        asset_frame = _asset_frame(frame, asset)
        total_missing = 0
        by_feature: dict[str, int] = {}
        for feature in asset_frame.columns:
            count = int(asset_frame[feature].isna().sum())
            total_missing += count
            if count > 0:
                by_feature[str(feature)] = count
        payload[asset] = {
            "total": total_missing,
            "by_feature": by_feature,
        }
    return payload


def _missing_by_feature(frame: pd.DataFrame) -> dict[str, int]:
    payload: dict[str, int] = {}
    for feature in frame.columns:
        count = int(frame[feature].isna().sum())
        if count > 0:
            payload[str(feature)] = count
    return payload


def _asset_frame(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    asset_frame = frame.xs(asset, axis=1, level=0)
    if isinstance(asset_frame, pd.Series):
        return asset_frame.to_frame()
    return asset_frame


def _nan_series(index: pd.Index, *, name: str) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype=float, name=name)
