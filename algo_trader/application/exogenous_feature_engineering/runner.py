from __future__ import annotations

import logging
from dataclasses import dataclass
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
from algo_trader.application.feature_catalog import FeatureCatalogConfig
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
_CHANGE_1W_ALIASES = frozenset(
    {
        "credit_us_baa10y",
        "ust_2y",
        "yield_curve_slope_us",
        "anfci",
        "us_real_10y",
    }
)


@dataclass(frozen=True)
class RunRequest:
    config_path: Path | None = None


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
    }


@log_boundary("exogenous_feature_engineering.run", context=_run_context)
def run(*, request: RunRequest) -> list[Path]:
    feature_catalog = FeatureCatalogConfig.load()
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
    common_index = _resolve_returns_index(returns_frame.index)
    aligned_returns = returns_frame.loc[common_index].copy()
    aligned_exogenous = exogenous_frame.reindex(common_index).copy()
    asset_block = _build_asset_block(
        config=config,
        returns_frame=aligned_returns,
        exogenous_frame=aligned_exogenous,
        configured_feature_names=feature_catalog.exogenous.asset_features,
    )
    global_block = _build_global_block(
        config=config,
        exogenous_frame=aligned_exogenous,
        configured_feature_names=feature_catalog.exogenous.global_features,
    )
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
    )
    _write_global_outputs(
        block=global_block,
        paths=global_paths,
        writer=writer,
        sources=sources,
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


def _resolve_returns_index(returns_index: pd.Index) -> pd.DatetimeIndex:
    return require_utc_hourly_index(
        pd.DatetimeIndex(returns_index),
        label="exogenous_feature_engineering_returns_index",
        timezone=_TENSOR_TIMEZONE,
    )


def _build_asset_block(
    *,
    config: FredRequestConfig,
    returns_frame: pd.DataFrame,
    exogenous_frame: pd.DataFrame,
    configured_feature_names: Sequence[str],
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
    return _filter_asset_block(
        AssetFeatureBlock(
            frame=frame,
            feature_names=[_CARRY_FEATURE_NAME],
            feature_specs=specs,
            assets=assets,
        ),
        configured_feature_names=configured_feature_names,
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
    *,
    config: FredRequestConfig,
    exogenous_frame: pd.DataFrame,
    configured_feature_names: Sequence[str],
) -> GlobalFeatureBlock:
    columns, specs = _derive_global_columns_and_specs(
        config=config, exogenous_frame=exogenous_frame
    )
    if not columns:
        raise DataProcessingError(
            "No global exogenous features were derived from cleaned inputs"
        )
    frame = pd.DataFrame(columns, index=exogenous_frame.index)
    return _filter_global_block(
        GlobalFeatureBlock(
            frame=frame,
            feature_names=list(columns.keys()),
            feature_specs=specs,
        ),
        configured_feature_names=configured_feature_names,
    )


def _derive_global_columns_and_specs(
    *,
    config: FredRequestConfig,
    exogenous_frame: pd.DataFrame,
) -> tuple[dict[str, pd.Series], list[DerivedFeatureSpec]]:
    columns: dict[str, pd.Series] = {}
    specs: list[DerivedFeatureSpec] = []
    for family_key, series_list in _global_series_by_family(config).items():
        family = _require_global_family(config.families, family_key)
        family_columns, family_specs = _derive_family_global_outputs(
            config=config,
            exogenous_frame=exogenous_frame,
            family=family,
            series_list=series_list,
        )
        columns.update(family_columns)
        specs.extend(family_specs)
    return columns, specs


def _require_global_family(
    families: Sequence[FredFamilyConfig], family_key: str
) -> FredFamilyConfig:
    family = _family_by_key(families, family_key)
    if family is None:
        raise ConfigError(
            "Configured global series requires a matching family definition",
            context={"family_key": family_key},
        )
    return family


def _derive_family_global_outputs(
    *,
    config: FredRequestConfig,
    exogenous_frame: pd.DataFrame,
    family: FredFamilyConfig,
    series_list: Sequence[FredSeriesConfig],
) -> tuple[dict[str, pd.Series], list[DerivedFeatureSpec]]:
    columns: dict[str, pd.Series] = {}
    specs: list[DerivedFeatureSpec] = []
    for series in series_list:
        feature_name, values, transform = _global_feature_from_series(
            series,
            raw=exogenous_frame[_cleaned_series_name(config.provider, series)],
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
    return columns, specs


def _filter_asset_block(
    block: AssetFeatureBlock,
    *,
    configured_feature_names: Sequence[str],
) -> AssetFeatureBlock:
    feature_names = [
        name for name in configured_feature_names if name in block.feature_names
    ]
    if not feature_names:
        raise DataProcessingError(
            "No configured asset-level exogenous features remain after filtering"
        )
    filtered_frame = block.frame.loc[:, pd.IndexSlice[:, feature_names]]
    filtered_specs = [
        spec for spec in block.feature_specs if spec.feature_name in feature_names
    ]
    return AssetFeatureBlock(
        frame=filtered_frame,
        feature_names=feature_names,
        feature_specs=filtered_specs,
        assets=block.assets,
    )


def _filter_global_block(
    block: GlobalFeatureBlock,
    *,
    configured_feature_names: Sequence[str],
) -> GlobalFeatureBlock:
    feature_names = [
        name for name in configured_feature_names if name in block.feature_names
    ]
    if not feature_names:
        raise DataProcessingError(
            "No configured global exogenous features remain after filtering"
        )
    filtered_frame = block.frame.loc[:, feature_names]
    filtered_specs = [
        spec for spec in block.feature_specs if spec.feature_name in feature_names
    ]
    return GlobalFeatureBlock(
        frame=filtered_frame,
        feature_names=feature_names,
        feature_specs=filtered_specs,
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
    transform = _resolve_global_feature_transform(series, alias)
    if transform == "log_level":
        return f"log_{alias}", _log_series(raw, alias), "log_level"
    if transform == "change_1w":
        return (
            f"{alias}_change_1w",
            _change_1w_series(raw, alias),
            "change_1w",
        )
    return alias, raw.astype(float), "level"


def _resolve_global_feature_transform(
    series: FredSeriesConfig, alias: str
) -> str:
    family_key = series.family_key or ""
    if alias in _CHANGE_1W_ALIASES:
        return "change_1w"
    if family_key in _LOG_LEVEL_FAMILIES:
        return "log_level"
    return "level"


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


def _change_1w_series(raw: pd.Series, alias: str) -> pd.Series:
    values = raw.astype(float)
    return pd.Series(
        values.diff().to_numpy(dtype=float),
        index=values.index,
        dtype=float,
        name=f"{alias}_change_1w",
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
) -> None:
    _write_feature_frame(block.frame, paths.base.output_path)
    _write_asset_tensor(block, paths.tensor_path)
    writer.write_metadata(
        _asset_metadata(
            block=block,
            paths=paths,
            sources=sources,
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


def _write_feature_frame(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path, index=True, index_label="timestamp")
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write exogenous feature CSV",
            context={"path": str(path)},
        ) from exc


def _asset_metadata(
    *,
    block: AssetFeatureBlock,
    paths: ExogenousOutputPaths,
    sources: RunSources,
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
) -> None:
    _write_feature_frame(block.frame, paths.base.output_path)
    _write_global_tensor(block.frame, paths.tensor_path)
    writer.write_metadata(
        _global_metadata(
            block=block,
            paths=paths,
            sources=sources,
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
