from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd
import torch

from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputNames,
    OutputPaths,
    OutputWriter,
    build_preprocessor_output_paths,
    ensure_directory,
    format_run_at,
    log_boundary,
    require_env,
    resolve_latest_week_dir,
)
from algo_trader.infrastructure.paths import format_tilde_path
from algo_trader.preprocessing import (
    Preprocessor,
    PCAPreprocessor,
    default_registry,
    normalize_datetime_index,
)
from ..data_io import read_indexed_csv
from ..pipeline_utils import parse_pipeline_name

if TYPE_CHECKING:
    from algo_trader.preprocessing import PCAResult

logger = logging.getLogger(__name__)

_INPUT_NAME = "returns.csv"
_DEFAULT_OUTPUT_NAMES = OutputNames(
    output_name="processed.csv",
    metadata_name="metadata.json",
)
_PCA_OUTPUT_NAMES = OutputNames(
    output_name="factors.csv",
    metadata_name="metadata.json",
)


@dataclass(frozen=True)
class MetadataContext:
    input_path: Path
    output_path: Path
    preprocessor_name: str
    pipeline: str
    params: Mapping[str, str]
    frame: pd.DataFrame


@dataclass(frozen=True)
class PCAOutputPaths:
    factors_pt: Path
    loadings_csv: Path
    loadings_pt: Path
    eigenvalues_csv: Path


@dataclass(frozen=True)
class RunState:
    preprocessor_name: str
    input_path: Path
    version_label: str
    processed: pd.DataFrame
    preprocessor: Preprocessor


def _run_context(
    preprocessor_name: str,
    preprocessor_args: Sequence[str] | None = None,
) -> Mapping[str, str]:
    return {
        "preprocessor": preprocessor_name,
        "arg_count": str(len(preprocessor_args or [])),
    }


@log_boundary("data_processing.run", context=_run_context)
def run(
    *,
    preprocessor_name: str,
    preprocessor_args: Sequence[str] | None = None,
) -> Path:
    params = _parse_preprocessor_args(preprocessor_args)
    pipeline = _parse_pipeline(params)
    state = _build_run_state(preprocessor_name, params)
    output_paths = _write_outputs(state, pipeline, params)

    logger.info(
        "Saved output CSV path=%s rows=%s assets=%s metadata=%s",
        output_paths.output_path,
        len(state.processed),
        len(state.processed.columns),
        output_paths.metadata_path,
    )
    return output_paths.output_path


def _resolve_data_lake() -> Path:
    data_lake = Path(require_env("DATA_LAKE_SOURCE")).expanduser()
    if not data_lake.exists():
        raise DataSourceError(
            "DATA_LAKE_SOURCE does not exist",
            context={"path": str(data_lake)},
        )
    if not data_lake.is_dir():
        raise DataSourceError(
            "DATA_LAKE_SOURCE must be a directory",
            context={"path": str(data_lake)},
        )
    return data_lake


def _resolve_feature_store() -> Path:
    feature_store = Path(require_env("FEATURE_STORE_SOURCE")).expanduser()
    ensure_directory(
        feature_store,
        error_type=DataSourceError,
        invalid_message="FEATURE_STORE_SOURCE must be a directory",
        create_message="FEATURE_STORE_SOURCE cannot be created",
    )
    return feature_store


def _load_latest_returns() -> tuple[pd.DataFrame, Path, str]:
    data_lake = _resolve_data_lake()
    latest_dir = _resolve_latest_directory(data_lake)
    input_path = latest_dir / _INPUT_NAME
    frame = _load_returns(input_path)
    return frame, input_path, latest_dir.name


def _build_run_state(
    preprocessor_name: str, params: Mapping[str, str]
) -> RunState:
    normalized_name = _normalize_preprocessor_name(preprocessor_name)
    frame, input_path, version_label = _load_latest_returns()
    processed, preprocessor = _apply_preprocessor(
        normalized_name, frame, params
    )
    return RunState(
        preprocessor_name=normalized_name,
        input_path=input_path,
        version_label=version_label,
        processed=processed,
        preprocessor=preprocessor,
    )


def _resolve_latest_directory(base_dir: Path) -> Path:
    return resolve_latest_week_dir(
        base_dir,
        error_type=DataSourceError,
        error_message="No YYYY-WW data directories found",
    )


def _load_returns(path: Path) -> pd.DataFrame:
    frame = read_indexed_csv(
        path,
        missing_message="Input returns.csv not found",
        read_message="Failed to read returns CSV",
    )
    return normalize_datetime_index(
        frame, label="input", preprocessor_name=""
    )


def _apply_preprocessor(
    preprocessor_name: str,
    frame: pd.DataFrame,
    params: Mapping[str, str],
) -> tuple[pd.DataFrame, Preprocessor]:
    preprocessor = _resolve_preprocessor(preprocessor_name)
    processed = preprocessor.process(frame, params=params)
    normalized = normalize_datetime_index(
        processed,
        label="processed",
        preprocessor_name=preprocessor_name,
    )
    return normalized, preprocessor


def _write_outputs(
    state: RunState,
    pipeline: str,
    params: Mapping[str, str],
) -> OutputPaths:
    output_names = _output_names(state.preprocessor_name)
    output_paths = _prepare_output_paths(
        state.preprocessor_name,
        pipeline,
        state.version_label,
        output_names,
    )
    writer = _build_output_writer()
    writer.write_frame(state.processed, output_paths.output_path)
    artifact_payload = _maybe_write_pca_artifacts(
        state.preprocessor_name,
        state.preprocessor,
        state.processed,
        output_paths.output_dir,
        writer,
    )
    metadata_params = _metadata_params(state.preprocessor_name, params)
    metadata_payload = _build_metadata(
        MetadataContext(
            input_path=state.input_path,
            output_path=output_paths.output_path,
            preprocessor_name=state.preprocessor_name,
            pipeline=pipeline,
            params=metadata_params,
            frame=state.processed,
        )
    )
    if artifact_payload:
        metadata_payload.update(artifact_payload)
    writer.write_metadata(metadata_payload, output_paths.metadata_path)
    return output_paths


def _resolve_preprocessor(name: str) -> Preprocessor:
    registry = default_registry()
    return registry.get(name)


def _parse_preprocessor_args(
    args: Sequence[str] | None,
) -> dict[str, str]:
    if not args:
        return {}
    params: dict[str, str] = {}
    for raw in args:
        if "=" not in raw:
            raise ConfigError(
                "preprocessor-arg must be in key=value format",
                context={"value": raw},
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(
                "preprocessor-arg key must not be empty",
                context={"value": raw},
            )
        params[key] = value.strip()
    return params


def _parse_pipeline(params: dict[str, str]) -> str:
    raw = params.pop("pipeline", "")
    return parse_pipeline_name(raw)


def _normalize_preprocessor_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("preprocessor name must not be empty")
    return normalized


def _output_names(preprocessor_name: str) -> OutputNames:
    if preprocessor_name == "pca":
        return _PCA_OUTPUT_NAMES
    return _DEFAULT_OUTPUT_NAMES


def _prepare_output_paths(
    preprocessor_name: str,
    pipeline: str,
    version_label: str,
    names: OutputNames,
) -> OutputPaths:
    feature_store = _resolve_feature_store()
    output_paths = build_preprocessor_output_paths(
        feature_store,
        preprocessor_name,
        pipeline,
        version_label,
        names,
    )
    ensure_directory(
        output_paths.output_dir,
        error_type=DataProcessingError,
        invalid_message="Feature store output path must be a directory",
        create_message="Failed to prepare feature store output directory",
    )
    return output_paths


def _maybe_write_pca_artifacts(
    preprocessor_name: str,
    preprocessor: Preprocessor,
    factors: pd.DataFrame,
    output_dir: Path,
    writer: OutputWriter,
) -> dict[str, object]:
    if preprocessor_name != "pca":
        return {}
    if not isinstance(preprocessor, PCAPreprocessor):
        raise DataProcessingError(
            "PCA preprocessor is unavailable",
            context={"preprocessor": preprocessor_name},
        )
    result = preprocessor.result()
    paths = _pca_output_paths(output_dir)
    _write_tensor(_frame_to_tensor(factors), paths.factors_pt)
    writer.write_frame(result.loadings, paths.loadings_csv)
    _write_tensor(_frame_to_tensor(result.loadings), paths.loadings_pt)
    writer.write_frame(result.eigenvalues, paths.eigenvalues_csv)
    return _pca_metadata(paths, result)


def _pca_output_paths(output_dir: Path) -> PCAOutputPaths:
    return PCAOutputPaths(
        factors_pt=output_dir / "factors.pt",
        loadings_csv=output_dir / "loadings.csv",
        loadings_pt=output_dir / "loadings.pt",
        eigenvalues_csv=output_dir / "eigenvalues.csv",
    )


def _pca_metadata(
    paths: PCAOutputPaths, result: PCAResult
) -> dict[str, object]:
    return {
        "pca": {
            "selected_k": result.selected_k,
            "variance_target": result.variance_target,
        },
        "artifacts": {
            "factors_pt": format_tilde_path(paths.factors_pt),
            "loadings_csv": format_tilde_path(paths.loadings_csv),
            "loadings_pt": format_tilde_path(paths.loadings_pt),
            "eigenvalues_csv": format_tilde_path(paths.eigenvalues_csv),
        },
    }


def _frame_to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(frame.to_numpy(dtype=float), dtype=torch.float64)


def _write_tensor(tensor: torch.Tensor, path: Path) -> None:
    try:
        torch.save(tensor, path)
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write tensor output",
            context={"path": str(path)},
        ) from exc


def _metadata_params(
    preprocessor_name: str, params: Mapping[str, str]
) -> dict[str, str]:
    metadata = dict(params)
    if preprocessor_name == "identity":
        if not metadata.get("copy"):
            metadata["copy"] = "false"
    if preprocessor_name == "zscore":
        if not metadata.get("missing"):
            metadata["missing"] = "zero"
        if not metadata.get("start_date"):
            metadata["start_date"] = "full_range"
        if not metadata.get("end_date"):
            metadata["end_date"] = "full_range"
    if preprocessor_name == "pca":
        if not metadata.get("missing"):
            metadata["missing"] = "zero"
        if not metadata.get("start_date"):
            metadata["start_date"] = "full_range"
        if not metadata.get("end_date"):
            metadata["end_date"] = "full_range"
    return metadata


def _build_output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write CSV output",
        ),
        metadata_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write metadata JSON",
        ),
    )


def _build_metadata(context: MetadataContext) -> dict[str, object]:
    return {
        "pipeline": context.pipeline,
        "preprocessor": context.preprocessor_name,
        "params": dict(context.params),
        "rows": len(context.frame),
        "assets": len(context.frame.columns),
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source": format_tilde_path(context.input_path),
        "destination": format_tilde_path(context.output_path),
    }
