from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.infrastructure import log_boundary, require_env
from algo_trader.preprocessing import (
    Preprocessor,
    default_registry,
    normalize_datetime_index,
)

logger = logging.getLogger(__name__)

_WEEK_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_PIPELINE_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_INPUT_NAME = "returns.csv"
_OUTPUT_NAME = "processed.csv"
_METADATA_NAME = "metadata.json"


@dataclass(frozen=True)
class MetadataContext:
    input_path: Path
    output_path: Path
    preprocessor_name: str
    pipeline: str
    params: Mapping[str, str]
    frame: pd.DataFrame


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
    normalized_name = _normalize_preprocessor_name(preprocessor_name)
    frame, input_path, version_label = _load_latest_returns()
    normalized = _apply_preprocessor(normalized_name, frame, params)

    output_path, metadata_path = _prepare_output_paths(
        normalized_name,
        pipeline,
        version_label,
    )

    _write_processed(normalized, output_path)
    metadata_params = _metadata_params(normalized_name, params)
    metadata_payload = _build_metadata(
        MetadataContext(
            input_path=input_path,
            output_path=output_path,
            preprocessor_name=normalized_name,
            pipeline=pipeline,
            params=metadata_params,
            frame=normalized,
        )
    )
    _write_metadata(metadata_path, metadata_payload)

    logger.info(
        "Saved processed CSV path=%s rows=%s assets=%s metadata=%s",
        output_path,
        len(normalized),
        len(normalized.columns),
        metadata_path,
    )
    return output_path


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
    try:
        feature_store.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataSourceError(
            "FEATURE_STORE_SOURCE cannot be created",
            context={"path": str(feature_store)},
        ) from exc
    if not feature_store.is_dir():
        raise DataSourceError(
            "FEATURE_STORE_SOURCE must be a directory",
            context={"path": str(feature_store)},
        )
    return feature_store


def _load_latest_returns() -> tuple[pd.DataFrame, Path, str]:
    data_lake = _resolve_data_lake()
    latest_dir = _resolve_latest_directory(data_lake)
    input_path = latest_dir / _INPUT_NAME
    frame = _load_returns(input_path)
    return frame, input_path, latest_dir.name


def _resolve_latest_directory(base_dir: Path) -> Path:
    candidates = [
        entry
        for entry in base_dir.iterdir()
        if entry.is_dir() and _WEEK_PATTERN.match(entry.name)
    ]
    if not candidates:
        raise DataSourceError(
            "No YYYY-WW data directories found",
            context={"path": str(base_dir)},
        )
    latest = max(candidates, key=lambda item: item.name)
    return latest


def _load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataSourceError(
            "Input returns.csv not found",
            context={"path": str(path)},
        )
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    except Exception as exc:
        raise DataSourceError(
            "Failed to read returns CSV",
            context={"path": str(path)},
        ) from exc
    return normalize_datetime_index(
        frame, label="input", preprocessor_name=""
    )


def _apply_preprocessor(
    preprocessor_name: str,
    frame: pd.DataFrame,
    params: Mapping[str, str],
) -> pd.DataFrame:
    preprocessor = _resolve_preprocessor(preprocessor_name)
    processed = preprocessor.process(frame, params=params)
    return normalize_datetime_index(
        processed,
        label="processed",
        preprocessor_name=preprocessor_name,
    )


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
    normalized = raw.strip()
    if not normalized:
        return "debug"
    if not _PIPELINE_PATTERN.match(normalized):
        raise ConfigError(
            "pipeline contains invalid characters",
            context={"pipeline": raw},
        )
    return normalized


def _normalize_preprocessor_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("preprocessor name must not be empty")
    return normalized


def _resolve_output_dir(
    feature_store: Path,
    preprocessor_name: str,
    pipeline: str,
    version_label: str,
) -> Path:
    output_dir = feature_store / preprocessor_name
    if pipeline:
        output_dir = output_dir / pipeline
    return output_dir / version_label


def _prepare_output_paths(
    preprocessor_name: str,
    pipeline: str,
    version_label: str,
) -> tuple[Path, Path]:
    feature_store = _resolve_feature_store()
    output_dir = _resolve_output_dir(
        feature_store, preprocessor_name, pipeline, version_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / _OUTPUT_NAME, output_dir / _METADATA_NAME


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
    return metadata


def _write_processed(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path)
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write processed CSV",
            context={"path": str(path)},
        ) from exc


def _build_metadata(context: MetadataContext) -> dict[str, object]:
    return {
        "pipeline": context.pipeline,
        "preprocessor": context.preprocessor_name,
        "params": dict(context.params),
        "input_path": str(context.input_path),
        "output_path": str(context.output_path),
        "rows": len(context.frame),
        "assets": len(context.frame.columns),
        "run_at": _format_run_at(datetime.now(timezone.utc)),
    }


def _write_metadata(
    path: Path, payload: Mapping[str, object]
) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write metadata JSON",
            context={"path": str(path)},
        ) from exc


def _format_run_at(timestamp: datetime) -> str:
    return timestamp.isoformat(timespec="seconds").replace("T", "_")
