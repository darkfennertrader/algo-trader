from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from algo_trader.domain import SimulationError

from .data_versioning import resolve_dataset_version_context


@dataclass(frozen=True)
class DataSourceMetadata:
    version_label: str
    return_type: str
    return_frequency: str
    data_lake_dir: str


def write_data_source_metadata(
    *,
    base_dir: Path,
    dataset_params: Mapping[str, Any],
) -> DataSourceMetadata:
    metadata = _resolve_data_source_metadata(dataset_params)
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    path = inputs_dir / "data_source.json"
    try:
        path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    except Exception as exc:
        raise SimulationError(
            "Failed to write data source metadata",
            context={"path": str(path)},
        ) from exc
    return metadata


def load_data_source_metadata(
    *,
    base_dir: Path,
    dataset_params: Mapping[str, Any],
) -> DataSourceMetadata:
    path = base_dir / "inputs" / "data_source.json"
    if path.exists():
        return _load_saved_metadata(path)
    return _resolve_data_source_metadata(dataset_params)


def _load_saved_metadata(path: Path) -> DataSourceMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SimulationError(
            "Failed to read data source metadata",
            context={"path": str(path)},
        ) from exc
    return DataSourceMetadata(
        version_label=str(payload["version_label"]),
        return_type=str(payload["return_type"]),
        return_frequency=str(payload["return_frequency"]),
        data_lake_dir=str(payload["data_lake_dir"]),
    )


def _resolve_data_source_metadata(
    dataset_params: Mapping[str, Any],
) -> DataSourceMetadata:
    version_context = resolve_dataset_version_context(dataset_params)
    returns_meta_path = (
        version_context.data_lake
        / version_context.version_label
        / "returns_meta.json"
    )
    try:
        payload = json.loads(returns_meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SimulationError(
            "Failed to read returns metadata",
            context={"path": str(returns_meta_path)},
        ) from exc
    return DataSourceMetadata(
        version_label=version_context.version_label,
        return_type=str(payload["return_type"]),
        return_frequency=str(payload["return_frequency"]),
        data_lake_dir=str(
            (
                version_context.data_lake
                / version_context.version_label
            ).expanduser()
        ),
    )
