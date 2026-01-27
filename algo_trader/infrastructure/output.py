from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Protocol

from algo_trader.domain import AlgoTraderError

_WEEK_PATTERN = re.compile(r"^\d{4}-\d{2}$")


class CsvWritable(Protocol):
    def to_csv(self, path_or_buf: Path, *, index: bool = True) -> None:
        """Write a CSV representation to the target path."""


class OutputWriter(Protocol):
    def write_frame(self, frame: CsvWritable, path: Path) -> None:
        """Persist tabular data to the requested path."""

    def write_metadata(
        self, payload: Mapping[str, object], path: Path
    ) -> None:
        """Persist metadata to the requested path."""


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    output_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class OutputNames:
    output_name: str
    metadata_name: str


@dataclass(frozen=True)
class ErrorPolicy:
    error_type: type[AlgoTraderError]
    message: str
    context: Mapping[str, str] | None = None


@dataclass(frozen=True)
class FileOutputWriter:
    data_policy: ErrorPolicy
    metadata_policy: ErrorPolicy

    def write_frame(self, frame: CsvWritable, path: Path) -> None:
        write_csv(frame, path, error_policy=self.data_policy)

    def write_metadata(
        self, payload: Mapping[str, object], path: Path
    ) -> None:
        write_json(path, payload, error_policy=self.metadata_policy)


def format_run_at(timestamp: datetime) -> str:
    return timestamp.isoformat(timespec="seconds").replace("T", "_")


def build_weekly_output_paths(
    root: Path,
    run_date: date,
    names: OutputNames,
) -> OutputPaths:
    output_dir = _versioned_week_dir(root, run_date)
    return OutputPaths(
        output_dir=output_dir,
        output_path=output_dir / names.output_name,
        metadata_path=output_dir / names.metadata_name,
    )


def resolve_latest_week_dir(
    root: Path,
    *,
    error_type: type[AlgoTraderError],
    error_message: str,
) -> Path:
    candidates = [
        entry
        for entry in root.iterdir()
        if entry.is_dir() and _WEEK_PATTERN.match(entry.name)
    ]
    if not candidates:
        raise error_type(error_message, context={"path": str(root)})
    return max(candidates, key=lambda item: item.name)


def build_preprocessor_output_paths(
    feature_store: Path,
    preprocessor_name: str,
    pipeline: str,
    version_label: str,
    names: OutputNames,
) -> OutputPaths:
    output_dir = feature_store / preprocessor_name
    if pipeline:
        output_dir = output_dir / pipeline
    output_dir = output_dir / version_label
    return OutputPaths(
        output_dir=output_dir,
        output_path=output_dir / names.output_name,
        metadata_path=output_dir / names.metadata_name,
    )


def ensure_directory(
    path: Path,
    *,
    error_type: type[AlgoTraderError],
    invalid_message: str,
    create_message: str,
    context: Mapping[str, str] | None = None,
) -> None:
    if path.exists() and not path.is_dir():
        raise error_type(
            invalid_message,
            context=_merge_context(context, path),
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise error_type(
            create_message,
            context=_merge_context(context, path),
        ) from exc


def write_csv(
    frame: CsvWritable,
    path: Path,
    *,
    error_policy: ErrorPolicy,
    include_index: bool = True,
) -> None:
    try:
        frame.to_csv(path, index=include_index)
    except Exception as exc:
        _raise_with_policy(error_policy, path, exc)


def write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    error_policy: ErrorPolicy,
) -> None:
    try:
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        _raise_with_policy(error_policy, path, exc)


def _versioned_week_dir(root: Path, run_date: date) -> Path:
    iso = run_date.isocalendar()
    return root / f"{iso.year:04d}-{iso.week:02d}"


def _merge_context(
    context: Mapping[str, str] | None,
    path: Path | None,
) -> dict[str, str]:
    merged = dict(context) if context else {}
    if path is not None and "path" not in merged:
        merged["path"] = str(path)
    return merged


def _raise_with_policy(
    policy: ErrorPolicy,
    path: Path,
    exc: Exception,
) -> None:
    raise policy.error_type(
        policy.message,
        context=_merge_context(policy.context, path),
    ) from exc
