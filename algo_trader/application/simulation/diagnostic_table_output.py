from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from algo_trader.domain import BLOCK_ORDER, SimulationError
from algo_trader.infrastructure import ensure_directory


@dataclass(frozen=True)
class DiagnosticTableSpec:
    directory_name: str
    payload_key: str
    file_stem: str
    manifest_scope: str
    invalid_message: str
    create_message: str
    fields: Sequence[str]


@dataclass(frozen=True)
class PostprocessTableRequest:
    base_dir: Path
    outer_k: int
    candidate_id: int


@dataclass(frozen=True)
class GlobalTableRequest:
    base_dir: Path
    outer_ids: Sequence[int]
    candidate_id: int


def write_postprocess_diagnostic_table(
    *,
    request: PostprocessTableRequest,
    scores: Mapping[str, Mapping[str, float]],
    spec: DiagnosticTableSpec,
    extra_payload: Mapping[str, Any] | None = None,
) -> None:
    output_dir = _ensure_dir(
        request.base_dir
        / "inner"
        / f"outer_{request.outer_k}"
        / "postprocessing"
        / spec.directory_name,
        spec=spec,
    )
    serialized_scores = serialize_scores(scores=scores, fields=spec.fields)
    payload = {
        "scope": "outer_postprocess_selection",
        "outer_k": int(request.outer_k),
        "best_candidate_id": int(request.candidate_id),
        **dict(extra_payload or {}),
        spec.payload_key: serialized_scores,
    }
    payload.update(_top_level_score_aliases(serialized_scores))
    _write_json(output_dir / f"{spec.file_stem}.json", payload)
    _write_csv(
        path=output_dir / f"{spec.file_stem}.csv",
        scores=scores,
        fields=spec.fields,
    )


def write_global_diagnostic_table(
    *,
    request: GlobalTableRequest,
    scores: Mapping[str, Mapping[str, float]],
    spec: DiagnosticTableSpec,
    extra_payload: Mapping[str, Any] | None = None,
) -> None:
    output_dir = _ensure_dir(
        request.base_dir / "outer" / "diagnostics" / spec.directory_name,
        spec=spec,
    )
    extra = dict(extra_payload or {})
    serialized_scores = serialize_scores(scores=scores, fields=spec.fields)
    payload = {
        "scope": "global_selection",
        "aggregation": "median_over_outer_folds",
        "outer_ids": [int(item) for item in request.outer_ids],
        "best_candidate_id": int(request.candidate_id),
        **extra,
        spec.payload_key: serialized_scores,
    }
    payload.update(_top_level_score_aliases(serialized_scores))
    _write_json(output_dir / f"{spec.file_stem}.json", payload)
    _write_json(
        output_dir / "aggregate_manifest.json",
        {
            "aggregation": "median_over_outer_folds",
            "outer_ids": [int(item) for item in request.outer_ids],
            "scope": spec.manifest_scope,
            **extra,
        },
    )
    _write_csv(
        path=output_dir / f"{spec.file_stem}.csv",
        scores=scores,
        fields=spec.fields,
    )


def serialize_scores(
    *, scores: Mapping[str, Mapping[str, float]], fields: Sequence[str]
) -> dict[str, dict[str, float]]:
    return {
        block: {
            field: float(values.get(field, float("nan")))
            for field in fields
        }
        for block, values in ordered_block_items(scores)
    }


def ordered_block_items(
    scores: Mapping[str, Mapping[str, float]]
) -> list[tuple[str, Mapping[str, float]]]:
    ordered: list[tuple[str, Mapping[str, float]]] = []
    for block in BLOCK_ORDER:
        if block in scores:
            ordered.append((block, scores[block]))
    for block, values in scores.items():
        if block not in BLOCK_ORDER:
            ordered.append((str(block), values))
    return ordered


def _top_level_score_aliases(
    serialized_scores: Mapping[str, Mapping[str, float]]
) -> dict[str, Mapping[str, float]]:
    return dict(serialized_scores.items())


def _ensure_dir(path: Path, *, spec: DiagnosticTableSpec) -> Path:
    ensure_directory(
        path,
        error_type=SimulationError,
        invalid_message=spec.invalid_message,
        create_message=spec.create_message,
        context={"path": str(path)},
    )
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    *,
    path: Path,
    scores: Mapping[str, Mapping[str, float]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("block", *fields))
        for block, values in ordered_block_items(scores):
            writer.writerow(
                [block, *[float(values.get(field, float("nan"))) for field in fields]]
            )
