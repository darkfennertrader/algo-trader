from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .diagnostic_table_output import (
    DiagnosticTableSpec,
    GlobalTableRequest,
    PostprocessTableRequest,
    ordered_block_items,
    write_global_diagnostic_table,
    write_postprocess_diagnostic_table,
)

_BLOCK_FIELDS = (
    "es",
    "crps",
    "ql",
    "coverage_p50",
    "coverage_p90",
    "coverage_p95",
    "n_assets",
)
_DEPENDENCE_FIELDS = (
    "energy_score",
    "variogram_score",
    "variogram_p",
    "n_assets",
)
_BLOCK_SPEC = DiagnosticTableSpec(
    directory_name="block_scoring",
    payload_key="block_scores",
    file_stem="block_scores",
    manifest_scope="selected_candidate_block_scores",
    invalid_message="Block scoring output path is not a directory",
    create_message="Failed to create block scoring output",
    fields=_BLOCK_FIELDS,
)
_DEPENDENCE_SPEC = DiagnosticTableSpec(
    directory_name="dependence_scoring",
    payload_key="dependence_scores",
    file_stem="dependence_scores",
    manifest_scope="selected_candidate_dependence_scores",
    invalid_message="Dependence scoring output path is not a directory",
    create_message="Failed to create dependence scoring output",
    fields=_DEPENDENCE_FIELDS,
)


def write_postprocess_block_scores(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
    block_scores: Mapping[str, Mapping[str, float]],
) -> None:
    write_postprocess_diagnostic_table(
        request=PostprocessTableRequest(
            base_dir=base_dir,
            outer_k=outer_k,
            candidate_id=candidate_id,
        ),
        scores=block_scores,
        spec=_BLOCK_SPEC,
    )


def write_global_block_scores(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
    block_scores: Mapping[str, Mapping[str, float]],
) -> None:
    write_global_diagnostic_table(
        request=GlobalTableRequest(
            base_dir=base_dir,
            outer_ids=outer_ids,
            candidate_id=candidate_id,
        ),
        scores=block_scores,
        spec=_BLOCK_SPEC,
    )


def write_postprocess_dependence_scores(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
    dependence_scores: Mapping[str, Mapping[str, float]],
) -> None:
    write_postprocess_diagnostic_table(
        request=PostprocessTableRequest(
            base_dir=base_dir,
            outer_k=outer_k,
            candidate_id=candidate_id,
        ),
        scores=dependence_scores,
        spec=_DEPENDENCE_SPEC,
        extra_payload={"variogram_p": _resolve_variogram_p(dependence_scores)},
    )


def write_global_dependence_scores(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
    dependence_scores: Mapping[str, Mapping[str, float]],
) -> None:
    write_global_diagnostic_table(
        request=GlobalTableRequest(
            base_dir=base_dir,
            outer_ids=outer_ids,
            candidate_id=candidate_id,
        ),
        scores=dependence_scores,
        spec=_DEPENDENCE_SPEC,
        extra_payload={"variogram_p": _resolve_variogram_p(dependence_scores)},
    )


def _resolve_variogram_p(
    dependence_scores: Mapping[str, Mapping[str, float]]
) -> float:
    for _, values in ordered_block_items(dependence_scores):
        value = values.get("variogram_p")
        if value is not None:
            return float(value)
    return float("nan")
