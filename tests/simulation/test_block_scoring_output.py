from __future__ import annotations

import json
from pathlib import Path

from algo_trader.application.simulation.block_scoring_output import (
    write_global_block_scores,
    write_postprocess_block_scores,
)


def _block_scores() -> dict[str, dict[str, float]]:
    return {
        "fx": {
            "es": 1.1,
            "crps": 0.9,
            "ql": 0.8,
            "coverage_p50": 0.52,
            "coverage_p90": 0.88,
            "coverage_p95": 0.93,
            "n_assets": 32,
        },
        "indices": {
            "es": 1.4,
            "crps": 1.2,
            "ql": 1.1,
            "coverage_p50": 0.48,
            "coverage_p90": 0.83,
            "coverage_p95": 0.90,
            "n_assets": 10,
        },
        "commodities": {
            "es": 0.7,
            "crps": 0.6,
            "ql": 0.5,
            "coverage_p50": 0.55,
            "coverage_p90": 0.91,
            "coverage_p95": 0.96,
            "n_assets": 2,
        },
        "full": {
            "es": 1.0,
            "crps": 0.95,
            "ql": 0.9,
            "coverage_p50": 0.51,
            "coverage_p90": 0.87,
            "coverage_p95": 0.93,
            "n_assets": 44,
        },
    }


def test_write_postprocess_block_scores_creates_csv_and_json(
    tmp_path: Path,
) -> None:
    write_postprocess_block_scores(
        base_dir=tmp_path,
        outer_k=7,
        candidate_id=11,
        block_scores=_block_scores(),
    )

    output_dir = tmp_path / "inner" / "outer_7" / "postprocessing" / "block_scoring"
    payload = json.loads((output_dir / "block_scores.json").read_text(encoding="utf-8"))

    assert (output_dir / "block_scores.csv").exists()
    assert payload["outer_k"] == 7
    assert payload["best_candidate_id"] == 11
    assert payload["block_scores"]["full"]["n_assets"] == 44.0
    assert payload["indices"]["coverage_p90"] == 0.83


def test_write_global_block_scores_creates_manifest(
    tmp_path: Path,
) -> None:
    write_global_block_scores(
        base_dir=tmp_path,
        outer_ids=[17, 18],
        candidate_id=9,
        block_scores=_block_scores(),
    )

    output_dir = tmp_path / "outer" / "diagnostics" / "block_scoring"
    payload = json.loads((output_dir / "block_scores.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "aggregate_manifest.json").read_text(encoding="utf-8")
    )

    assert (output_dir / "block_scores.csv").exists()
    assert payload["best_candidate_id"] == 9
    assert payload["outer_ids"] == [17, 18]
    assert payload["full"]["n_assets"] == 44.0
    assert manifest["scope"] == "selected_candidate_block_scores"
