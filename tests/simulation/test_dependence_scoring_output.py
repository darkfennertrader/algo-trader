from __future__ import annotations

import json
from pathlib import Path

from algo_trader.application.simulation.dependence_scoring_output import (
    write_global_dependence_scores,
    write_postprocess_dependence_scores,
)


def _dependence_scores() -> dict[str, dict[str, float]]:
    return {
        "fx": {
            "energy_score": 1.1,
            "variogram_score": 0.4,
            "variogram_p": 0.5,
            "n_assets": 32,
        },
        "indices": {
            "energy_score": 1.4,
            "variogram_score": 0.7,
            "variogram_p": 0.5,
            "n_assets": 10,
        },
        "commodities": {
            "energy_score": 0.7,
            "variogram_score": 0.3,
            "variogram_p": 0.5,
            "n_assets": 2,
        },
        "full": {
            "energy_score": 1.0,
            "variogram_score": 0.6,
            "variogram_p": 0.5,
            "n_assets": 44,
        },
    }


def test_write_postprocess_dependence_scores_creates_csv_and_json(
    tmp_path: Path,
) -> None:
    write_postprocess_dependence_scores(
        base_dir=tmp_path,
        outer_k=7,
        candidate_id=11,
        dependence_scores=_dependence_scores(),
    )

    output_dir = (
        tmp_path / "inner" / "outer_7" / "postprocessing" / "dependence_scoring"
    )
    payload = json.loads(
        (output_dir / "dependence_scores.json").read_text(encoding="utf-8")
    )

    assert (output_dir / "dependence_scores.csv").exists()
    assert payload["outer_k"] == 7
    assert payload["best_candidate_id"] == 11
    assert payload["variogram_p"] == 0.5
    assert payload["dependence_scores"]["full"]["variogram_p"] == 0.5


def test_write_global_dependence_scores_creates_manifest(
    tmp_path: Path,
) -> None:
    write_global_dependence_scores(
        base_dir=tmp_path,
        outer_ids=[17, 18],
        candidate_id=9,
        dependence_scores=_dependence_scores(),
    )

    output_dir = tmp_path / "outer" / "diagnostics" / "dependence_scoring"
    payload = json.loads(
        (output_dir / "dependence_scores.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "aggregate_manifest.json").read_text(encoding="utf-8")
    )

    assert (output_dir / "dependence_scores.csv").exists()
    assert payload["best_candidate_id"] == 9
    assert payload["outer_ids"] == [17, 18]
    assert manifest["scope"] == "selected_candidate_dependence_scores"
    assert manifest["variogram_p"] == 0.5
