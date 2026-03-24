from __future__ import annotations

from pathlib import Path

from algo_trader.application.simulation.artifacts import SimulationArtifacts
from algo_trader.application.simulation.model_selection import (
    GlobalSelectionContext,
    select_best_candidate_global,
)
from algo_trader.domain.simulation import CandidateSpec, ModelSelectionConfig


def _write_outer_metrics(
    artifacts: SimulationArtifacts, outer_k: int, payload: dict
) -> None:
    artifacts.write_postprocess_metrics(
        outer_k=outer_k,
        metrics=payload,
    )


def test_global_selection_uses_median_across_outer_folds(
    tmp_path: Path,
) -> None:
    artifacts = SimulationArtifacts(tmp_path)
    candidates = [
        CandidateSpec(candidate_id=0, params={"alpha": 1}),
        CandidateSpec(candidate_id=1, params={"alpha": 2}),
    ]
    _write_outer_metrics(
        artifacts,
        0,
        {
            "0": {
                "es_model": 1.0,
                "se_es": 0.1,
                "calibration_score": 0.4,
                "mean_abs_coverage_error": 0.25,
                "max_abs_coverage_error": 0.4,
                "pit_uniform_rmse": 0.08,
                "crps_model": 2.0,
                "ql_model": 3.0,
            },
            "1": {
                "es_model": 2.0,
                "se_es": 0.1,
                "calibration_score": 0.1,
                "mean_abs_coverage_error": 0.08,
                "max_abs_coverage_error": 0.12,
                "pit_uniform_rmse": 0.03,
                "crps_model": 1.0,
                "ql_model": 1.0,
            },
        },
    )
    _write_outer_metrics(
        artifacts,
        1,
        {
            "0": {
                "es_model": 3.0,
                "se_es": 0.2,
                "calibration_score": 0.5,
                "mean_abs_coverage_error": 0.3,
                "max_abs_coverage_error": 0.45,
                "pit_uniform_rmse": 0.09,
                "crps_model": 4.0,
                "ql_model": 5.0,
            },
            "1": {
                "es_model": 1.0,
                "se_es": 0.2,
                "calibration_score": 0.11,
                "mean_abs_coverage_error": 0.07,
                "max_abs_coverage_error": 0.14,
                "pit_uniform_rmse": 0.02,
                "crps_model": 2.0,
                "ql_model": 2.0,
            },
        },
    )
    selection = select_best_candidate_global(
        GlobalSelectionContext(
            artifacts=artifacts,
            outer_ids=[0, 1],
            candidates=candidates,
            model_selection=ModelSelectionConfig(enable=True),
        )
    )
    assert selection.best_candidate_id == 1

    metrics_path = tmp_path / "outer" / "metrics.json"
    selection_path = tmp_path / "outer" / "selection.json"
    assert metrics_path.exists()
    assert selection_path.exists()
    metrics_payload = metrics_path.read_text(encoding="utf-8")
    assert '"0"' in metrics_payload
    assert '"1"' in metrics_payload
