from __future__ import annotations

from pathlib import Path

import torch

from algo_trader.application.simulation.model_selection import (
    _complexity_scores_post_tune,
    FinalSelectionInputs,
    _select_final_candidate,
    _variogram_weekly,
)
from algo_trader.domain.simulation import (
    CandidateSpec,
    ModelSelectionCalibration,
    ModelSelectionComplexity,
    ModelSelectionConfig,
)


def test_posterior_l1_complexity_prefers_more_shrunk_candidate(
    tmp_path: Path,
) -> None:
    debug_dir = (
        tmp_path / "inner" / "outer_7" / "postprocessing" / "debug"
    )
    debug_dir.mkdir(parents=True)
    _write_debug_payload(
        debug_dir / "candidate_0000_split_0000_state.pt",
        weight_scale=0.01,
    )
    _write_debug_payload(
        debug_dir / "candidate_0001_split_0000_state.pt",
        weight_scale=0.10,
    )

    scores = _complexity_scores_post_tune(
        base_dir=tmp_path,
        outer_k=7,
        candidates=(
            CandidateSpec(candidate_id=0, params={}),
            CandidateSpec(candidate_id=1, params={}),
        ),
        model_selection=ModelSelectionConfig(
            complexity=ModelSelectionComplexity(method="posterior_l1")
        ),
    )

    assert scores[0] < scores[1]


def _write_debug_payload(path: Path, *, weight_scale: float) -> None:
    payload = {
        "structural_posterior_means": {
            "alpha": torch.full((2,), 0.5),
            "sigma_idio": torch.full((2,), 0.2),
            "w": torch.full((2, 3), weight_scale),
            "beta": torch.full((2, 1), weight_scale),
            "B": torch.full((2, 2), weight_scale),
            "s_u_mean": torch.tensor(0.1),
        }
    }
    torch.save(payload, path)


def test_select_final_candidate_prefers_more_calibrated_survivor() -> None:
    selection = _select_final_candidate(
        inputs=FinalSelectionInputs(
            es_metrics={
                0: {"es_model": 1.0, "se_es": 0.1},
                1: {"es_model": 1.05, "se_es": 0.1},
            },
            calibration_metrics={
                0: {
                    "calibration_score": 0.50,
                    "mean_abs_coverage_error": 0.30,
                    "max_abs_coverage_error": 0.40,
                    "pit_uniform_rmse": 0.08,
                },
                1: {
                    "calibration_score": 0.12,
                    "mean_abs_coverage_error": 0.08,
                    "max_abs_coverage_error": 0.15,
                    "pit_uniform_rmse": 0.03,
                },
            },
            crps_metrics={1: 1.0},
            ql_metrics={1: 1.0},
            basket_diagnostics={1: {}},
            complexity={0: 0.1, 1: 0.2},
        ),
        model_selection=ModelSelectionConfig(
            calibration=ModelSelectionCalibration(top_k=1)
        ),
    )

    assert selection["best_candidate_id"] == 1
    assert selection["survivors_calibration"] == [1]
    assert selection["survivors_es"] == [1]


def test_select_final_candidate_uses_basket_aware_mode() -> None:
    selection = _select_final_candidate(
        inputs=FinalSelectionInputs(
            es_metrics={
                0: {"es_model": 1.0, "se_es": 0.1},
                1: {"es_model": 1.0, "se_es": 0.1},
            },
            calibration_metrics={
                0: {
                    "calibration_score": 0.10,
                    "mean_abs_coverage_error": 0.08,
                    "max_abs_coverage_error": 0.12,
                    "pit_uniform_rmse": 0.03,
                },
                1: {
                    "calibration_score": 0.11,
                    "mean_abs_coverage_error": 0.08,
                    "max_abs_coverage_error": 0.12,
                    "pit_uniform_rmse": 0.03,
                },
            },
            crps_metrics={0: 1.0, 1: 1.0},
            ql_metrics={0: 1.0, 1: 1.0},
            basket_diagnostics={
                0: {
                    "us_index": {
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.90,
                        "coverage_p95": 0.95,
                        "pit_uniform_rmse": 0.02,
                    },
                    "europe_index": {
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.90,
                        "coverage_p95": 0.95,
                        "pit_uniform_rmse": 0.02,
                    },
                    "us_minus_europe": {
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.90,
                        "coverage_p95": 0.95,
                        "pit_uniform_rmse": 0.02,
                    },
                    "index_equal_weight": {
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.90,
                        "coverage_p95": 0.95,
                        "pit_uniform_rmse": 0.02,
                    },
                },
                1: {
                    "us_index": {
                        "coverage_p50": 0.45,
                        "coverage_p90": 0.85,
                        "coverage_p95": 0.90,
                        "pit_uniform_rmse": 0.05,
                    },
                    "europe_index": {
                        "coverage_p50": 0.45,
                        "coverage_p90": 0.85,
                        "coverage_p95": 0.90,
                        "pit_uniform_rmse": 0.05,
                    },
                    "us_minus_europe": {
                        "coverage_p50": 0.45,
                        "coverage_p90": 0.85,
                        "coverage_p95": 0.90,
                        "pit_uniform_rmse": 0.05,
                    },
                    "index_equal_weight": {
                        "coverage_p50": 0.45,
                        "coverage_p90": 0.85,
                        "coverage_p95": 0.90,
                        "pit_uniform_rmse": 0.05,
                    },
                },
            },
            complexity={0: 0.2, 1: 0.1},
        ),
        model_selection=ModelSelectionConfig(
            mode="basket_aware",
            calibration=ModelSelectionCalibration(top_k=2),
        ),
    )

    assert selection["best_candidate_id"] == 0
    assert selection["survivors_calibration"] == [0, 1]
    assert selection["survivors_es"] == [0, 1]
    assert selection["survivors_secondary"] == [0]
    assert selection["basket_scores"][0] < selection["basket_scores"][1]


def test_variogram_weekly_matches_manual_two_asset_case() -> None:
    z_true = torch.tensor([[1.0, 3.0]])
    z_samples = torch.tensor(
        [
            [[0.0, 1.0]],
            [[2.0, 5.0]],
        ]
    )

    score = _variogram_weekly(
        z_true=z_true,
        z_samples=z_samples,
        variogram_p=0.5,
    )

    expected = torch.tensor(
        [(2.0 ** 0.5 - (1.0 + 3.0 ** 0.5) / 2.0) ** 2]
    )
    assert torch.allclose(score, expected)
