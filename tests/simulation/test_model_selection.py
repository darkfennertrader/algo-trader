from __future__ import annotations

from pathlib import Path

import torch

from algo_trader.application.simulation.model_selection import (
    _complexity_scores_post_tune,
    FinalSelectionInputs,
    PostTuneSelectionContext,
    _select_es_survivors,
    _select_final_candidate,
    select_best_candidate_post_tune,
    _variogram_weekly,
)
from algo_trader.application.simulation.artifacts import SimulationArtifacts
from algo_trader.domain.simulation import (
    CandidateSpec,
    ModelSelectionCalibration,
    ModelSelectionComplexity,
    ModelSelectionConfig,
    ModelSelectionESBand,
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
            signal_metrics={1: {}},
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
            signal_metrics={0: {}, 1: {}},
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


def test_select_final_candidate_uses_signal_aware_mode() -> None:
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
            signal_metrics={
                0: {
                    "mean_rank_ic": 0.05,
                    "positive_rank_ic_fraction": 0.60,
                    "mean_linear_ic": 0.04,
                    "mean_top_k_spread": 0.01,
                    "mean_top_k_hit_rate": 0.55,
                    "mean_brier_score": 0.18,
                    "calibration_rmse": 0.12,
                },
                1: {
                    "mean_rank_ic": 0.01,
                    "positive_rank_ic_fraction": 0.52,
                    "mean_linear_ic": 0.01,
                    "mean_top_k_spread": 0.002,
                    "mean_top_k_hit_rate": 0.49,
                    "mean_brier_score": 0.24,
                    "calibration_rmse": 0.18,
                },
            },
            basket_diagnostics={0: {}, 1: {}},
            complexity={0: 0.2, 1: 0.1},
        ),
        model_selection=ModelSelectionConfig(
            mode="signal_aware",
            calibration=ModelSelectionCalibration(top_k=2),
        ),
    )

    assert selection["best_candidate_id"] == 0
    assert selection["survivors_calibration"] == [0, 1]
    assert selection["survivors_es"] == [0, 1]
    assert selection["survivors_secondary"] == [0]
    assert selection["signal_scores"][0] < selection["signal_scores"][1]


def test_select_es_survivors_enforces_min_keep() -> None:
    survivors = _select_es_survivors(
        {
            0: {"es_model": 1.0, "se_es": 0.1},
            1: {"es_model": 1.5, "se_es": 0.1},
            2: {"es_model": 1.7, "se_es": 0.1},
        },
        ModelSelectionConfig(es_band=ModelSelectionESBand(min_keep=3)),
        candidate_ids=[0, 1, 2],
    )

    assert survivors == [0, 1, 2]


def test_select_best_candidate_post_tune_persists_metrics_for_all_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secondary_candidate_ids: list[int] = []
    signal_candidate_ids: list[int] = []

    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._resolve_device",
        lambda _use_gpu: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._compute_es_metrics",
        lambda **_kwargs: {
            0: {"es_model": 1.0, "se_es": 0.1},
            1: {"es_model": 1.1, "se_es": 0.1},
            2: {"es_model": 2.0, "se_es": 0.1},
        },
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._compute_calibration_metrics",
        lambda **_kwargs: {
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
            2: {
                "calibration_score": 0.12,
                "mean_abs_coverage_error": 0.08,
                "max_abs_coverage_error": 0.12,
                "pit_uniform_rmse": 0.03,
            },
        },
    )

    def _fake_secondary_metrics(**kwargs):
        secondary_candidate_ids.extend(kwargs["candidate_ids"])
        return ({0: 1.0, 1: 1.1, 2: 1.2}, {0: 1.0, 1: 1.1, 2: 1.2})

    def _fake_signal_metrics(**kwargs):
        signal_candidate_ids.extend(kwargs["candidate_ids"])
        return {
            0: {
                "mean_rank_ic": 0.05,
                "positive_rank_ic_fraction": 0.60,
                "mean_linear_ic": 0.04,
                "mean_top_k_spread": 0.01,
                "mean_top_k_hit_rate": 0.55,
                "mean_brier_score": 0.18,
                "calibration_rmse": 0.12,
            },
            1: {
                "mean_rank_ic": 0.03,
                "positive_rank_ic_fraction": 0.55,
                "mean_linear_ic": 0.02,
                "mean_top_k_spread": 0.005,
                "mean_top_k_hit_rate": 0.52,
                "mean_brier_score": 0.20,
                "calibration_rmse": 0.14,
            },
            2: {
                "mean_rank_ic": 0.01,
                "positive_rank_ic_fraction": 0.51,
                "mean_linear_ic": 0.01,
                "mean_top_k_spread": 0.001,
                "mean_top_k_hit_rate": 0.49,
                "mean_brier_score": 0.24,
                "calibration_rmse": 0.18,
            },
        }

    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._compute_secondary_metrics",
        _fake_secondary_metrics,
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._compute_signal_metrics",
        _fake_signal_metrics,
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._build_block_metric_context",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._compute_diagnostic_scores",
        lambda **_kwargs: (
            {0: {}, 1: {}, 2: {}},
            {0: {}, 1: {}, 2: {}},
            {0: {}, 1: {}, 2: {}},
        ),
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._complexity_scores_post_tune",
        lambda **_kwargs: {0: 0.2, 1: 0.1, 2: 0.3},
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._with_best_candidate_block_scores",
        lambda **kwargs: kwargs["selection"],
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._with_best_candidate_dependence_scores",
        lambda **kwargs: kwargs["selection"],
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._with_best_candidate_basket_diagnostics",
        lambda **kwargs: kwargs["selection"],
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._write_postprocess_block_scores_report",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._write_postprocess_dependence_scores_report",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._write_postprocess_basket_diagnostics_report",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.model_selection._write_postprocess_residual_dependence_report",
        lambda **_kwargs: None,
    )

    result = select_best_candidate_post_tune(
        PostTuneSelectionContext(
            artifacts=SimulationArtifacts(tmp_path),
            outer_k=40,
            candidates=(
                CandidateSpec(candidate_id=0, params={}),
                CandidateSpec(candidate_id=1, params={}),
                CandidateSpec(candidate_id=2, params={}),
            ),
            model_selection=ModelSelectionConfig(mode="signal_aware"),
            score_spec={},
            use_gpu=False,
        )
    )

    assert secondary_candidate_ids == [0, 1, 2]
    assert signal_candidate_ids == [0, 1, 2]
    assert result.metrics["2"]["mean_rank_ic"] == 0.01


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
