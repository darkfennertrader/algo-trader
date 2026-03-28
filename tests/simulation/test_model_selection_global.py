from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

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
    _write_basket_inputs(tmp_path)
    _write_basket_candidate_payload(tmp_path, outer_k=0, candidate_id=1, offset=0.0)
    _write_basket_candidate_payload(tmp_path, outer_k=1, candidate_id=1, offset=0.2)
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
                "block_scores": {
                    "fx": {
                        "es": 1.2,
                        "crps": 2.2,
                        "ql": 3.2,
                        "coverage_p50": 0.45,
                        "coverage_p90": 0.83,
                        "coverage_p95": 0.88,
                        "n_assets": 32,
                    },
                    "full": {
                        "es": 1.0,
                        "crps": 2.0,
                        "ql": 3.0,
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.85,
                        "coverage_p95": 0.90,
                        "n_assets": 44,
                    },
                },
                "dependence_scores": {
                    "fx": {
                        "energy_score": 1.2,
                        "variogram_score": 0.5,
                        "variogram_p": 0.5,
                        "n_assets": 32,
                    },
                    "full": {
                        "energy_score": 1.0,
                        "variogram_score": 0.7,
                        "variogram_p": 0.5,
                        "n_assets": 44,
                    },
                },
                "basket_diagnostics": {
                    "us_index": {
                        "crps": 0.8,
                        "quantile_loss_p05": 0.1,
                        "quantile_loss_p25": 0.2,
                        "quantile_loss_p75": 0.3,
                        "quantile_loss_p95": 0.4,
                        "coverage_p50": 0.48,
                        "coverage_p90": 0.86,
                        "coverage_p95": 0.92,
                        "pit_uniform_rmse": 0.04,
                        "sharpness_p50": 0.5,
                        "sharpness_p90": 0.9,
                        "sharpness_p95": 1.1,
                        "n_assets": 3,
                        "n_time": 24,
                    },
                },
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
                "block_scores": {
                    "fx": {
                        "es": 0.9,
                        "crps": 0.8,
                        "ql": 0.9,
                        "coverage_p50": 0.51,
                        "coverage_p90": 0.91,
                        "coverage_p95": 0.95,
                        "n_assets": 32,
                    },
                    "full": {
                        "es": 2.0,
                        "crps": 1.0,
                        "ql": 1.0,
                        "coverage_p50": 0.52,
                        "coverage_p90": 0.92,
                        "coverage_p95": 0.96,
                        "n_assets": 44,
                    },
                },
                "dependence_scores": {
                    "fx": {
                        "energy_score": 0.9,
                        "variogram_score": 0.4,
                        "variogram_p": 0.5,
                        "n_assets": 32,
                    },
                    "full": {
                        "energy_score": 2.0,
                        "variogram_score": 0.5,
                        "variogram_p": 0.5,
                        "n_assets": 44,
                    },
                },
                "basket_diagnostics": {
                    "us_index": {
                        "crps": 0.5,
                        "quantile_loss_p05": 0.05,
                        "quantile_loss_p25": 0.1,
                        "quantile_loss_p75": 0.2,
                        "quantile_loss_p95": 0.25,
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.90,
                        "coverage_p95": 0.95,
                        "pit_uniform_rmse": 0.02,
                        "sharpness_p50": 0.4,
                        "sharpness_p90": 0.8,
                        "sharpness_p95": 1.0,
                        "n_assets": 3,
                        "n_time": 24,
                    },
                },
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
                "block_scores": {
                    "fx": {
                        "es": 3.2,
                        "crps": 4.2,
                        "ql": 5.2,
                        "coverage_p50": 0.47,
                        "coverage_p90": 0.81,
                        "coverage_p95": 0.87,
                        "n_assets": 32,
                    },
                    "full": {
                        "es": 3.0,
                        "crps": 4.0,
                        "ql": 5.0,
                        "coverage_p50": 0.48,
                        "coverage_p90": 0.82,
                        "coverage_p95": 0.88,
                        "n_assets": 44,
                    },
                },
                "dependence_scores": {
                    "fx": {
                        "energy_score": 3.2,
                        "variogram_score": 0.9,
                        "variogram_p": 0.5,
                        "n_assets": 32,
                    },
                    "full": {
                        "energy_score": 3.0,
                        "variogram_score": 1.2,
                        "variogram_p": 0.5,
                        "n_assets": 44,
                    },
                },
                "basket_diagnostics": {
                    "us_index": {
                        "crps": 1.2,
                        "quantile_loss_p05": 0.2,
                        "quantile_loss_p25": 0.25,
                        "quantile_loss_p75": 0.35,
                        "quantile_loss_p95": 0.45,
                        "coverage_p50": 0.46,
                        "coverage_p90": 0.84,
                        "coverage_p95": 0.90,
                        "pit_uniform_rmse": 0.05,
                        "sharpness_p50": 0.6,
                        "sharpness_p90": 1.0,
                        "sharpness_p95": 1.2,
                        "n_assets": 3,
                        "n_time": 24,
                    },
                },
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
                "block_scores": {
                    "fx": {
                        "es": 1.1,
                        "crps": 1.8,
                        "ql": 1.9,
                        "coverage_p50": 0.53,
                        "coverage_p90": 0.89,
                        "coverage_p95": 0.94,
                        "n_assets": 32,
                    },
                    "full": {
                        "es": 1.0,
                        "crps": 2.0,
                        "ql": 2.0,
                        "coverage_p50": 0.50,
                        "coverage_p90": 0.90,
                        "coverage_p95": 0.94,
                        "n_assets": 44,
                    },
                },
                "dependence_scores": {
                    "fx": {
                        "energy_score": 1.1,
                        "variogram_score": 0.6,
                        "variogram_p": 0.5,
                        "n_assets": 32,
                    },
                    "full": {
                        "energy_score": 1.0,
                        "variogram_score": 0.9,
                        "variogram_p": 0.5,
                        "n_assets": 44,
                    },
                },
                "basket_diagnostics": {
                    "us_index": {
                        "crps": 0.6,
                        "quantile_loss_p05": 0.06,
                        "quantile_loss_p25": 0.11,
                        "quantile_loss_p75": 0.21,
                        "quantile_loss_p95": 0.26,
                        "coverage_p50": 0.52,
                        "coverage_p90": 0.91,
                        "coverage_p95": 0.96,
                        "pit_uniform_rmse": 0.03,
                        "sharpness_p50": 0.45,
                        "sharpness_p90": 0.85,
                        "sharpness_p95": 1.05,
                        "n_assets": 3,
                        "n_time": 24,
                    },
                },
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
    block_report_path = (
        tmp_path / "outer" / "diagnostics" / "block_scoring" / "block_scores.json"
    )
    dependence_report_path = (
        tmp_path
        / "outer"
        / "diagnostics"
        / "dependence_scoring"
        / "dependence_scores.json"
    )
    basket_report_path = (
        tmp_path
        / "outer"
        / "diagnostics"
        / "basket_diagnostics"
        / "basket_scores.json"
    )
    assert metrics_path.exists()
    assert selection_path.exists()
    assert block_report_path.exists()
    assert dependence_report_path.exists()
    assert basket_report_path.exists()
    metrics_payload = metrics_path.read_text(encoding="utf-8")
    assert '"0"' in metrics_payload
    assert '"1"' in metrics_payload
    assert '"block_scores"' in metrics_payload
    assert '"dependence_scores"' in metrics_payload
    assert '"basket_diagnostics"' in metrics_payload

    selection_payload = selection_path.read_text(encoding="utf-8")
    assert '"best_candidate_block_scores"' in selection_payload
    assert '"best_candidate_dependence_scores"' in selection_payload
    assert '"best_candidate_basket_diagnostics"' in selection_payload

    block_report = block_report_path.read_text(encoding="utf-8")
    assert '"best_candidate_id": 1' in block_report
    assert '"fx"' in block_report
    dependence_report = dependence_report_path.read_text(encoding="utf-8")
    assert '"best_candidate_id": 1' in dependence_report
    assert '"variogram_score"' in dependence_report
    basket_report = basket_report_path.read_text(encoding="utf-8")
    assert '"best_candidate_id": 1' in basket_report
    assert '"us_index"' in basket_report


def _write_basket_inputs(base_dir: Path) -> None:
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        columns=[
            "timestamp",
            "IBUS30",
            "IBUS500",
            "IBUST100",
            "IBEU50",
            "IBFR40",
            "IBGB100",
            "IBCH20",
            "EUR.USD",
            "USD.JPY",
            "XAUUSD",
        ],
    )
    frame.to_csv(inputs_dir / "targets.csv", index=False)


def _write_basket_candidate_payload(
    base_dir: Path,
    *,
    outer_k: int,
    candidate_id: int,
    offset: float,
) -> None:
    candidates_dir = (
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "candidates"
    )
    candidates_dir.mkdir(parents=True, exist_ok=True)
    asset_weights = torch.tensor(
        [1.0, 1.2, 0.8, -0.9, -1.1, -0.7, 0.4, 0.2, -0.2, 0.6],
        dtype=torch.float64,
    )
    sample_multipliers = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    true_multipliers = torch.tensor([0.5, -0.5, 1.5, -1.5], dtype=torch.float64) + offset
    z_samples = sample_multipliers[:, None, None] * asset_weights[None, None, :]
    z_samples = z_samples.repeat(1, 4, 1)
    z_true = true_multipliers[:, None] * asset_weights[None, :]
    torch.save(
        {
            "z_true": z_true,
            "z_samples": z_samples,
            "scale": torch.ones(10, dtype=torch.float64),
            "test_idx": [0, 1, 2, 3],
        },
        candidates_dir / f"candidate_{candidate_id:04d}_split_0000.pt",
    )
