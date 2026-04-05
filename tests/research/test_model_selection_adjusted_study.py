from __future__ import annotations

import json
from pathlib import Path

from algo_trader.application.research.model_selection_adjusted import (
    ModelSelectionAdjustedRunSpec,
    ModelSelectionAdjustedStudyConfig,
    run_model_selection_adjusted_study,
)
from algo_trader.domain.simulation import ModelSelectionConfig


def test_model_selection_adjusted_study_reselects_saved_metrics(
    tmp_path: Path,
) -> None:
    simulation_dir = tmp_path / "simulation" / "example_run"
    outer_dir = simulation_dir / "outer"
    outer_dir.mkdir(parents=True)
    _write_json(outer_dir / "metrics.json", _metrics_payload())
    _write_json(outer_dir / "selection.json", {"best_candidate_id": 1})

    result = run_model_selection_adjusted_study(
        ModelSelectionAdjustedStudyConfig(
            output_dir=tmp_path / "out",
            model_selection=ModelSelectionConfig(enable=True, mode="basket_aware"),
            runs=(
                ModelSelectionAdjustedRunSpec(
                    label="example",
                    simulation_dir=simulation_dir,
                ),
            ),
        )
    )

    summary = result.tables["summary"]
    original_best = summary["original_best_candidate_id"].tolist()
    global_best = summary["global_calibrated_best_candidate_id"].tolist()
    basket_best = summary["basket_aware_best_candidate_id"].tolist()
    changed = summary["changed_under_basket_aware"].tolist()

    assert original_best == [1]
    assert global_best == [1]
    assert basket_best == [2]
    assert changed == [True]

    comparison = result.tables["example_candidate_comparison"]
    selected = comparison.loc[
        comparison["selected_basket_aware"], "candidate_id"
    ].tolist()
    assert selected == [2]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics_payload() -> dict[str, object]:
    return {
        "1": _candidate_payload(
            {
                "es_model": 1.0,
                "calibration_score": 0.10,
                "mean_abs": 0.020,
                "max_abs": 0.030,
                "pit": 0.010,
                "crps": 0.10,
                "ql": 0.21,
                "us_index": 0.86,
                "europe_index": 0.89,
                "us_minus_europe": 0.995,
                "index_equal_weight": 0.84,
            }
        ),
        "2": _candidate_payload(
            {
                "es_model": 1.02,
                "calibration_score": 0.11,
                "mean_abs": 0.021,
                "max_abs": 0.031,
                "pit": 0.011,
                "crps": 0.11,
                "ql": 0.20,
                "us_index": 0.90,
                "europe_index": 0.90,
                "us_minus_europe": 0.93,
                "index_equal_weight": 0.89,
            }
        ),
        "3": _candidate_payload(
            {
                "es_model": 1.50,
                "calibration_score": 0.50,
                "mean_abs": 0.100,
                "max_abs": 0.120,
                "pit": 0.050,
                "crps": 0.50,
                "ql": 0.60,
                "us_index": 0.70,
                "europe_index": 0.70,
                "us_minus_europe": 0.70,
                "index_equal_weight": 0.70,
            }
        ),
    }


def _candidate_payload(values: dict[str, float]) -> dict[str, object]:
    return {
        "es_model": values["es_model"],
        "se_es": 0.05,
        "calibration_score": values["calibration_score"],
        "mean_abs_coverage_error": values["mean_abs"],
        "max_abs_coverage_error": values["max_abs"],
        "pit_uniform_rmse": values["pit"],
        "crps_model": values["crps"],
        "ql_model": values["ql"],
        "complexity": 0.01,
        "basket_diagnostics": {
            "us_index": _basket_payload(values["us_index"]),
            "europe_index": _basket_payload(values["europe_index"]),
            "us_minus_europe": _basket_payload(values["us_minus_europe"]),
            "index_equal_weight": _basket_payload(values["index_equal_weight"]),
        },
    }


def _basket_payload(coverage_p90: float) -> dict[str, float]:
    return {
        "coverage_p50": 0.50,
        "coverage_p90": coverage_p90,
        "coverage_p95": min(coverage_p90 + 0.03, 1.0),
        "pit_uniform_rmse": 0.02,
    }
