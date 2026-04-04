from __future__ import annotations

import pandas as pd

from algo_trader.application.research.state_conditioned_measurement_postmortem import (
    build_counterfactual_decision_summary,
    default_output_dir,
)


def test_default_output_dir_uses_env_root() -> None:
    path = default_output_dir("demo")
    assert path.name == "demo"
    assert "state_conditioned_measurement_postmortem" in str(path)


def test_decision_summary_prefers_better_counterfactual() -> None:
    basket_summary = pd.DataFrame(
        {
            "scenario": [
                "baseline",
                "baseline",
                "gate_off",
                "gate_off",
            ],
            "basket": [
                "us_index",
                "europe_index",
                "us_index",
                "europe_index",
            ],
            "gate_bucket": ["all", "all", "all", "all"],
            "coverage_p50": [0.40, 0.40, 0.48, 0.49],
            "coverage_p90": [0.75, 0.78, 0.88, 0.91],
            "coverage_p95": [0.81, 0.84, 0.94, 0.95],
            "pit_uniform_rmse": [0.020, 0.021, 0.011, 0.012],
            "sharpness_p90": [0.10, 0.11, 0.12, 0.13],
            "n_time": [12, 12, 12, 12],
        }
    )
    summary = build_counterfactual_decision_summary(basket_summary)
    assert summary.iloc[0]["scenario"] == "gate_off"
    assert float(summary.iloc[0]["coverage_improvement_vs_baseline"]) > 0.0
