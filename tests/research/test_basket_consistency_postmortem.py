import pandas as pd

from algo_trader.application.research.basket_consistency_postmortem import (
    build_decision_summary_from_frames,
    default_output_dir,
)


def test_default_output_dir_uses_env_root() -> None:
    path = default_output_dir("demo")
    assert path.name == "demo"
    assert "basket_consistency_postmortem" in str(path)


def test_decision_summary_flags_directional_tradeoff() -> None:
    comparisons = pd.DataFrame(
        {
            "coordinate": [
                "us_index",
                "europe_index",
                "us_minus_europe",
                "index_equal_weight",
            ],
            "delta_coverage_p90": [-0.01, 0.04, 0.03, 0.01],
        }
    )
    axis_variance = pd.DataFrame(
        {
            "axis_name": ["us_minus_europe"],
            "variance_ratio_v13_vs_v4": [0.98],
        }
    )
    axis_loading = pd.DataFrame(
        {
            "axis_name": ["us_minus_europe"],
            "dominant_basket": ["us_minus_europe"],
        }
    )
    summary = build_decision_summary_from_frames(
        basket_metrics=comparisons,
        axis_variance=axis_variance,
        axis_loading=axis_loading,
    )
    assert summary.iloc[-1]["answer"] == "not_clearly"
