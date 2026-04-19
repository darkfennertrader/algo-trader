from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.pair_postmortem_study import (
    run_pair_postmortem_study,
    write_pair_postmortem_outputs,
    write_pair_postmortem_plots,
)
from algo_trader.application.research.posterior_signal import PosteriorSignalObservation
from tests.research.observation_support import build_observations


def test_pair_postmortem_study_writes_expected_outputs(tmp_path: Path) -> None:
    result = run_pair_postmortem_study(
        _observations(),
        curated_pairs=(
            "IBCH20_minus_IBDE40",
            "IBUS30_minus_IBUST100",
        ),
    )

    assert sorted(result.pair_summary["pair_label"].unique().tolist()) == [
        "IBCH20_minus_IBDE40",
        "IBUS30_minus_IBUST100",
    ]
    assert "range" in set(result.pair_state_summary["state_label"].astype(str))
    assert set(result.pair_confidence_summary["confidence_bucket"].astype(str)) <= {
        "weak",
        "medium",
        "strong",
    }

    write_pair_postmortem_outputs(result=result, output_dir=tmp_path)
    write_pair_postmortem_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "pair_by_week.csv").exists()
    assert (tmp_path / "pair_summary.csv").exists()
    assert (tmp_path / "pair_state_summary.csv").exists()
    assert (tmp_path / "pair_confidence_summary.csv").exists()
    assert (tmp_path / "pair_calibration.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "rmse_by_pair.png").exists()
    assert (tmp_path / "plots" / "direction_hit_rate_by_pair.png").exists()
    assert (tmp_path / "plots" / "calibration_rmse_by_pair.png").exists()


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBUS30",
        "IBUST100",
        "EUR.USD",
    )
    predictive_rows = (
        (
            np.array([0.04, 0.01, 0.03, 0.00, 0.0]),
            np.array([0.03, 0.03, 0.02, 0.02, 0.02]),
            np.array([0.75, 0.55, 0.72, 0.48, 0.50]),
            np.array([0.03, 0.01, 0.02, -0.01, 0.0]),
        ),
        (
            np.array([0.03, 0.01, 0.02, 0.00, 0.0]),
            np.array([0.03, 0.03, 0.02, 0.02, 0.02]),
            np.array([0.70, 0.54, 0.68, 0.49, 0.50]),
            np.array([0.02, 0.00, 0.03, 0.00, 0.0]),
        ),
        (
            np.array([0.02, 0.01, 0.01, 0.00, 0.0]),
            np.array([0.03, 0.03, 0.02, 0.02, 0.02]),
            np.array([0.62, 0.53, 0.60, 0.50, 0.50]),
            np.array([0.01, 0.00, 0.01, 0.00, 0.0]),
        ),
        (
            np.array([0.03, 0.01, 0.04, 0.01, 0.0]),
            np.array([0.03, 0.03, 0.02, 0.02, 0.02]),
            np.array([0.73, 0.55, 0.76, 0.52, 0.50]),
            np.array([0.03, 0.01, 0.04, 0.00, 0.0]),
        ),
        (
            np.array([0.02, 0.00, 0.03, 0.01, 0.0]),
            np.array([0.03, 0.03, 0.02, 0.02, 0.02]),
            np.array([0.68, 0.49, 0.74, 0.53, 0.50]),
            np.array([0.01, -0.01, 0.02, 0.00, 0.0]),
        ),
    )
    return build_observations(
        asset_names=asset_names,
        predictive_rows=predictive_rows,
        outer_k_start=20,
        timestamp_prefix="2025-09",
    )
