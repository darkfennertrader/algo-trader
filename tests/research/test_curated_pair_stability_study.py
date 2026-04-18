from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.curated_pair_stability_study import (
    run_curated_pair_stability_study,
    write_curated_pair_stability_outputs,
    write_curated_pair_stability_plots,
)
from algo_trader.application.research.posterior_signal import PosteriorSignalObservation
from tests.research.observation_support import build_observations


def test_curated_pair_stability_study_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    result = run_curated_pair_stability_study(
        observations_by_experiment={
            "v16_l1_11y_reduced_index_universe": _observations(scale=1.0),
            "v16_l1_5y_reduced_index_universe": _observations(scale=0.8),
        },
        curated_pairs=(
            "IBCH20_minus_IBDE40",
            "IBDE40_minus_IBFR40",
            "IBUS30_minus_IBUST100",
        ),
    )

    assert sorted(result.by_experiment["pair_label"].unique().tolist()) == [
        "IBCH20_minus_IBDE40",
        "IBDE40_minus_IBFR40",
        "IBUS30_minus_IBUST100",
    ]
    assert set(result.by_period["period_label"].unique().tolist()) == {
        "first_half",
        "second_half",
    }
    best_pair = str(result.stability.iloc[0]["pair_label"])
    assert best_pair == "IBCH20_minus_IBDE40"

    write_curated_pair_stability_outputs(result=result, output_dir=tmp_path)
    write_curated_pair_stability_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "pair_by_experiment.csv").exists()
    assert (tmp_path / "pair_by_period.csv").exists()
    assert (tmp_path / "pair_stability.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "rank_ic_by_pair.png").exists()
    assert (tmp_path / "plots" / "directional_spread_by_pair.png").exists()
    assert (tmp_path / "plots" / "stability_by_pair.png").exists()


def _observations(scale: float) -> tuple[PosteriorSignalObservation, ...]:
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBFR40",
        "IBUS30",
        "IBUST100",
        "EUR.USD",
    )
    predictive_rows = (
        (
            np.array([0.05, 0.00, -0.01, 0.04, -0.01, 0.0]) * scale,
            np.array([0.03, 0.02, 0.02, 0.03, 0.02, 0.02]),
            np.array([0.82, 0.48, 0.43, 0.76, 0.44, 0.50]),
            np.array([0.04, -0.01, -0.02, 0.03, -0.02, 0.0]) * scale,
        ),
        (
            np.array([0.04, 0.00, -0.01, 0.03, -0.01, 0.0]) * scale,
            np.array([0.03, 0.02, 0.02, 0.03, 0.02, 0.02]),
            np.array([0.79, 0.49, 0.44, 0.73, 0.45, 0.50]),
            np.array([0.03, -0.01, -0.01, 0.02, -0.02, 0.0]) * scale,
        ),
        (
            np.array([0.06, 0.01, 0.00, 0.02, -0.01, 0.0]) * scale,
            np.array([0.03, 0.02, 0.02, 0.03, 0.02, 0.02]),
            np.array([0.84, 0.52, 0.48, 0.69, 0.46, 0.50]),
            np.array([0.04, 0.00, -0.01, 0.01, -0.02, 0.0]) * scale,
        ),
        (
            np.array([0.05, 0.01, 0.00, 0.01, 0.00, 0.0]) * scale,
            np.array([0.03, 0.02, 0.02, 0.03, 0.02, 0.02]),
            np.array([0.81, 0.51, 0.49, 0.58, 0.50, 0.50]),
            np.array([0.03, 0.00, -0.01, 0.00, -0.01, 0.0]) * scale,
        ),
    )
    return build_observations(
        asset_names=asset_names,
        predictive_rows=predictive_rows,
        outer_k_start=10,
        timestamp_prefix="2025-08",
    )
