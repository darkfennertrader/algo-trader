from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.pairwise_index_study import (
    run_pairwise_index_study,
    write_pairwise_index_outputs,
    write_pairwise_index_plots,
)
from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
)


def test_pairwise_index_study_writes_expected_outputs(tmp_path: Path) -> None:
    result = run_pairwise_index_study(_observations())

    assert sorted(result.summary["pair_label"].tolist()) == [
        "IBDE40_minus_IBFR40",
        "IBUS30_minus_IBDE40",
        "IBUS30_minus_IBFR40",
    ]
    best_pair = str(result.summary.iloc[0]["pair_label"])
    assert best_pair == "IBUS30_minus_IBFR40"

    write_pairwise_index_outputs(result=result, output_dir=tmp_path)
    write_pairwise_index_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "pair_by_week.csv").exists()
    assert (tmp_path / "pair_summary.csv").exists()
    assert (tmp_path / "pair_calibration.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "directional_spread_by_pair.png").exists()
    assert (tmp_path / "plots" / "rank_ic_by_pair.png").exists()
    assert (tmp_path / "plots" / "direction_hit_rate_by_pair.png").exists()
    assert (tmp_path / "plots" / "calibration_rmse_by_pair.png").exists()


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = ("IBUS30", "IBDE40", "IBFR40", "EUR.USD")
    observations = []
    predictive_rows = (
        (
            np.array([0.06, 0.01, -0.01, 0.0]),
            np.array([0.03, 0.02, 0.02, 0.02]),
            np.array([0.88, 0.55, 0.45, 0.50]),
            np.array([0.05, 0.02, -0.01, 0.0]),
        ),
        (
            np.array([0.05, 0.00, -0.01, 0.0]),
            np.array([0.03, 0.02, 0.02, 0.02]),
            np.array([0.84, 0.50, 0.44, 0.50]),
            np.array([0.04, -0.01, -0.02, 0.0]),
        ),
        (
            np.array([0.04, 0.01, 0.00, 0.0]),
            np.array([0.03, 0.02, 0.02, 0.02]),
            np.array([0.81, 0.53, 0.49, 0.50]),
            np.array([0.03, 0.00, -0.01, 0.0]),
        ),
    )
    for week_index, (
        posterior_mean,
        posterior_std,
        p_positive,
        realized,
    ) in enumerate(predictive_rows, start=1):
        samples = np.vstack(
            [
                posterior_mean - 0.01,
                posterior_mean,
                posterior_mean + 0.01,
            ]
        )
        observations.append(
            PosteriorSignalObservation(
                outer_k=40 + week_index,
                timestamp=f"2025-07-{week_index:02d}",
                asset_names=asset_names,
                predictive=PosteriorPredictiveSnapshot(
                    posterior_mean=posterior_mean,
                    posterior_std=posterior_std,
                    p_positive=p_positive,
                    posterior_samples=samples,
                ),
                realized_returns=realized,
            )
        )
    return tuple(observations)
