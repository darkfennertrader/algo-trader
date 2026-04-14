from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.index_universe_study import (
    run_index_universe_study,
    write_index_universe_outputs,
    write_index_universe_plots,
)
from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
)


def test_index_universe_study_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    result = run_index_universe_study(_observations())

    labels = result.summary["universe_label"].tolist()
    assert labels == [
        "full_indices",
        "drop_ibus500",
        "drop_ibeu50",
        "drop_both_benchmarks",
    ]
    full_rank_ic = float(
        result.summary.loc[
            result.summary["universe_label"] == "full_indices",
            "mean_rank_ic",
        ].iloc[0]
    )
    reduced_rank_ic = float(
        result.summary.loc[
            result.summary["universe_label"] == "drop_both_benchmarks",
            "mean_rank_ic",
        ].iloc[0]
    )
    assert reduced_rank_ic > full_rank_ic

    write_index_universe_outputs(result=result, output_dir=tmp_path)
    write_index_universe_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "universe_by_week.csv").exists()
    assert (tmp_path / "universe_summary.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "rank_ic_by_universe.png").exists()
    assert (tmp_path / "plots" / "top_k_spread_by_universe.png").exists()
    assert (tmp_path / "plots" / "calibration_rmse_by_universe.png").exists()


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = ("IBUS30", "IBUS500", "IBDE40", "IBEU50")
    return (
        PosteriorSignalObservation(
            outer_k=40,
            timestamp="2025-07-04",
            asset_names=asset_names,
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.05, 0.10, -0.02, -0.09]),
                posterior_std=np.array([0.03, 0.03, 0.03, 0.03]),
                p_positive=np.array([0.70, 0.80, 0.45, 0.20]),
                posterior_samples=np.array(
                    [
                        [0.05, 0.11, -0.01, -0.08],
                        [0.06, 0.10, -0.02, -0.09],
                        [0.04, 0.09, -0.03, -0.10],
                    ],
                    dtype=float,
                ),
            ),
            realized_returns=np.array([0.06, -0.02, -0.01, -0.08]),
        ),
        PosteriorSignalObservation(
            outer_k=41,
            timestamp="2025-07-11",
            asset_names=asset_names,
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.04, 0.09, -0.01, -0.08]),
                posterior_std=np.array([0.03, 0.03, 0.03, 0.03]),
                p_positive=np.array([0.68, 0.78, 0.48, 0.22]),
                posterior_samples=np.array(
                    [
                        [0.04, 0.10, -0.01, -0.08],
                        [0.05, 0.09, 0.00, -0.09],
                        [0.03, 0.08, -0.02, -0.07],
                    ],
                    dtype=float,
                ),
            ),
            realized_returns=np.array([0.05, -0.01, 0.00, -0.07]),
        ),
    )
