from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.index_subset_study import (
    run_index_subset_study,
    write_index_subset_outputs,
    write_index_subset_plots,
)
from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
)


def test_index_subset_study_writes_expected_outputs(tmp_path: Path) -> None:
    result = run_index_subset_study(_observations())

    labels = result.summary["subset_label"].tolist()
    assert labels == [
        "full_indices",
        "drop_ibus30",
        "drop_ibde40",
        "drop_ibfr40",
    ]
    full_rank_ic = float(
        result.summary.loc[
            result.summary["subset_label"] == "full_indices",
            "mean_rank_ic",
        ].iloc[0]
    )
    reduced_rank_ic = float(
        result.summary.loc[
            result.summary["subset_label"] == "drop_ibus30",
            "mean_rank_ic",
        ].iloc[0]
    )
    assert reduced_rank_ic > full_rank_ic

    write_index_subset_outputs(result=result, output_dir=tmp_path)
    write_index_subset_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "subset_by_week.csv").exists()
    assert (tmp_path / "subset_summary.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "rank_ic_by_subset.png").exists()
    assert (tmp_path / "plots" / "top_k_spread_by_subset.png").exists()
    assert (tmp_path / "plots" / "top_k_hit_rate_by_subset.png").exists()


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = ("IBUS30", "IBDE40", "IBFR40")
    return (
        PosteriorSignalObservation(
            outer_k=40,
            timestamp="2025-07-04",
            asset_names=asset_names,
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.12, 0.01, 0.00]),
                posterior_std=np.array([0.03, 0.03, 0.03]),
                p_positive=np.array([0.85, 0.52, 0.49]),
                posterior_samples=np.array(
                    [
                        [0.12, 0.01, -0.01],
                        [0.11, 0.02, 0.00],
                        [0.13, 0.00, 0.01],
                    ],
                    dtype=float,
                ),
            ),
            realized_returns=np.array([-0.05, 0.02, 0.01]),
        ),
        PosteriorSignalObservation(
            outer_k=41,
            timestamp="2025-07-11",
            asset_names=asset_names,
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.11, 0.02, 0.01]),
                posterior_std=np.array([0.03, 0.03, 0.03]),
                p_positive=np.array([0.82, 0.55, 0.50]),
                posterior_samples=np.array(
                    [
                        [0.11, 0.03, 0.01],
                        [0.10, 0.02, 0.00],
                        [0.12, 0.01, 0.02],
                    ],
                    dtype=float,
                ),
            ),
            realized_returns=np.array([-0.04, 0.03, 0.02]),
        ),
    )
