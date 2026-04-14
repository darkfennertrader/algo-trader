from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.index_target_study import (
    run_index_target_study,
    write_index_target_outputs,
    write_index_target_plots,
)
from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
)


def test_index_target_study_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    result = run_index_target_study(_observations())

    target_names = result.summary["target_name"].tolist()
    assert target_names == sorted(
        target_names,
        key={
            "raw_return": 0,
            "equal_weight_relative_return": 1,
            "regional_relative_return": 2,
            "raw_sign": 3,
            "regional_relative_sign": 4,
        }.__getitem__,
    )
    raw_row = result.summary.loc[
        result.summary["target_name"] == "raw_return"
    ].iloc[0]
    assert float(raw_row["mean_rank_ic"]) > 0.0
    assert float(raw_row["mean_top_k_hit_rate"]) > 0.0

    write_index_target_outputs(result=result, output_dir=tmp_path)
    write_index_target_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "target_by_week.csv").exists()
    assert (tmp_path / "target_summary.csv").exists()
    assert (tmp_path / "target_calibration.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "rank_ic_by_target.png").exists()
    assert (tmp_path / "plots" / "top_k_spread_by_target.png").exists()
    assert (tmp_path / "plots" / "calibration_rmse_by_target.png").exists()


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = ("IBUS30", "IBUS500", "IBDE40", "IBEU50")
    return (
        PosteriorSignalObservation(
            outer_k=40,
            timestamp="2025-07-04",
            asset_names=asset_names,
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.09, 0.06, -0.01, -0.03]),
                posterior_std=np.array([0.03, 0.03, 0.04, 0.04]),
                p_positive=np.array([0.85, 0.75, 0.40, 0.25]),
                posterior_samples=np.array(
                    [
                        [0.10, 0.07, 0.00, -0.02],
                        [0.08, 0.05, -0.01, -0.03],
                        [0.09, 0.06, -0.02, -0.04],
                    ],
                    dtype=float,
                ),
            ),
            realized_returns=np.array([0.08, 0.03, -0.01, -0.04]),
        ),
        PosteriorSignalObservation(
            outer_k=41,
            timestamp="2025-07-11",
            asset_names=asset_names,
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.07, 0.04, 0.00, -0.02]),
                posterior_std=np.array([0.03, 0.03, 0.04, 0.04]),
                p_positive=np.array([0.80, 0.70, 0.52, 0.30]),
                posterior_samples=np.array(
                    [
                        [0.08, 0.05, 0.02, -0.01],
                        [0.07, 0.04, 0.00, -0.02],
                        [0.06, 0.03, -0.01, -0.03],
                    ],
                    dtype=float,
                ),
            ),
            realized_returns=np.array([0.05, 0.02, 0.01, -0.03]),
        ),
    )
