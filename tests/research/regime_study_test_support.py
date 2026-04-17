from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
)


def assert_regime_outputs_exist(output_dir: Path) -> None:
    assert (output_dir / "regime_by_week.csv").exists()
    assert (output_dir / "regime_summary.csv").exists()
    assert (output_dir / "regime_definition.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "plots" / "rank_ic_by_regime.png").exists()
    assert (output_dir / "plots" / "top_k_spread_by_regime.png").exists()
    assert (output_dir / "plots" / "hit_rate_by_regime.png").exists()
    assert (output_dir / "plots" / "calibration_rmse_by_regime.png").exists()


def sample_regime_observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = ("IBUS30", "IBUST100", "IBDE40", "EUR.USD")
    realized_by_week = (
        np.array([0.05, 0.04, 0.01, 0.0]),
        np.array([0.04, 0.03, 0.00, 0.0]),
        np.array([0.01, 0.00, -0.01, 0.0]),
        np.array([-0.03, -0.04, -0.02, 0.0]),
        np.array([-0.04, -0.05, -0.03, 0.0]),
        np.array([0.00, 0.01, 0.00, 0.0]),
    )
    observations: list[PosteriorSignalObservation] = []
    for week_index, realized in enumerate(realized_by_week, start=1):
        posterior_mean = realized + np.array([0.01, 0.0, -0.005, 0.0])
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
                    posterior_std=np.full(4, 0.02),
                    p_positive=np.clip(0.5 + posterior_mean * 5.0, 0.05, 0.95),
                    posterior_samples=samples,
                ),
                realized_returns=realized,
            )
        )
    return tuple(observations)
