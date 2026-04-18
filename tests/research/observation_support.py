from __future__ import annotations

import numpy as np

from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
)


def build_observations(
    *,
    asset_names: tuple[str, ...],
    predictive_rows: tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ...,
    ],
    outer_k_start: int,
    timestamp_prefix: str,
) -> tuple[PosteriorSignalObservation, ...]:
    observations = []
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
                outer_k=outer_k_start + week_index,
                timestamp=f"{timestamp_prefix}-{week_index:02d}",
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
