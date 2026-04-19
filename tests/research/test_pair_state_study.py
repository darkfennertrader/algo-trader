from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.pair_state_study import (
    run_pair_state_study,
    write_pair_state_outputs,
    write_pair_state_plots,
)
from algo_trader.application.research.posterior_signal import PosteriorSignalObservation
from tests.research.observation_support import build_observations


def test_pair_state_study_writes_expected_outputs(tmp_path: Path) -> None:
    result = run_pair_state_study(
        _observations(),
        curated_pairs=(
            "IBCH20_minus_IBDE40",
            "IBUS30_minus_IBUST100",
        ),
    )

    assert sorted(result.by_week["pair_label"].unique().tolist()) == [
        "IBCH20_minus_IBDE40",
        "IBUS30_minus_IBUST100",
    ]
    assert set(result.state_summary["state_label"].astype(str).tolist()) >= {
        "insufficient_history",
        "continuation_up",
        "range",
    }
    continuation_up = result.state_summary.loc[
        result.state_summary["state_label"].astype(str) == "continuation_up"
    ].reset_index(drop=True)
    assert not continuation_up.empty
    continuation_value = continuation_up["mean_directional_spread"].to_numpy(
        dtype=float
    )[0]
    assert continuation_value > 0.0

    write_pair_state_outputs(result=result, output_dir=tmp_path)
    write_pair_state_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "state_by_week.csv").exists()
    assert (tmp_path / "state_summary.csv").exists()
    assert (tmp_path / "pair_state_summary.csv").exists()
    assert (tmp_path / "state_definition.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "rank_ic_by_state.png").exists()
    assert (tmp_path / "plots" / "directional_spread_by_state.png").exists()
    assert (tmp_path / "plots" / "hit_rate_by_state.png").exists()


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBUS30",
        "IBUST100",
        "EUR.USD",
    )
    predictive_rows = (
        _row((0.04, 0.00, 0.03, -0.01), (0.03, 0.00, 0.02, -0.01)),
        _row((0.05, 0.00, 0.02, -0.01), (0.04, 0.00, 0.03, -0.02)),
        _row((0.06, 0.01, 0.03, -0.01), (0.02, -0.01, 0.01, -0.02)),
        _row((0.05, 0.01, 0.04, -0.01), (-0.01, 0.00, -0.02, 0.01)),
        _row((-0.02, 0.01, -0.03, 0.00), (0.03, 0.00, 0.02, -0.01)),
        _row((-0.01, 0.00, -0.02, 0.00), (0.04, 0.01, 0.03, -0.01)),
    )
    return build_observations(
        asset_names=asset_names,
        predictive_rows=predictive_rows,
        outer_k_start=30,
        timestamp_prefix="2025-11",
    )


def _row(
    pair_one: tuple[float, float, float, float],
    pair_two: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_a, mean_b, realized_a, realized_b = pair_one
    mean_c, mean_d, realized_c, realized_d = pair_two
    posterior_mean = np.array([mean_a, mean_b, mean_c, mean_d, 0.0], dtype=float)
    posterior_std = np.array([0.03, 0.02, 0.03, 0.02, 0.02], dtype=float)
    p_positive = np.array([0.78, 0.48, 0.74, 0.44, 0.50], dtype=float)
    realized = np.array(
        [realized_a, realized_b, realized_c, realized_d, 0.0],
        dtype=float,
    )
    return posterior_mean, posterior_std, p_positive, realized
