from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_trader.application.research.posterior_signal import (
    PosteriorPredictiveSnapshot,
    PosteriorSignalObservation,
    run_posterior_signal_slice_study,
    run_posterior_signal_study,
    write_posterior_signal_outputs,
    write_posterior_signal_plots,
    write_slice_outputs,
    write_slice_plots,
)
from algo_trader.application.research.posterior_signal.runner import (
    resolve_source_study_dir,
)


def test_posterior_signal_study_reports_positive_signal(
    tmp_path: Path,
) -> None:
    observations = _observations()
    result = run_posterior_signal_study(observations)
    slice_result = run_posterior_signal_slice_study(observations)

    summary = result.summary.iloc[0]

    assert float(summary["mean_rank_ic"]) > 0.0
    assert float(summary["mean_top_k_spread"]) > 0.0
    assert float(summary["mean_top_k_hit_rate"]) > 0.5

    write_posterior_signal_outputs(result=result, output_dir=tmp_path)
    write_posterior_signal_plots(result=result, output_dir=tmp_path)
    write_slice_outputs(result=slice_result, output_dir=tmp_path)
    write_slice_plots(result=slice_result, output_dir=tmp_path)

    assert (tmp_path / "signal_by_week.csv").exists()
    assert (tmp_path / "sign_calibration.csv").exists()
    assert (tmp_path / "signal_summary.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "slice_diagnostics" / "by_block.csv").exists()
    assert (
        tmp_path / "slice_diagnostics" / "by_confidence_bucket.csv"
    ).exists()
    assert (tmp_path / "slice_diagnostics" / "by_sign_bucket.csv").exists()
    assert (tmp_path / "slice_diagnostics" / "summary.json").exists()
    assert (tmp_path / "plots" / "rank_ic_over_time.png").exists()
    assert (tmp_path / "plots" / "top_k_spread_over_time.png").exists()
    assert (tmp_path / "plots" / "sign_calibration.png").exists()
    assert (
        tmp_path / "slice_diagnostics" / "plots" / "rank_ic_by_block.png"
    ).exists()
    assert (
        tmp_path
        / "slice_diagnostics"
        / "plots"
        / "confidence_bucket_realized_return.png"
    ).exists()
    assert (
        tmp_path / "slice_diagnostics" / "plots" / "sign_bucket_calibration.png"
    ).exists()


def test_resolve_source_study_dir_prefers_walkforward_layout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "experiment"
    walkforward_inputs = source_root / "walkforward" / "inputs"
    walkforward_inputs.mkdir(parents=True)
    (walkforward_inputs / "panel_tensor.pt").write_bytes(b"test")

    assert resolve_source_study_dir(source_root) == source_root / "walkforward"


def _observations() -> tuple[PosteriorSignalObservation, ...]:
    return (
        PosteriorSignalObservation(
            outer_k=40,
            timestamp="2025-07-04",
            asset_names=("EUR.USD", "SPX", "XAU.USD", "DAX"),
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.10, 0.06, 0.02, -0.03]),
                posterior_std=np.array([0.04, 0.05, 0.05, 0.06]),
                p_positive=np.array([0.90, 0.75, 0.60, 0.20]),
            ),
            realized_returns=np.array([0.08, 0.05, 0.01, -0.02]),
        ),
        PosteriorSignalObservation(
            outer_k=41,
            timestamp="2025-07-11",
            asset_names=("EUR.USD", "SPX", "XAU.USD", "DAX"),
            predictive=PosteriorPredictiveSnapshot(
                posterior_mean=np.array([0.07, 0.04, 0.01, -0.01]),
                posterior_std=np.array([0.04, 0.05, 0.05, 0.05]),
                p_positive=np.array([0.85, 0.70, 0.55, 0.35]),
            ),
            realized_returns=np.array([0.06, 0.03, 0.00, -0.01]),
        ),
    )
