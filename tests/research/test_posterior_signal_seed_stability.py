from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_trader.application.research.posterior_signal import seed_stability
from algo_trader.application.simulation.seed_stability_common import SeedStudyResult


def test_write_seed_outputs_stages_slice_diagnostics_separately(
    tmp_path: Path,
) -> None:
    results = tuple(
        _seed_result(
            root=tmp_path,
            seed=seed,
            mean_rank_ic=mean_rank_ic,
        )
        for seed, mean_rank_ic in ((7, 0.10), (9, -0.02))
    )

    write_outputs = getattr(seed_stability, "_write_seed_outputs")
    payload = write_outputs(
        base_dir=tmp_path / "study",
        study_dir=tmp_path / "study" / "seed_stability",
        source_label="validated_model",
        results=results,
    )

    slice_dir = tmp_path / "study" / "slice_diagnostics"
    assert payload["posterior_signal_seed_stability"]["slice_diagnostics_dir"] == str(
        slice_dir
    )
    assert (slice_dir / "by_block.csv").exists()
    assert (slice_dir / "by_confidence_bucket.csv").exists()
    assert (slice_dir / "by_sign_bucket.csv").exists()
    assert (slice_dir / "plots" / "rank_ic_by_block.png").exists()
    assert (
        slice_dir / "seed_stability" / "seed_7" / "by_block.csv"
    ).exists()
    assert (
        slice_dir / "seed_stability" / "seed_9" / "by_sign_bucket.csv"
    ).exists()


def _seed_result(
    *,
    root: Path,
    seed: int,
    mean_rank_ic: float,
) -> SeedStudyResult:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "n_weeks": 26,
                "asset_count": 4,
                "top_k": 1,
                "mean_rank_ic": mean_rank_ic,
                "std_rank_ic": 0.1,
                "positive_rank_ic_fraction": 0.5,
                "mean_linear_ic": mean_rank_ic,
                "std_linear_ic": 0.1,
                "mean_top_k_spread": 0.01,
                "mean_top_k_hit_rate": 0.5,
                "mean_confidence_top_k_spread": 0.0,
                "mean_brier_score": 0.3,
                "calibration_rmse": 0.2,
                "mean_posterior_std": 0.02,
                "start_timestamp": "2025-07-04",
                "end_timestamp": "2025-12-26",
            }
        ]
    ).to_csv(seed_dir / "signal_summary.csv", index=False)
    slice_dir = seed_dir / "slice_diagnostics"
    slice_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "block": "full",
                "block_asset_count": 4,
                "n_weeks": 26,
                "top_k": 1,
                "mean_rank_ic": mean_rank_ic,
                "std_rank_ic": 0.1,
                "positive_rank_ic_fraction": 0.5,
                "mean_linear_ic": mean_rank_ic,
                "std_linear_ic": 0.1,
                "mean_top_k_spread": 0.01,
                "mean_top_k_hit_rate": 0.5,
                "mean_brier_score": 0.3,
                "calibration_rmse": 0.2,
                "mean_posterior_std": 0.02,
            }
        ]
    ).to_csv(slice_dir / "by_block.csv", index=False)
    _bucket_frame().to_csv(slice_dir / "by_confidence_bucket.csv", index=False)
    _bucket_frame().to_csv(slice_dir / "by_sign_bucket.csv", index=False)
    return SeedStudyResult(
        seed=seed,
        output_dir=seed_dir,
        log_path=seed_dir / "seed_run.log",
    )


def _bucket_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bucket_label": "[0.0, 0.1]",
                "count": 10,
                "mean_confidence": 0.05,
                "mean_predicted_positive": 0.5,
                "realized_positive_rate": 0.6,
                "mean_realized_return": 0.01,
                "mean_posterior_std": 0.02,
            }
        ]
    )
