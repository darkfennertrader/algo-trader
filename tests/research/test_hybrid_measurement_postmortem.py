from __future__ import annotations

import torch

from algo_trader.application.research.hybrid_measurement_postmortem import (
    compute_global_overlap_summary,
    compute_loading_drift_summary,
    compute_variance_decomposition,
    default_output_dir,
)


def test_default_output_dir_uses_env_root() -> None:
    path = default_output_dir("demo")
    assert path.name == "demo"
    assert "hybrid_measurement_postmortem" in str(path)


def test_loading_drift_summary_flags_composite_rows() -> None:
    summary = compute_loading_drift_summary(
        anchor_loadings=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        loading_delta=torch.tensor([[0.1, 0.0], [0.0, -0.2]], dtype=torch.float64),
        asset_names=("IBUS500", "IBDE40"),
        outer_id=17,
        split_id=3,
    )
    assert summary.loc[summary["asset"] == "IBUS500", "composite_row"].item()
    assert float(summary["drift_max_abs"].max()) == 0.2


def test_variance_decomposition_returns_valid_shares() -> None:
    asset_frame, block_frame = compute_variance_decomposition(
        global_block=torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
        measurement_block=torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64),
        residual_var=torch.tensor([0.25, 0.50], dtype=torch.float64),
        asset_names=("IBUS500", "IBEU50"),
        fold_key=(18, 4),
    )
    assert set(asset_frame["asset"]) == {"IBUS500", "IBEU50"}
    assert set(block_frame["scope"]) == {"all_indices", "composite_rows"}
    assert (asset_frame["global_share"] + asset_frame["measurement_share"] + asset_frame["residual_share"]).round(8).eq(1.0).all()


def test_global_overlap_summary_reports_abs_overlap() -> None:
    summary = compute_global_overlap_summary(
        global_vector=torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64),
        measurement_block=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        outer_id=17,
        split_id=5,
    )
    assert len(summary) == 5
    assert float(summary["global_overlap_abs"].max()) <= 1.0
