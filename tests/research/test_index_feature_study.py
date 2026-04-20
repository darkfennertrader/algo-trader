from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_trader.application.research.index_feature_study import (
    build_feature_catalog_frame,
    run_index_feature_study,
    write_index_feature_study_outputs,
    write_index_feature_study_plots,
)


def test_index_feature_study_writes_expected_outputs(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(
        ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"],
        utc=True,
    )
    feature_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "DGS2_change_1w": [0.1, 0.2, -0.1, -0.2],
            "DGS10_change_1w": [0.1, 0.1, -0.1, -0.1],
            "DGS10_minus_DGS2_change_1w": [0.0, -0.1, 0.1, 0.0],
            "BAA10Y_change_1w": [0.2, 0.1, -0.2, -0.1],
            "BAMLH0A0HYM2_change_1w": [0.2, 0.2, -0.2, -0.2],
            "VIXCLS_change_1w": [0.3, 0.2, -0.1, -0.2],
            "DTWEXBGS_change_1w": [0.1, 0.0, -0.1, 0.0],
            "reduced_index_realized_volatility_z": [0.5, 0.4, -0.2, -0.3],
            "reduced_index_rolling_correlation_z": [0.3, 0.2, -0.1, -0.2],
            "index_vs_gold_relative_momentum_4w": [0.1, 0.2, -0.1, -0.2],
            "index_vs_silver_relative_momentum_4w": [0.0, 0.2, -0.2, 0.0],
            "index_vs_fx_relative_momentum_4w": [0.1, 0.1, -0.1, -0.1],
        }
    )
    target_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "reduced_index_block": [0.02, 0.01, -0.01, -0.02],
            "IBCH20_minus_IBDE40": [0.03, 0.02, -0.01, -0.03],
            "IBUS30_minus_IBUST100": [0.01, 0.01, -0.02, -0.01],
        }
    )

    result = run_index_feature_study(
        feature_frame=feature_frame,
        target_frame=target_frame,
    )

    assert "DGS2_change_1w" in set(result.feature_catalog["feature_key"].astype(str))
    assert "internal_cross_asset" in set(
        result.feature_group_scores["group_name"].astype(str)
    )
    assert "IBCH20_minus_IBDE40" in set(
        result.feature_target_scores["target_label"].astype(str)
    )

    write_index_feature_study_outputs(result=result, output_dir=tmp_path)
    write_index_feature_study_plots(result=result, output_dir=tmp_path)

    assert (tmp_path / "feature_catalog.csv").exists()
    assert (tmp_path / "feature_matrix_summary.csv").exists()
    assert (tmp_path / "feature_target_scores.csv").exists()
    assert (tmp_path / "feature_group_scores.csv").exists()
    assert (tmp_path / "feature_stability.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "plots" / "linear_ic_by_feature_group.png").exists()


def test_index_feature_catalog_contains_expected_sources() -> None:
    catalog = build_feature_catalog_frame()

    assert "DGS2" in set(catalog["source_id"].astype(str))
    assert "VIXCLS" in set(catalog["source_id"].astype(str))
    assert "index_vs_gold_relative_momentum" in set(
        catalog["source_id"].astype(str)
    )
