from pathlib import Path

import pandas as pd
import torch

from algo_trader.application.research.residual_copula import (
    DEFAULT_RESEARCH_ROOT,
    CopulaFitPhase3SplitConfig,
    CopulaFitPhase3StudyConfig,
    CopulaFitSplitConfig,
    CopulaFitStudyConfig,
    ResidualStudyConfig,
    build_asset_region_series,
    build_residual_study_dataset,
    compute_tail_dependence_matrix,
    default_fit_output_dir,
    default_fit_phase3_output_dir,
    default_output_dir,
    load_selected_candidate_dataset,
    run_phase3_fitted_copula_study,
    run_selected_candidate_fit_phase3_study,
    run_fitted_copula_study,
    run_selected_candidate_fit_study,
    run_selected_candidate_study,
    run_residual_copula_study,
)


def test_build_asset_region_series_labels_us_and_europe() -> None:
    result = build_asset_region_series(["IBUS500", "IBDE40", "IBHKG50"])
    assert result.loc["IBUS500"] == "US"
    assert result.loc["IBDE40"] == "Europe"
    assert result.loc["IBHKG50"] == "Other"


def test_build_residual_study_dataset_rejects_non_index_columns() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1, -0.1], "IBUS500": [0.2, -0.2]},
        index=pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
    )
    pseudo = pd.DataFrame(
        {"EUR.USD": [0.6, 0.4], "IBUS500": [0.7, 0.3]},
        index=frame.index,
    )
    try:
        build_residual_study_dataset(frame, frame, pseudo)
    except ValueError as error:
        assert "index-only" in str(error)
    else:
        raise AssertionError("expected ValueError for non-index columns")


def test_run_residual_copula_study_writes_expected_artifacts(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    residuals = pd.DataFrame(
        {
            "IBUS500": [0.2, -0.1, 0.3, -0.2],
            "IBDE40": [0.1, -0.2, 0.2, -0.1],
            "IBEU50": [0.15, -0.1, 0.25, -0.15],
        },
        index=index,
    )
    standardized = residuals / 0.1
    pseudo = pd.DataFrame(
        {
            "IBUS500": [0.8, 0.2, 0.85, 0.15],
            "IBDE40": [0.75, 0.25, 0.8, 0.2],
            "IBEU50": [0.78, 0.22, 0.82, 0.18],
        },
        index=index,
    )
    dataset = build_residual_study_dataset(residuals, standardized, pseudo)
    outputs = run_residual_copula_study(
        dataset, ResidualStudyConfig(output_dir=tmp_path / "study")
    )
    assert outputs.dataset.residuals_path.exists()
    assert outputs.dataset.standardized_residuals_path.exists()
    assert outputs.dataset.pseudo_observations_path.exists()
    assert outputs.dataset.asset_regions_path.exists()
    assert outputs.dataset.stress_regime_path.exists()
    assert outputs.summary.rank_correlation_path.exists()
    assert outputs.summary.upper_tail_path.exists()
    assert outputs.summary.lower_tail_path.exists()
    assert outputs.summary.region_summary_path.exists()
    assert outputs.summary.conditional_links_path.exists()
    assert outputs.summary.regime_summary_path.exists()


def test_compute_tail_dependence_matrix_returns_square_dataframe() -> None:
    pseudo = pd.DataFrame(
        {"IBUS500": [0.95, 0.1, 0.92], "IBDE40": [0.93, 0.2, 0.91]},
        index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
    )
    result = compute_tail_dependence_matrix(pseudo, alpha=0.9, upper=True)
    assert list(result.index) == ["IBUS500", "IBDE40"]
    assert list(result.columns) == ["IBUS500", "IBDE40"]
    assert result.at["IBUS500", "IBUS500"] == 1.0


def test_default_output_dir_uses_data_sources_root() -> None:
    path = default_output_dir("v4_l1")
    assert path == DEFAULT_RESEARCH_ROOT / "v4_l1"


def test_load_selected_candidate_dataset_reads_simulation_payloads(
    tmp_path: Path,
) -> None:
    base_dir = _build_study_base_dir(tmp_path)
    dataset = load_selected_candidate_dataset(base_dir)
    assert list(dataset.residuals.columns) == ["IBUS500", "IBDE40"]
    assert dataset.residuals.shape == (2, 2)
    assert dataset.pseudo_observations.min().min() > 0.0
    assert dataset.pseudo_observations.max().max() < 1.0


def test_run_selected_candidate_study_writes_external_artifacts(
    tmp_path: Path,
) -> None:
    base_dir = _build_study_base_dir(tmp_path)
    outputs = run_selected_candidate_study(
        base_dir,
        study_label="unit_test_v4_l1",
    )
    assert outputs.summary.rank_correlation_path.exists()
    assert outputs.summary.rank_correlation_path.parent.name == "unit_test_v4_l1"


def test_default_fit_output_dir_uses_phase2_subdirectory() -> None:
    path = default_fit_output_dir("v4_l1")
    assert path == DEFAULT_RESEARCH_ROOT / "v4_l1" / "fit_phase2"


def test_run_fitted_copula_study_writes_expected_artifacts(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    residuals = pd.DataFrame(
        {
            "IBUS500": [0.2, -0.1, 0.3, -0.2, 0.25, -0.12, 0.18, -0.21, 0.22, -0.05],
            "IBUS30": [0.15, -0.08, 0.25, -0.18, 0.21, -0.11, 0.16, -0.2, 0.19, -0.04],
            "IBDE40": [0.1, -0.2, 0.2, -0.1, 0.12, -0.18, 0.16, -0.14, 0.18, -0.09],
            "IBEU50": [0.15, -0.1, 0.25, -0.15, 0.19, -0.13, 0.2, -0.16, 0.21, -0.1],
        },
        index=index,
    )
    standardized = residuals / 0.1
    pseudo = pd.DataFrame(
        {
            "IBUS500": [0.8, 0.2, 0.85, 0.15, 0.79, 0.21, 0.77, 0.19, 0.81, 0.23],
            "IBUS30": [0.78, 0.22, 0.82, 0.18, 0.76, 0.24, 0.75, 0.2, 0.8, 0.26],
            "IBDE40": [0.72, 0.28, 0.76, 0.24, 0.7, 0.3, 0.74, 0.26, 0.75, 0.25],
            "IBEU50": [0.74, 0.26, 0.79, 0.21, 0.73, 0.27, 0.77, 0.23, 0.78, 0.22],
        },
        index=index,
    )
    dataset = build_residual_study_dataset(residuals, standardized, pseudo)
    outputs = run_fitted_copula_study(
        dataset,
        CopulaFitStudyConfig(
            output_dir=tmp_path / "study" / "fit_phase2",
            split=CopulaFitSplitConfig(min_train_size=6, n_folds=2, min_regime_size=3),
        ),
    )
    assert outputs.model_comparison_path.exists()
    assert outputs.fold_scores_path.exists()
    assert outputs.parameter_summary_path.exists()
    assert outputs.question_summary_path.exists()


def test_run_selected_candidate_fit_study_writes_external_artifacts(
    tmp_path: Path,
) -> None:
    base_dir = _build_study_base_dir(tmp_path)
    outputs = run_selected_candidate_fit_study(
        base_dir,
        study_label="unit_test_v4_l1",
    )
    assert outputs.model_comparison_path.exists()
    assert outputs.model_comparison_path.parent.name == "fit_phase2"


def test_default_fit_phase3_output_dir_uses_phase3_subdirectory() -> None:
    path = default_fit_phase3_output_dir("v4_l1")
    assert path == DEFAULT_RESEARCH_ROOT / "v4_l1" / "fit_phase3"


def test_run_phase3_fitted_copula_study_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    dataset = _build_fit_dataset()
    outputs = run_phase3_fitted_copula_study(
        dataset,
        CopulaFitPhase3StudyConfig(
            output_dir=tmp_path / "study" / "fit_phase3",
            split=CopulaFitPhase3SplitConfig(
                min_train_size=6,
                n_folds=2,
                min_component_size=3,
                high_stress_quantile=0.7,
            ),
        ),
    )
    assert outputs.model_comparison_path.exists()
    assert outputs.fold_scores_path.exists()
    assert outputs.parameter_summary_path.exists()
    assert outputs.question_summary_path.exists()


def test_run_selected_candidate_fit_phase3_study_writes_external_artifacts(
    tmp_path: Path,
) -> None:
    base_dir = _build_study_base_dir(tmp_path)
    outputs = run_selected_candidate_fit_phase3_study(
        base_dir,
        study_label="unit_test_v4_l1",
    )
    assert outputs.model_comparison_path.exists()
    assert outputs.model_comparison_path.parent.name == "fit_phase3"


def _build_fit_dataset():
    index = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
    residuals = pd.DataFrame(
        {
            "IBUS500": [0.2, -0.1, 0.3, -0.2, 0.25, -0.12, 0.18, -0.21, 0.22, -0.05, 0.31, -0.17],
            "IBUS30": [0.15, -0.08, 0.25, -0.18, 0.21, -0.11, 0.16, -0.2, 0.19, -0.04, 0.28, -0.14],
            "IBDE40": [0.1, -0.2, 0.2, -0.1, 0.12, -0.18, 0.16, -0.14, 0.18, -0.09, 0.22, -0.16],
            "IBEU50": [0.15, -0.1, 0.25, -0.15, 0.19, -0.13, 0.2, -0.16, 0.21, -0.1, 0.26, -0.18],
        },
        index=index,
    )
    standardized = residuals / 0.1
    pseudo = pd.DataFrame(
        {
            "IBUS500": [0.8, 0.2, 0.85, 0.15, 0.79, 0.21, 0.77, 0.19, 0.81, 0.23, 0.86, 0.18],
            "IBUS30": [0.78, 0.22, 0.82, 0.18, 0.76, 0.24, 0.75, 0.2, 0.8, 0.26, 0.84, 0.21],
            "IBDE40": [0.72, 0.28, 0.76, 0.24, 0.7, 0.3, 0.74, 0.26, 0.75, 0.25, 0.79, 0.22],
            "IBEU50": [0.74, 0.26, 0.79, 0.21, 0.73, 0.27, 0.77, 0.23, 0.78, 0.22, 0.82, 0.19],
        },
        index=index,
    )
    return build_residual_study_dataset(residuals, standardized, pseudo)


def _build_study_base_dir(tmp_path: Path) -> Path:
    base_dir = tmp_path / "simulation" / "v4_l1_5y"
    inputs_dir = base_dir / "inputs"
    candidate_dir = base_dir / "inner" / "outer_17" / "postprocessing" / "candidates"
    outer_dir = base_dir / "outer"
    candidate_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    outer_dir.mkdir(parents=True)
    (outer_dir / "selection.json").write_text(
        '{"best_candidate_id": 4}',
        encoding="utf-8",
    )
    (inputs_dir / "targets.csv").write_text(
        "timestamp,IBUS500,IBDE40\n",
        encoding="utf-8",
    )
    (inputs_dir / "timestamps.csv").write_text(
        "datetime_utc\n2024-01-05T00:00:00Z\n2024-01-12T00:00:00Z\n",
        encoding="utf-8",
    )
    payload = {
        "z_true": torch.tensor([[0.1, -0.2], [0.3, -0.1]], dtype=torch.float64),
        "z_samples": torch.tensor(
            [
                [[0.0, -0.1], [0.2, 0.0]],
                [[0.1, -0.3], [0.4, -0.2]],
                [[0.2, -0.2], [0.3, -0.1]],
            ],
            dtype=torch.float64,
        ),
        "scale": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "test_idx": [0, 1],
    }
    torch.save(payload, candidate_dir / "candidate_0004_split_0000.pt")
    return base_dir
