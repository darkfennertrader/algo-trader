from __future__ import annotations
# pylint: disable=duplicate-code

from pathlib import Path

import numpy as np
import pandas as pd

from algo_trader.application.research.index_representation.coordinates import (
    build_coordinate_specs,
)
from algo_trader.application.research.index_representation.basket_gaussian_test import (
    _BasketModelInputs,
    _basis_matrix,
    _build_basket_specs,
    _fit_coordinate_scale_factors,
    _summarize_baskets,
    _transformed_moments,
)
from algo_trader.application.research.index_representation.basket_student_t_test import (
    _fit_coordinate_student_t_params,
    _summarize_student_t_baskets,
)
from algo_trader.application.research.index_representation.diagnostics import (
    run_index_representation_study,
)
from algo_trader.application.research.index_representation.targeted_student_t_test import (
    _summarize_targeted_student_t_baskets,
    _targeted_correlation_matrix,
)
from algo_trader.application.research.index_representation.types import (
    IndexPosteriorDataset,
    RepresentationStudyConfig,
)


def test_build_coordinate_specs_creates_expected_names() -> None:
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBUS30",
        "IBUS500",
        "IBUST100",
    )
    specs = build_coordinate_specs(asset_names)
    assert [item.name for item in specs] == [
        "global_level",
        "us_minus_europe",
        "us_internal_style",
        "euro_core_vs_uk_ch",
        "spain_vs_euro_core",
    ]


def test_run_index_representation_study_returns_coordinate_and_whitening_outputs(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2024-01-05", periods=4, freq="W-FRI", tz="UTC")
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBUS30",
        "IBUS500",
        "IBUST100",
    )
    truth = pd.DataFrame(
        np.arange(40, dtype=float).reshape(4, 10) / 1000.0,
        index=timestamps,
        columns=asset_names,
    )
    samples = tuple(
        np.vstack([truth.iloc[row].to_numpy(dtype=float) + offset for offset in (-0.01, 0.0, 0.01)])
        for row in range(len(truth))
    )
    dataset = IndexPosteriorDataset(
        timestamps=timestamps,
        asset_names=asset_names,
        truth=truth,
        samples=samples,
    )
    result = run_index_representation_study(
        dataset=dataset,
        coordinate_specs=build_coordinate_specs(asset_names),
        config=RepresentationStudyConfig(output_dir=tmp_path),
    )
    assert list(result.coordinate_diagnostics.truth.columns) == [
        "global_level",
        "us_minus_europe",
        "us_internal_style",
        "euro_core_vs_uk_ch",
        "spain_vs_euro_core",
    ]
    assert not result.coordinate_diagnostics.summary.empty
    assert not result.whitening.summary.empty


def test_transformed_gaussian_basket_summary_returns_expected_baskets() -> None:
    timestamps = pd.date_range("2024-01-05", periods=6, freq="W-FRI", tz="UTC")
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBUS30",
        "IBUS500",
        "IBUST100",
    )
    truth = pd.DataFrame(
        np.arange(60, dtype=float).reshape(6, 10) / 1000.0,
        index=timestamps,
        columns=asset_names,
    )
    samples = tuple(
        np.vstack(
            [
                truth.iloc[row].to_numpy(dtype=float) + offset
                for offset in (-0.02, -0.01, 0.0, 0.01, 0.02)
            ]
        )
        for row in range(len(truth))
    )
    dataset = IndexPosteriorDataset(
        timestamps=timestamps,
        asset_names=asset_names,
        truth=truth,
        samples=samples,
    )
    specs = build_coordinate_specs(asset_names)
    basis = _basis_matrix(specs)
    means, stds = _transformed_moments(dataset, basis)
    scale_factors = _fit_coordinate_scale_factors(
        dataset=dataset,
        means=means,
        stds=stds,
        basis=basis,
    )
    summary = _summarize_baskets(
        basket_specs=_build_basket_specs(asset_names),
        model_inputs=_BasketModelInputs(
            dataset=dataset,
            basis=basis,
            means=means,
            stds=stds,
            scale_factors=scale_factors["scale_factor"].to_numpy(dtype=float),
            ridge=1.0e-8,
        ),
    )
    assert list(summary["basket"]) == [
        "index_equal_weight",
        "us_index",
        "europe_index",
        "us_minus_europe",
    ]


def test_transformed_student_t_basket_summary_returns_expected_baskets() -> None:
    timestamps = pd.date_range("2024-01-05", periods=6, freq="W-FRI", tz="UTC")
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBUS30",
        "IBUS500",
        "IBUST100",
    )
    truth = pd.DataFrame(
        np.arange(60, dtype=float).reshape(6, 10) / 1000.0,
        index=timestamps,
        columns=asset_names,
    )
    samples = tuple(
        np.vstack(
            [
                truth.iloc[row].to_numpy(dtype=float) + offset
                for offset in (-0.02, -0.01, 0.0, 0.01, 0.02)
            ]
        )
        for row in range(len(truth))
    )
    dataset = IndexPosteriorDataset(
        timestamps=timestamps,
        asset_names=asset_names,
        truth=truth,
        samples=samples,
    )
    specs = build_coordinate_specs(asset_names)
    basis = _basis_matrix(specs)
    means, stds = _transformed_moments(dataset, basis)
    fitted = _fit_coordinate_student_t_params(
        coordinate_names=[spec.name for spec in specs],
        truth=dataset.truth.to_numpy(dtype=float) @ basis,
        means=means,
        stds=stds,
    )
    summary = _summarize_student_t_baskets(
        basket_specs=_build_basket_specs(asset_names),
        model_inputs=_build_model_inputs(dataset, basis, means, stds, fitted),
        dfs=fitted["df"].to_numpy(dtype=float),
        rng_seed=7,
    )
    assert list(summary["basket"]) == [
        "index_equal_weight",
        "us_index",
        "europe_index",
        "us_minus_europe",
    ]


def test_targeted_student_t_basket_summary_returns_expected_baskets() -> None:
    timestamps = pd.date_range("2024-01-05", periods=6, freq="W-FRI", tz="UTC")
    asset_names = (
        "IBCH20",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBUS30",
        "IBUS500",
        "IBUST100",
    )
    truth = pd.DataFrame(
        np.arange(60, dtype=float).reshape(6, 10) / 1000.0,
        index=timestamps,
        columns=asset_names,
    )
    samples = tuple(
        np.vstack(
            [
                truth.iloc[row].to_numpy(dtype=float) + offset
                for offset in (-0.02, -0.01, 0.0, 0.01, 0.02)
            ]
        )
        for row in range(len(truth))
    )
    dataset = IndexPosteriorDataset(
        timestamps=timestamps,
        asset_names=asset_names,
        truth=truth,
        samples=samples,
    )
    specs = build_coordinate_specs(asset_names)
    basis = _basis_matrix(specs)
    means, stds = _transformed_moments(dataset, basis)
    fitted = _fit_coordinate_student_t_params(
        coordinate_names=[spec.name for spec in specs],
        truth=dataset.truth.to_numpy(dtype=float) @ basis,
        means=means,
        stds=stds,
    )
    corr = _targeted_correlation_matrix(
        coordinate_names=[spec.name for spec in specs],
        baseline_corr=np.asarray(
            [
                [1.0, 0.0, 0.25, 0.0, 0.0],
                [0.0, 1.0, 0.40, 0.0, 0.0],
                [0.25, 0.40, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -0.30],
                [0.0, 0.0, 0.0, -0.30, 1.0],
            ],
            dtype=float,
        ),
    )
    summary = _summarize_targeted_student_t_baskets(
        basket_specs=_build_basket_specs(asset_names),
        model_inputs=_build_model_inputs(dataset, basis, means, stds, fitted),
        dfs=fitted["df"].to_numpy(dtype=float),
        target_corr=corr,
        rng_seed=7,
    )
    assert list(summary["basket"]) == [
        "index_equal_weight",
        "us_index",
        "europe_index",
        "us_minus_europe",
    ]


def _build_model_inputs(
    dataset: IndexPosteriorDataset,
    basis: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    fitted: pd.DataFrame,
) -> _BasketModelInputs:
    return _BasketModelInputs(
        dataset=dataset,
        basis=basis,
        means=means,
        stds=stds,
        scale_factors=fitted["scale_factor"].to_numpy(dtype=float),
        ridge=1.0e-8,
    )
