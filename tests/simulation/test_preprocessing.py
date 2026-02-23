import numpy as np
import torch

from algo_trader.application.simulation.preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)
from algo_trader.domain.simulation import (
    CleaningSpec,
    GuardrailSpec,
    PreprocessSpec,
    ScalingInputSpec,
    ScalingSpec,
    WinsorSpec,
)


def test_feature_cleaning_drops_low_usable() -> None:
    X = torch.tensor(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
        ]
    )
    M = torch.tensor(
        [
            [[False, True]],
            [[False, False]],
        ]
    )
    spec = PreprocessSpec(
        cleaning=CleaningSpec(
            min_usable_ratio=1.0,
            min_variance=0.0,
            max_abs_corr=0.99,
            corr_subsample=None,
        ),
        scaling=ScalingSpec(
            mad_eps=1e-12,
            inputs=ScalingInputSpec(impute_missing_to_zero=True),
        ),
    )
    cleaning = fit_feature_cleaning(
        X=X, M=M, train_idx=np.array([0, 1]), spec=spec
    )
    assert cleaning.feature_idx.tolist() == [0]
    assert cleaning.dropped_low_usable.tolist() == [1]


def test_transform_imputes_missing_and_scales() -> None:
    X = torch.tensor(
        [
            [[1.0, 10.0]],
            [[3.0, 14.0]],
        ]
    )
    M = torch.tensor(
        [
            [[False, False]],
            [[False, True]],
        ]
    )
    spec = PreprocessSpec(
        cleaning=CleaningSpec(
            min_usable_ratio=0.0,
            min_variance=0.0,
            max_abs_corr=0.99,
            corr_subsample=None,
        ),
        scaling=ScalingSpec(
            mad_eps=1e-12,
            inputs=ScalingInputSpec(impute_missing_to_zero=True),
        ),
    )
    cleaning = fit_feature_cleaning(
        X=X, M=M, train_idx=np.array([0, 1]), spec=spec
    )
    scaler = fit_robust_scaler(
        X=X, M=M, train_idx=np.array([0, 1]), cleaning=cleaning, spec=spec
    )
    transformed = transform_X(
        X=X,
        M=M,
        idx=np.array([0, 1]),
        state=TransformState(cleaning=cleaning, scaler=scaler, spec=spec),
    )
    assert transformed.shape == (2, 1, 2)
    assert transformed[1, 0, 1].item() == 0.0


def test_breakout_features_center_and_variance_scale() -> None:
    X = torch.tensor(
        [
            [[0.0, 10.0], [1.0, 12.0]],
            [[1.0, 14.0], [1.0, 16.0]],
        ],
        dtype=torch.float64,
    )
    M = torch.zeros_like(X, dtype=torch.bool)
    spec = PreprocessSpec(
        cleaning=CleaningSpec(
            min_usable_ratio=0.0,
            min_variance=0.0,
            max_abs_corr=0.99,
            corr_subsample=None,
        ),
        scaling=ScalingSpec(
            mad_eps=1e-12,
            breakout_var_floor=1e-3,
            inputs=ScalingInputSpec(
                impute_missing_to_zero=True,
                feature_names=["breakout::brk_up_4w", "momentum::z_mom_4w"],
            ),
        ),
    )
    cleaning = fit_feature_cleaning(
        X=X, M=M, train_idx=np.array([0, 1]), spec=spec
    )
    scaler = fit_robust_scaler(
        X=X, M=M, train_idx=np.array([0, 1]), cleaning=cleaning, spec=spec
    )
    transformed = transform_X(
        X=X,
        M=M,
        idx=np.array([0, 1]),
        state=TransformState(cleaning=cleaning, scaler=scaler, spec=spec),
    )
    denom = torch.sqrt(torch.tensor(0.25 + 1e-3, dtype=torch.float64))
    expected = torch.tensor([-0.5, 0.5], dtype=torch.float64) / denom
    assert torch.allclose(transformed[:, 0, 0], expected, atol=1e-8)
    assert torch.allclose(
        transformed[:, 1, 0], torch.zeros(2, dtype=torch.float64)
    )


def test_near_constant_guardrail_zeroes_feature() -> None:
    X = torch.tensor(
        [
            [[1.0, 2.0]],
            [[1.0, 4.0]],
            [[1.0, 6.0]],
        ],
        dtype=torch.float64,
    )
    M = torch.zeros_like(X, dtype=torch.bool)
    spec = PreprocessSpec(
        cleaning=CleaningSpec(
            min_usable_ratio=0.0,
            min_variance=0.0,
            max_abs_corr=0.99,
            corr_subsample=None,
        ),
        scaling=ScalingSpec(
            mad_eps=1e-12,
            inputs=ScalingInputSpec(impute_missing_to_zero=True),
            guardrail=GuardrailSpec(
                abs_eps=1.0e-6,
                rel_eps=1.0e-3,
                rel_offset=1.0e-8,
            ),
        ),
    )
    cleaning = fit_feature_cleaning(
        X=X, M=M, train_idx=np.array([0, 1, 2]), spec=spec
    )
    scaler = fit_robust_scaler(
        X=X, M=M, train_idx=np.array([0, 1, 2]), cleaning=cleaning, spec=spec
    )
    transformed = transform_X(
        X=X,
        M=M,
        idx=np.array([0, 1, 2]),
        state=TransformState(cleaning=cleaning, scaler=scaler, spec=spec),
    )
    assert torch.allclose(transformed[:, 0, 0], torch.zeros(3, dtype=torch.float64))


def test_winsorization_clips_before_scaling() -> None:
    X = torch.tensor(
        [
            [[0.0]],
            [[100.0]],
            [[0.0]],
            [[0.0]],
        ],
        dtype=torch.float64,
    )
    M = torch.zeros_like(X, dtype=torch.bool)
    spec = PreprocessSpec(
        cleaning=CleaningSpec(
            min_usable_ratio=0.0,
            min_variance=0.0,
            max_abs_corr=0.99,
            corr_subsample=None,
        ),
        scaling=ScalingSpec(
            mad_eps=1e-12,
            inputs=ScalingInputSpec(
                impute_missing_to_zero=True,
                feature_names=["momentum::z_mom_4w"],
            ),
            winsor=WinsorSpec(lower_q=0.5, upper_q=0.5),
        ),
    )
    cleaning = fit_feature_cleaning(
        X=X, M=M, train_idx=np.array([0, 1, 2, 3]), spec=spec
    )
    scaler = fit_robust_scaler(
        X=X,
        M=M,
        train_idx=np.array([0, 1, 2, 3]),
        cleaning=cleaning,
        spec=spec,
    )
    transformed = transform_X(
        X=X,
        M=M,
        idx=np.array([0, 1, 2, 3]),
        state=TransformState(cleaning=cleaning, scaler=scaler, spec=spec),
    )
    assert torch.allclose(
        transformed[:, 0, 0], torch.zeros(4, dtype=torch.float64)
    )
