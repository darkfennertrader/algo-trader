import numpy as np
import torch

from algo_trader.application.simulation.preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)
from algo_trader.domain.simulation import CleaningSpec, PreprocessSpec, ScalingSpec


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
            impute_missing_to_zero=True,
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
            impute_missing_to_zero=True,
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
