from __future__ import annotations

import pandas as pd
import pytest

from algo_trader.domain import ConfigError
from algo_trader.preprocessing import PCAPreprocessor


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [4.0, 3.0, 2.0, 1.0],
        },
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            utc=True,
        ),
    )


def test_pca_requires_k_or_variance() -> None:
    preprocessor = PCAPreprocessor()
    frame = _sample_frame()

    with pytest.raises(ConfigError):
        preprocessor.process(frame, params={})


def test_pca_rejects_k_and_variance() -> None:
    preprocessor = PCAPreprocessor()
    frame = _sample_frame()

    with pytest.raises(ConfigError):
        preprocessor.process(frame, params={"k": "1", "variance": "0.9"})


def test_pca_returns_factors_and_artifacts() -> None:
    preprocessor = PCAPreprocessor()
    frame = _sample_frame()

    processed = preprocessor.process(frame, params={"variance": "0.9"})
    result = preprocessor.result()

    assert processed.shape[0] == len(frame)
    assert list(processed.columns) == ["factor_1"]
    assert result.selected_k == 1
    assert result.variance_target == 0.9
    assert result.loadings.shape == (2, 1)
    assert list(result.loadings.columns) == ["factor_1"]
    assert len(result.eigenvalues) == len(frame.columns)
    assert list(result.eigenvalues.columns) == [
        "eigenvalue",
        "explained_variance",
        "cumulative_variance",
    ]
