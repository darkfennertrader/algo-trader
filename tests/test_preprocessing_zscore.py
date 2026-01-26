from __future__ import annotations

import pandas as pd

from algo_trader.preprocessing import ZScorePreprocessor


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [1.0, None, 5.0],
        },
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
        ),
    )


def test_zscore_filters_dates_and_fills_missing_by_default() -> None:
    preprocessor = ZScorePreprocessor()
    frame = _sample_frame()

    processed = preprocessor.process(
        frame,
        params={"start_date": "2024-01-02", "end_date": "2024-01-03"},
    )

    assert processed.index.min() == pd.Timestamp("2024-01-02", tz="UTC")
    assert processed.index.max() == pd.Timestamp("2024-01-03", tz="UTC")
    assert not processed.isna().any().any()
    assert (processed.mean().abs() < 1e-9).all()
    assert ((processed.std(ddof=0) - 1.0).abs() < 1e-9).all()


def test_zscore_drops_missing_rows_when_requested() -> None:
    preprocessor = ZScorePreprocessor()
    frame = _sample_frame()

    processed = preprocessor.process(frame, params={"missing": "drop"})

    assert len(processed) == 2
    assert not processed.isna().any().any()
