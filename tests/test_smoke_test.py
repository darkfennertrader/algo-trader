from __future__ import annotations

from algo_trader.application.simulation.smoke_test import (
    SMOKE_TEST_A,
    SMOKE_TEST_F,
    SMOKE_TEST_G,
    SMOKE_TEST_T,
    build_smoke_test_dataset,
)


def test_build_smoke_test_dataset_includes_global_block() -> None:
    dataset = build_smoke_test_dataset("cpu")

    assert tuple(dataset.data.shape) == (SMOKE_TEST_T, SMOKE_TEST_A, SMOKE_TEST_F)
    assert dataset.global_data is not None
    assert tuple(dataset.global_data.shape) == (SMOKE_TEST_T, SMOKE_TEST_G)
    assert dataset.global_missing_mask is not None
    assert tuple(dataset.global_missing_mask.shape) == (
        SMOKE_TEST_T,
        SMOKE_TEST_G,
    )
    assert list(dataset.global_features) == [
        "GlobalFeature0",
        "GlobalFeature1",
    ]
