from __future__ import annotations

import torch

from algo_trader.application.simulation.smoke_test import (
    SMOKE_TEST_A,
    SMOKE_TEST_F,
    SMOKE_TEST_G,
    SMOKE_TEST_T,
    build_smoke_test_dataset,
)
from algo_trader.infrastructure.data import PanelTensorDataset


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


def test_build_smoke_test_dataset_clears_cuda_cache_on_cuda(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def _fake_clear_cuda_memory() -> None:
        calls.append("cleared")

    monkeypatch.setattr(
        "algo_trader.application.simulation.smoke_test.clear_cuda_memory",
        _fake_clear_cuda_memory,
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.smoke_test.build_synthetic_panel_dataset",
        lambda **_: PanelTensorDataset(
            data=torch.zeros((1, 1, 1), dtype=torch.float32),
            targets=torch.zeros((1, 1), dtype=torch.float32),
            missing_mask=torch.zeros((1, 1, 1), dtype=torch.bool),
            global_data=None,
            global_missing_mask=None,
            dates=[0],
            assets=["Asset0"],
            features=["Feature0"],
            global_features=(),
            device="cuda",
        ),
    )
    monkeypatch.setattr(
        "algo_trader.application.simulation.smoke_test._build_smoke_test_global_block",
        lambda _device: (
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.bool),
            ["GlobalFeature0"],
        ),
    )

    build_smoke_test_dataset("cuda")

    assert calls == ["cleared"]
