from __future__ import annotations

from dataclasses import dataclass

import torch

from algo_trader.application.simulation import runner


@dataclass(frozen=True)
class _NestedPayload:
    value: torch.Tensor


def test_summarize_cuda_refs_ignores_cpu_tensors() -> None:
    summary = runner._summarize_cuda_refs(  # pylint: disable=protected-access
        {
            "cpu_tensor": torch.zeros((2, 3), dtype=torch.float32),
            "nested": _NestedPayload(value=torch.ones((1,), dtype=torch.float32)),
        }
    )

    assert summary.count == 0
    assert summary.total_bytes == 0
    assert summary.examples == ()
