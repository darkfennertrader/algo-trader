from __future__ import annotations

import torch

from algo_trader.application.simulation.prediction_handoff import (
    build_prediction_packet,
)
from algo_trader.domain.simulation import AllocationRequest


def test_build_prediction_packet_uses_first_horizon_step() -> None:
    mean = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    covariance = torch.tensor(
        [[[1.0, 0.1], [0.1, 2.0]]],
        dtype=torch.float32,
    )
    samples = torch.tensor(
        [
            [[0.1, 0.2]],
            [[0.3, 0.4]],
        ],
        dtype=torch.float32,
    )

    packet = build_prediction_packet(
        pred={
            "mean": mean,
            "covariance": covariance,
            "samples": samples,
        },
        asset_names=("A", "B"),
        rebalance_index=7,
        rebalance_timestamp="2026-01-02",
    )

    assert packet.rebalance_index == 7
    assert packet.rebalance_timestamp == "2026-01-02"
    assert packet.asset_names == ("A", "B")
    assert torch.equal(packet.mu, mean[0])
    assert torch.equal(packet.covariance, covariance[0])
    assert packet.samples is not None
    assert torch.equal(packet.samples, samples[:, 0, :])


def test_prediction_packet_supports_allocation_request() -> None:
    previous = torch.tensor([0.2, 0.8], dtype=torch.float32)
    prediction = build_prediction_packet(
        pred={"mean": torch.tensor([0.1, 0.2], dtype=torch.float32)},
        asset_names=("A", "B"),
        rebalance_index=3,
        rebalance_timestamp=None,
    )

    request = AllocationRequest(
        prediction=prediction,
        allocation_spec={"method": "equal_weight"},
        previous_weights=previous,
    )

    assert request.previous_weights is previous
    assert request.allocation_spec["method"] == "equal_weight"
    assert request.prediction.rebalance_index == 3
