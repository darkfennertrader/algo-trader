from __future__ import annotations

from typing import Any, cast

import torch
import pytest

from algo_trader.application.simulation import hooks
from algo_trader.domain import ConfigError

from algo_trader.domain.simulation import AllocationRequest, PredictionPacket


def _request(
    *,
    alloc_spec: dict[str, Any],
    samples: torch.Tensor | None = None,
    asset_names: tuple[str, ...] = (),
) -> AllocationRequest:
    asset_count = int(
        alloc_spec.get(
            "n_assets",
            len(asset_names) or (0 if samples is None else int(samples.shape[-1])),
        )
    )
    names = asset_names or tuple(f"asset_{idx}" for idx in range(asset_count))
    prediction = PredictionPacket(
        rebalance_index=0,
        rebalance_timestamp=None,
        asset_names=names,
        tradable_mask=cast(
            torch.BoolTensor,
            torch.ones((asset_count,), dtype=torch.bool),
        ),
        mu=torch.zeros((asset_count,), dtype=torch.float32),
        covariance=torch.eye(asset_count, dtype=torch.float32),
        samples=samples,
    )
    return AllocationRequest(
        prediction=prediction,
        allocation_spec=alloc_spec,
        previous_weights=None,
    )


def test_allocate_equal_weight_long_only() -> None:
    result = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(
            alloc_spec={
            "method": "equal_weight",
            "portfolio_style": "long_only",
            "n_assets": 4,
            },
        )
    )

    weights = result.weights
    assert torch.allclose(weights, torch.full((4,), 0.25))


def test_allocate_equal_weight_long_short_bounded_net() -> None:
    result = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(
            alloc_spec={
            "method": "equal_weight",
            "portfolio_style": "long_short_bounded_net",
            "n_assets": 4,
            },
        )
    )

    weights = result.weights
    assert weights.shape == (4,)
    assert torch.isclose(weights.sum(), torch.tensor(0.0))
    assert torch.isclose(weights.abs().sum(), torch.tensor(1.0))


def test_allocate_random_is_seeded() -> None:
    alloc_spec = {
        "method": "random",
        "portfolio_style": "long_short_bounded_net",
        "n_assets": 5,
        "random_seed": 11,
    }

    w1 = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(alloc_spec=alloc_spec),
    ).weights
    w2 = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(alloc_spec=alloc_spec),
    ).weights

    assert torch.allclose(w1, w2)
    assert torch.isclose(w1.sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(w1.abs().sum(), torch.tensor(1.0), atol=1e-6)


def test_allocate_random_uses_stable_default_seed() -> None:
    alloc_spec = {
        "method": "random",
        "portfolio_style": "long_short_bounded_net",
        "n_assets": 5,
    }

    w1 = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(alloc_spec=alloc_spec),
    ).weights
    w2 = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(alloc_spec=alloc_spec),
    ).weights

    assert torch.allclose(w1, w2)
    assert torch.isclose(w1.sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(w1.abs().sum(), torch.tensor(1.0), atol=1e-6)


def test_allocate_risk_budgeting_uses_predictive_samples() -> None:
    samples = torch.randn((16, 1, 3), dtype=torch.float32) * 0.01

    result = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(
            alloc_spec={
            "method": "skfolio_risk_budgeting",
            "portfolio_style": "long_only",
            "risk_measure": "cvar",
            },
            samples=samples[:, 0, :],
            asset_names=("A", "B", "C"),
        )
    )

    weights = result.weights
    assert weights.shape == (3,)
    assert torch.all(weights >= 0.0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_allocate_herc_uses_predictive_samples() -> None:
    samples = torch.randn((24, 3), dtype=torch.float32) * 0.01

    result = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(
            alloc_spec={
                "family": "herc",
                "risk_measure": "cvar",
                "distance_estimator": "pearson",
                "min_weight": 0.0,
                "max_weight": 0.8,
            },
            samples=samples,
            asset_names=("A", "B", "C"),
        )
    )

    weights = result.weights
    assert weights.shape == (3,)
    assert torch.all(weights >= 0.0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_allocate_risk_budgeting_rejects_non_long_only() -> None:
    with pytest.raises(ConfigError):
        hooks._allocate_weights(  # pylint: disable=protected-access
            request=_request(
                alloc_spec={
                    "method": "skfolio_risk_budgeting",
                    "portfolio_style": "long_short_bounded_net",
                },
                samples=torch.randn((8, 3), dtype=torch.float32),
                asset_names=("A", "B", "C"),
            ),
        )


def test_allocate_factor_neutral_style_requires_dedicated_backend() -> None:
    with pytest.raises(ConfigError):
        hooks._allocate_weights(  # pylint: disable=protected-access
            request=_request(
                alloc_spec={
                    "method": "equal_weight",
                    "portfolio_style": "factor_neutral_long_short",
                    "n_assets": 4,
                },
            ),
        )


def test_allocate_de_risked_defaults_to_low_gross_long_only() -> None:
    result = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(
            alloc_spec={
            "method": "de_risked",
            "portfolio_style": "long_only",
            "n_assets": 4,
            },
        )
    )

    weights = result.weights
    assert torch.all(weights >= 0.0)
    assert torch.isclose(weights.sum(), torch.tensor(0.10), atol=1e-6)


def test_allocate_de_risked_allows_flat_portfolio() -> None:
    result = hooks._allocate_weights(  # pylint: disable=protected-access
        request=_request(
            alloc_spec={
            "method": "de_risked",
            "portfolio_style": "long_only",
            "gross_exposure": 0.0,
            "n_assets": 3,
            },
        )
    )

    weights = result.weights
    assert torch.allclose(weights, torch.zeros(3))


def test_allocate_long_only_respects_cap_and_sum() -> None:
    prediction = PredictionPacket(
        rebalance_index=0,
        rebalance_timestamp=None,
        asset_names=("A", "B", "C"),
        tradable_mask=cast(
            torch.BoolTensor,
            torch.tensor([True, True, True], dtype=torch.bool),
        ),
        mu=torch.tensor([0.20, 0.10, -0.05], dtype=torch.float32),
        covariance=torch.diag(torch.tensor([0.01, 0.04, 0.09])),
        samples=None,
    )
    request = AllocationRequest(
        prediction=prediction,
        allocation_spec={
            "family": "long_only",
            "min_weight": 0.0,
            "max_weight": 0.60,
        },
        previous_weights=None,
    )

    result = hooks.default_hooks().allocate(request)

    weights = result.weights
    assert torch.all(weights >= 0.0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert float(weights.max().item()) <= 0.600001
    assert float(weights[0].item()) >= float(weights[1].item())
    assert float(weights[1].item()) >= float(weights[2].item())


def test_allocate_long_only_rejects_gross_exposure() -> None:
    with pytest.raises(ConfigError):
        hooks._allocate_weights(  # pylint: disable=protected-access
            request=_request(
                alloc_spec={
                    "family": "long_only",
                    "gross_exposure": 1.0,
                    "max_weight": 0.60,
                    "n_assets": 3,
                },
            ),
        )


def test_allocate_long_only_rejects_use_previous_weights() -> None:
    with pytest.raises(ConfigError):
        hooks._allocate_weights(  # pylint: disable=protected-access
            request=_request(
                alloc_spec={
                    "family": "long_only",
                    "use_previous_weights": True,
                    "max_weight": 0.60,
                    "n_assets": 3,
                },
            ),
        )
