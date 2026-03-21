from __future__ import annotations

import torch
import pytest

from algo_trader.application.simulation import hooks
from algo_trader.domain import ConfigError


def test_allocate_equal_weight_long_only() -> None:
    weights = hooks._allocate_weights(  # pylint: disable=protected-access
        pred={},
        alloc_spec={
            "method": "equal_weight",
            "portfolio_style": "long_only",
            "n_assets": 4,
        },
    )

    assert torch.allclose(weights, torch.full((4,), 0.25))


def test_allocate_equal_weight_long_short_bounded_net() -> None:
    weights = hooks._allocate_weights(  # pylint: disable=protected-access
        pred={},
        alloc_spec={
            "method": "equal_weight",
            "portfolio_style": "long_short_bounded_net",
            "n_assets": 4,
        },
    )

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
        pred={},
        alloc_spec=alloc_spec,
    )
    w2 = hooks._allocate_weights(  # pylint: disable=protected-access
        pred={},
        alloc_spec=alloc_spec,
    )

    assert torch.allclose(w1, w2)
    assert torch.isclose(w1.sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(w1.abs().sum(), torch.tensor(1.0), atol=1e-6)


def test_allocate_risk_budgeting_uses_predictive_samples() -> None:
    samples = torch.randn((16, 1, 3), dtype=torch.float32) * 0.01

    weights = hooks._allocate_weights(  # pylint: disable=protected-access
        pred={"samples": samples},
        alloc_spec={
            "method": "skfolio_risk_budgeting",
            "portfolio_style": "long_only",
            "risk_measure": "cvar",
        },
        asset_names=("A", "B", "C"),
    )

    assert weights.shape == (3,)
    assert torch.all(weights >= 0.0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_allocate_risk_budgeting_rejects_non_long_only() -> None:
    with pytest.raises(ConfigError):
        hooks._allocate_weights(  # pylint: disable=protected-access
            pred={"samples": torch.randn((8, 1, 3), dtype=torch.float32)},
            alloc_spec={
                "method": "skfolio_risk_budgeting",
                "portfolio_style": "long_short_bounded_net",
            },
            asset_names=("A", "B", "C"),
        )


def test_allocate_factor_neutral_style_requires_dedicated_backend() -> None:
    with pytest.raises(ConfigError):
        hooks._allocate_weights(  # pylint: disable=protected-access
            pred={},
            alloc_spec={
                "method": "equal_weight",
                "portfolio_style": "factor_neutral_long_short",
                "n_assets": 4,
            },
        )


def test_allocate_de_risked_defaults_to_low_gross_long_only() -> None:
    weights = hooks._allocate_weights(  # pylint: disable=protected-access
        pred={},
        alloc_spec={
            "method": "de_risked",
            "portfolio_style": "long_only",
            "n_assets": 4,
        },
    )

    assert torch.all(weights >= 0.0)
    assert torch.isclose(weights.sum(), torch.tensor(0.10), atol=1e-6)


def test_allocate_de_risked_allows_flat_portfolio() -> None:
    weights = hooks._allocate_weights(  # pylint: disable=protected-access
        pred={},
        alloc_spec={
            "method": "de_risked",
            "portfolio_style": "long_only",
            "gross_exposure": 0.0,
            "n_assets": 3,
        },
    )

    assert torch.allclose(weights, torch.zeros(3))
