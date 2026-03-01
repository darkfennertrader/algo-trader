import numpy as np
import pytest
import torch

from algo_trader.domain import SimulationError
from algo_trader.application.simulation.posterior_scale_diagnostics import (
    _ratio_in_space,
)
from algo_trader.application.simulation.target_space_transform import (
    TargetSpaceTransform,
)


def test_target_space_transform_roundtrip_value_spaces() -> None:
    transform = TargetSpaceTransform(
        model_center=torch.tensor(0.1, dtype=torch.float64),
        model_scale=torch.tensor(2.0, dtype=torch.float64),
        mad_scale=torch.tensor([0.5, 2.0], dtype=torch.float64),
    )
    values = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    model = transform.forward_y_to_model(values)
    roundtrip_y = transform.inverse_model_to_y(model)
    z_values = transform.forward_y_to_z(values)
    roundtrip_y_from_z = transform.inverse_z_to_y(z_values)
    assert torch.allclose(values, roundtrip_y)
    assert torch.allclose(values, roundtrip_y_from_z)


def test_target_space_transform_convert_dispersion() -> None:
    transform = TargetSpaceTransform(
        model_center=torch.tensor(0.0, dtype=torch.float64),
        model_scale=torch.tensor(2.0, dtype=torch.float64),
        mad_scale=torch.tensor([0.5, 2.0], dtype=torch.float64),
    )
    residual_y = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    residual_model = transform.convert_dispersion(
        residual_y, source="y", target="model"
    )
    residual_z = transform.convert_dispersion(
        residual_y, source="y", target="z"
    )
    assert torch.allclose(
        residual_model, torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    )
    assert torch.allclose(
        residual_z, torch.tensor([[2.0, 0.5]], dtype=torch.float64)
    )


def test_ratio_in_space_rejects_mixed_spaces() -> None:
    with pytest.raises(SimulationError, match="Space mismatch"):
        _ratio_in_space(
            numerator=1.0,
            denominator=2.0,
            numerator_space="y",
            denominator_space="z",
            space="y",
        )


def test_ratio_in_space_returns_nan_for_zero_denominator() -> None:
    result = _ratio_in_space(
        numerator=1.0,
        denominator=0.0,
        numerator_space="y",
        denominator_space="y",
        space="y",
    )
    assert np.isnan(result)
