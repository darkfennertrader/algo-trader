from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Literal

import torch

from algo_trader.domain import SimulationError

TargetSpace = Literal["y", "z", "model"]


@dataclass(frozen=True)
class TargetSpaceTransform:
    model_center: torch.Tensor
    model_scale: torch.Tensor
    mad_scale: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor(self.model_center, "model_center")
        _validate_tensor(self.model_scale, "model_scale")
        _validate_tensor(self.mad_scale, "mad_scale")
        if self.mad_scale.ndim != 1:
            raise SimulationError("Target transform mad_scale must be 1D")
        if not torch.isfinite(self.model_scale).all():
            raise SimulationError("Target transform model_scale must be finite")
        if not torch.isfinite(self.mad_scale).all():
            raise SimulationError("Target transform mad_scale must be finite")
        if bool((self.model_scale <= 0).any()):
            raise SimulationError("Target transform model_scale must be positive")
        if bool((self.mad_scale <= 0).any()):
            raise SimulationError("Target transform mad_scale must be positive")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        asset_count: int | None = None,
        device: torch.device | None = None,
    ) -> "TargetSpaceTransform":
        model_center = _require_payload_tensor(payload, "model_center")
        model_scale = _require_payload_tensor(payload, "model_scale")
        mad_scale = _require_payload_tensor(payload, "mad_scale")
        if device is not None:
            model_center = model_center.to(device)
            model_scale = model_scale.to(device)
            mad_scale = mad_scale.to(device)
        if asset_count is not None and int(mad_scale.shape[0]) != int(asset_count):
            raise SimulationError(
                "Target transform mad_scale has wrong asset count",
                context={
                    "expected": str(asset_count),
                    "actual": str(int(mad_scale.shape[0])),
                },
            )
        return cls(
            model_center=model_center,
            model_scale=model_scale,
            mad_scale=mad_scale,
        )

    @classmethod
    def identity(cls, *, asset_count: int) -> "TargetSpaceTransform":
        return cls(
            model_center=torch.tensor(0.0),
            model_scale=torch.tensor(1.0),
            mad_scale=torch.ones(asset_count),
        )

    def to_payload(self) -> Mapping[str, torch.Tensor]:
        return {
            "model_center": self.model_center.detach().cpu(),
            "model_scale": self.model_scale.detach().cpu(),
            "mad_scale": self.mad_scale.detach().cpu(),
        }

    def forward_y_to_model(self, values: torch.Tensor) -> torch.Tensor:
        center = _expand_to_values(self.model_center, values, "model_center")
        scale = _expand_to_values(self.model_scale, values, "model_scale")
        return (values - center) / scale

    def inverse_model_to_y(self, values: torch.Tensor) -> torch.Tensor:
        center = _expand_to_values(self.model_center, values, "model_center")
        scale = _expand_to_values(self.model_scale, values, "model_scale")
        return values * scale + center

    def forward_y_to_z(self, values: torch.Tensor) -> torch.Tensor:
        scale = _expand_asset_scale(self.mad_scale, values, "mad_scale")
        return values / scale

    def inverse_z_to_y(self, values: torch.Tensor) -> torch.Tensor:
        scale = _expand_asset_scale(self.mad_scale, values, "mad_scale")
        return values * scale

    def convert(
        self, values: torch.Tensor, *, source: TargetSpace, target: TargetSpace
    ) -> torch.Tensor:
        result = values
        if source == target:
            return result
        if source == "y" and target == "z":
            result = self.forward_y_to_z(values)
        elif source == "z" and target == "y":
            result = self.inverse_z_to_y(values)
        elif source == "y" and target == "model":
            result = self.forward_y_to_model(values)
        elif source == "model" and target == "y":
            result = self.inverse_model_to_y(values)
        elif source == "z" and target == "model":
            result = self.forward_y_to_model(self.inverse_z_to_y(values))
        elif source == "model" and target == "z":
            result = self.forward_y_to_z(self.inverse_model_to_y(values))
        else:
            raise SimulationError(
                "Unknown target-space conversion",
                context={"source": source, "target": target},
            )
        return result

    def convert_dispersion(
        self, values: torch.Tensor, *, source: TargetSpace, target: TargetSpace
    ) -> torch.Tensor:
        if source == target:
            return values
        y_values = values
        if source == "z":
            y_values = values * _expand_asset_scale(self.mad_scale, values, "mad_scale")
        elif source == "model":
            y_values = values * _expand_to_values(
                self.model_scale, values, "model_scale"
            )
        if target == "y":
            return y_values
        if target == "z":
            return y_values / _expand_asset_scale(self.mad_scale, y_values, "mad_scale")
        if target == "model":
            return y_values / _expand_to_values(
                self.model_scale, y_values, "model_scale"
            )
        raise SimulationError(
            "Unknown target-space conversion",
            context={"source": source, "target": target},
        )


def _validate_tensor(values: Any, field: str) -> None:
    if not isinstance(values, torch.Tensor):
        raise SimulationError(
            "Target transform field must be a tensor",
            context={"field": field},
        )


def _require_payload_tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError(
            "Target transform payload missing tensor",
            context={"key": key},
        )
    return value


def _expand_asset_scale(
    scale: torch.Tensor, values: torch.Tensor, label: str
) -> torch.Tensor:
    if values.ndim == 0:
        raise SimulationError(
            "Target transform values must have at least 1 dimension",
            context={"label": label},
        )
    if int(values.shape[-1]) != int(scale.shape[0]):
        raise SimulationError(
            "Target transform asset dimension mismatch",
            context={
                "label": label,
                "values_asset_dim": str(int(values.shape[-1])),
                "scale_asset_dim": str(int(scale.shape[0])),
            },
        )
    view_shape = (1,) * (values.ndim - 1) + (int(scale.shape[0]),)
    return scale.to(device=values.device, dtype=values.dtype).view(view_shape)


def _expand_to_values(
    param: torch.Tensor, values: torch.Tensor, label: str
) -> torch.Tensor:
    if param.ndim == 0:
        return param.to(device=values.device, dtype=values.dtype)
    if param.ndim == 1:
        return _expand_asset_scale(param, values, label)
    raise SimulationError(
        "Target transform field must be scalar or 1D",
        context={"label": label, "ndim": str(int(param.ndim))},
    )
