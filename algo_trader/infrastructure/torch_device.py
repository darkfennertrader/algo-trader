from __future__ import annotations

import torch

from algo_trader.domain import ConfigError
from .env import require_env


def resolve_torch_device() -> torch.device:
    raw = require_env("TORCH_DEVICE")
    normalized = raw.strip()
    try:
        device = torch.device(normalized)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigError(
            "TORCH_DEVICE is not a valid torch device",
            context={"device": raw},
        ) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError(
                "TORCH_DEVICE requests CUDA but CUDA is not available",
                context={"device": str(device)},
            )
        _validate_cuda_index(device)

    return device


def move_tensor_to_device(
    data: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if data.device == device:
        return data
    return data.to(device)


def _validate_cuda_index(device: torch.device) -> None:
    index = device.index
    if index is None:
        return
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise ConfigError(
            "TORCH_DEVICE CUDA index is out of range",
            context={"device": str(device), "available": str(count)},
        )
