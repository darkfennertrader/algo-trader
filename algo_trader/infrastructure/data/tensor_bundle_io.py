from __future__ import annotations

from pathlib import Path

import torch

from algo_trader.domain import DataSourceError


def load_tensor_bundle(path: Path, *, error_message: str) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise DataSourceError(
            error_message,
            context={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise DataSourceError(
            "Tensor bundle must be a mapping",
            context={"path": str(path)},
        )
    return payload


def require_tensor(value: object, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DataSourceError(
            f"Missing '{label}' tensor",
            context={"label": label},
        )
    return value
