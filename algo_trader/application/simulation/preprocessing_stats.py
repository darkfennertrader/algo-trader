from __future__ import annotations

import torch


def _nanmedian(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nanmedian(x, dim=dim).values
