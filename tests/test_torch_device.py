from __future__ import annotations

import torch
import pytest

from algo_trader.domain import ConfigError, EnvVarError
from algo_trader.infrastructure import resolve_torch_device


def test_resolve_torch_device_requires_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCH_DEVICE", raising=False)

    with pytest.raises(EnvVarError):
        resolve_torch_device()


def test_resolve_torch_device_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_DEVICE", "cpu")

    device = resolve_torch_device()

    assert device.type == "cpu"


def test_resolve_torch_device_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_DEVICE", "cuda")

    if torch.cuda.is_available():
        device = resolve_torch_device()
        assert device.type == "cuda"
    else:
        with pytest.raises(ConfigError):
            resolve_torch_device()
