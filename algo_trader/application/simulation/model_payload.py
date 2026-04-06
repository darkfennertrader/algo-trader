from __future__ import annotations

from dataclasses import asdict
from typing import Any

from algo_trader.domain.simulation import ModelConfig


def model_to_payload(
    model: ModelConfig, *, include_prebuild: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": model.model_name,
        "guide_name": model.guide_name,
        "predict_name": model.predict_name,
        "params": dict(model.params),
        "guide_params": dict(model.guide_params),
        "predict_params": dict(model.predict_params),
    }
    if include_prebuild and model.prebuild is not None:
        payload["prebuild"] = asdict(model.prebuild)
    return payload
