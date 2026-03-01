from __future__ import annotations

from typing import Any, Mapping


def resolve_dof_shift(config: Mapping[str, Any], *, default: float = 2.0) -> float:
    model = config.get("model")
    if not isinstance(model, Mapping):
        return default
    params = model.get("params")
    if not isinstance(params, Mapping):
        return default
    dof = params.get("dof")
    if not isinstance(dof, Mapping):
        return default
    try:
        return float(dof.get("shift", default))
    except (TypeError, ValueError):
        return default
