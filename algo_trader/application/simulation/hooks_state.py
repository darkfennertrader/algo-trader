from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, cast

import pyro
import torch

from algo_trader.domain import SimulationError
from algo_trader.pipeline.stages import modeling


def _snapshot_param_store_state() -> Mapping[str, Any]:
    return cast(Mapping[str, Any], pyro.get_param_store().get_state())


def _serialize_filtering_state(
    filtering_state: object | None,
) -> Mapping[str, Any] | None:
    if filtering_state is None:
        return None
    if isinstance(filtering_state, Mapping):
        return _coerce_filtering_state_mapping(filtering_state)
    h_loc = getattr(filtering_state, "h_loc", None)
    h_scale = getattr(filtering_state, "h_scale", None)
    steps_seen = getattr(filtering_state, "steps_seen", None)
    return _coerce_filtering_state_mapping(
        {
            "h_loc": h_loc,
            "h_scale": h_scale,
            "steps_seen": steps_seen,
        }
    )


def _coerce_filtering_state_payload(
    state: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if state is None:
        return None
    raw_state = state.get("filtering_state")
    if raw_state is None:
        return None
    if not isinstance(raw_state, Mapping):
        raise SimulationError("filtering_state must be a mapping")
    return _coerce_filtering_state_mapping(raw_state)


def _coerce_filtering_state_mapping(
    raw_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    h_loc = raw_state.get("h_loc")
    h_scale = raw_state.get("h_scale")
    if not isinstance(h_loc, torch.Tensor) or not isinstance(
        h_scale, torch.Tensor
    ):
        raise SimulationError(
            "filtering_state must include tensor h_loc and h_scale"
        )
    return {
        "h_loc": h_loc.detach(),
        "h_scale": h_scale.detach(),
        "steps_seen": _resolve_steps_seen(raw_state.get("steps_seen", 0)),
    }


def _resolve_steps_seen(value: object) -> int:
    if isinstance(value, bool):
        raise SimulationError("filtering_state.steps_seen must be an integer")
    try:
        steps_seen = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise SimulationError(
            "filtering_state.steps_seen must be an integer"
        ) from exc
    if steps_seen < 0:
        raise SimulationError(
            "filtering_state.steps_seen must be non-negative"
        )
    return steps_seen


def _export_structural_posterior_means(
    guide: modeling.PyroGuide,
) -> Mapping[str, Any] | None:
    for attr_name in (
        "structural_predictive_summaries",
        "structural_posterior_means",
    ):
        exporter = getattr(guide, attr_name, None)
        if callable(exporter):
            structural = cast(Any, exporter)()
            break
    else:
        return None
    serializer = getattr(structural, "to_mapping", None)
    if callable(serializer):
        return cast(Mapping[str, Any], serializer())
    if isinstance(structural, Mapping):
        return dict(structural)
    return None


def _with_filtering_state(
    batch: modeling.ModelBatch, filtering_state: object | None
) -> modeling.ModelBatch:
    if filtering_state is None:
        return batch
    return replace(batch, filtering_state=filtering_state)


def _build_filtering_state(
    *,
    guide: modeling.PyroGuide,
    batch: modeling.ModelBatch,
    default: object | None,
) -> object | None:
    builder = getattr(guide, "build_filtering_state", None)
    if not callable(builder):
        return default
    return cast(Any, builder)(batch)


def _restore_param_store_state(state: Mapping[str, Any]) -> None:
    raw_state = state.get("param_store_state")
    if raw_state is None:
        return
    if not isinstance(raw_state, Mapping):
        raise SimulationError("param_store_state must be a mapping")
    pyro.get_param_store().set_state(cast(Any, raw_state))


def _reset_param_store(
    init_state: Mapping[str, Any] | None,
) -> None:
    pyro.clear_param_store()
    if init_state is not None:
        _restore_param_store_state(init_state)
