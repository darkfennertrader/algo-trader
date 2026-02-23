from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


def shape_str(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    return str(tuple(value.shape))


@dataclass(frozen=True)
class DebugMetadata:
    run_timestamp: str
    model_name: str
    guide_name: str
    model_file: str | None
    guide_file: str | None


@dataclass(frozen=True)
class _DebugState:
    output_path: Path
    metadata: DebugMetadata
    seen_keys: set[str] = field(default_factory=set)


_DEBUG_STATE: _DebugState | None = None


def configure_debug_sink(
    *,
    output_dir: str,
    metadata: DebugMetadata,
) -> None:
    output_path = Path(output_dir).expanduser() / "debug.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = _DebugState(output_path=output_path, metadata=metadata)
    _set_debug_state(state)
    _write_debug_header(state)


def debug_log(debug: bool, message: str, *args: object) -> None:
    state = _require_state(debug)
    if state is None:
        return
    key = f"log:{message}"
    if not _mark_first_seen(state, key):
        return
    text = _render_message(message, args)
    _write_lines(state, [text])


def debug_log_shapes(
    debug: bool,
    values: Mapping[str, torch.Tensor | None]
    | tuple[str, torch.Tensor | None],
) -> None:
    state = _require_state(debug)
    if state is None:
        return
    payload = _normalize_shapes(values)
    key = f"shapes:{','.join(sorted(payload))}"
    if not _mark_first_seen(state, key):
        return
    _write_lines(state, _format_shape_lines(payload))


def _set_debug_state(state: _DebugState) -> None:
    global _DEBUG_STATE  # pylint: disable=global-statement
    _DEBUG_STATE = state


def _require_state(debug: bool) -> _DebugState | None:
    if not debug:
        return None
    return _DEBUG_STATE


def _normalize_shapes(
    values: Mapping[str, torch.Tensor | None]
    | tuple[str, torch.Tensor | None],
) -> dict[str, str | None]:
    if isinstance(values, tuple):
        name, value = values
        return {name: shape_str(value)}
    return {
        name: shape_str(value) for name, value in values.items()
    }


def _mark_first_seen(state: _DebugState, key: str) -> bool:
    if key in state.seen_keys:
        return False
    state.seen_keys.add(key)
    return True


def _write_debug_header(state: _DebugState) -> None:
    lines = [
        f"run_timestamp: {state.metadata.run_timestamp}",
        f"model_name: {state.metadata.model_name}",
        f"guide_name: {state.metadata.guide_name}",
        f"model_file: {state.metadata.model_file}",
        f"guide_file: {state.metadata.guide_file}",
        "",
    ]
    _write_lines(state, lines, overwrite=True)


def _render_message(message: str, args: Sequence[object]) -> str:
    if not args:
        return message
    try:
        return message % args
    except (TypeError, ValueError):
        return f"{message} {args!r}"


def _format_shape_lines(
    payload: Mapping[str, str | None]
) -> list[str]:
    return [_format_shape_line(name, shape) for name, shape in payload.items()]


def _format_shape_line(name: str, shape: str | None) -> str:
    label = name
    if label.endswith(":"):
        return f"{label} {shape}"
    return f"{label}: {shape}"


def _write_lines(
    state: _DebugState, lines: Iterable[str], *, overwrite: bool = False
) -> None:
    mode = "w" if overwrite else "a"
    with state.output_path.open(mode, encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
