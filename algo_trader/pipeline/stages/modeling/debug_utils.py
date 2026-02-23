from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Iterable, Sequence


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
    key = _callsite_key()
    if not _mark_first_seen(state, key):
        return
    text = _render_message(message, args)
    _write_lines(state, [text])


def _set_debug_state(state: _DebugState) -> None:
    global _DEBUG_STATE  # pylint: disable=global-statement
    _DEBUG_STATE = state


def _require_state(debug: bool) -> _DebugState | None:
    if not debug:
        return None
    return _DEBUG_STATE


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


def _callsite_key() -> str:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return "log:unknown"
    try:
        caller = _caller_frame(frame)
        if caller is None:
            return "log:unknown"
        code = caller.f_code
        return f"log:{code.co_filename}:{caller.f_lineno}"
    finally:
        del frame


def _caller_frame(frame: FrameType) -> FrameType | None:
    # debug_log -> _callsite_key -> caller
    return frame.f_back.f_back if frame.f_back is not None else None


def _write_lines(
    state: _DebugState, lines: Iterable[str], *, overwrite: bool = False
) -> None:
    mode = "w" if overwrite else "a"
    with state.output_path.open(mode, encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
