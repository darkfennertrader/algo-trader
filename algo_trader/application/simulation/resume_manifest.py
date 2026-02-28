from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from algo_trader.domain import ConfigError, SimulationError

_MANIFEST_VERSION = 1
_MANIFEST_STATUS_RUNNING = "running"
_MANIFEST_STATUS_COMPLETED = "completed"
_MANIFEST_MODE_PER_OUTER = "per_outer"
_MANIFEST_MODE_GLOBAL = "global"


@dataclass
class ResumeManifestState:
    version: int
    run_id: str
    status: str
    mode: str
    outer_ids: tuple[int, ...]
    progress: "ResumeProgressState"
    updated_at: str


@dataclass
class ResumeProgressState:
    completed_inner_outer_ids: set[int]
    completed_outer_outer_ids: set[int]
    active_outer_k: int | None
    active_phase: str | None


class SimulationResumeTracker:
    def __init__(
        self,
        *,
        base_dir: Path,
        outer_ids: Sequence[int],
        model_selection_enabled: bool,
        resume_requested: bool,
    ) -> None:
        self._path = _manifest_path(base_dir)
        self._outer_ids = tuple(int(value) for value in outer_ids)
        self._mode = (
            _MANIFEST_MODE_GLOBAL
            if model_selection_enabled
            else _MANIFEST_MODE_PER_OUTER
        )
        if resume_requested:
            self._state = self._load_for_resume()
        else:
            self._state = self._build_new_state()
            self._write_state()

    @property
    def run_id(self) -> str:
        return self._state.run_id

    def experiment_name_for_outer(self, outer_k: int) -> str:
        return f"algotrader_{self._state.run_id}_outer_{int(outer_k)}"

    def is_inner_completed(self, outer_k: int) -> bool:
        return int(outer_k) in self._state.progress.completed_inner_outer_ids

    def is_outer_completed(self, outer_k: int) -> bool:
        return int(outer_k) in self._state.progress.completed_outer_outer_ids

    def mark_inner_started(self, outer_k: int) -> None:
        self._state.progress.active_outer_k = int(outer_k)
        self._state.progress.active_phase = "inner"
        self._write_state()

    def mark_inner_completed(self, outer_k: int) -> None:
        self._state.progress.completed_inner_outer_ids.add(int(outer_k))
        self._state.progress.active_outer_k = None
        self._state.progress.active_phase = None
        self._write_state()

    def mark_outer_started(self, outer_k: int) -> None:
        self._state.progress.active_outer_k = int(outer_k)
        self._state.progress.active_phase = "outer"
        self._write_state()

    def mark_outer_completed(self, outer_k: int) -> None:
        outer_id = int(outer_k)
        self._state.progress.completed_outer_outer_ids.add(outer_id)
        self._state.progress.completed_inner_outer_ids.add(outer_id)
        self._state.progress.active_outer_k = None
        self._state.progress.active_phase = None
        self._write_state()

    def mark_run_completed(self) -> None:
        self._state.status = _MANIFEST_STATUS_COMPLETED
        self._state.progress.active_outer_k = None
        self._state.progress.active_phase = None
        self._state.progress.completed_outer_outer_ids.update(self._outer_ids)
        self._state.progress.completed_inner_outer_ids.update(self._outer_ids)
        self._write_state()

    def _build_new_state(self) -> ResumeManifestState:
        return ResumeManifestState(
            version=_MANIFEST_VERSION,
            run_id=_new_run_id(),
            status=_MANIFEST_STATUS_RUNNING,
            mode=self._mode,
            outer_ids=self._outer_ids,
            progress=ResumeProgressState(
                completed_inner_outer_ids=set(),
                completed_outer_outer_ids=set(),
                active_outer_k=None,
                active_phase=None,
            ),
            updated_at=_utc_now(),
        )

    def _load_for_resume(self) -> ResumeManifestState:
        payload = _read_manifest_payload(self._path)
        state = _parse_manifest_payload(payload)
        if state.status != _MANIFEST_STATUS_RUNNING:
            raise ConfigError("No interrupted Ray Tune experiment to resume")
        if state.mode != self._mode:
            raise ConfigError(
                "Resume manifest does not match model_selection mode"
            )
        if state.outer_ids != self._outer_ids:
            raise ConfigError(
                "Resume manifest does not match current outer folds"
            )
        if state.progress.completed_outer_outer_ids.issuperset(self._outer_ids):
            raise ConfigError("No interrupted Ray Tune experiment to resume")
        return state

    def _write_state(self) -> None:
        payload = {
            "version": int(self._state.version),
            "run_id": self._state.run_id,
            "status": self._state.status,
            "mode": self._state.mode,
            "outer_ids": [int(value) for value in self._state.outer_ids],
            "completed_inner_outer_ids": sorted(
                int(value)
                for value in self._state.progress.completed_inner_outer_ids
            ),
            "completed_outer_outer_ids": sorted(
                int(value)
                for value in self._state.progress.completed_outer_outer_ids
            ),
            "active_outer_k": self._state.progress.active_outer_k,
            "active_phase": self._state.progress.active_phase,
            "updated_at": _utc_now(),
        }
        self._state.updated_at = payload["updated_at"]
        _write_manifest_payload(self._path, payload)


def _manifest_path(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "resume" / "ray_tune_resume_manifest.json"


def _new_run_id() -> str:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{stamp}_{suffix}"


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _read_manifest_payload(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigError("No interrupted Ray Tune experiment to resume")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError("Simulation resume manifest is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("Simulation resume manifest must be a mapping")
    return payload


def _parse_manifest_payload(payload: Mapping[str, Any]) -> ResumeManifestState:
    version = _require_int(payload, "version")
    if version != _MANIFEST_VERSION:
        raise ConfigError("Simulation resume manifest version is not supported")
    run_id = _require_string(payload, "run_id")
    status = _require_string(payload, "status")
    mode = _require_string(payload, "mode")
    outer_ids = tuple(_require_int_list(payload, "outer_ids"))
    completed_inner = set(
        _require_int_list(payload, "completed_inner_outer_ids")
    )
    completed_outer = set(
        _require_int_list(payload, "completed_outer_outer_ids")
    )
    active_outer = payload.get("active_outer_k")
    if active_outer is not None:
        try:
            active_outer = int(active_outer)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "Simulation resume manifest field active_outer_k must be an int"
            ) from exc
    active_phase = payload.get("active_phase")
    if active_phase is not None and not isinstance(active_phase, str):
        raise ConfigError(
            "Simulation resume manifest field active_phase must be a string"
        )
    updated_at = _require_string(payload, "updated_at")
    return ResumeManifestState(
        version=version,
        run_id=run_id,
        status=status,
        mode=mode,
        outer_ids=outer_ids,
        progress=ResumeProgressState(
            completed_inner_outer_ids=completed_inner,
            completed_outer_outer_ids=completed_outer,
            active_outer_k=active_outer,
            active_phase=active_phase,
        ),
        updated_at=updated_at,
    )


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"Simulation resume manifest field {key} must be a string"
        )
    return value


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ConfigError(
            f"Simulation resume manifest field {key} must be an int"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Simulation resume manifest field {key} must be an int"
        ) from exc


def _require_int_list(payload: Mapping[str, Any], key: str) -> list[int]:
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        raise ConfigError(
            f"Simulation resume manifest field {key} must be a list"
        )
    parsed: list[int] = []
    for item in raw:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Simulation resume manifest field {key} must contain ints"
            ) from exc
    return parsed


def _write_manifest_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        raise SimulationError(
            "Failed to write simulation resume manifest",
            context={"path": str(path)},
        ) from exc
