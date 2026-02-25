from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import CandidateSpec, TuningResourcesConfig
from algo_trader.infrastructure.env import require_env

from .hooks import SimulationHooks
from .inner_objective import (
    InnerObjectiveContext,
    aggregate_scores,
    evaluate_inner_splits,
    SplitEvaluationRequest,
    SplitEvaluationResume,
)
from .tuning import apply_param_updates, with_candidate_context


@dataclass(frozen=True)
class RayTuneInputs:
    objective: Any
    inner_context: InnerObjectiveContext
    hooks: SimulationHooks


@dataclass(frozen=True)
class RayTuneSpec:
    base_config: Mapping[str, Any]
    candidates: Sequence[CandidateSpec]
    resources: TuningResourcesConfig
    use_gpu: bool
    address: str | None
    resume: bool


@dataclass(frozen=True)
class RayTuneContext:
    inputs: RayTuneInputs
    spec: RayTuneSpec


@dataclass(frozen=True)
class TrialCheckpointState:
    candidate_id: int
    candidate_fingerprint: str
    last_completed_split_id: int
    split_scores: tuple[float, ...]


def select_best_with_ray(context: RayTuneContext) -> Mapping[str, Any]:
    if not context.spec.candidates:
        raise ConfigError("No candidates available for Ray Tune")
    storage_path = _resolve_storage_path()
    _, tune = _import_tune()
    trainable = _build_trainable(context, tune)
    if context.spec.resume:
        tuner = _restore_latest_tuner(
            trainable=trainable,
            candidates=context.spec.candidates,
            tune=tune,
            storage_path=storage_path,
        )
    else:
        tuner = _build_tuner(
            trainable, context.spec.candidates, tune, storage_path
        )
    best_candidate = _extract_best_candidate(tuner.fit())
    return apply_param_updates(context.spec.base_config, best_candidate)


def init_ray_for_tuning(address: str | None) -> None:
    ray, _ = _import_tune()
    _init_ray(address, ray)


def shutdown_ray_for_tuning() -> None:
    ray, _ = _import_tune()
    if ray.is_initialized():
        ray.shutdown()


def _import_tune():
    try:
        import ray  # pylint: disable=import-outside-toplevel
        from ray import tune  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ConfigError(
            "ray[tune] is required when tuning.engine=ray; install with `uv add \"ray[tune]\"`"
        ) from exc
    return ray, tune


def _init_ray(address: str | None, ray: Any) -> None:
    if ray.is_initialized():
        return
    if address:
        ray.init(
            address=address,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
    else:
        ray.init(ignore_reinit_error=True, include_dashboard=False)


def _resolve_storage_path() -> Path:
    raw = require_env("RAY_TUNE_STORAGE_PATH")
    path = Path(raw).expanduser()
    if path.exists() and not path.is_dir():
        raise ConfigError("RAY_TUNE_STORAGE_PATH must be a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _aggregate_for_context(
    scores: Sequence[float], context: InnerObjectiveContext
) -> float:
    return aggregate_scores(
        scores,
        method=context.params.aggregate,
        penalty=context.params.aggregate_lambda,
    )


def _candidate_fingerprint(params: Mapping[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_start_split_id(
    state: TrialCheckpointState | None,
) -> int:
    if state is None:
        return 0
    return int(state.last_completed_split_id) + 1


def _resolve_prior_scores(
    state: TrialCheckpointState | None,
) -> list[float]:
    if state is None:
        return []
    return [float(value) for value in state.split_scores]


def _load_checkpoint_state(
    checkpoint: Any | None,
    candidate_id: int,
    fingerprint: str,
) -> TrialCheckpointState | None:
    if checkpoint is None:
        return None
    try:
        context = checkpoint.as_directory()
    except AttributeError:
        return None
    with context as path_str:
        state_path = Path(path_str) / "checkpoint.json"
        if not state_path.exists():
            return None
        data = _load_checkpoint_payload(state_path)
    stored_id = _require_int_field(data, "candidate_id")
    stored_fingerprint = _require_str_field(
        data, "candidate_fingerprint"
    )
    if stored_id != candidate_id:
        raise ConfigError("Ray Tune checkpoint candidate_id mismatch")
    if stored_fingerprint != fingerprint:
        raise ConfigError("Ray Tune checkpoint fingerprint mismatch")
    last_completed = _require_int_field(
        data, "last_completed_split_id"
    )
    split_scores = _require_scores_field(data, "split_scores")
    return TrialCheckpointState(
        candidate_id=stored_id,
        candidate_fingerprint=stored_fingerprint,
        last_completed_split_id=last_completed,
        split_scores=tuple(split_scores),
    )


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError("Ray Tune checkpoint payload is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("Ray Tune checkpoint payload must be a mapping")
    return payload


def _require_int_field(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ConfigError(f"Ray Tune checkpoint missing {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Ray Tune checkpoint field {key} must be an int"
        ) from exc


def _require_str_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"Ray Tune checkpoint field {key} must be a string"
        )
    return value


def _require_scores_field(
    payload: Mapping[str, Any], key: str
) -> list[float]:
    raw = payload.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(
            f"Ray Tune checkpoint field {key} must be a list"
        )
    scores: list[float] = []
    for value in raw:
        try:
            scores.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Ray Tune checkpoint field {key} must be numeric"
            ) from exc
    return scores


def _write_checkpoint_state(
    tune: Any, state: TrialCheckpointState, score: float
) -> None:
    payload = {
        "candidate_id": int(state.candidate_id),
        "candidate_fingerprint": state.candidate_fingerprint,
        "last_completed_split_id": int(state.last_completed_split_id),
        "split_scores": [float(value) for value in state.split_scores],
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "checkpoint.json"
        path.write_text(json.dumps(payload))
        checkpoint = tune.Checkpoint.from_directory(temp_dir)
        tune.report({"score": float(score)}, checkpoint=checkpoint)


def _build_trainable(context: RayTuneContext, tune: Any):
    def trainable(config: Mapping[str, Any]) -> None:
        candidate = _require_candidate_payload(config)
        candidate_id = int(candidate["candidate_id"])
        candidate_params = candidate["params"]
        merged = apply_param_updates(
            context.spec.base_config, candidate_params
        )
        merged = with_candidate_context(merged, candidate_id)
        fingerprint = _candidate_fingerprint(candidate_params)
        checkpoint_state = _load_checkpoint_state(
            tune.get_checkpoint(), candidate_id, fingerprint
        )
        start_split_id = _resolve_start_split_id(checkpoint_state)
        prior_scores = _resolve_prior_scores(checkpoint_state)

        def on_split(split_id: int, scores: list[float]) -> None:
            aggregate = _aggregate_for_context(
                scores, context.inputs.inner_context
            )
            state = TrialCheckpointState(
                candidate_id=candidate_id,
                candidate_fingerprint=fingerprint,
                last_completed_split_id=split_id,
                split_scores=tuple(scores),
            )
            _write_checkpoint_state(tune, state, aggregate)

        request = SplitEvaluationRequest(
            context=context.inputs.inner_context,
            hooks=context.inputs.hooks,
            config=merged,
            candidate_id=candidate_id,
        )
        resume = SplitEvaluationResume(
            start_split_id=start_split_id,
            prior_scores=prior_scores,
            on_split=on_split,
        )
        fold_scores = evaluate_inner_splits(request, resume=resume)
        score = _aggregate_for_context(
            fold_scores, context.inputs.inner_context
        )
        tune.report({"score": score})

    return _with_resources(
        trainable=trainable,
        resources=context.spec.resources,
        use_gpu=context.spec.use_gpu,
        tune=tune,
    )


def _build_tuner(
    trainable,
    candidates: Sequence[CandidateSpec],
    tune: Any,
    storage_path: Path,
):
    return tune.Tuner(
        trainable,
        param_space=_build_param_space(candidates, tune),
        tune_config=tune.TuneConfig(metric="score", mode="max"),
        run_config=tune.RunConfig(storage_path=str(storage_path)),
    )


def _restore_latest_tuner(
    *,
    trainable,
    candidates: Sequence[CandidateSpec],
    tune: Any,
    storage_path: Path,
):
    experiment_dir = _latest_experiment_dir(storage_path)
    if experiment_dir is None:
        raise ConfigError("No Ray Tune experiment to resume")
    if not _is_interrupted_experiment(experiment_dir):
        raise ConfigError("No interrupted Ray Tune experiment to resume")
    return tune.Tuner.restore(
        str(experiment_dir),
        trainable=trainable,
        param_space=_build_param_space(candidates, tune),
        resume_unfinished=True,
        resume_errored=True,
    )


def _build_param_space(
    candidates: Sequence[CandidateSpec], tune: Any
) -> Mapping[str, Any]:
    return {
        "candidate": tune.grid_search(_candidate_payloads(candidates))
    }


def _latest_experiment_dir(storage_path: Path) -> Path | None:
    if not storage_path.exists():
        return None
    candidates = [path for path in storage_path.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _is_interrupted_experiment(experiment_dir: Path) -> bool:
    state = _load_experiment_state(experiment_dir)
    if state is None:
        return False
    statuses = _extract_trial_statuses(state)
    if not statuses:
        return False
    terminal = {"TERMINATED", "COMPLETED"}
    return any(status not in terminal for status in statuses)


def _load_experiment_state(
    experiment_dir: Path,
) -> Mapping[str, Any] | None:
    state_path = experiment_dir / "experiment_state.json"
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError("Ray Tune experiment_state.json is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("Ray Tune experiment_state.json must be a mapping")
    return payload


def _extract_trial_statuses(
    payload: Mapping[str, Any],
) -> list[str]:
    statuses: list[str] = []
    for key in ("trials", "trial_data", "trial_states"):
        statuses.extend(_extract_statuses(payload.get(key)))
    return [status.upper() for status in statuses if status]


def _extract_statuses(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        items = value.values()
    elif isinstance(value, list):
        items = value
    else:
        return []
    statuses: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        status = item.get("status") or item.get("trial_state") or item.get(
            "state"
        )
        if isinstance(status, str):
            statuses.append(status)
    return statuses


def _extract_best_candidate(results) -> Mapping[str, Any]:
    best_result = results.get_best_result(metric="score", mode="max")
    if best_result is None:
        raise ConfigError("Ray Tune did not return a best result")
    config = best_result.config
    if not isinstance(config, Mapping):
        raise ConfigError("Ray Tune best config is missing candidate data")
    best_candidate = config.get("candidate")
    if not isinstance(best_candidate, Mapping):
        raise ConfigError("Ray Tune best config is missing candidate data")
    params = best_candidate.get("params")
    if not isinstance(params, Mapping):
        raise ConfigError("Ray Tune best candidate missing params")
    return params


def _candidate_payloads(
    candidates: Sequence[CandidateSpec],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        payloads.append(
            {
                "candidate_id": int(candidate.candidate_id),
                "params": dict(candidate.params),
            }
        )
    return payloads


def _require_candidate_payload(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ConfigError("Ray Tune candidate payload is invalid")
    if "candidate_id" not in candidate or "params" not in candidate:
        raise ConfigError("Ray Tune candidate payload is missing fields")
    return candidate


def _with_resources(
    *,
    trainable,
    resources: TuningResourcesConfig,
    use_gpu: bool,
    tune: Any,
):
    resource_spec: dict[str, float] = {}
    if resources.cpu is not None:
        resource_spec["cpu"] = float(resources.cpu)
    if use_gpu and resources.gpu is not None and resources.gpu > 0:
        resource_spec["gpu"] = float(resources.gpu)
    if not resource_spec:
        return trainable
    return tune.with_resources(trainable, resource_spec)
