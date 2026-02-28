from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
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

_BOOTSTRAP_SCORE_SENTINEL = -1e30
_EXPERIMENT_ANALYSIS_METRICS_WARNING = "Failed to fetch metrics for"


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
    runtime: "RayTuneRuntimeSpec"


@dataclass(frozen=True)
class RayTuneRuntimeSpec:
    storage_path: Path
    experiment_name: str
    address: str | None
    resume_experiment_dir: Path | None
    logs_enabled: bool


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
    _configure_ray_tune_restore_warnings(
        logs_enabled=context.spec.runtime.logs_enabled
    )
    _, tune = _import_tune()
    trainable = _build_trainable(context, tune)
    if context.spec.runtime.resume_experiment_dir is not None:
        tuner = _restore_tuner(
            trainable=trainable,
            candidates=context.spec.candidates,
            tune=tune,
            experiment_dir=context.spec.runtime.resume_experiment_dir,
        )
    else:
        tuner = _build_tuner(
            trainable=trainable,
            candidates=context.spec.candidates,
            tune=tune,
            runtime=context.spec.runtime,
        )
    best_candidate = _extract_best_candidate(tuner.fit())
    return apply_param_updates(context.spec.base_config, best_candidate)


def init_ray_for_tuning(
    address: str | None, *, logs_enabled: bool
) -> None:
    _configure_ray_tune_restore_warnings(logs_enabled=logs_enabled)
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


def resolve_ray_tune_storage_path() -> Path:
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


def _maybe_report_bootstrap_checkpoint(
    *,
    tune: Any,
    state: TrialCheckpointState | None,
    candidate_id: int,
    fingerprint: str,
) -> None:
    if state is not None:
        return
    bootstrap_state = TrialCheckpointState(
        candidate_id=candidate_id,
        candidate_fingerprint=fingerprint,
        last_completed_split_id=-1,
        split_scores=tuple(),
    )
    # TensorBoardX logs noisy warnings when metrics are NaN/Inf.
    # Use a finite sentinel while keeping bootstrap trial ranking effectively minimal.
    _write_checkpoint_state(
        tune, bootstrap_state, _BOOTSTRAP_SCORE_SENTINEL
    )


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
        _maybe_report_bootstrap_checkpoint(
            tune=tune,
            state=checkpoint_state,
            candidate_id=candidate_id,
            fingerprint=fingerprint,
        )

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
    *,
    trainable,
    candidates: Sequence[CandidateSpec],
    tune: Any,
    runtime: RayTuneRuntimeSpec,
):
    return tune.Tuner(
        trainable,
        param_space=_build_param_space(candidates, tune),
        tune_config=tune.TuneConfig(metric="score", mode="max"),
        run_config=tune.RunConfig(
            name=runtime.experiment_name,
            storage_path=str(runtime.storage_path),
            verbose=1 if runtime.logs_enabled else 0,
        ),
    )


def _restore_tuner(
    *,
    trainable,
    candidates: Sequence[CandidateSpec],
    tune: Any,
    experiment_dir: Path,
):
    if not tune.Tuner.can_restore(str(experiment_dir)):
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


def _configure_ray_tune_restore_warnings(*, logs_enabled: bool) -> None:
    logger = logging.getLogger("ray.tune.analysis.experiment_analysis")
    _ensure_experiment_analysis_filter(logger)
    level = logging.WARNING if logs_enabled else logging.ERROR
    logger.setLevel(level)


class _ExperimentAnalysisFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return _EXPERIMENT_ANALYSIS_METRICS_WARNING not in message


def _ensure_experiment_analysis_filter(logger: logging.Logger) -> None:
    for existing_filter in logger.filters:
        if isinstance(existing_filter, _ExperimentAnalysisFilter):
            return
    logger.addFilter(_ExperimentAnalysisFilter())
