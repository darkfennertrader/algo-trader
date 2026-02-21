from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import CandidateSpec, TuningResourcesConfig

from .tuning import apply_param_updates, with_candidate_context


@dataclass(frozen=True)
class RayTuneContext:
    objective: Any
    base_config: Mapping[str, Any]
    candidates: Sequence[CandidateSpec]
    resources: TuningResourcesConfig
    use_gpu: bool
    address: str | None


def select_best_with_ray(context: RayTuneContext) -> Mapping[str, Any]:
    if not context.candidates:
        raise ConfigError("No candidates available for Ray Tune")
    ray, tune = _import_tune()
    _init_ray(context.address, ray)
    trainable = _build_trainable(context, tune)
    tuner = _build_tuner(trainable, context.candidates, tune)
    best_candidate = _extract_best_candidate(tuner.fit())
    merged = apply_param_updates(context.base_config, best_candidate)
    return merged


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


def _build_trainable(context: RayTuneContext, tune: Any):
    def trainable(config: Mapping[str, Any]) -> None:
        candidate = _require_candidate_payload(config)
        merged = apply_param_updates(context.base_config, candidate["params"])
        merged = with_candidate_context(
            merged, int(candidate["candidate_id"])
        )
        score = float(context.objective(merged))
        tune.report({"score": score})

    return _with_resources(
        trainable=trainable,
        resources=context.resources,
        use_gpu=context.use_gpu,
        tune=tune,
    )


def _build_tuner(
    trainable, candidates: Sequence[CandidateSpec], tune: Any
):
    return tune.Tuner(
        trainable,
        param_space={
            "candidate": tune.grid_search(_candidate_payloads(candidates))
        },
        tune_config=tune.TuneConfig(metric="score", mode="max"),
    )


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
