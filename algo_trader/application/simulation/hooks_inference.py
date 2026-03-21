from __future__ import annotations
# pylint: disable=too-many-lines

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import pyro
import torch
from pyro import poutine
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer import util as infer_util

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.pipeline.stages import modeling

from .config_utils import require_bool
from .hooks_batching import (
    _DebugParticipants,
    _apply_target_denorm,
    _build_online_filtering_batches,
    _build_prediction_batch,
    _build_tbptt_batches,
    _configure_debug_sink,
    _debug_config,
    _prepare_training_batches,
)
from .hooks_state import (
    _build_filtering_state,
    _coerce_filtering_state_payload,
    _export_structural_posterior_means,
    _reset_param_store,
    _restore_param_store_state,
    _serialize_filtering_state,
    _snapshot_param_store_state,
    _with_filtering_state,
)
from .hooks_types import (
    _FitInputs,
    _OnlineFilteringParams,
    _SVIParams,
    _TBPTTParams,
    _TargetNormState,
    _TrainingArtifacts,
    _TrainingParams,
)
from .metrics import build_metric_scorer
from .model_params import resolve_dof_shift

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = modeling.default_model_registry()
_GUIDE_REGISTRY = modeling.default_guide_registry()
_PREDICTOR_REGISTRY = modeling.default_predictor_registry()


@dataclass(frozen=True)
class _ResolvedModelConfig:
    model_name: str
    guide_name: str
    predict_name: str | None
    model_params: Mapping[str, Any]
    guide_params: Mapping[str, Any]
    predict_params: Mapping[str, Any]


@dataclass(frozen=True)
class _ResolvedModeling:
    model_name: str
    guide_name: str
    predict_name: str | None
    model: modeling.PyroModel
    guide: modeling.PyroGuide
    predictor: modeling.PyroPredictor | None


def _log_missing_targets(y_train: torch.Tensor) -> None:
    if torch.isfinite(y_train).all():
        return
    missing = int((~torch.isfinite(y_train)).sum().item())
    total = int(y_train.numel())
    logger.warning(
        "Training targets contain missing values; missing=%s total=%s",
        missing,
        total,
    )


def _resolve_model_and_guide(
    config: Mapping[str, Any],
) -> _ResolvedModeling:
    config_values = _resolve_model_config(config)
    training_method = _config_training_method(config)
    model = _MODEL_REGISTRY.get(
        config_values.model_name,
        config_values.model_params,
    )
    _validate_training_method_support(
        model_name=config_values.model_name,
        model=model,
        training_method=training_method,
    )
    guide = _GUIDE_REGISTRY.get(
        config_values.guide_name,
        config_values.guide_params,
    )
    predictor = None
    if config_values.predict_name is not None:
        predictor = _PREDICTOR_REGISTRY.get(
            config_values.predict_name,
            config_values.predict_params,
        )
    return _ResolvedModeling(
        model_name=config_values.model_name,
        guide_name=config_values.guide_name,
        predict_name=config_values.predict_name,
        model=model,
        guide=guide,
        predictor=predictor,
    )


def _config_training_method(config: Mapping[str, Any]) -> str:
    training = config.get("training")
    if not isinstance(training, Mapping):
        return "tbptt"
    return _resolve_training_method(training.get("method", "tbptt"))


def _validate_training_method_support(
    *,
    model_name: str,
    model: modeling.PyroModel,
    training_method: str,
) -> None:
    supported_methods = getattr(model, "supported_training_methods", None)
    if not callable(supported_methods):
        return
    raw_supported = cast(Iterable[object], supported_methods())
    supported = tuple(str(value) for value in raw_supported)
    if training_method in supported:
        return
    raise ConfigError(
        f"{model_name} does not support training.method={training_method}",
        context={"supported_methods": ", ".join(sorted(supported))},
    )


def _fit_pyro(
    X_train: torch.Tensor,
    X_train_global: torch.Tensor | None,
    y_train: torch.Tensor,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    params = _training_params(config)
    inputs = _FitInputs(
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=y_train,
    )
    if params.method == "online_filtering":
        return _fit_pyro_online_filtering(
            inputs=inputs,
            config=config,
            init_state=init_state,
            params=params,
        )
    return _fit_pyro_svi(inputs=inputs, config=config, params=params)


def _fit_pyro_svi(
    inputs: _FitInputs,
    config: Mapping[str, Any],
    *,
    params: _TrainingParams | None = None,
) -> Mapping[str, Any]:
    _log_missing_targets(inputs.y_train)
    resolved = _resolve_model_and_guide(config)
    debug_config = _debug_config(config)
    _configure_debug_sink(
        debug_config=debug_config,
        participants=_DebugParticipants(
            model_name=resolved.model_name,
            guide_name=resolved.guide_name,
            predictor_name=resolved.predict_name,
            model=resolved.model,
            guide=resolved.guide,
            predictor=resolved.predictor,
        ),
    )
    params = params or _training_params(config)
    batches, norm_state = _prepare_training_batches(
        inputs.X_train,
        inputs.X_train_global,
        inputs.y_train,
        params,
        debug_config.enabled,
    )
    pyro.clear_param_store()
    svi_loss_history = _run_svi_steps(
        svi=_build_svi(model=resolved.model, guide=resolved.guide, params=params),
        batches=batches,
        params=params,
        context=_run_context(config),
    )
    return _build_training_state(
        resolved.model_name,
        resolved.guide_name,
        _TrainingArtifacts(
            norm_state=norm_state,
            posterior_summary=_summarize_posterior_params(config),
            training_diagnostics=_build_training_diagnostics(
                svi_loss_history=svi_loss_history,
                params=params,
                num_batches=len(batches),
            ),
        ),
    )


def _fit_pyro_online_filtering(
    inputs: _FitInputs,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None,
    *,
    params: _TrainingParams,
) -> Mapping[str, Any]:
    _log_missing_targets(inputs.y_train)
    resolved = _resolve_model_and_guide(config)
    debug_config = _debug_config(config)
    _configure_debug_sink(
        debug_config=debug_config,
        participants=_DebugParticipants(
            model_name=resolved.model_name,
            guide_name=resolved.guide_name,
            predictor_name=resolved.predict_name,
            model=resolved.model,
            guide=resolved.guide,
            predictor=resolved.predictor,
        ),
    )
    batches, norm_state = _prepare_training_batches(
        inputs.X_train,
        inputs.X_train_global,
        inputs.y_train,
        params,
        debug_config.enabled,
    )
    _reset_param_store(init_state)
    filtering_state = _coerce_filtering_state_payload(init_state)
    svi_loss_history, final_filtering_state = _run_online_filtering_steps(
        svi=_build_svi(
            model=resolved.model,
            guide=resolved.guide,
            params=params,
        ),
        batches=batches,
        params=params,
        context=_run_context(config),
        initial_filtering_state=filtering_state,
    )
    structural_posterior_means = _export_structural_posterior_means(
        resolved.guide
    )
    return _build_training_state(
        resolved.model_name,
        resolved.guide_name,
        _TrainingArtifacts(
            norm_state=norm_state,
            posterior_summary=_summarize_posterior_params(config),
            training_diagnostics=_build_training_diagnostics(
                svi_loss_history=svi_loss_history,
                params=params,
                num_batches=len(batches),
            ),
            filtering_state=_serialize_filtering_state(final_filtering_state),
            structural_posterior_means=structural_posterior_means,
            param_store_state=_snapshot_param_store_state(),
        ),
    )


def _build_training_state(
    model_name: str,
    guide_name: str,
    artifacts: _TrainingArtifacts,
) -> Mapping[str, Any]:
    state: dict[str, Any] = {
        "model_name": model_name,
        "guide_name": guide_name,
    }
    if artifacts.norm_state is not None:
        state["target_norm"] = {
            "center": artifacts.norm_state.center,
            "scale": artifacts.norm_state.scale,
        }
    if artifacts.posterior_summary:
        state["posterior_summary"] = dict(artifacts.posterior_summary)
    if artifacts.training_diagnostics:
        state["training_diagnostics"] = dict(
            artifacts.training_diagnostics
        )
    if artifacts.filtering_state is not None:
        state["filtering_state"] = dict(artifacts.filtering_state)
    if artifacts.structural_posterior_means is not None:
        state["structural_posterior_means"] = dict(
            artifacts.structural_posterior_means
        )
    if artifacts.param_store_state is not None:
        state["param_store_state"] = dict(artifacts.param_store_state)
    return state


def _build_training_diagnostics(
    *,
    svi_loss_history: Sequence[float],
    params: _TrainingParams,
    num_batches: int,
) -> Mapping[str, Any]:
    if not svi_loss_history:
        return {}
    diagnostics: dict[str, Any] = {
        "method": params.method,
        "svi_loss_history": [float(value) for value in svi_loss_history],
        "grad_accum_steps": int(params.svi.grad_accum_steps),
        "num_batches": int(num_batches),
    }
    if params.method == "online_filtering":
        diagnostics["steps_per_observation"] = int(
            params.online_filtering.steps_per_observation
        )
        diagnostics["num_updates"] = int(len(svi_loss_history))
        return diagnostics
    diagnostics["num_steps"] = int(params.svi.steps)
    return diagnostics


def _predict_pyro(
    X_pred: torch.Tensor,
    X_pred_global: torch.Tensor | None,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    num_samples: int,
) -> Mapping[str, Any]:
    resolved = _resolve_model_and_guide(config)
    num_samples = max(int(num_samples), 1)
    _restore_param_store_state(state)
    batch = _build_prediction_batch(
        X_pred,
        X_pred_global,
        filtering_state=_coerce_filtering_state_payload(state),
    )
    request = modeling.PredictiveRequest(
        model=resolved.model,
        guide=resolved.guide,
        batch=batch,
        num_samples=num_samples,
        state=state,
    )
    pred = _predict_model_specific(
        request=request,
        predictor=resolved.predictor,
    )
    if pred is None:
        pred = _predict_with_pyro_predictive(
            model=request.model,
            guide=request.guide,
            batch=request.batch,
            num_samples=request.num_samples,
        )
    samples = _require_prediction_tensor(pred, key="samples")
    mean = _require_prediction_tensor(pred, key="mean")
    mean, samples = _apply_target_denorm(mean, samples, state)
    result = dict(pred)
    result["samples"] = samples
    result["mean"] = mean
    if "covariance" in result:
        result["covariance"] = modeling.predictive_covariance(samples)
    return result


def _score_metrics(
    y_true: torch.Tensor,
    pred: Mapping[str, Any],
    score_spec: Mapping[str, Any],
) -> float:
    scorer = build_metric_scorer(score_spec, scope="inner")
    return float(scorer(y_true, pred, score_spec))


def _predict_model_specific(
    *,
    request: modeling.PredictiveRequest,
    predictor: modeling.PyroPredictor | None = None,
) -> Mapping[str, Any] | None:
    if predictor is not None:
        return cast(Mapping[str, Any] | None, predictor(request))
    model_predictor = getattr(request.model, "posterior_predict", None)
    if not callable(model_predictor):
        return None
    return cast(
        Mapping[str, Any] | None,
        model_predictor(
            guide=request.guide,
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        ),
    )


def _predict_with_pyro_predictive(
    *,
    model: modeling.PyroModel,
    guide: modeling.PyroGuide,
    batch: modeling.ModelBatch,
    num_samples: int,
) -> Mapping[str, Any]:
    unconditioned = poutine.uncondition(model)
    predictive = Predictive(
        unconditioned,
        guide=guide,
        num_samples=num_samples,
        return_sites=("obs",),
    )
    samples = predictive(batch)["obs"]
    return {"samples": samples, "mean": samples.mean(dim=0)}


def _require_prediction_tensor(
    pred: Mapping[str, Any], *, key: str
) -> torch.Tensor:
    value = pred.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError(f"Prediction output must include tensor {key!r}")
    return value


def _resolve_model_config(
    config: Mapping[str, Any],
) -> _ResolvedModelConfig:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("model config missing for simulation hooks")
    model_name = str(model.get("model_name", "")).strip().lower()
    guide_name = str(model.get("guide_name", "")).strip().lower()
    if not model_name:
        raise ConfigError("model.model_name must be set for simulation hooks")
    if not guide_name:
        raise ConfigError("model.guide_name must be set for simulation hooks")
    return _ResolvedModelConfig(
        model_name=model_name,
        guide_name=guide_name,
        predict_name=_optional_name(model.get("predict_name")),
        model_params=_coerce_mapping(model.get("params", {}), label="model.params"),
        guide_params=_coerce_mapping(
            model.get("guide_params", {}),
            label="model.guide_params",
        ),
        predict_params=_coerce_mapping(
            model.get("predict_params", {}),
            label="model.predict_params",
        ),
    )


def _coerce_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping for simulation hooks")
    return dict(value)


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _training_params(config: Mapping[str, Any]) -> _TrainingParams:
    training = _require_mapping(
        config.get("training"),
        message="training config missing for simulation hooks",
    )
    method = _resolve_training_method(training.get("method", "tbptt"))
    raw_svi_shared = _require_mapping(
        training.get("svi_shared"),
        message="training.svi_shared config missing for simulation hooks",
    )
    raw_tbptt = _require_mapping(
        training.get("tbptt"),
        message="training.tbptt config missing for simulation hooks",
    )
    raw_online_filtering = _optional_mapping(
        training.get("online_filtering"),
        message="training.online_filtering config missing for simulation hooks",
    )
    target_norm = require_bool(
        training.get("target_normalization"),
        field="training.target_normalization",
        config_path=None,
    )
    if method == "online_filtering" and target_norm:
        raise ConfigError(
            "training.target_normalization must be false when "
            "training.method=online_filtering"
        )
    grad_accum_steps = _resolve_grad_accum_steps(
        raw_svi_shared=raw_svi_shared,
        method=method,
    )
    return _TrainingParams(
        method=method,
        svi=_SVIParams(
            steps=int(raw_tbptt.get("num_steps", 0)),
            learning_rate=float(raw_svi_shared.get("learning_rate", 1e-3)),
            num_elbo_particles=int(
                raw_svi_shared.get("num_elbo_particles", 1)
            ),
            log_every=_resolve_log_every(raw_svi_shared),
            grad_accum_steps=grad_accum_steps,
        ),
        tbptt=_resolve_tbptt_params(raw_tbptt),
        online_filtering=_OnlineFilteringParams(
            steps_per_observation=_resolve_steps_per_observation(
                raw_online_filtering
            )
        ),
        log_prob_scaling=require_bool(
            training.get("log_prob_scaling"),
            field="training.log_prob_scaling",
            config_path=None,
        ),
        target_normalization=target_norm,
    )


def _require_mapping(
    value: object, *, message: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(message)
    return value


def _optional_mapping(
    value: object, *, message: str
) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, message=message)


def _resolve_training_method(value: object) -> str:
    method = str(value).strip().lower()
    if method in {"tbptt", "online_filtering"}:
        return method
    raise ConfigError("training.method must be tbptt or online_filtering")


def _resolve_log_every(raw_svi_shared: Mapping[str, Any]) -> int | None:
    raw_log_every = raw_svi_shared.get("log_every")
    if raw_log_every is None:
        return None
    log_every = int(raw_log_every)
    if log_every <= 0:
        raise ConfigError(
            "training.svi_shared.log_every must be >= 1 or null"
        )
    return log_every


def _resolve_tbptt_params(raw_tbptt: Mapping[str, Any]) -> _TBPTTParams:
    window_len = _resolve_optional_positive_int(
        raw_tbptt.get("window_len"),
        field="training.tbptt.window_len",
    )
    burn_in_len = int(raw_tbptt.get("burn_in_len", 0))
    if burn_in_len < 0:
        raise ConfigError("training.tbptt.burn_in_len must be >= 0")
    if window_len is not None and burn_in_len >= window_len:
        raise ConfigError(
            "training.tbptt.burn_in_len must be < "
            "training.tbptt.window_len"
        )
    return _TBPTTParams(window_len=window_len, burn_in_len=burn_in_len)


def _resolve_optional_positive_int(
    value: object, *, field: str
) -> int | None:
    if value is None:
        return None
    resolved = int(cast(Any, value))
    if resolved <= 0:
        raise ConfigError(f"{field} must be positive or null")
    return resolved


def _resolve_grad_accum_steps(
    *, raw_svi_shared: Mapping[str, Any], method: str
) -> int:
    grad_accum_steps = int(raw_svi_shared.get("grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ConfigError("training.svi_shared.grad_accum_steps must be >= 1")
    if method == "online_filtering" and grad_accum_steps != 1:
        raise ConfigError(
            "training.svi_shared.grad_accum_steps must be 1 when "
            "training.method=online_filtering"
        )
    return grad_accum_steps


def _resolve_steps_per_observation(
    raw_online_filtering: Mapping[str, Any]
) -> int:
    steps_per_observation = int(
        raw_online_filtering.get("steps_per_observation", 1)
    )
    if steps_per_observation <= 0:
        raise ConfigError(
            "training.online_filtering.steps_per_observation must be >= 1"
        )
    return steps_per_observation


def _run_context(config: Mapping[str, Any]) -> Mapping[str, Any]:
    context = config.get("run_context")
    if not isinstance(context, Mapping):
        return {}
    return dict(context)


def _log_svi_progress(
    *, step: int, loss: float, context: Mapping[str, Any], start: float
) -> None:
    payload = dict(context)
    payload.update(
        {
            "step": step,
            "loss": loss,
            "elapsed_s": round(time.perf_counter() - start, 4),
        }
    )
    logger.info("event=progress boundary=simulation.svi context=%s", payload)


def _build_svi(
    *,
    model: modeling.PyroModel,
    guide: modeling.PyroGuide,
    params: _TrainingParams,
) -> SVI:
    return SVI(
        model,
        guide,
        pyro.optim.Adam({"lr": params.svi.learning_rate}),  # type: ignore  # pylint: disable=no-member
        loss=Trace_ELBO(num_particles=params.svi.num_elbo_particles),
    )


def _run_svi_steps(
    *,
    svi: SVI,
    batches: list[modeling.ModelBatch],
    params: _TrainingParams,
    context: Mapping[str, Any],
) -> list[float]:
    start = time.perf_counter()
    if not batches:
        raise SimulationError("No training batches available for SVI")
    loss_history: list[float] = []
    for step in range(params.svi.steps):
        total_loss, params_in_step = _accumulate_svi_grads(
            svi=svi,
            batches=batches,
            step_index=step,
            grad_accum_steps=params.svi.grad_accum_steps,
        )
        mean_loss = total_loss / float(params.svi.grad_accum_steps)
        loss_history.append(mean_loss)
        _apply_svi_update(svi=svi, params_in_step=params_in_step)
        if params.svi.log_every and (step + 1) % params.svi.log_every == 0:
            _log_svi_progress(
                step=step + 1,
                loss=mean_loss,
                context=context,
                start=start,
            )
    return loss_history


def _run_online_filtering_steps(
    *,
    svi: SVI,
    batches: list[modeling.ModelBatch],
    params: _TrainingParams,
    context: Mapping[str, Any],
    initial_filtering_state: object | None = None,
) -> tuple[list[float], object | None]:
    start = time.perf_counter()
    if not batches:
        raise SimulationError(
            "No training batches available for online filtering"
        )
    loss_history: list[float] = []
    update_index = 0
    total_batches = len(batches)
    current_filtering_state = initial_filtering_state
    for batch_index, batch in enumerate(batches, start=1):
        batch = _with_filtering_state(batch, current_filtering_state)
        for _ in range(params.online_filtering.steps_per_observation):
            update_index += 1
            loss, params_in_step = _loss_and_params(svi, batch)
            loss_history.append(loss)
            _apply_svi_update(svi=svi, params_in_step=params_in_step)
            if params.svi.log_every and update_index % params.svi.log_every == 0:
                _log_svi_progress(
                    step=update_index,
                    loss=loss,
                    context=_online_filtering_context(
                        context=context,
                        observation_index=batch_index,
                        num_observations=total_batches,
                    ),
                    start=start,
                )
        current_filtering_state = _build_filtering_state(
            guide=svi.guide,
            batch=batch,
            default=current_filtering_state,
        )
    return loss_history, current_filtering_state


def _online_filtering_context(
    *,
    context: Mapping[str, Any],
    observation_index: int,
    num_observations: int,
) -> Mapping[str, Any]:
    payload = dict(context)
    payload.update(
        {
            "observation_index": observation_index,
            "num_observations": num_observations,
        }
    )
    return payload


def _accumulate_svi_grads(
    *,
    svi: SVI,
    batches: list[modeling.ModelBatch],
    step_index: int,
    grad_accum_steps: int,
) -> tuple[float, set[torch.Tensor]]:
    total_loss = 0.0
    params_in_step: set[torch.Tensor] = set()
    total_batches = len(batches)
    for micro in range(grad_accum_steps):
        batch = batches[(step_index * grad_accum_steps + micro) % total_batches]
        loss, params = _loss_and_params(svi, batch)
        total_loss += loss
        params_in_step.update(params)
    return total_loss, params_in_step


def _loss_and_params(
    svi: SVI,
    batch: modeling.ModelBatch,
) -> tuple[float, set[torch.Tensor]]:
    with poutine.trace(  # pylint: disable=not-context-manager
        param_only=True
    ) as param_capture:
        loss = svi.loss_and_grads(  # type: ignore[attr-defined]
            svi.model, svi.guide, batch
        )
    return _loss_to_float(loss), _collect_step_params(param_capture)


def _collect_step_params(
    param_capture: Any,
) -> set[torch.Tensor]:
    params_in_step: set[torch.Tensor] = set()
    for site in param_capture.trace.nodes.values():
        value = site.get("value")
        if value is None:
            continue
        raw_value = cast(Any, value)
        if hasattr(raw_value, "unconstrained"):
            params_in_step.add(raw_value.unconstrained())
        elif isinstance(value, torch.Tensor):
            params_in_step.add(value)
    return params_in_step


def _apply_svi_update(*, svi: SVI, params_in_step: set[torch.Tensor]) -> None:
    if not params_in_step:
        return
    svi.optim(params_in_step)
    infer_util.zero_grads(params_in_step)


def _loss_to_float(loss: object) -> float:
    if isinstance(loss, tuple):
        if not loss:
            return 0.0
        return float(cast(Any, infer_util.torch_item(loss[0])))
    return float(cast(Any, infer_util.torch_item(loss)))


def _summarize_posterior_params(config: Mapping[str, Any]) -> Mapping[str, Any]:
    store = pyro.get_param_store()
    summary: dict[str, Any] = {}
    dof_shift = resolve_dof_shift(config)

    nu_raw_loc = _get_param(store, "nu_raw_loc")
    nu_raw_scale = _get_param(store, "nu_raw_scale")
    nu_raw_mean = _lognormal_mean(nu_raw_loc, nu_raw_scale)
    if nu_raw_mean is not None:
        summary["nu_raw"] = _summarize_tensor(nu_raw_mean)
    nu_stats = _lognormal_quantile_summary(nu_raw_loc, nu_raw_scale)
    if nu_stats is not None:
        summary["nu_raw_posterior"] = nu_stats
        summary["nu_posterior"] = {
            "q10": float(nu_stats["q10"] + dof_shift),
            "median": float(nu_stats["median"] + dof_shift),
            "q90": float(nu_stats["q90"] + dof_shift),
        }

    sigma_loc = _get_param(store, "sigma_loc")
    sigma_scale = _get_param(store, "sigma_scale")
    sigma_mean = _lognormal_mean(sigma_loc, sigma_scale)
    if sigma_mean is not None:
        summary["sigma"] = _summarize_tensor(sigma_mean)
    sigma_by_asset = _lognormal_quantiles_by_asset(
        sigma_loc=sigma_loc, sigma_scale=sigma_scale
    )
    if sigma_by_asset is not None:
        summary["sigma_by_asset"] = sigma_by_asset

    w_loc = _get_param(store, "w_loc")
    w_scale = _get_param(store, "w_scale")
    if isinstance(w_loc, torch.Tensor):
        summary["w"] = {
            "abs_mean": _safe_stat(torch.mean, w_loc.abs()),
            "abs_max": _safe_stat(torch.max, w_loc.abs()),
        }
        if isinstance(w_scale, torch.Tensor):
            summary["w"]["scale_mean"] = _safe_stat(torch.mean, w_scale)
            summary["w"]["scale_max"] = _safe_stat(torch.max, w_scale)
    return summary


def _get_param(store: Any, name: str) -> torch.Tensor | None:
    if name not in store:
        return None
    value = store.get_param(name)
    if not isinstance(value, torch.Tensor):
        return None
    return value


def _lognormal_mean(
    loc: torch.Tensor | None, scale: torch.Tensor | None
) -> torch.Tensor | None:
    if not isinstance(loc, torch.Tensor) or not isinstance(
        scale, torch.Tensor
    ):
        return None
    return torch.exp(loc + 0.5 * scale.pow(2))


def _lognormal_quantile_summary(
    loc: torch.Tensor | None, scale: torch.Tensor | None
) -> Mapping[str, float] | None:
    if not isinstance(loc, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return None
    if loc.numel() == 0 or scale.numel() == 0:
        return None
    q = _normal_quantiles(loc.device, loc.dtype)
    median = torch.exp(loc)
    q10 = torch.exp(loc + scale * q[0])
    q90 = torch.exp(loc + scale * q[2])
    if not torch.isfinite(median).any():
        return None
    return {
        "q10": _safe_stat(torch.median, q10),
        "median": _safe_stat(torch.median, median),
        "q90": _safe_stat(torch.median, q90),
    }


def _lognormal_quantiles_by_asset(
    *, sigma_loc: torch.Tensor | None, sigma_scale: torch.Tensor | None
) -> Mapping[str, list[float]] | None:
    if not isinstance(sigma_loc, torch.Tensor) or not isinstance(
        sigma_scale, torch.Tensor
    ):
        return None
    if sigma_loc.numel() == 0 or sigma_scale.numel() == 0:
        return None
    loc = sigma_loc.reshape(-1)
    scale = sigma_scale.reshape(-1)
    if loc.shape != scale.shape:
        return None
    q = _normal_quantiles(loc.device, loc.dtype)
    return {
        "q10": [float(v) for v in torch.exp(loc + scale * q[0]).tolist()],
        "median": [float(v) for v in torch.exp(loc).tolist()],
        "q90": [float(v) for v in torch.exp(loc + scale * q[2]).tolist()],
    }


def _normal_quantiles(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    standard = torch.distributions.Normal(  # pylint: disable=not-callable
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )
    return standard.icdf(
        torch.tensor([0.10, 0.50, 0.90], device=device, dtype=dtype)
    )


def _summarize_tensor(values: torch.Tensor) -> Mapping[str, float]:
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return {}
    return {
        "mean": _safe_stat(torch.mean, flat),
        "median": _safe_stat(torch.median, flat),
        "min": _safe_stat(torch.min, flat),
        "max": _safe_stat(torch.max, flat),
    }


def _safe_stat(
    fn: Callable[[torch.Tensor], torch.Tensor], values: torch.Tensor
) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(fn(finite).item())


def _fit_bayes_svi_stub(
    X_train: torch.Tensor,
    X_train_global: torch.Tensor | None,
    y_train: torch.Tensor,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _ = (X_train, X_train_global, y_train, config, init_state)
    return {}


def _posterior_predict_stub(
    X_pred: torch.Tensor,
    X_pred_global: torch.Tensor | None,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    num_samples: int,
) -> Mapping[str, Any]:
    _ = (X_pred, X_pred_global, state, config, num_samples)
    return {}


def _score_predictive_stub(
    y_true: torch.Tensor,
    pred: Mapping[str, Any],
    score_spec: Mapping[str, Any],
) -> float:
    _ = (y_true, pred, score_spec)
    return 0.0


def _allocate_weights_stub(
    pred: Mapping[str, Any],
    alloc_spec: Mapping[str, Any],
) -> torch.Tensor:
    _ = pred
    n_assets = int(alloc_spec.get("n_assets", 1))
    return torch.full((n_assets,), 1.0 / max(n_assets, 1))


def _compute_weekly_pnl_stub(
    w: torch.Tensor,
    y_t: torch.Tensor,
    w_prev: torch.Tensor | None = None,
    cost_spec: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    gross = (w * y_t).sum()
    if w_prev is None or cost_spec is None:
        return gross
    turnover = torch.abs(w - w_prev).sum()
    return gross - float(cost_spec.get("tc_per_unit_turnover", 0.0)) * turnover
