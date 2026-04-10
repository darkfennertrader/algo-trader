from __future__ import annotations
# pylint: disable=too-many-lines

from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, cast

import yaml

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    CleaningSpec,
    CostConfig,
    CPCVParams,
    CVLeakage,
    CVParams,
    CVWindow,
    DataConfig,
    DiagnosticsConfig,
    ExecutionMode,
    EvaluationSpec,
    GuardrailSpec,
    FanChartsConfig,
    SviLossConfig,
    ModelConfig,
    ModelingSpec,
    OuterConfig,
    PredictiveConfig,
    PreprocessSpec,
    WinsorSpec,
    ClipSpec,
    ScalingInputSpec,
    ScalingSpec,
    ScoringConfig,
    SimulationConfig,
    SimulationFlags,
    TrainingConfig,
    TrainingMethod,
    TrainingOnlineFilteringConfig,
    TrainingSVISharedConfig,
    TrainingTBPTTConfig,
    ModelSelectionBatching,
    ModelSelectionBasket,
    ModelSelectionBootstrap,
    ModelSelectionCalibration,
    ModelSelectionComplexity,
    ModelSelectionConfig,
    ModelSelectionESBand,
    ModelSelectionTail,
    ModelPrebuildConfig,
    WalkforwardConfig,
)
from .config_allocation import _build_allocation_config
from .config_tuning import build_tuning_config
from .config_utils import coerce_mapping, require_bool, require_string
from .model_payload import model_to_payload

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "simulation.yml"
)


def load_config(path: Path | None = None) -> SimulationConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml_mapping(config_path)
    return _build_config(raw, config_path)


def config_to_dict(config: SimulationConfig) -> dict[str, object]:
    return asdict(config)


def config_to_input_dict(config: SimulationConfig) -> dict[str, object]:
    return {
        "use_feature_names_for_scaling": (
            config.flags.use_feature_names_for_scaling
        ),
        "use_gpu": config.flags.use_gpu,
        "simulation_mode": config.flags.simulation_mode,
        "execution": {"mode": config.flags.execution_mode},
        "smoke_test": {
            "enabled": config.flags.smoke_test_enabled,
            "debug": config.flags.smoke_test_debug,
        },
        "data": {
            "simulation_output_path": config.data.simulation_output_path,
            "portfolio_output_path": config.data.portfolio_output_path,
            "posterior_output_path": config.data.posterior_output_path,
            "dataset_params": dict(config.data.dataset_params),
        },
        "cv": {
            "warmup_len": config.cv.window.warmup_len,
            "group_len": config.cv.window.group_len,
            "horizon": config.cv.leakage.horizon,
            "embargo_len": config.cv.leakage.embargo_len,
            "q": config.cv.cpcv.q,
            "max_inner_combos": config.cv.cpcv.max_inner_combos,
            "seed": config.cv.cpcv.seed,
            "exclude_warmup": config.cv.exclude_warmup,
        },
        "outer": {
            "test_group_ids": config.outer.test_group_ids,
            "last_n": config.outer.last_n,
        },
        "walkforward": {
            "num_seeds": config.walkforward.num_seeds,
            "seeds": list(config.walkforward.seeds),
            "max_parallel_seeds_per_gpu": (
                config.walkforward.max_parallel_seeds_per_gpu
            ),
        },
        "preprocessing": {
            "min_usable_ratio": config.preprocessing.cleaning.min_usable_ratio,
            "min_variance": config.preprocessing.cleaning.min_variance,
            "max_abs_corr": config.preprocessing.cleaning.max_abs_corr,
            "corr_subsample": config.preprocessing.cleaning.corr_subsample,
            "mad_eps": config.preprocessing.scaling.mad_eps,
            "breakout_var_floor": (
                config.preprocessing.scaling.breakout_var_floor
            ),
            "winsor_low_q": config.preprocessing.scaling.winsor.lower_q,
            "winsor_high_q": config.preprocessing.scaling.winsor.upper_q,
            "scale_floor": config.preprocessing.scaling.scale_floor,
            "guard_abs_eps": config.preprocessing.scaling.guardrail.abs_eps,
            "guard_rel_eps": config.preprocessing.scaling.guardrail.rel_eps,
            "guard_rel_offset": (
                config.preprocessing.scaling.guardrail.rel_offset
            ),
            "clip_min": config.preprocessing.scaling.clip.min_value,
            "clip_max": config.preprocessing.scaling.clip.max_value,
            "max_abs_fail": config.preprocessing.scaling.clip.max_abs_fail,
            "impute_missing_to_zero": (
                config.preprocessing.scaling.inputs.impute_missing_to_zero
            ),
            "append_mask_as_features": (
                config.preprocessing.scaling.inputs.append_mask_as_features
            ),
            "append_exogenous_mask_as_features": (
                config.preprocessing.scaling.inputs.append_exogenous_mask_as_features
            ),
        },
        "model": _model_to_input_dict(config.modeling.model),
        "training": _training_to_input_dict(config.modeling.training),
        "tuning": _tuning_to_input_dict(config.modeling.tuning),
        "scoring": {"spec": dict(config.evaluation.scoring.spec)},
        "predictive": asdict(config.evaluation.predictive),
        "allocation": _allocation_to_input_dict(config.evaluation.allocation),
        "cost": {"spec": dict(config.evaluation.cost.spec)},
        "model_selection": _model_selection_to_input_dict(
            config.evaluation.model_selection
        ),
        "diagnostics": _diagnostics_to_input_dict(
            config.evaluation.diagnostics
        ),
    }


def _model_to_input_dict(model: ModelConfig) -> dict[str, object]:
    return model_to_payload(model, include_prebuild=True)


def _training_to_input_dict(training: TrainingConfig) -> dict[str, object]:
    return {
        "method": training.method,
        "svi_shared": asdict(training.svi_shared),
        "tbptt": asdict(training.tbptt),
        "online_filtering": asdict(training.online_filtering),
        "target_normalization": training.target_normalization,
        "log_prob_scaling": training.log_prob_scaling,
    }


def _tuning_to_input_dict(tuning: Any) -> dict[str, object]:
    return {
        "space": [
            _tuning_param_to_input_dict(param) for param in tuning.space
        ],
        "num_samples": tuning.num_samples,
        "seed": tuning.seed,
        "kwargs": dict(tuning.kwargs),
        "engine": tuning.engine,
        "aggregate": asdict(tuning.aggregate),
        "ray": {
            "address": tuning.ray.address,
            "logs_enabled": tuning.ray.logs_enabled,
            "resources": asdict(tuning.ray.resources),
            "early_stopping": asdict(tuning.ray.early_stopping),
        },
    }


def _tuning_param_to_input_dict(param: Any) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": param.path,
        "type": param.param_type,
        "transform": param.transform,
    }
    if param.bounds is not None:
        payload["bounds"] = list(param.bounds)
    if param.values is not None:
        payload["values"] = list(param.values)
    if param.when:
        payload["when"] = {
            key: list(values) for key, values in param.when.items()
        }
    return payload


def _allocation_to_input_dict(
    allocation: Any,
) -> dict[str, object]:
    return {
        "primary": {
            "family": allocation.primary.family,
            **dict(allocation.primary.params),
        },
        "baselines": [
            {"family": baseline.family, **dict(baseline.params)}
            for baseline in allocation.baselines
        ],
    }


def _model_selection_to_input_dict(
    selection: ModelSelectionConfig,
) -> dict[str, object]:
    return {
        "enable": selection.enable,
        "phase_name": selection.phase_name,
        "mode": selection.mode,
        "calibration": asdict(selection.calibration),
        "basket": {
            "baskets": list(selection.basket.baskets),
            "mean_abs_weight": selection.basket.mean_abs_weight,
            "max_abs_weight": selection.basket.max_abs_weight,
            "pit_weight": selection.basket.pit_weight,
        },
        "es_band": asdict(selection.es_band),
        "bootstrap": asdict(selection.bootstrap),
        "tail": asdict(selection.tail),
        "batching": asdict(selection.batching),
        "complexity": asdict(selection.complexity),
    }


def _diagnostics_to_input_dict(
    diagnostics: DiagnosticsConfig,
) -> dict[str, object]:
    assets: object
    if diagnostics.fan_charts.assets_mode == "all":
        assets = "all"
    else:
        assets = list(diagnostics.fan_charts.assets)
    return {
        "fan_charts": {
            "enable": diagnostics.fan_charts.enable,
            "assets_mode": diagnostics.fan_charts.assets_mode,
            "assets": assets,
            "rolling_mean": list(diagnostics.fan_charts.rolling_mean),
            "quantiles": list(diagnostics.fan_charts.quantiles),
            "coverage_levels": list(
                diagnostics.fan_charts.coverage_levels
            ),
        },
        "svi_loss": asdict(diagnostics.svi_loss),
    }


def _build_config(
    raw: Mapping[str, Any], config_path: Path
) -> SimulationConfig:
    data = _build_data_config(raw, config_path)
    cv = _build_cv_params(raw, config_path)
    preprocessing = _build_preprocess_spec(raw, config_path)
    modeling = _build_modeling_spec(raw, config_path)
    evaluation = _build_evaluation_spec(raw, config_path)
    outer = _build_outer_config(raw, config_path)
    walkforward = _build_walkforward_config(raw, config_path)
    flags = _build_flags(raw, config_path)

    return SimulationConfig(
        data=data,
        cv=cv,
        preprocessing=preprocessing,
        modeling=modeling,
        evaluation=evaluation,
        outer=outer,
        walkforward=walkforward,
        flags=flags,
    )


def _build_flags(
    raw: Mapping[str, Any], config_path: Path
) -> SimulationFlags:
    if "stop_after" in raw:
        raise ConfigError(
            f"stop_after is no longer supported in {config_path}; "
            "use execution.mode instead"
        )
    execution_mode = _read_execution_mode(raw, config_path)
    mode = raw.get("simulation_mode", "full")
    smoke_enabled, debug_enabled = _read_smoke_test_flags(
        raw, config_path
    )
    return SimulationFlags(
        use_feature_names_for_scaling=bool(
            raw.get("use_feature_names_for_scaling", True)
        ),
        use_gpu=bool(raw.get("use_gpu", False)),
        smoke_test_enabled=smoke_enabled,
        smoke_test_debug=debug_enabled,
        simulation_mode=_normalize_simulation_mode(mode),
        execution_mode=execution_mode,
    )


def _read_execution_mode(
    raw: Mapping[str, Any], config_path: Path
) -> Literal[
    "full",
    "model_research",
    "posterior_signal",
    "walkforward",
    "results_aggregation",
]:
    section = raw.get("execution", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ConfigError(f"execution must be a mapping in {config_path}")
    extra = set(section) - {"mode"}
    if extra:
        raise ConfigError(
            f"execution contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    value = section.get("mode", "full")
    return _normalize_execution_mode(value)


def _read_smoke_test_flags(
    raw: Mapping[str, Any], config_path: Path
) -> tuple[bool, bool]:
    section = raw.get("smoke_test", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"smoke_test must be a mapping in {config_path}"
        )
    extra = set(section) - {"enabled", "debug"}
    if extra:
        raise ConfigError(
            f"smoke_test contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    enabled = require_bool(
        section.get("enabled"),
        field="smoke_test.enabled",
        config_path=config_path,
    )
    debug_raw = section.get("debug")
    if debug_raw is None:
        return enabled, False
    return enabled, require_bool(
        debug_raw, field="smoke_test.debug", config_path=config_path
    )


SimulationMode = Literal["dry_run", "stub", "full"]
def _normalize_simulation_mode(value: object) -> SimulationMode:
    raw = str(value).strip().lower()
    if raw == "dry_run":
        return "dry_run"
    if raw == "stub":
        return "stub"
    if raw == "full":
        return "full"
    raise ConfigError("simulation_mode must be dry_run, stub, or full")


def _normalize_execution_mode(value: object) -> ExecutionMode:
    raw = str(value).strip().lower()
    if raw == "full":
        return "full"
    if raw == "model_research":
        return "model_research"
    if raw == "posterior_signal":
        return "posterior_signal"
    if raw in {"walkforward", "outer_evaluation"}:
        return "walkforward"
    if raw == "results_aggregation":
        return "results_aggregation"
    raise ConfigError(
        "execution.mode must be full, model_research, "
        "posterior_signal, walkforward, or results_aggregation"
    )


def _build_outer_config(
    raw: Mapping[str, Any], config_path: Path
) -> OuterConfig:
    outer = raw.get("outer", {})
    if outer is None:
        outer = {}
    if not isinstance(outer, Mapping):
        raise ConfigError(f"outer must be a mapping in {config_path}")
    test_group_ids = outer.get("test_group_ids")
    last_n = outer.get("last_n")
    if test_group_ids is not None:
        if not isinstance(test_group_ids, list) or not all(
            isinstance(item, int) for item in test_group_ids
        ):
            raise ConfigError(
                f"outer.test_group_ids must be a list of ints in {config_path}"
            )
    if last_n is not None:
        try:
            last_n = int(last_n)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"outer.last_n must be an int in {config_path}"
            ) from exc
    if test_group_ids is not None and last_n is not None:
        raise ConfigError(
            f"Specify only one of outer.test_group_ids or outer.last_n in {config_path}"
        )
    return OuterConfig(test_group_ids=test_group_ids, last_n=last_n)


def _build_walkforward_config(
    raw: Mapping[str, Any], config_path: Path
) -> WalkforwardConfig:
    section = raw.get("walkforward", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ConfigError(f"walkforward must be a mapping in {config_path}")
    extra = set(section) - {
        "num_seeds",
        "seeds",
        "max_parallel_seeds_per_gpu",
    }
    if extra:
        raise ConfigError(
            f"walkforward contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    seeds = _build_walkforward_seed_list(
        section.get("seeds", (7,)),
        field="walkforward.seeds",
        config_path=config_path,
    )
    num_seeds = _require_positive_int(
        section.get("num_seeds", len(seeds)),
        field="walkforward.num_seeds",
        config_path=config_path,
    )
    if len(seeds) != num_seeds:
        raise ConfigError(
            "walkforward.num_seeds must match the number of explicit seeds",
            context={
                "num_seeds": str(num_seeds),
                "seed_count": str(len(seeds)),
            },
        )
    return WalkforwardConfig(
        num_seeds=num_seeds,
        seeds=seeds,
        max_parallel_seeds_per_gpu=_require_positive_int(
            section.get("max_parallel_seeds_per_gpu", 1),
            field="walkforward.max_parallel_seeds_per_gpu",
            config_path=config_path,
        ),
    )


def _build_walkforward_seed_list(
    value: object,
    *,
    field: str,
    config_path: Path,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{field} must be a list of ints in {config_path}")
    seeds = tuple(_require_int_field(item, field=field, config_path=config_path) for item in value)
    if not seeds:
        raise ConfigError(f"{field} must not be empty in {config_path}")
    return seeds


def _require_positive_int(
    value: object,
    *,
    field: str,
    config_path: Path,
) -> int:
    parsed = _require_int_field(value, field=field, config_path=config_path)
    if parsed <= 0:
        raise ConfigError(f"{field} must be positive in {config_path}")
    return parsed


def _require_int_field(
    value: object,
    *,
    field: str,
    config_path: Path,
) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be an int in {config_path}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(
                f"{field} must be an int in {config_path}"
            ) from exc
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be an int in {config_path}") from exc


def _build_modeling_spec(
    raw: Mapping[str, Any], config_path: Path
) -> ModelingSpec:
    model = _build_model_config(raw, config_path)
    training = _build_training_config(raw, config_path)
    tuning = build_tuning_config(raw, config_path)
    return ModelingSpec(model=model, training=training, tuning=tuning)


def _build_model_config(
    raw: Mapping[str, Any], config_path: Path
) -> ModelConfig:
    section = raw.get("model")
    if not isinstance(section, Mapping):
        raise ConfigError(f"model must be a mapping in {config_path}")
    extra = set(section) - {
        "model_name",
        "guide_name",
        "predict_name",
        "params",
        "guide_params",
        "predict_params",
        "prebuild",
    }
    if extra:
        raise ConfigError(
            f"model contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    model_name = require_string(
        section.get("model_name"),
        field="model.model_name",
        config_path=config_path,
    )
    guide_name = require_string(
        section.get("guide_name"),
        field="model.guide_name",
        config_path=config_path,
    )
    predict_name = _optional_string(
        section.get("predict_name"),
        field="model.predict_name",
        config_path=config_path,
    )
    params = coerce_mapping(
        section.get("params"),
        field="model.params",
        config_path=config_path,
    )
    guide_params = coerce_mapping(
        section.get("guide_params"),
        field="model.guide_params",
        config_path=config_path,
    )
    predict_params = coerce_mapping(
        section.get("predict_params"),
        field="model.predict_params",
        config_path=config_path,
    )
    prebuild = _build_model_prebuild(
        section.get("prebuild"), config_path
    )
    return ModelConfig(
        model_name=model_name,
        guide_name=guide_name,
        predict_name=predict_name,
        params=params,
        guide_params=guide_params,
        predict_params=predict_params,
        prebuild=prebuild,
    )


def _optional_string(
    value: object,
    *,
    field: str,
    config_path: Path,
) -> str | None:
    if value is None:
        return None
    return require_string(value, field=field, config_path=config_path)


def _build_model_prebuild(
    raw: object, config_path: Path
) -> ModelPrebuildConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"model.prebuild must be a mapping in {config_path}")
    extra = set(raw) - {"name", "params", "enabled"}
    if extra:
        raise ConfigError(
            f"model.prebuild contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    name = require_string(
        raw.get("name"),
        field="model.prebuild.name",
        config_path=config_path,
    )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError(
            f"model.prebuild.enabled must be a boolean in {config_path}"
        )
    params = coerce_mapping(
        raw.get("params"),
        field="model.prebuild.params",
        config_path=config_path,
    )
    return ModelPrebuildConfig(
        name=name,
        params=params,
        enabled=enabled,
    )


def _build_training_config(
    raw: Mapping[str, Any], config_path: Path
) -> TrainingConfig:
    section = raw.get("training")
    if not isinstance(section, Mapping):
        raise ConfigError(f"training must be a mapping in {config_path}")
    raw_svi_shared, raw_tbptt, raw_online_filtering = (
        _resolve_training_sections(section, config_path)
    )
    method = _normalize_training_method(
        section.get("method", "tbptt"), config_path
    )
    target_norm, log_prob_scaling = _resolve_training_bools(
        section, config_path
    )
    svi_shared, tbptt, online_filtering = _build_training_blocks(
        raw_svi_shared=raw_svi_shared,
        raw_tbptt=raw_tbptt,
        raw_online_filtering=raw_online_filtering,
        config_path=config_path,
    )
    training = TrainingConfig(
        method=method,
        svi_shared=svi_shared,
        tbptt=tbptt,
        online_filtering=online_filtering,
        target_normalization=target_norm,
        log_prob_scaling=log_prob_scaling,
    )
    _validate_training_config(training=training, config_path=config_path)
    return training


def _resolve_training_sections(
    section: Mapping[str, Any], config_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    extra_training = set(section) - {
        "method",
        "svi_shared",
        "tbptt",
        "online_filtering",
        "target_normalization",
        "log_prob_scaling",
    }
    if extra_training:
        raise ConfigError(
            f"training contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra_training))},
        )
    return (
        _require_training_mapping(
            section.get("svi_shared"),
            field="training.svi_shared",
            config_path=config_path,
            allowed_keys={
                "learning_rate",
                "grad_accum_steps",
                "num_elbo_particles",
                "log_every",
            },
        ),
        _require_training_mapping(
            section.get("tbptt"),
            field="training.tbptt",
            config_path=config_path,
            allowed_keys={"num_steps", "window_len", "burn_in_len"},
        ),
        _require_training_mapping(
            section.get("online_filtering", {}),
            field="training.online_filtering",
            config_path=config_path,
            allowed_keys={"steps_per_observation"},
        ),
    )


def _require_training_mapping(
    raw: object,
    *,
    field: str,
    config_path: Path,
    allowed_keys: set[str],
) -> Mapping[str, Any]:
    value = {} if raw is None else raw
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping in {config_path}")
    extra = set(value) - allowed_keys
    if extra:
        raise ConfigError(
            f"{field} contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    return value


def _resolve_training_bools(
    section: Mapping[str, Any], config_path: Path
) -> tuple[bool, bool]:
    target_norm = require_bool(
        section.get("target_normalization"),
        field="training.target_normalization",
        config_path=config_path,
    )
    log_prob_scaling = require_bool(
        section.get("log_prob_scaling"),
        field="training.log_prob_scaling",
        config_path=config_path,
    )
    return target_norm, log_prob_scaling


def _build_training_blocks(
    *,
    raw_svi_shared: Mapping[str, Any],
    raw_tbptt: Mapping[str, Any],
    raw_online_filtering: Mapping[str, Any],
    config_path: Path,
) -> tuple[
    TrainingSVISharedConfig,
    TrainingTBPTTConfig,
    TrainingOnlineFilteringConfig,
]:
    try:
        return (
            TrainingSVISharedConfig(
                learning_rate=float(raw_svi_shared.get("learning_rate", 1e-3)),
                grad_accum_steps=int(raw_svi_shared.get("grad_accum_steps", 1)),
                num_elbo_particles=int(
                    raw_svi_shared.get("num_elbo_particles", 1)
                ),
                log_every=(
                    int(raw_svi_shared["log_every"])
                    if raw_svi_shared.get("log_every") is not None
                    else None
                ),
            ),
            TrainingTBPTTConfig(
                num_steps=int(raw_tbptt.get("num_steps", 2_000)),
                window_len=(
                    int(raw_tbptt["window_len"])
                    if raw_tbptt.get("window_len") is not None
                    else None
                ),
                burn_in_len=int(raw_tbptt.get("burn_in_len", 0)),
            ),
            TrainingOnlineFilteringConfig(
                steps_per_observation=int(
                    raw_online_filtering.get("steps_per_observation", 1)
                )
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid training configuration in {config_path}",
            context={"section": "training"},
        ) from exc


def _validate_training_config(
    *,
    training: TrainingConfig,
    config_path: Path,
) -> None:
    if training.online_filtering.steps_per_observation <= 0:
        raise ConfigError(
            f"training.online_filtering.steps_per_observation must be >= 1 in {config_path}"
        )
    if training.method == "online_filtering" and training.target_normalization:
        raise ConfigError(
            f"training.target_normalization must be false when training.method=online_filtering in {config_path}"
        )
    if (
        training.method == "online_filtering"
        and training.svi_shared.grad_accum_steps != 1
    ):
        raise ConfigError(
            f"training.svi_shared.grad_accum_steps must be 1 when training.method=online_filtering in {config_path}"
        )
    if training.tbptt.burn_in_len < 0:
        raise ConfigError(
            f"training.tbptt.burn_in_len must be >= 0 in {config_path}"
        )
    if (
        training.tbptt.window_len is not None
        and training.tbptt.burn_in_len >= training.tbptt.window_len
    ):
        raise ConfigError(
            f"training.tbptt.burn_in_len must be < training.tbptt.window_len in {config_path}"
        )


def _normalize_training_method(
    value: object, config_path: Path
) -> TrainingMethod:
    raw = str(value).strip().lower()
    if raw == "tbptt":
        return "tbptt"
    if raw == "online_filtering":
        return "online_filtering"
    raise ConfigError(
        f"training.method must be tbptt or online_filtering in {config_path}"
    )


def _build_evaluation_spec(
    raw: Mapping[str, Any], config_path: Path
) -> EvaluationSpec:
    scoring = _build_section(raw, "scoring", ScoringConfig, config_path)
    predictive = _build_section(raw, "predictive", PredictiveConfig, config_path)
    allocation = _build_allocation_config(raw, config_path)
    cost = _build_section(raw, "cost", CostConfig, config_path)
    model_selection = _build_model_selection_config(raw, config_path)
    diagnostics = _build_diagnostics_config(raw, config_path)
    return EvaluationSpec(
        scoring=scoring,
        predictive=predictive,
        allocation=allocation,
        cost=cost,
        model_selection=model_selection,
        diagnostics=diagnostics,
    )


def _build_model_selection_config(
    raw: Mapping[str, Any], config_path: Path
) -> ModelSelectionConfig:
    section = raw.get("model_selection", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"model_selection must be a mapping in {config_path}"
        )
    enable = bool(section.get("enable", False))
    phase_name = str(
        section.get("phase_name", "post_tune_model_selection")
    )
    mode = _build_model_selection_mode(section, config_path)
    es_band = _build_model_selection_es_band(section, config_path)
    calibration = _build_model_selection_calibration(section, config_path)
    basket = _build_model_selection_basket(section, config_path)
    bootstrap = _build_model_selection_bootstrap(section, config_path)
    tail = _build_model_selection_tail(section, config_path)
    batching = _build_model_selection_batching(section, config_path)
    complexity = _build_model_selection_complexity(section, config_path)
    return ModelSelectionConfig(
        enable=enable,
        phase_name=phase_name,
        mode=mode,
        calibration=calibration,
        basket=basket,
        es_band=es_band,
        bootstrap=bootstrap,
        tail=tail,
        batching=batching,
        complexity=complexity,
    )


def _build_diagnostics_config(
    raw: Mapping[str, Any], config_path: Path
) -> DiagnosticsConfig:
    section = raw.get("diagnostics", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"diagnostics must be a mapping in {config_path}"
        )
    fan_charts = _build_fan_charts_config(section, config_path)
    svi_loss = _build_svi_loss_config(section, config_path)
    return DiagnosticsConfig(fan_charts=fan_charts, svi_loss=svi_loss)


def _build_svi_loss_config(
    section: Mapping[str, Any], config_path: Path
) -> SviLossConfig:
    raw_svi_loss = section.get("svi_loss", {})
    if raw_svi_loss is None:
        raw_svi_loss = {}
    if isinstance(raw_svi_loss, bool):
        return SviLossConfig(enable=raw_svi_loss)
    if not isinstance(raw_svi_loss, Mapping):
        raise ConfigError(
            f"diagnostics.svi_loss must be a mapping or bool in {config_path}"
        )
    extra = set(raw_svi_loss) - {"enable"}
    if extra:
        raise ConfigError(
            f"diagnostics.svi_loss contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    return SviLossConfig(
        enable=require_bool(
            raw_svi_loss.get("enable"),
            field="diagnostics.svi_loss.enable",
            config_path=config_path,
        )
    )


def _build_model_selection_mode(
    section: Mapping[str, Any], config_path: Path
) -> Literal["global_calibrated", "basket_aware", "signal_aware"]:
    raw_mode = str(section.get("mode", "global_calibrated")).strip().lower()
    if raw_mode not in {"global_calibrated", "basket_aware", "signal_aware"}:
        raise ConfigError(
            "model_selection.mode must be 'global_calibrated', "
            f"'basket_aware', or 'signal_aware' in {config_path}"
        )
    return cast(
        Literal["global_calibrated", "basket_aware", "signal_aware"],
        raw_mode,
    )


def _build_fan_charts_config(
    section: Mapping[str, Any], config_path: Path
) -> FanChartsConfig:
    raw_fan = section.get("fan_charts", {})
    if raw_fan is None:
        raw_fan = {}
    if not isinstance(raw_fan, Mapping):
        raise ConfigError(
            f"diagnostics.fan_charts must be a mapping in {config_path}"
        )
    enable = bool(raw_fan.get("enable", False))
    assets_mode, assets = _parse_fan_assets(
        raw_fan.get("assets", "all"), config_path
    )
    quantiles = _parse_float_list(
        raw_fan.get("quantiles"),
        field="diagnostics.fan_charts.quantiles",
        config_path=config_path,
        default=FanChartsConfig().quantiles,
    )
    coverage_levels = _parse_float_list(
        raw_fan.get("coverage_levels"),
        field="diagnostics.fan_charts.coverage_levels",
        config_path=config_path,
        default=FanChartsConfig().coverage_levels,
    )
    rolling_mean = _parse_positive_int_list(
        raw_fan.get("rolling_mean"),
        field="diagnostics.fan_charts.rolling_mean",
        config_path=config_path,
        default=FanChartsConfig().rolling_mean,
        allow_empty=True,
    )
    return FanChartsConfig(
        enable=enable,
        assets_mode=assets_mode,
        assets=assets,
        rolling_mean=rolling_mean,
        quantiles=quantiles,
        coverage_levels=coverage_levels,
    )


def _parse_fan_assets(
    raw_assets: Any, config_path: Path
) -> tuple[Literal["all", "list"], tuple[str, ...]]:
    if raw_assets is None:
        return "all", tuple()
    if isinstance(raw_assets, str):
        if raw_assets.strip().lower() == "all":
            return "all", tuple()
        return "list", (raw_assets.strip(),)
    if isinstance(raw_assets, (list, tuple)):
        assets: list[str] = []
        for item in raw_assets:
            if item is None:
                continue
            name = str(item).strip()
            if not name:
                raise ConfigError(
                    f"diagnostics.fan_charts.assets must be non-empty in {config_path}"
                )
            assets.append(name)
        if not assets:
            raise ConfigError(
                f"diagnostics.fan_charts.assets must be non-empty in {config_path}"
            )
        return "list", tuple(assets)
    raise ConfigError(
        f"diagnostics.fan_charts.assets must be 'all' or a list in {config_path}"
    )


def _parse_float_list(
    raw: Any,
    *,
    field: str,
    config_path: Path,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, (list, tuple)):
        values = [float(item) for item in raw]
    else:
        values = [float(raw)]
    if not values:
        raise ConfigError(f"{field} must be non-empty in {config_path}")
    for value in values:
        if not 0.0 < float(value) < 1.0:
            raise ConfigError(
                f"{field} entries must be in (0, 1) in {config_path}"
            )
    unique = sorted(set(values))
    return tuple(unique)


def _parse_nonempty_string_list(
    raw: Any,
    *,
    field: str,
    config_path: Path,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw]
    else:
        values = [str(raw).strip()]
    if not values or any(not value for value in values):
        raise ConfigError(f"{field} must be non-empty in {config_path}")
    return tuple(values)


def _parse_positive_int_list(
    raw: Any,
    *,
    field: str,
    config_path: Path,
    default: tuple[int, ...],
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, (list, tuple)):
        values = [
            _parse_positive_int(
                item,
                field=field,
                config_path=config_path,
            )
            for item in raw
        ]
    else:
        values = [
            _parse_positive_int(
                raw,
                field=field,
                config_path=config_path,
            )
        ]
    if not values:
        if allow_empty:
            return tuple()
        raise ConfigError(f"{field} must be non-empty in {config_path}")
    unique = sorted(set(values))
    return tuple(unique)


def _parse_positive_int(
    raw: Any,
    *,
    field: str,
    config_path: Path,
) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{field} entries must be integers in {config_path}")
    if raw <= 0:
        raise ConfigError(
            f"{field} entries must be positive integers in {config_path}"
        )
    return int(raw)


def _build_model_selection_es_band(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionESBand:
    raw_band = section.get("es_band", {})
    if raw_band is None:
        raw_band = {}
    if not isinstance(raw_band, Mapping):
        raise ConfigError(
            f"model_selection.es_band must be a mapping in {config_path}"
        )
    c = float(raw_band.get("c", 1.0))
    min_keep = int(raw_band.get("min_keep", 1))
    max_keep = int(raw_band.get("max_keep", 10))
    if min_keep <= 0:
        raise ConfigError(
            f"model_selection.es_band.min_keep must be positive in {config_path}"
        )
    if max_keep <= 0:
        raise ConfigError(
            f"model_selection.es_band.max_keep must be positive in {config_path}"
        )
    if min_keep > max_keep:
        raise ConfigError(
            f"model_selection.es_band.min_keep must be <= max_keep in {config_path}"
        )
    return ModelSelectionESBand(c=c, min_keep=min_keep, max_keep=max_keep)


def _build_model_selection_calibration(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionCalibration:
    raw_calibration = section.get("calibration", {})
    if raw_calibration is None:
        raw_calibration = {}
    if not isinstance(raw_calibration, Mapping):
        raise ConfigError(
            f"model_selection.calibration must be a mapping in {config_path}"
        )
    defaults = ModelSelectionCalibration()
    top_k = int(raw_calibration.get("top_k", defaults.top_k))
    coverage_levels = _parse_float_list(
        raw_calibration.get("coverage_levels"),
        field="model_selection.calibration.coverage_levels",
        config_path=config_path,
        default=defaults.coverage_levels,
    )
    mean_abs_weight = float(
        raw_calibration.get("mean_abs_weight", defaults.mean_abs_weight)
    )
    max_abs_weight = float(
        raw_calibration.get("max_abs_weight", defaults.max_abs_weight)
    )
    pit_weight = float(raw_calibration.get("pit_weight", defaults.pit_weight))
    if top_k <= 0:
        raise ConfigError(
            f"model_selection.calibration.top_k must be positive in {config_path}"
        )
    weights = (mean_abs_weight, max_abs_weight, pit_weight)
    if any(weight < 0.0 for weight in weights):
        raise ConfigError(
            "model_selection.calibration weights must be non-negative "
            f"in {config_path}"
        )
    if all(weight == 0.0 for weight in weights):
        raise ConfigError(
            "model_selection.calibration requires at least one positive weight "
            f"in {config_path}"
        )
    return ModelSelectionCalibration(
        top_k=top_k,
        coverage_levels=coverage_levels,
        mean_abs_weight=mean_abs_weight,
        max_abs_weight=max_abs_weight,
        pit_weight=pit_weight,
    )


def _build_model_selection_basket(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionBasket:
    raw_basket = section.get("basket", {})
    if raw_basket is None:
        raw_basket = {}
    if not isinstance(raw_basket, Mapping):
        raise ConfigError(
            f"model_selection.basket must be a mapping in {config_path}"
        )
    defaults = ModelSelectionBasket()
    baskets = _parse_nonempty_string_list(
        raw_basket.get("baskets"),
        field="model_selection.basket.baskets",
        config_path=config_path,
        default=defaults.baskets,
    )
    mean_abs_weight = float(
        raw_basket.get("mean_abs_weight", defaults.mean_abs_weight)
    )
    max_abs_weight = float(
        raw_basket.get("max_abs_weight", defaults.max_abs_weight)
    )
    pit_weight = float(raw_basket.get("pit_weight", defaults.pit_weight))
    weights = (mean_abs_weight, max_abs_weight, pit_weight)
    if any(weight < 0.0 for weight in weights):
        raise ConfigError(
            "model_selection.basket weights must be non-negative "
            f"in {config_path}"
        )
    if all(weight == 0.0 for weight in weights):
        raise ConfigError(
            "model_selection.basket requires at least one positive weight "
            f"in {config_path}"
        )
    return ModelSelectionBasket(
        baskets=baskets,
        mean_abs_weight=mean_abs_weight,
        max_abs_weight=max_abs_weight,
        pit_weight=pit_weight,
    )


def _build_model_selection_bootstrap(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionBootstrap:
    raw_boot = section.get("bootstrap", {})
    if raw_boot is None:
        raw_boot = {}
    if not isinstance(raw_boot, Mapping):
        raise ConfigError(
            f"model_selection.bootstrap must be a mapping in {config_path}"
        )
    num_samples = int(raw_boot.get("num_samples", 500))
    seed = int(raw_boot.get("seed", 123))
    if num_samples <= 0:
        raise ConfigError(
            f"model_selection.bootstrap.num_samples must be positive in {config_path}"
        )
    return ModelSelectionBootstrap(num_samples=num_samples, seed=seed)


def _build_model_selection_tail(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionTail:
    raw_tail = section.get("tail", {})
    if raw_tail is None:
        raw_tail = {}
    if not isinstance(raw_tail, Mapping):
        raise ConfigError(
            f"model_selection.tail must be a mapping in {config_path}"
        )
    alpha = float(raw_tail.get("alpha", 0.1))
    if not 0.0 < alpha < 1.0:
        raise ConfigError(
            f"model_selection.tail.alpha must be in (0, 1) in {config_path}"
        )
    return ModelSelectionTail(alpha=alpha)


def _build_model_selection_batching(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionBatching:
    raw_batch = section.get("batching", {})
    if raw_batch is None:
        raw_batch = {}
    if not isinstance(raw_batch, Mapping):
        raise ConfigError(
            f"model_selection.batching must be a mapping in {config_path}"
        )
    candidates = int(raw_batch.get("candidates", 1))
    splits = int(raw_batch.get("splits", 1))
    if candidates <= 0:
        raise ConfigError(
            f"model_selection.batching.candidates must be positive in {config_path}"
        )
    if splits <= 0:
        raise ConfigError(
            f"model_selection.batching.splits must be positive in {config_path}"
        )
    return ModelSelectionBatching(candidates=candidates, splits=splits)


def _build_model_selection_complexity(
    section: Mapping[str, Any], config_path: Path
) -> ModelSelectionComplexity:
    raw_complexity = section.get("complexity", {})
    if raw_complexity is None:
        raw_complexity = {}
    if not isinstance(raw_complexity, Mapping):
        raise ConfigError(
            f"model_selection.complexity must be a mapping in {config_path}"
        )
    method = str(
        raw_complexity.get("method", "posterior_l1")
    ).strip().lower()
    if method not in {"random", "posterior_l1"}:
        raise ConfigError(
            "model_selection.complexity.method must be "
            f"'random' or 'posterior_l1' in {config_path}"
        )
    seed = int(raw_complexity.get("seed", 123))
    return ModelSelectionComplexity(
        method=cast(Literal["random", "posterior_l1"], method),
        seed=seed,
    )


def _build_section(
    raw: Mapping[str, Any],
    key: str,
    constructor: type[Any],
    config_path: Path,
) -> Any:
    section = raw.get(key)
    if not isinstance(section, Mapping):
        raise ConfigError(f"{key} must be a mapping in {config_path}")
    try:
        return constructor(**section)
    except TypeError as exc:
        raise ConfigError(
            f"Invalid {key} configuration in {config_path}",
            context={"section": key},
        ) from exc


def _build_cv_params(
    raw: Mapping[str, Any], config_path: Path
) -> CVParams:
    section = raw.get("cv")
    if not isinstance(section, Mapping):
        raise ConfigError(f"cv must be a mapping in {config_path}")
    try:
        window = CVWindow(
            warmup_len=int(section.get("warmup_len", 0)),
            group_len=int(section.get("group_len", 0)),
        )
        leakage = CVLeakage(
            horizon=int(section.get("horizon", 1)),
            embargo_len=int(section.get("embargo_len", 0)),
        )
        cpcv = CPCVParams(
            q=int(section.get("q", 0)),
            max_inner_combos=(
                int(section["max_inner_combos"])
                if section.get("max_inner_combos") is not None
                else None
            ),
            seed=int(section.get("seed", 0)),
        )
        exclude_warmup = bool(section.get("exclude_warmup", False))
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid cv configuration in {config_path}",
            context={"section": "cv"},
        ) from exc
    return CVParams(
        window=window,
        leakage=leakage,
        cpcv=cpcv,
        exclude_warmup=exclude_warmup,
    )


def _build_preprocess_spec(
    raw: Mapping[str, Any], config_path: Path
) -> PreprocessSpec:
    section = raw.get("preprocessing")
    if not isinstance(section, Mapping):
        raise ConfigError(f"preprocessing must be a mapping in {config_path}")
    try:
        cleaning = CleaningSpec(
            min_usable_ratio=float(section.get("min_usable_ratio", 0.0)),
            min_variance=float(section.get("min_variance", 0.0)),
            max_abs_corr=float(section.get("max_abs_corr", 0.0)),
            corr_subsample=(
                int(section["corr_subsample"])
                if section.get("corr_subsample") is not None
                else None
            ),
        )
        guardrail = GuardrailSpec(
            abs_eps=float(section.get("guard_abs_eps", 1.0e-6)),
            rel_eps=float(section.get("guard_rel_eps", 1.0e-3)),
            rel_offset=float(section.get("guard_rel_offset", 1.0e-8)),
        )
        clip = ClipSpec(
            min_value=float(section.get("clip_min", -10.0)),
            max_value=float(section.get("clip_max", 10.0)),
            max_abs_fail=float(section.get("max_abs_fail", 50.0)),
        )
        winsor = WinsorSpec(
            lower_q=float(section.get("winsor_low_q", 0.005)),
            upper_q=float(section.get("winsor_high_q", 0.995)),
        )
        inputs = ScalingInputSpec(
            impute_missing_to_zero=bool(section.get("impute_missing_to_zero", True)),
            feature_names=section.get("feature_names"),
            append_mask_as_features=bool(
                section.get("append_mask_as_features", False)
            ),
            append_exogenous_mask_as_features=bool(
                section.get("append_exogenous_mask_as_features", False)
            ),
        )
        scaling = ScalingSpec(
            mad_eps=float(section.get("mad_eps", 0.0)),
            breakout_var_floor=float(section.get("breakout_var_floor", 1.0e-3)),
            scale_floor=float(section.get("scale_floor", 1.0e-3)),
            guardrail=guardrail,
            clip=clip,
            winsor=winsor,
            inputs=inputs,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid preprocessing configuration in {config_path}",
            context={"section": "preprocessing"},
        ) from exc
    return PreprocessSpec(cleaning=cleaning, scaling=scaling)


def _normalize_relative_output_path(
    value: object,
    config_path: Path,
    field_name: str,
    root_name: str = "SIMULATION_SOURCE",
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            f"data.{field_name} must be a string in {config_path}"
        )
    label = value.strip()
    if not label:
        raise ConfigError(
            f"data.{field_name} must not be empty in {config_path}"
        )
    if "\\" in label:
        raise ConfigError(
            f"data.{field_name} must use forward slashes in {config_path}"
        )
    path = PurePosixPath(label)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ConfigError(
            f"data.{field_name} must be a relative path under {root_name} in {config_path}"
        )
    return label


def _build_data_config(
    raw: Mapping[str, Any], config_path: Path
) -> DataConfig:
    section = raw.get("data")
    if not isinstance(section, Mapping):
        raise ConfigError(f"data must be a mapping in {config_path}")
    if "paths" in section:
        raise ConfigError(
            f"data.paths is no longer supported in {config_path}"
        )
    if "selection" in section:
        raise ConfigError(
            f"data.selection is no longer supported in {config_path}"
        )
    dataset_params = section.get("dataset_params", {})
    if dataset_params is None:
        dataset_params = {}
    if not isinstance(dataset_params, Mapping):
        raise ConfigError(
            f"data.dataset_params must be a mapping in {config_path}"
        )
    if "version_label" in dataset_params:
        raise ConfigError(
            f"data.dataset_params.version_label is no longer supported in {config_path}"
        )
    simulation_output_path = _normalize_relative_output_path(
        section.get("simulation_output_path"),
        config_path,
        "simulation_output_path",
    )
    portfolio_output_path = _normalize_relative_output_path(
        section.get("portfolio_output_path"),
        config_path,
        "portfolio_output_path",
    )
    posterior_output_path = _normalize_relative_output_path(
        section.get("posterior_output_path"),
        config_path,
        "posterior_output_path",
        root_name="POSTERIOR_SIGNAL_SOURCE",
    )
    try:
        return DataConfig(
            dataset_params=dataset_params,
            simulation_output_path=simulation_output_path,
            portfolio_output_path=portfolio_output_path,
            posterior_output_path=posterior_output_path,
        )
    except TypeError as exc:
        raise ConfigError(
            f"Invalid data configuration in {config_path}",
            context={"section": "data"},
        ) from exc




def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read config file {config_path}") from exc
    try:
        raw_config: Any = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in {config_path}",
            context={"path": str(config_path)},
        ) from exc
    if not isinstance(raw_config, Mapping):
        raise ConfigError(f"Config file must contain a mapping: {config_path}")
    return raw_config
