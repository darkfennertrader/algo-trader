from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import yaml

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    AllocationConfig,
    CleaningSpec,
    CostConfig,
    CPCVParams,
    CVLeakage,
    CVParams,
    CVWindow,
    DataConfig,
    DiagnosticsConfig,
    EvaluationSpec,
    FanChartsConfig,
    ModelConfig,
    ModelingSpec,
    OuterConfig,
    PredictiveConfig,
    PreprocessSpec,
    ScalingSpec,
    ScoringConfig,
    SimulationConfig,
    SimulationFlags,
    TrainingConfig,
    ModelSelectionBatching,
    ModelSelectionBootstrap,
    ModelSelectionComplexity,
    ModelSelectionConfig,
    ModelSelectionESBand,
    ModelSelectionTail,
    TuningAggregateConfig,
    TuningConfig,
    TuningRayConfig,
    TuningResourcesConfig,
    TuningParamSpec,
    TuningParamType,
    TuningTransform,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "simulation.yml"
)


def load_config(path: Path | None = None) -> SimulationConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml_mapping(config_path)
    return _build_config(raw, config_path)


def config_to_dict(config: SimulationConfig) -> dict[str, object]:
    return asdict(config)


def _build_config(
    raw: Mapping[str, Any], config_path: Path
) -> SimulationConfig:
    data = _build_data_config(raw, config_path)
    cv = _build_cv_params(raw, config_path)
    preprocessing = _build_preprocess_spec(raw, config_path)
    modeling = _build_modeling_spec(raw, config_path)
    evaluation = _build_evaluation_spec(raw, config_path)
    outer = _build_outer_config(raw, config_path)
    flags = _build_flags(raw)

    return SimulationConfig(
        data=data,
        cv=cv,
        preprocessing=preprocessing,
        modeling=modeling,
        evaluation=evaluation,
        outer=outer,
        flags=flags,
    )


def _build_flags(raw: Mapping[str, Any]) -> SimulationFlags:
    mode = raw.get("simulation_mode", "full")
    stop_after = raw.get("stop_after")
    return SimulationFlags(
        use_feature_names_for_scaling=bool(
            raw.get("use_feature_names_for_scaling", True)
        ),
        use_gpu=bool(raw.get("use_gpu", False)),
        simulation_mode=_normalize_simulation_mode(mode),
        stop_after=_normalize_stop_after(stop_after),
    )


SimulationMode = Literal["dry_run", "stub", "full"]
StopAfter = Literal["inputs", "cv", "inner", "outer", "results"] | None
TuningEngine = Literal["local", "ray"]
AggregateMethod = Literal["mean", "median", "mean_minus_std"]


def _normalize_simulation_mode(value: object) -> SimulationMode:
    raw = str(value).strip().lower()
    if raw == "dry_run":
        return "dry_run"
    if raw == "stub":
        return "stub"
    if raw == "full":
        return "full"
    raise ConfigError("simulation_mode must be dry_run, stub, or full")


def _normalize_stop_after(value: object) -> StopAfter:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw == "inputs":
        return "inputs"
    if raw == "cv":
        return "cv"
    if raw == "inner":
        return "inner"
    if raw == "outer":
        return "outer"
    if raw == "results":
        return "results"
    raise ConfigError(
        "stop_after must be inputs, cv, inner, outer, results, or null"
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


def _build_modeling_spec(
    raw: Mapping[str, Any], config_path: Path
) -> ModelingSpec:
    model = _build_section(raw, "model", ModelConfig, config_path)
    training = _build_section(raw, "training", TrainingConfig, config_path)
    tuning = _build_tuning_config(raw, config_path)
    return ModelingSpec(model=model, training=training, tuning=tuning)


def _build_tuning_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningConfig:
    section = raw.get("tuning")
    if not isinstance(section, Mapping):
        raise ConfigError(f"tuning must be a mapping in {config_path}")
    try:
        space = _build_tuning_space(
            section.get("space", None), config_path
        )
        aggregate = _build_tuning_aggregate_config(
            section.get("aggregate", {}), config_path
        )
        resources = _build_tuning_resources_config(
            section.get("resources", {}), config_path
        )
        ray = _build_tuning_ray_config(
            section.get("ray", {}), resources, config_path
        )
        tuning = TuningConfig(
            space=space,
            num_samples=int(section.get("num_samples", 1)),
            seed=int(section.get("seed", 0)),
            kwargs=section.get("kwargs", {}),
            engine=section.get("engine", "local"),
            aggregate=aggregate,
            ray=ray,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid tuning configuration in {config_path}",
            context={"section": "tuning"},
        ) from exc
    return _normalize_tuning_config(tuning, config_path)


def _normalize_tuning_config(
    tuning: TuningConfig, config_path: Path
) -> TuningConfig:
    if tuning.num_samples <= 0:
        raise ConfigError(
            f"tuning.num_samples must be positive in {config_path}"
        )
    if tuning.seed < 0:
        raise ConfigError(
            f"tuning.seed must be non-negative in {config_path}"
        )
    return replace(
        tuning,
        engine=_normalize_tuning_engine(tuning.engine),
        aggregate=_normalize_aggregate_config(
            tuning.aggregate, config_path
        ),
        ray=_normalize_tuning_ray_config(tuning.ray, config_path),
    )


def _build_tuning_space(
    raw: object, config_path: Path
) -> tuple[TuningParamSpec, ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, list):
        raise ConfigError(f"tuning.space must be a list in {config_path}")
    specs: list[TuningParamSpec] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConfigError(
                f"tuning.space[{idx}] must be a mapping in {config_path}"
            )
        spec = _build_tuning_param_spec(
            item, config_path=config_path, index=idx
        )
        if spec.path in seen:
            raise ConfigError(
                f"Duplicate tuning.space path '{spec.path}' in {config_path}"
            )
        seen.add(spec.path)
        specs.append(spec)
    return tuple(specs)


def _build_tuning_param_spec(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
    index: int,
) -> TuningParamSpec:
    path = _require_string(
        raw.get("path"),
        field=f"tuning.space[{index}].path",
        config_path=config_path,
    )
    param_type = _normalize_param_type(
        raw.get("type"),
        field=f"tuning.space[{index}].type",
    )
    transform = _normalize_transform(
        raw.get("transform"),
        field=f"tuning.space[{index}].transform",
        config_path=config_path,
    )
    when = _parse_when(
        raw.get("when"),
        field=f"tuning.space[{index}].when",
        config_path=config_path,
    )
    bounds = raw.get("bounds")
    values = raw.get("values")
    if param_type in {"float", "int"}:
        if values is not None:
            raise ConfigError(
                f"tuning.space[{index}].values is not allowed for {param_type} params in {config_path}"
            )
        bounds_tuple = _parse_bounds(
            bounds,
            field=f"tuning.space[{index}].bounds",
            config_path=config_path,
        )
        _validate_continuous_transform(
            transform=transform,
            bounds=bounds_tuple,
            field=f"tuning.space[{index}].transform",
            config_path=config_path,
        )
        return TuningParamSpec(
            path=path,
            param_type=param_type,
            bounds=bounds_tuple,
            values=None,
            transform=transform,
            when=when,
        )
    values_tuple = _parse_values(
        values,
        field=f"tuning.space[{index}].values",
        config_path=config_path,
    )
    if bounds is not None:
        raise ConfigError(
            f"tuning.space[{index}].bounds is not allowed for {param_type} params in {config_path}"
        )
    if transform != "none":
        raise ConfigError(
            f"tuning.space[{index}].transform must be 'none' for {param_type} params in {config_path}"
        )
    if param_type == "bool" and not all(
        isinstance(item, bool) for item in values_tuple
    ):
        raise ConfigError(
            f"tuning.space[{index}].values must be booleans in {config_path}"
        )
    return TuningParamSpec(
        path=path,
        param_type=param_type,
        bounds=None,
        values=values_tuple,
        transform=transform,
        when=when,
    )


def _parse_bounds(
    raw: object, *, field: str, config_path: Path
) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError(
            f"{field} must be a 2-item list in {config_path}"
        )
    try:
        min_value = float(raw[0])
        max_value = float(raw[1])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be numeric in {config_path}") from exc
    if min_value >= max_value:
        raise ConfigError(
            f"{field} min must be less than max in {config_path}"
        )
    return min_value, max_value


def _parse_values(
    raw: object, *, field: str, config_path: Path
) -> tuple[Any, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            f"{field} must be a non-empty list in {config_path}"
        )
    return tuple(raw)


def _parse_when(
    raw: object, *, field: str, config_path: Path
) -> Mapping[str, tuple[Any, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{field} must be a mapping in {config_path}")
    when: dict[str, tuple[Any, ...]] = {}
    for key, value in raw.items():
        name = _require_string(
            key, field=f"{field} key", config_path=config_path
        )
        allowed = _normalize_when_values(
            value, field=f"{field}.{name}", config_path=config_path
        )
        when[name] = allowed
    return when


def _normalize_when_values(
    raw: object, *, field: str, config_path: Path
) -> tuple[Any, ...]:
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ConfigError(
                f"{field} must not be empty in {config_path}"
            )
        return tuple(raw)
    if raw is None:
        raise ConfigError(f"{field} must not be null in {config_path}")
    return (raw,)


def _normalize_param_type(
    value: object, *, field: str
) -> TuningParamType:
    raw = str(value).strip().lower()
    if raw in {"float", "int", "categorical", "bool"}:
        return cast(TuningParamType, raw)
    raise ConfigError(f"{field} must be float, int, categorical, or bool")


def _normalize_transform(
    value: object, *, field: str, config_path: Path
) -> TuningTransform:
    if value is None:
        raise ConfigError(f"{field} is required in {config_path}")
    raw = str(value).strip().lower()
    if raw in {"linear", "log", "log10", "none"}:
        return cast(TuningTransform, raw)
    raise ConfigError(
        f"{field} must be linear, log, log10, or none in {config_path}"
    )


def _validate_continuous_transform(
    *,
    transform: str,
    bounds: tuple[float, float],
    field: str,
    config_path: Path,
) -> None:
    if transform == "none":
        raise ConfigError(
            f"{field} must not be 'none' for continuous params in {config_path}"
        )
    if transform in {"log", "log10"} and bounds[0] <= 0:
        raise ConfigError(
            f"{field} requires positive bounds in {config_path}"
        )


def _build_tuning_aggregate_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningAggregateConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"tuning.aggregate must be a mapping in {config_path}"
        )
    return TuningAggregateConfig(
        method=raw.get("method", "mean"),
        penalty=float(raw.get("penalty", 0.5)),
    )


def _normalize_aggregate_config(
    aggregate: TuningAggregateConfig, config_path: Path
) -> TuningAggregateConfig:
    if aggregate.penalty < 0:
        raise ConfigError(
            f"tuning.aggregate.penalty must be >= 0 in {config_path}"
        )
    return replace(
        aggregate,
        method=_normalize_aggregate_method(aggregate.method),
    )


def _build_tuning_resources_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningResourcesConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"tuning.resources must be a mapping in {config_path}"
        )
    cpu = raw.get("cpu")
    gpu = raw.get("gpu")
    return TuningResourcesConfig(
        cpu=float(cpu) if cpu is not None else None,
        gpu=float(gpu) if gpu is not None else None,
    )


def _normalize_tuning_resources_config(
    resources: TuningResourcesConfig, config_path: Path
) -> TuningResourcesConfig:
    if resources.cpu is not None and resources.cpu <= 0:
        raise ConfigError(
            f"tuning.resources.cpu must be > 0 in {config_path}"
        )
    if resources.gpu is not None and resources.gpu < 0:
        raise ConfigError(
            f"tuning.resources.gpu must be >= 0 in {config_path}"
        )
    return resources


def _build_tuning_ray_config(
    raw: Mapping[str, Any],
    resources: TuningResourcesConfig,
    config_path: Path,
) -> TuningRayConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"tuning.ray must be a mapping in {config_path}")
    raw_resources = raw.get("resources")
    if raw_resources is not None:
        if not isinstance(raw_resources, Mapping):
            raise ConfigError(
                f"tuning.ray.resources must be a mapping in {config_path}"
            )
        resources = _build_tuning_resources_config(
            raw_resources, config_path
        )
    address = raw.get("address")
    if address is not None:
        address = str(address).strip()
    return TuningRayConfig(address=address or None, resources=resources)


def _normalize_tuning_ray_config(
    ray: TuningRayConfig,
    config_path: Path,
) -> TuningRayConfig:
    resources = _normalize_tuning_resources_config(
        ray.resources, config_path
    )
    if ray.address is None:
        return replace(ray, resources=resources)
    normalized = ray.address.strip()
    if not normalized:
        return TuningRayConfig(address=None, resources=resources)
    return TuningRayConfig(address=normalized, resources=resources)


def _normalize_tuning_engine(value: object) -> TuningEngine:
    raw = str(value).strip().lower()
    if raw == "local":
        return "local"
    if raw == "ray":
        return "ray"
    raise ConfigError("tuning.engine must be local or ray")


def _normalize_aggregate_method(value: object) -> AggregateMethod:
    raw = str(value).strip().lower()
    if raw == "mean":
        return "mean"
    if raw == "median":
        return "median"
    if raw in {"mean_minus_std", "mean-std", "mean_std"}:
        return "mean_minus_std"
    raise ConfigError(
        "tuning.aggregate must be mean, median, or mean_minus_std"
    )


def _require_string(
    value: object, *, field: str, config_path: Path
) -> str:
    if value is None:
        raise ConfigError(f"{field} is required in {config_path}")
    raw = str(value).strip()
    if not raw:
        raise ConfigError(f"{field} must be non-empty in {config_path}")
    return raw


def _build_evaluation_spec(
    raw: Mapping[str, Any], config_path: Path
) -> EvaluationSpec:
    scoring = _build_section(raw, "scoring", ScoringConfig, config_path)
    predictive = _build_section(raw, "predictive", PredictiveConfig, config_path)
    allocation = _build_section(raw, "allocation", AllocationConfig, config_path)
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
    es_band = _build_model_selection_es_band(section, config_path)
    bootstrap = _build_model_selection_bootstrap(section, config_path)
    tail = _build_model_selection_tail(section, config_path)
    batching = _build_model_selection_batching(section, config_path)
    complexity = _build_model_selection_complexity(section, config_path)
    return ModelSelectionConfig(
        enable=enable,
        phase_name=phase_name,
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
    return DiagnosticsConfig(fan_charts=fan_charts)


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
    return FanChartsConfig(
        enable=enable,
        assets_mode=assets_mode,
        assets=assets,
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
    method = str(raw_complexity.get("method", "random")).strip().lower()
    if method != "random":
        raise ConfigError(
            f"model_selection.complexity.method must be 'random' in {config_path}"
        )
    seed = int(raw_complexity.get("seed", 123))
    return ModelSelectionComplexity(method="random", seed=seed)


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
        scaling = ScalingSpec(
            mad_eps=float(section.get("mad_eps", 0.0)),
            impute_missing_to_zero=bool(section.get("impute_missing_to_zero", True)),
            feature_names=section.get("feature_names"),
            append_mask_as_features=bool(
                section.get("append_mask_as_features", False)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid preprocessing configuration in {config_path}",
            context={"section": "preprocessing"},
        ) from exc
    return PreprocessSpec(cleaning=cleaning, scaling=scaling)


def _normalize_simulation_output_path(
    value: object, config_path: Path
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            f"data.simulation_output_path must be a string in {config_path}"
        )
    label = value.strip()
    if not label:
        raise ConfigError(
            f"data.simulation_output_path must not be empty in {config_path}"
        )
    if Path(label).name != label or "/" in label or "\\" in label:
        raise ConfigError(
            f"data.simulation_output_path must be a single directory name in {config_path}"
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
    simulation_output_path = _normalize_simulation_output_path(
        section.get("simulation_output_path"), config_path
    )
    try:
        return DataConfig(
            dataset_params=dataset_params,
            simulation_output_path=simulation_output_path,
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
