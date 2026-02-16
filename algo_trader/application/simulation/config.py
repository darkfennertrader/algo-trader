from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, Mapping

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
    DataPaths,
    EvaluationSpec,
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
    TuningAggregateConfig,
    TuningConfig,
    TuningRayConfig,
    TuningResourcesConfig,
    TuningSamplingConfig,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "model_selection.yml"
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
SamplingMethod = Literal["grid", "random", "sobol", "lhs"]
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
        sampling = _build_tuning_sampling_config(
            section.get("sampling", {}), config_path
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
            param_space=section.get("param_space", {}),
            num_samples=int(section.get("num_samples", 1)),
            kwargs=section.get("kwargs", {}),
            engine=section.get("engine", "local"),
            sampling=sampling,
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
    return replace(
        tuning,
        engine=_normalize_tuning_engine(tuning.engine),
        sampling=replace(
            tuning.sampling,
            method=_normalize_sampling_method(tuning.sampling.method),
        ),
        aggregate=_normalize_aggregate_config(
            tuning.aggregate, config_path
        ),
        ray=_normalize_tuning_ray_config(tuning.ray, config_path),
    )


def _build_tuning_sampling_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningSamplingConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"tuning.sampling must be a mapping in {config_path}")
    return TuningSamplingConfig(
        method=raw.get("method", "grid"),
        seed=int(raw.get("seed", 0)),
        pre_sampled_path=raw.get("pre_sampled_path"),
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


def _normalize_sampling_method(value: object) -> SamplingMethod:
    raw = str(value).strip().lower()
    if raw == "grid":
        return "grid"
    if raw == "random":
        return "random"
    if raw == "sobol":
        return "sobol"
    if raw == "lhs":
        return "lhs"
    raise ConfigError(
        "tuning.sampling_method must be grid, random, sobol, or lhs"
    )


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


def _build_evaluation_spec(
    raw: Mapping[str, Any], config_path: Path
) -> EvaluationSpec:
    scoring = _build_section(raw, "scoring", ScoringConfig, config_path)
    predictive = _build_section(raw, "predictive", PredictiveConfig, config_path)
    allocation = _build_section(raw, "allocation", AllocationConfig, config_path)
    cost = _build_section(raw, "cost", CostConfig, config_path)
    return EvaluationSpec(
        scoring=scoring,
        predictive=predictive,
        allocation=allocation,
        cost=cost,
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


def _build_data_config(
    raw: Mapping[str, Any], config_path: Path
) -> DataConfig:
    section = raw.get("data")
    if not isinstance(section, Mapping):
        raise ConfigError(f"data must be a mapping in {config_path}")
    paths = section.get("paths", {})
    if paths is None:
        paths = {}
    if not isinstance(paths, Mapping):
        raise ConfigError(f"data.paths must be a mapping in {config_path}")
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
    dataset_name = section.get("dataset_name", "")
    if not dataset_name:
        raise ConfigError(f"data.dataset_name is required in {config_path}")
    try:
        return DataConfig(
            dataset_name=str(dataset_name),
            paths=DataPaths(**paths),
            dataset_params=dataset_params,
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
