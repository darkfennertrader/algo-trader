from __future__ import annotations

from dataclasses import asdict
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
    DiagnosticsConfig,
    EvaluationSpec,
    GuardrailSpec,
    FanChartsConfig,
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
    TrainingSVIConfig,
    ModelSelectionBatching,
    ModelSelectionBootstrap,
    ModelSelectionComplexity,
    ModelSelectionConfig,
    ModelSelectionESBand,
    ModelSelectionTail,
    ModelPrebuildConfig,
)
from .config_tuning import build_tuning_config
from .config_utils import coerce_mapping, require_bool, require_string

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
        "params",
        "guide_params",
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
    prebuild = _build_model_prebuild(
        section.get("prebuild"), config_path
    )
    return ModelConfig(
        model_name=model_name,
        guide_name=guide_name,
        params=params,
        guide_params=guide_params,
        prebuild=prebuild,
    )


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
    extra_training = set(section) - {
        "svi",
        "target_normalization",
        "log_prob_scaling",
    }
    if extra_training:
        raise ConfigError(
            f"training contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra_training))},
        )
    raw_svi = section.get("svi")
    if not isinstance(raw_svi, Mapping):
        raise ConfigError(f"training.svi must be a mapping in {config_path}")
    extra_svi = set(raw_svi) - {
        "num_steps",
        "learning_rate",
        "tbptt_window_len",
        "tbptt_burn_in_len",
        "grad_accum_steps",
        "num_elbo_particles",
        "log_every",
    }
    if extra_svi:
        raise ConfigError(
            f"training.svi contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra_svi))},
        )
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
    try:
        svi = TrainingSVIConfig(
            num_steps=int(raw_svi.get("num_steps", 2_000)),
            learning_rate=float(raw_svi.get("learning_rate", 1e-3)),
            tbptt_window_len=(
                int(raw_svi["tbptt_window_len"])
                if raw_svi.get("tbptt_window_len") is not None
                else None
            ),
            tbptt_burn_in_len=int(raw_svi.get("tbptt_burn_in_len", 0)),
            grad_accum_steps=int(raw_svi.get("grad_accum_steps", 1)),
            num_elbo_particles=int(raw_svi.get("num_elbo_particles", 1)),
            log_every=(
                int(raw_svi["log_every"])
                if raw_svi.get("log_every") is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid training configuration in {config_path}",
            context={"section": "training.svi"},
        ) from exc
    return TrainingConfig(
        svi=svi,
        target_normalization=target_norm,
        log_prob_scaling=log_prob_scaling,
    )


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
