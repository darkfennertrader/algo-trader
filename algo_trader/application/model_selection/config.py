from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from algo_trader.domain import ConfigError
from algo_trader.domain.model_selection import (
    CVConfig,
    CVGuards,
    CVSampling,
    DataConfig,
    DataPaths,
    DataSelection,
    ExperimentConfig,
    MetricConfig,
    ModelConfig,
    TrainingConfig,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "model_selection.yml"
)


def load_config(path: Path | None = None) -> ExperimentConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml_mapping(config_path)
    return _build_config(raw, config_path)


def config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    return asdict(config)


def _build_config(
    raw: Mapping[str, Any], config_path: Path
) -> ExperimentConfig:
    data = _build_data_config(raw, config_path)
    cv = _build_cv_config(raw, config_path)
    model = _build_section(raw, "model", ModelConfig, config_path)
    training = _build_section(raw, "training", TrainingConfig, config_path)
    metrics = _build_section(raw, "metrics", MetricConfig, config_path)
    use_gpu = bool(raw.get("use_gpu", False))
    return ExperimentConfig(
        data=data,
        cv=cv,
        model=model,
        training=training,
        metrics=metrics,
        use_gpu=use_gpu,
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


def _build_cv_config(
    raw: Mapping[str, Any], config_path: Path
) -> CVConfig:
    section = raw.get("cv")
    if not isinstance(section, Mapping):
        raise ConfigError(f"cv must be a mapping in {config_path}")
    sampling = section.get("sampling", {})
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, Mapping):
        raise ConfigError(f"cv.sampling must be a mapping in {config_path}")
    guards = section.get("guards", {})
    if guards is None:
        guards = {}
    if not isinstance(guards, Mapping):
        raise ConfigError(f"cv.guards must be a mapping in {config_path}")
    try:
        return CVConfig(
            cv_name=str(section.get("cv_name", "cpcv")),
            n_blocks=int(section.get("n_blocks", 0)),
            test_block_size=int(section.get("test_block_size", 0)),
            embargo_size=int(section.get("embargo_size", 0)),
            purge_size=int(section.get("purge_size", 0)),
            guards=CVGuards(
                min_train_size=(
                    int(guards["min_train_size"])
                    if guards.get("min_train_size") is not None
                    else None
                ),
                min_block_size=(
                    int(guards["min_block_size"])
                    if guards.get("min_block_size") is not None
                    else None
                ),
            ),
            sampling=CVSampling(**sampling),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid cv configuration in {config_path}",
            context={"section": "cv"},
        ) from exc


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
    selection = section.get("selection", {})
    if selection is None:
        selection = {}
    if not isinstance(selection, Mapping):
        raise ConfigError(
            f"data.selection must be a mapping in {config_path}"
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
            selection=DataSelection(**selection),
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
        raise ConfigError(
            f"Failed to read config file {config_path}"
        ) from exc
    try:
        raw_config: Any = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in {config_path}",
            context={"path": str(config_path)},
        ) from exc
    if not isinstance(raw_config, Mapping):
        raise ConfigError(
            f"Config file must contain a mapping: {config_path}"
        )
    return raw_config
