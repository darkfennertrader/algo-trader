from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from algo_trader.domain import ConfigError


def load_yaml_mapping(
    config_path: Path,
    *,
    missing_message: str,
    invalid_mapping_message: str,
) -> Mapping[str, Any]:
    if not config_path.exists():
        raise ConfigError(missing_message)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded: Any = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML content in {config_path}") from exc
    mapping = loaded if loaded is not None else {}
    if not isinstance(mapping, Mapping):
        raise ConfigError(invalid_mapping_message)
    return mapping


__all__ = ["load_yaml_mapping"]
