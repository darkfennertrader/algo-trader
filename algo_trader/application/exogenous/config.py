from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from algo_trader.domain import ConfigError

_DIR_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_VALID_UNITS = {
    "lin",
    "chg",
    "ch1",
    "pch",
    "pc1",
    "pca",
    "cch",
    "cca",
    "log",
}
_VALID_FREQUENCIES = {
    "d",
    "w",
    "bw",
    "m",
    "q",
    "sa",
    "a",
    "wef",
    "weth",
    "wew",
    "wetu",
    "wem",
    "wesu",
    "wesa",
    "bwew",
    "bwem",
}
_VALID_AGGREGATIONS = {"avg", "sum", "eop"}


@dataclass(frozen=True)
class FredSeriesConfig:
    series_id: str
    dir_name: str
    units: str | None
    frequency: str | None
    aggregation_method: str | None


@dataclass(frozen=True)
class FredRequestConfig:
    provider: str
    start_date: str
    end_date: str
    series: Sequence[FredSeriesConfig]
    config_path: Path

    @classmethod
    def load(cls, config_path: Path) -> "FredRequestConfig":
        mapping = _load_yaml_mapping(config_path)
        provider = _normalize_provider(mapping, config_path)
        start_date, end_date = _load_window(mapping, config_path)
        series = _load_series(mapping, config_path)
        return cls(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            series=series,
            config_path=config_path,
        )


def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"FRED config not found at {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded: Any = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML content in {config_path}") from exc
    raw_config = loaded if loaded is not None else {}
    if not isinstance(raw_config, Mapping):
        raise ConfigError(
            f"FRED config must be a mapping in {config_path}"
        )
    return raw_config


def _normalize_provider(
    mapping: Mapping[str, Any], config_path: Path
) -> str:
    provider = str(mapping.get("provider", "fred")).strip().lower()
    if provider != "fred":
        raise ConfigError(
            f"Unsupported provider '{provider}' in {config_path}",
            context={"provider": provider},
        )
    return provider


def _load_window(
    mapping: Mapping[str, Any], config_path: Path
) -> tuple[str, str]:
    start_date = _require_non_empty(mapping.get("start_date"), "start_date")
    end_date = _require_non_empty(mapping.get("end_date"), "end_date")
    start_value = _parse_date(start_date, "start_date", config_path)
    end_value = _parse_date(end_date, "end_date", config_path)
    if start_value > end_value:
        raise ConfigError(
            f"start_date must be <= end_date in {config_path}",
            context={"start_date": start_date, "end_date": end_date},
        )
    return start_date, end_date


def _load_series(
    mapping: Mapping[str, Any], config_path: Path
) -> list[FredSeriesConfig]:
    raw_series = mapping.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        raise ConfigError(
            f"series must be a non-empty list in {config_path}"
        )
    parsed: list[FredSeriesConfig] = []
    for item in raw_series:
        if not isinstance(item, Mapping):
            raise ConfigError(
                f"series entries must be mappings in {config_path}"
            )
        parsed.append(_parse_series_entry(item, config_path))
    return parsed


def _parse_series_entry(
    entry: Mapping[str, Any], config_path: Path
) -> FredSeriesConfig:
    series_id = _require_non_empty(entry.get("id"), "series.id")
    dir_name = _require_non_empty(entry.get("dir_name"), "series.dir_name")
    _validate_dir_name(dir_name, config_path)
    units = _optional_value(entry.get("units"))
    if units is not None and units not in _VALID_UNITS:
        raise ConfigError(
            f"Invalid units '{units}' in {config_path}",
            context={"units": units},
        )
    frequency = _optional_value(entry.get("frequency"))
    if frequency is not None and frequency not in _VALID_FREQUENCIES:
        raise ConfigError(
            f"Invalid frequency '{frequency}' in {config_path}",
            context={"frequency": frequency},
        )
    aggregation = _optional_value(entry.get("aggregation_method"))
    if aggregation is not None and aggregation not in _VALID_AGGREGATIONS:
        raise ConfigError(
            f"Invalid aggregation_method '{aggregation}' in {config_path}",
            context={"aggregation_method": aggregation},
        )
    return FredSeriesConfig(
        series_id=series_id,
        dir_name=dir_name,
        units=units,
        frequency=frequency,
        aggregation_method=aggregation,
    )


def _optional_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def _require_non_empty(value: Any, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ConfigError(f"{field} is required")
    return text


def _parse_date(raw: str, field: str, config_path: Path) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid {field} '{raw}' in {config_path}; expected YYYY-MM-DD"
        ) from exc


def _validate_dir_name(dir_name: str, config_path: Path) -> None:
    normalized = dir_name.strip()
    if not normalized:
        raise ConfigError(
            f"Invalid dir_name '{dir_name}' in {config_path}",
            context={"dir_name": dir_name},
        )
    if normalized.startswith(("/", "\\")):
        raise ConfigError(
            f"Invalid dir_name '{dir_name}' in {config_path}",
            context={"dir_name": dir_name},
        )
    parts = normalized.split("/")
    if any(not part for part in parts):
        raise ConfigError(
            f"Invalid dir_name '{dir_name}' in {config_path}",
            context={"dir_name": dir_name},
        )
    for part in parts:
        if part in {".", ".."} or not _DIR_SEGMENT_PATTERN.match(part):
            raise ConfigError(
                f"Invalid dir_name '{dir_name}' in {config_path}",
                context={"dir_name": dir_name},
            )
