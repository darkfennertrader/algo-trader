from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from algo_trader.domain import ConfigError

_DIR_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
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
_VALID_PRIORITIES = {"core", "optional"}
_VALID_FUTURE_ROLES = {"global", "asset", "both"}
_VALID_CHANNELS = {"mean"}
_VALID_CALENDAR_SOURCES = {"data_lake_returns"}
_VALID_CALENDAR_FREQUENCIES = {"weekly"}
_VALID_FILL_METHODS = {"forward_fill_only"}
_VALID_CORE_POLICIES = {"fail"}
_VALID_OPTIONAL_POLICIES = {"drop"}


@dataclass(frozen=True)
class FredCleaningOutputConfig:
    subdir: str
    cleaned_csv: str
    metadata_json: str
    tensor_pt: str


@dataclass(frozen=True)
class FredCoveragePolicyConfig:
    require_no_missing_after_cleaning: bool
    core_series_policy: str
    optional_series_policy: str


@dataclass(frozen=True)
class FredFillPolicyConfig:
    method: str
    allow_backfill: bool
    weekly_max_ffill_weeks: int
    monthly_max_ffill_weeks: int


@dataclass(frozen=True)
class FredLagPolicyConfig:
    default_weekly_release_lag_weeks: int
    default_monthly_release_lag_weeks: int


@dataclass(frozen=True)
class FredCleaningConfig:
    calendar_source: str
    calendar_frequency: str
    calendar_index_file: str
    output: FredCleaningOutputConfig
    coverage_policy: FredCoveragePolicyConfig
    fill_policy: FredFillPolicyConfig
    lag_policy: FredLagPolicyConfig


@dataclass(frozen=True)
class FredFamilyConfig:
    key: str
    priority: str
    future_role: str
    channel: str
    description: str | None


@dataclass(frozen=True)
class FredSeriesMetadataConfig:
    family_key: str | None = None
    alias: str | None = None
    currency: str | None = None
    priority: str = "core"
    future_role: str | None = None
    release_lag_weeks: int | None = None


@dataclass(frozen=True)
class FredSeriesConfig:
    series_id: str
    dir_name: str
    units: str | None
    frequency: str | None
    aggregation_method: str | None
    metadata: FredSeriesMetadataConfig = field(
        default_factory=FredSeriesMetadataConfig
    )

    @property
    def family_key(self) -> str | None:
        return self.metadata.family_key

    @property
    def alias(self) -> str | None:
        return self.metadata.alias

    @property
    def currency(self) -> str | None:
        return self.metadata.currency

    @property
    def priority(self) -> str:
        return self.metadata.priority

    @property
    def future_role(self) -> str | None:
        return self.metadata.future_role

    @property
    def release_lag_weeks(self) -> int | None:
        return self.metadata.release_lag_weeks


@dataclass(frozen=True)
class FredRequestConfig:
    provider: str
    start_date: str
    end_date: str
    series: Sequence[FredSeriesConfig]
    config_path: Path
    cleaning: FredCleaningConfig
    families: Sequence[FredFamilyConfig]

    @classmethod
    def load(cls, config_path: Path) -> "FredRequestConfig":
        mapping = _load_yaml_mapping(config_path)
        provider = _normalize_provider(mapping, config_path)
        start_date, end_date = _load_window(mapping, config_path)
        cleaning = _load_cleaning(mapping, config_path)
        families = _load_families(mapping, config_path)
        family_map = {family.key: family for family in families}
        series = _load_series(
            mapping,
            config_path=config_path,
            family_map=family_map,
            cleaning=cleaning,
        )
        return cls(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            series=series,
            config_path=config_path,
            cleaning=cleaning,
            families=families,
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


def _load_cleaning(
    mapping: Mapping[str, Any], config_path: Path
) -> FredCleaningConfig:
    raw = _coerce_mapping(mapping.get("cleaning"), label="cleaning")
    output = _load_cleaning_output(raw.get("output"), config_path)
    coverage = _load_coverage_policy(
        raw.get("coverage_policy"), config_path
    )
    fill = _load_fill_policy(raw.get("fill_policy"), config_path)
    lag = _load_lag_policy(raw.get("lag_policy"), config_path)
    calendar_source = _normalize_choice(
        raw.get("calendar_source", "data_lake_returns"),
        field="cleaning.calendar_source",
        config_path=config_path,
        valid=_VALID_CALENDAR_SOURCES,
    )
    calendar_frequency = _normalize_choice(
        raw.get("calendar_frequency", "weekly"),
        field="cleaning.calendar_frequency",
        config_path=config_path,
        valid=_VALID_CALENDAR_FREQUENCIES,
    )
    calendar_index_file = _require_filename(
        raw.get("calendar_index_file", "returns.csv"),
        field="cleaning.calendar_index_file",
        config_path=config_path,
    )
    return FredCleaningConfig(
        calendar_source=calendar_source,
        calendar_frequency=calendar_frequency,
        calendar_index_file=calendar_index_file,
        output=output,
        coverage_policy=coverage,
        fill_policy=fill,
        lag_policy=lag,
    )


def _load_cleaning_output(
    value: object, config_path: Path
) -> FredCleaningOutputConfig:
    raw = _coerce_mapping(value, label="cleaning.output")
    subdir = _require_dir_name(
        raw.get("subdir", "exogenous"),
        field="cleaning.output.subdir",
        config_path=config_path,
    )
    cleaned_csv = _require_filename(
        raw.get("cleaned_csv", "exogenous_cleaned.csv"),
        field="cleaning.output.cleaned_csv",
        config_path=config_path,
    )
    metadata_json = _require_filename(
        raw.get("metadata_json", "exogenous_metadata.json"),
        field="cleaning.output.metadata_json",
        config_path=config_path,
    )
    tensor_pt = _require_filename(
        raw.get("tensor_pt", "exogenous_tensor.pt"),
        field="cleaning.output.tensor_pt",
        config_path=config_path,
    )
    return FredCleaningOutputConfig(
        subdir=subdir,
        cleaned_csv=cleaned_csv,
        metadata_json=metadata_json,
        tensor_pt=tensor_pt,
    )


def _load_coverage_policy(
    value: object, config_path: Path
) -> FredCoveragePolicyConfig:
    raw = _coerce_mapping(value, label="cleaning.coverage_policy")
    require_no_missing = _coerce_bool(
        raw.get("require_no_missing_after_cleaning", True),
        field="cleaning.coverage_policy.require_no_missing_after_cleaning",
        config_path=config_path,
    )
    core_policy = _normalize_choice(
        raw.get("core_series_policy", "fail"),
        field="cleaning.coverage_policy.core_series_policy",
        config_path=config_path,
        valid=_VALID_CORE_POLICIES,
    )
    optional_policy = _normalize_choice(
        raw.get("optional_series_policy", "drop"),
        field="cleaning.coverage_policy.optional_series_policy",
        config_path=config_path,
        valid=_VALID_OPTIONAL_POLICIES,
    )
    return FredCoveragePolicyConfig(
        require_no_missing_after_cleaning=require_no_missing,
        core_series_policy=core_policy,
        optional_series_policy=optional_policy,
    )


def _load_fill_policy(
    value: object, config_path: Path
) -> FredFillPolicyConfig:
    raw = _coerce_mapping(value, label="cleaning.fill_policy")
    method = _normalize_choice(
        raw.get("method", "forward_fill_only"),
        field="cleaning.fill_policy.method",
        config_path=config_path,
        valid=_VALID_FILL_METHODS,
    )
    allow_backfill = _coerce_bool(
        raw.get("allow_backfill", False),
        field="cleaning.fill_policy.allow_backfill",
        config_path=config_path,
    )
    weekly_limit = _coerce_non_negative_int(
        raw.get("weekly_max_ffill_weeks", 2),
        field="cleaning.fill_policy.weekly_max_ffill_weeks",
        config_path=config_path,
    )
    monthly_limit = _coerce_non_negative_int(
        raw.get("monthly_max_ffill_weeks", 8),
        field="cleaning.fill_policy.monthly_max_ffill_weeks",
        config_path=config_path,
    )
    return FredFillPolicyConfig(
        method=method,
        allow_backfill=allow_backfill,
        weekly_max_ffill_weeks=weekly_limit,
        monthly_max_ffill_weeks=monthly_limit,
    )


def _load_lag_policy(
    value: object, config_path: Path
) -> FredLagPolicyConfig:
    raw = _coerce_mapping(value, label="cleaning.lag_policy")
    weekly_lag = _coerce_non_negative_int(
        raw.get("default_weekly_release_lag_weeks", 0),
        field="cleaning.lag_policy.default_weekly_release_lag_weeks",
        config_path=config_path,
    )
    monthly_lag = _coerce_non_negative_int(
        raw.get("default_monthly_release_lag_weeks", 1),
        field="cleaning.lag_policy.default_monthly_release_lag_weeks",
        config_path=config_path,
    )
    return FredLagPolicyConfig(
        default_weekly_release_lag_weeks=weekly_lag,
        default_monthly_release_lag_weeks=monthly_lag,
    )


def _load_families(
    mapping: Mapping[str, Any], config_path: Path
) -> list[FredFamilyConfig]:
    raw_families = mapping.get("families")
    if raw_families is None:
        return []
    if not isinstance(raw_families, list):
        raise ConfigError(
            f"families must be a list in {config_path}"
        )
    families: list[FredFamilyConfig] = []
    seen: set[str] = set()
    for item in raw_families:
        if not isinstance(item, Mapping):
            raise ConfigError(
                f"families entries must be mappings in {config_path}"
            )
        family = _parse_family_entry(item, config_path)
        if family.key in seen:
            raise ConfigError(
                f"Duplicate family key '{family.key}' in {config_path}",
                context={"family_key": family.key},
            )
        seen.add(family.key)
        families.append(family)
    return families


def _parse_family_entry(
    entry: Mapping[str, Any], config_path: Path
) -> FredFamilyConfig:
    key = _require_non_empty(entry.get("key"), "families.key")
    priority = _normalize_choice(
        entry.get("priority", "core"),
        field=f"families[{key}].priority",
        config_path=config_path,
        valid=_VALID_PRIORITIES,
    )
    future_role = _normalize_choice(
        entry.get("future_role", "global"),
        field=f"families[{key}].future_role",
        config_path=config_path,
        valid=_VALID_FUTURE_ROLES,
    )
    channel = _normalize_choice(
        entry.get("channel", "mean"),
        field=f"families[{key}].channel",
        config_path=config_path,
        valid=_VALID_CHANNELS,
    )
    description = _optional_text(entry.get("description"))
    return FredFamilyConfig(
        key=key,
        priority=priority,
        future_role=future_role,
        channel=channel,
        description=description,
    )


def _load_series(
    mapping: Mapping[str, Any],
    *,
    config_path: Path,
    family_map: Mapping[str, FredFamilyConfig],
    cleaning: FredCleaningConfig,
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
        parsed.append(
            _parse_series_entry(
                item,
                config_path=config_path,
                family_map=family_map,
                cleaning=cleaning,
            )
        )
    return parsed


def _parse_series_entry(
    entry: Mapping[str, Any],
    *,
    config_path: Path,
    family_map: Mapping[str, FredFamilyConfig],
    cleaning: FredCleaningConfig,
) -> FredSeriesConfig:
    series_id = _require_non_empty(entry.get("id"), "series.id")
    dir_name = _require_dir_name(
        entry.get("dir_name"),
        field="series.dir_name",
        config_path=config_path,
    )
    units = _optional_choice(
        entry.get("units"),
        field="series.units",
        config_path=config_path,
        valid=_VALID_UNITS,
    )
    frequency = _optional_choice(
        entry.get("frequency"),
        field="series.frequency",
        config_path=config_path,
        valid=_VALID_FREQUENCIES,
    )
    aggregation = _optional_choice(
        entry.get("aggregation_method"),
        field="series.aggregation_method",
        config_path=config_path,
        valid=_VALID_AGGREGATIONS,
    )
    family_key = _optional_text(entry.get("family_key"))
    family = _resolve_family(family_key, family_map, config_path)
    priority = _resolve_series_priority(entry, family, config_path)
    future_role = _resolve_future_role(entry, family, config_path)
    release_lag = _resolve_release_lag(
        entry,
        cleaning=cleaning,
        frequency=frequency,
        config_path=config_path,
    )
    metadata = FredSeriesMetadataConfig(
        family_key=family_key,
        alias=_optional_text(entry.get("alias")),
        currency=_optional_upper(entry.get("currency")),
        priority=priority,
        future_role=future_role,
        release_lag_weeks=release_lag,
    )
    return FredSeriesConfig(
        series_id=series_id,
        dir_name=dir_name,
        units=units,
        frequency=frequency,
        aggregation_method=aggregation,
        metadata=metadata,
    )


def _resolve_family(
    family_key: str | None,
    family_map: Mapping[str, FredFamilyConfig],
    config_path: Path,
) -> FredFamilyConfig | None:
    if family_key is None:
        return None
    family = family_map.get(family_key)
    if family is None:
        raise ConfigError(
            f"Unknown family_key '{family_key}' in {config_path}",
            context={"family_key": family_key},
        )
    return family


def _resolve_series_priority(
    entry: Mapping[str, Any],
    family: FredFamilyConfig | None,
    config_path: Path,
) -> str:
    default = family.priority if family is not None else "core"
    return _normalize_choice(
        entry.get("priority", default),
        field="series.priority",
        config_path=config_path,
        valid=_VALID_PRIORITIES,
    )


def _resolve_future_role(
    entry: Mapping[str, Any],
    family: FredFamilyConfig | None,
    config_path: Path,
) -> str | None:
    if "future_role" not in entry and family is None:
        return None
    default = family.future_role if family is not None else "global"
    return _normalize_choice(
        entry.get("future_role", default),
        field="series.future_role",
        config_path=config_path,
        valid=_VALID_FUTURE_ROLES,
    )


def _resolve_release_lag(
    entry: Mapping[str, Any],
    *,
    cleaning: FredCleaningConfig,
    frequency: str | None,
    config_path: Path,
) -> int | None:
    if "release_lag_weeks" in entry:
        return _coerce_non_negative_int(
            entry.get("release_lag_weeks"),
            field="series.release_lag_weeks",
            config_path=config_path,
        )
    if frequency is None:
        return None
    if frequency == "m":
        return cleaning.lag_policy.default_monthly_release_lag_weeks
    return cleaning.lag_policy.default_weekly_release_lag_weeks


def _coerce_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _optional_upper(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return text.upper()


def _normalize_choice(
    value: Any,
    *,
    field: str,
    config_path: Path,
    valid: set[str],
) -> str:
    text = _require_non_empty(value, field).lower()
    if text not in valid:
        raise ConfigError(
            f"Invalid {field} '{text}' in {config_path}",
            context={field: text},
        )
    return text


def _optional_choice(
    value: Any,
    *,
    field: str,
    config_path: Path,
    valid: set[str],
) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in valid:
        raise ConfigError(
            f"Invalid {field} '{normalized}' in {config_path}",
            context={field: normalized},
        )
    return normalized


def _coerce_bool(value: Any, *, field: str, config_path: Path) -> bool:
    if isinstance(value, bool):
        return value
    text = _optional_text(value)
    if text is None:
        raise ConfigError(f"{field} is required in {config_path}")
    normalized = text.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ConfigError(
        f"Invalid {field} '{text}' in {config_path}; expected true/false"
    )


def _coerce_non_negative_int(
    value: Any, *, field: str, config_path: Path
) -> int:
    text = _require_non_empty(value, field)
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid {field} '{text}' in {config_path}; expected int"
        ) from exc
    if parsed < 0:
        raise ConfigError(
            f"Invalid {field} '{text}' in {config_path}; expected >= 0"
        )
    return parsed


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


def _require_dir_name(
    value: Any, *, field: str, config_path: Path
) -> str:
    dir_name = _require_non_empty(value, field)
    _validate_dir_name(dir_name, config_path)
    return dir_name


def _require_filename(
    value: Any, *, field: str, config_path: Path
) -> str:
    filename = _require_non_empty(value, field)
    if "/" in filename or "\\" in filename or not _FILENAME_PATTERN.match(
        filename
    ):
        raise ConfigError(
            f"Invalid {field} '{filename}' in {config_path}",
            context={field: filename},
        )
    return filename


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
