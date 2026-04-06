from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    AllocationConfig,
    AllocationFamilyConfig,
)

from .allocation_common import (
    VALID_ALLOCATION_FAMILIES,
    VALID_PORTFOLIO_STYLES,
    optional_float_value,
)
from .config_utils import coerce_mapping, require_bool


def _build_allocation_config(
    raw: Mapping[str, Any], config_path: Path
) -> AllocationConfig:
    section = raw.get("allocation")
    if not isinstance(section, Mapping):
        raise ConfigError(f"allocation must be a mapping in {config_path}")
    if _uses_primary_schema(section):
        return _build_primary_allocation_config(section, config_path)
    return _build_legacy_allocation_config(section, config_path)


def _uses_primary_schema(section: Mapping[str, Any]) -> bool:
    return "primary" in section or "baselines" in section


def _build_primary_allocation_config(
    section: Mapping[str, Any], config_path: Path
) -> AllocationConfig:
    extra = set(section) - {"primary", "baselines"}
    if extra:
        raise ConfigError(
            f"allocation contains unknown keys in {config_path}",
            context={"keys": ", ".join(sorted(extra))},
        )
    primary = _build_family_config(
        section.get("primary"),
        field="allocation.primary",
        config_path=config_path,
    )
    baselines = _build_baselines(
        section.get("baselines"), config_path=config_path
    )
    return AllocationConfig(primary=primary, baselines=baselines)


def _build_legacy_allocation_config(
    section: Mapping[str, Any], config_path: Path
) -> AllocationConfig:
    spec = coerce_mapping(
        section.get("spec"),
        field="allocation.spec",
        config_path=config_path,
    )
    primary = _build_family_config(
        spec,
        field="allocation.spec",
        config_path=config_path,
    )
    return AllocationConfig(primary=primary)


def _build_baselines(
    raw: Any,
    *,
    config_path: Path,
) -> tuple[AllocationFamilyConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(
            f"allocation.baselines must be a list in {config_path}"
        )
    baselines: list[AllocationFamilyConfig] = []
    for index, item in enumerate(raw):
        baselines.append(
            _build_family_config(
                item,
                field=f"allocation.baselines[{index}]",
                config_path=config_path,
            )
        )
    return tuple(baselines)


def _build_family_config(
    raw: Any,
    *,
    field: str,
    config_path: Path,
) -> AllocationFamilyConfig:
    spec = coerce_mapping(raw, field=field, config_path=config_path)
    family = _resolve_family(spec, field=field, config_path=config_path)
    params = _extract_params(spec)
    _validate_family_params(
        family=family,
        params=params,
        field=field,
        config_path=config_path,
    )
    return AllocationFamilyConfig(family=family, params=params)


def _resolve_family(
    spec: Mapping[str, Any],
    *,
    field: str,
    config_path: Path,
) -> str:
    raw = spec.get("family", spec.get("method"))
    if raw is None:
        raise ConfigError(f"{field}.family is required in {config_path}")
    family = str(raw).strip().lower()
    if family not in VALID_ALLOCATION_FAMILIES:
        raise ConfigError(
            f"{field}.family must be one of {', '.join(VALID_ALLOCATION_FAMILIES)} "
            f"in {config_path}"
        )
    return family


def _extract_params(spec: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(spec)
    params.pop("family", None)
    params.pop("method", None)
    return params


def _validate_family_params(
    *,
    family: str,
    params: Mapping[str, Any],
    field: str,
    config_path: Path,
) -> None:
    if family == "long_only":
        _validate_long_only_params(
            params=params, field=field, config_path=config_path
        )
        return
    _validate_legacy_family_params(
        family=family,
        params=params,
        field=field,
        config_path=config_path,
    )


def _validate_long_only_params(
    *,
    params: Mapping[str, Any],
    field: str,
    config_path: Path,
) -> None:
    gross = _require_non_negative_float(
        params.get("gross_exposure", 1.0),
        field=f"{field}.gross_exposure",
        config_path=config_path,
    )
    min_weight = _optional_float(
        params.get("min_weight"),
        field=f"{field}.min_weight",
        config_path=config_path,
    )
    max_weight = _optional_float(
        params.get("max_weight"),
        field=f"{field}.max_weight",
        config_path=config_path,
    )
    if min_weight is not None and min_weight < 0.0:
        raise ConfigError(
            f"{field}.min_weight must be >= 0 in {config_path}"
        )
    if max_weight is not None and max_weight <= 0.0:
        raise ConfigError(f"{field}.max_weight must be > 0 in {config_path}")
    if min_weight is not None and max_weight is not None and min_weight > max_weight:
        raise ConfigError(
            f"{field}.min_weight must be <= max_weight in {config_path}"
        )
    _ = gross
    if "use_previous_weights" in params:
        require_bool(
            params.get("use_previous_weights"),
            field=f"{field}.use_previous_weights",
            config_path=config_path,
        )


def _validate_legacy_family_params(
    *,
    family: str,
    params: Mapping[str, Any],
    field: str,
    config_path: Path,
) -> None:
    portfolio_style = str(
        params.get("portfolio_style", "long_only")
    ).strip().lower()
    if portfolio_style not in VALID_PORTFOLIO_STYLES:
        raise ConfigError(
            f"{field}.portfolio_style must be long_only, "
            f"long_short_bounded_net, or factor_neutral_long_short "
            f"in {config_path}"
        )
    if family == "skfolio_risk_budgeting" and portfolio_style != "long_only":
        raise ConfigError(
            f"{field}.family=skfolio_risk_budgeting currently supports only "
            f"{field}.portfolio_style=long_only in {config_path}"
        )
    gross = params.get("gross_exposure")
    if gross is not None:
        gross_value = _require_non_negative_float(
            gross,
            field=f"{field}.gross_exposure",
            config_path=config_path,
        )
        if gross_value == 0.0 and family != "de_risked":
            raise ConfigError(
                f"{field}.gross_exposure must be > 0 unless "
                f"{field}.family=de_risked in {config_path}"
            )
    if "random_seed" in params:
        _require_int(
            params.get("random_seed"),
            field=f"{field}.random_seed",
            config_path=config_path,
        )
    if family == "de_risked" and portfolio_style != "long_only":
        raise ConfigError(
            f"{field}.family=de_risked currently supports only "
            f"{field}.portfolio_style=long_only in {config_path}"
        )
    if "use_previous_weights" in params:
        require_bool(
            params.get("use_previous_weights"),
            field=f"{field}.use_previous_weights",
            config_path=config_path,
        )
    if "min_weight" in params:
        _optional_float(
            params.get("min_weight"),
            field=f"{field}.min_weight",
            config_path=config_path,
        )
    if "max_weight" in params:
        _optional_float(
            params.get("max_weight"),
            field=f"{field}.max_weight",
            config_path=config_path,
        )
    if "transaction_costs" in params:
        _optional_float(
            params.get("transaction_costs"),
            field=f"{field}.transaction_costs",
            config_path=config_path,
        )


def _require_non_negative_float(
    raw: Any,
    *,
    field: str,
    config_path: Path,
) -> float:
    value = _optional_float(raw, field=field, config_path=config_path)
    if value is None:
        raise ConfigError(f"{field} is required in {config_path}")
    if value < 0.0:
        raise ConfigError(f"{field} must be >= 0 in {config_path}")
    return value


def _optional_float(
    raw: Any,
    *,
    field: str,
    config_path: Path,
) -> float | None:
    return optional_float_value(
        raw, location=f"{field} in {config_path}"
    )


def _require_int(
    raw: Any,
    *,
    field: str,
    config_path: Path,
) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{field} must be an integer in {config_path}")
    return int(raw)


__all__ = ["_build_allocation_config"]
