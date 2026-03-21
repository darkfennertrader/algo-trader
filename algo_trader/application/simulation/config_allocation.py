from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import AllocationConfig

from .config_utils import coerce_mapping


def _build_allocation_config(
    raw: Mapping[str, Any], config_path: Path
) -> AllocationConfig:
    section = raw.get("allocation")
    if not isinstance(section, Mapping):
        raise ConfigError(f"allocation must be a mapping in {config_path}")
    spec = coerce_mapping(
        section.get("spec"),
        field="allocation.spec",
        config_path=config_path,
    )
    _validate_allocation_spec(spec, config_path)
    return AllocationConfig(spec=dict(spec))


def _validate_allocation_spec(
    spec: Mapping[str, Any], config_path: Path
) -> None:
    method = str(spec.get("method", "equal_weight")).strip().lower()
    valid_methods = {
        "equal_weight",
        "random",
        "de_risked",
        "skfolio_risk_budgeting",
    }
    if method not in valid_methods:
        raise ConfigError(
            f"allocation.spec.method must be equal_weight, random, "
            f"de_risked, or skfolio_risk_budgeting in {config_path}"
        )
    portfolio_style = str(
        spec.get("portfolio_style", "long_only")
    ).strip().lower()
    valid_styles = {
        "long_only",
        "long_short_bounded_net",
        "factor_neutral_long_short",
    }
    if portfolio_style not in valid_styles:
        raise ConfigError(
            f"allocation.spec.portfolio_style must be long_only, "
            f"long_short_bounded_net, or factor_neutral_long_short "
            f"in {config_path}"
        )
    if method == "skfolio_risk_budgeting" and portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=skfolio_risk_budgeting currently supports "
            f"only allocation.spec.portfolio_style=long_only in {config_path}"
        )
    gross_exposure = spec.get("gross_exposure")
    if gross_exposure is not None:
        gross_value = float(gross_exposure)
        if gross_value < 0.0:
            raise ConfigError(
                f"allocation.spec.gross_exposure must be >= 0 in {config_path}"
            )
        if gross_value == 0.0 and method != "de_risked":
            raise ConfigError(
                f"allocation.spec.gross_exposure must be > 0 unless "
                f"allocation.spec.method=de_risked in {config_path}"
            )
    random_seed = spec.get("random_seed")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise ConfigError(
            f"allocation.spec.random_seed must be an integer in {config_path}"
        )
    if method == "de_risked" and portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=de_risked currently supports only "
            f"allocation.spec.portfolio_style=long_only in {config_path}"
        )


__all__ = ["_build_allocation_config"]
