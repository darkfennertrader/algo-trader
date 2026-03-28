from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_FACTOR_COUNT_DEFAULTS = {
    "global_factor_count": 2,
    "fx_broad_factor_count": 1,
    "fx_cross_factor_count": 1,
    "index_factor_count": 2,
    "commodity_factor_count": 1,
}

_GUIDE_PHI_DEFAULTS = {
    "phi_fx_broad": 0.95,
    "phi_fx_cross": 0.985,
    "phi_index": 0.97,
    "phi_commodity": 0.97,
}


def build_guide_defaults(
    factor_count_defaults: Mapping[str, int],
) -> dict[str, Any]:
    return {**factor_count_defaults, **_GUIDE_PHI_DEFAULTS}


def guide_default_params_v3_l2() -> dict[str, Any]:
    return build_guide_defaults(_FACTOR_COUNT_DEFAULTS)


def model_default_params_v3_l2() -> dict[str, Any]:
    return {
        "mean": {
            "alpha_scale": 0.03,
            "sigma_fx_scale": 0.03,
            "sigma_index_scale": 0.08,
            "sigma_commodity_scale": 0.05,
            "beta_scale": 0.05,
            "tau0_scale": 0.05,
        },
        "factors": {
            **_FACTOR_COUNT_DEFAULTS,
            "global_b_scale": 0.10,
            "fx_broad_b_scale": 0.10,
            "fx_cross_b_scale": 0.06,
            "index_b_scale": 0.10,
            "commodity_b_scale": 0.08,
        },
        "regime": {
            "fx_broad": {"phi": 0.95, "s_u_scale": 0.03},
            "fx_cross": {"phi": 0.985, "s_u_scale": 0.01},
            "index": {"phi": 0.97, "s_u_scale": 0.03},
            "commodity": {"phi": 0.97, "s_u_scale": 0.02},
            "eps": 1e-6,
        },
    }


def merge_nested_params(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(defaults))
    _update_nested(merged, overrides)
    return merged


def _update_nested(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _update_nested(current, value)
            continue
        target[key] = value


__all__ = [
    "build_guide_defaults",
    "guide_default_params_v3_l2",
    "merge_nested_params",
    "model_default_params_v3_l2",
]
