from __future__ import annotations
# pylint: disable=duplicate-code

from typing import Any

from algo_trader.pipeline.stages.modeling.multi_asset_block.v3_l2_defaults import (
    build_guide_defaults,
    merge_nested_params,
)

_FACTOR_COUNT_DEFAULTS_V10_L1 = {
    "global_factor_count": 1,
    "fx_broad_factor_count": 1,
    "fx_cross_factor_count": 1,
    "index_factor_count": 0,
    "index_static_factor_count": 0,
    "commodity_factor_count": 1,
}


def guide_default_params_v10_l1() -> dict[str, Any]:
    return {
        **build_guide_defaults(_FACTOR_COUNT_DEFAULTS_V10_L1),
        "index_group_enabled": False,
        "state_conditioned_measurement_enabled": True,
    }


def model_default_params_v10_l1() -> dict[str, Any]:
    return {
        "mean": {
            "alpha_scale": 0.03,
            "sigma_fx_scale": 0.03,
            "sigma_index_scale": 0.06,
            "sigma_commodity_scale": 0.04,
            "beta_scale": 0.05,
            "tau0_scale": 0.05,
        },
        "factors": {
            **_FACTOR_COUNT_DEFAULTS_V10_L1,
            "global_b_scale": 0.08,
            "fx_broad_b_scale": 0.10,
            "fx_cross_b_scale": 0.06,
            "index_b_scale": 0.0,
            "index_static_b_scale": 0.0,
            "commodity_b_scale": 0.08,
            "index_group_scale": 0.0,
        },
        "regime": {
            "fx_broad": {"phi": 0.95, "s_u_scale": 0.03},
            "fx_cross": {"phi": 0.985, "s_u_scale": 0.01},
            "index": {"phi": 0.97, "s_u_scale": 0.02},
            "commodity": {"phi": 0.97, "s_u_scale": 0.02},
            "eps": 1e-6,
        },
        "state_conditioned_measurement": {
            "enabled": True,
            "state_df": 20.0,
            "prior_scales": {
                "state_scale": 0.05,
                "loading_deviation": 0.04,
                "composite_loading_deviation": 0.008,
                "primitive_residual_scale": 0.80,
                "composite_residual_scale": 0.35,
                "residual_log_scale": 0.12,
                "composite_residual_log_scale": 0.05,
                "correlation_concentration": 30.0,
            },
            "gate": {
                "center": 0.50,
                "scale": 0.75,
            },
            "condition_prior_scales": {
                "bias": 1.0,
                "global_weight": 0.50,
                "index_weight": 0.50,
                "contrast_strength": 0.12,
                "composite_residual_strength": 0.10,
            },
            "eps": 1e-6,
        },
    }


__all__ = [
    "guide_default_params_v10_l1",
    "merge_nested_params",
    "model_default_params_v10_l1",
]
