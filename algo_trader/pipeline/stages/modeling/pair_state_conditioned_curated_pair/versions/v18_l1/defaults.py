from __future__ import annotations

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    guide_default_params_dependence_followup,
    make_dependence_followup_model_defaults,
)

_AUXILIARY_DEFAULTS = {
    "enabled": True,
    "df": 8.0,
    "obs_weight": 0.05,
    "range_obs_weight": 0.14,
    "residual_obs_weight": 0.02,
    "state_window": 4,
    "mad_floor": 1e-4,
    "eps": 1e-6,
}

guide_default_params_v18_l1 = guide_default_params_dependence_followup
model_default_params_v18_l1 = make_dependence_followup_model_defaults(
    param_key="pair_state_conditioned_curated_pair",
    overrides=_AUXILIARY_DEFAULTS,
)

__all__ = [
    "guide_default_params_v18_l1",
    "model_default_params_v18_l1",
]
