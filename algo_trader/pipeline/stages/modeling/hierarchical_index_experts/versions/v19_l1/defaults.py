from __future__ import annotations

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    guide_default_params_dependence_followup,
    make_dependence_followup_model_defaults,
)

_AUXILIARY_DEFAULTS = {
    "enabled": True,
    "df": 8.0,
    "obs_weight": 0.06,
    "broad_obs_weight": 0.08,
    "anchor_pair_obs_weight": 0.16,
    "residual_obs_weight": 0.03,
    "mad_floor": 1e-4,
    "eps": 1e-6,
}

guide_default_params_v19_l1 = guide_default_params_dependence_followup
model_default_params_v19_l1 = make_dependence_followup_model_defaults(
    param_key="hierarchical_index_experts",
    overrides=_AUXILIARY_DEFAULTS,
)

__all__ = [
    "guide_default_params_v19_l1",
    "model_default_params_v19_l1",
]
