from __future__ import annotations

from typing import Any, Mapping

from algo_trader.domain.simulation import PreprocessSpec


def build_cleaning_thresholds(spec: PreprocessSpec) -> Mapping[str, Any]:
    return {
        "min_usable_ratio": spec.cleaning.min_usable_ratio,
        "min_variance": spec.cleaning.min_variance,
        "max_abs_corr": spec.cleaning.max_abs_corr,
        "corr_subsample": spec.cleaning.corr_subsample,
    }
