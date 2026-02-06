from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError
from ..horizons import HorizonSpec
from ..momentum import MomentumConfig, MomentumFeatureGroup
from ..mean_reversion import MeanReversionConfig, MeanReversionFeatureGroup
from ..protocols import FeatureGroup, FeatureInputs, FeatureOutput
from ..utils import normalize_feature_set, ordered_assets, reindex_asset_features
from ..volatility import VolatilityConfig, VolatilityFeatureGroup

DEFAULT_HORIZON_DAYS: tuple[int, ...] = (5, 20, 60, 130)
DEFAULT_EPSILON = 1e-8
SUPPORTED_FEATURES: tuple[str, ...] = (
    "cs_centered",
    "cs_rank",
)

_BASE_X_FEATURES: tuple[str, ...] = (
    "mom_4w",
    "mom_12w",
    "z_mom_4w",
    "z_mom_12w",
    "z_price_ema_12w",
    "z_price_ema_26w",
    "z_price_med_26w",
    "rev_1w",
    "shock_4w",
)

_BASE_Y_FEATURES: tuple[str, ...] = (
    "vol_cc_d_1w",
    "vol_cc_d_4w",
    "vol_cc_d_12w",
    "vol_cc_d_26w",
    "downside_vol_d_4w",
    "upside_vol_d_4w",
    "realized_skew_d_12w",
    "realized_kurt_d_12w",
    "tail_5p_sigma_ratio_12w",
    "jump_freq_4w",
)

_BASE_FEATURES: tuple[str, ...] = _BASE_X_FEATURES + _BASE_Y_FEATURES


@dataclass(frozen=True)
class CrossSectionalConfig:
    horizons: Sequence[HorizonSpec]
    eps: float = DEFAULT_EPSILON
    features: Sequence[str] | None = None


class CrossSectionalFeatureGroup(FeatureGroup):
    name = "cross_sectional"
    supported_features = SUPPORTED_FEATURES
    error_message = "Unknown cross-sectional features requested"

    def __init__(self, config: CrossSectionalConfig) -> None:
        self._config = config

    def compute(self, inputs: FeatureInputs) -> FeatureOutput:
        feature_set = normalize_feature_set(
            self._config.features,
            type(self).supported_features,
            error_message=type(self).error_message,
        )
        base_frame = _build_base_frame(inputs, config=self._config)
        if base_frame.empty:
            return FeatureOutput(frame=base_frame, feature_names=[])
        _require_base_features(base_frame, _BASE_FEATURES)
        assets = list(ordered_assets(base_frame))
        output_frame, feature_names = _build_cross_sectional_features(
            base_frame,
            assets=assets,
            feature_set=feature_set,
        )
        return FeatureOutput(frame=output_frame, feature_names=feature_names)


def _build_base_frame(
    inputs: FeatureInputs, *, config: CrossSectionalConfig
) -> pd.DataFrame:
    horizons = list(config.horizons)
    momentum = MomentumFeatureGroup(
        MomentumConfig(
            horizons=horizons,
            eps=config.eps,
            features=["momentum", "vol_scaled_momentum"],
        )
    )
    mean_reversion = MeanReversionFeatureGroup(
        MeanReversionConfig(
            horizons=horizons,
            eps=config.eps,
            features=["z_price_ema", "z_price_med", "rev", "shock"],
        )
    )
    volatility = VolatilityFeatureGroup(
        VolatilityConfig(
            horizons=horizons,
            eps=config.eps,
            features=[
                "vol_cc_d",
                "downside_vol_d_4w",
                "upside_vol_d_4w",
                "realized_skew_d_12w",
                "realized_kurt_d_12w",
                "tail_5p_sigma_ratio_12w",
                "jump_freq_4w",
            ],
        )
    )
    frames = [
        momentum.compute(inputs).frame,
        mean_reversion.compute(inputs).frame,
        volatility.compute(inputs).frame,
    ]
    combined = pd.concat(frames, axis=1)
    combined.columns = combined.columns.set_names(["asset", "feature"])
    return combined.loc[
        :, combined.columns.get_level_values(1).isin(_BASE_FEATURES)
    ]


def _require_base_features(
    frame: pd.DataFrame, required: Iterable[str]
) -> None:
    available = set(frame.columns.get_level_values(1))
    missing = sorted(set(required).difference(available))
    if missing:
        raise ConfigError(
            "Cross-sectional base features missing",
            context={"features": ",".join(missing)},
        )


def _build_cross_sectional_features(
    frame: pd.DataFrame,
    *,
    assets: list[str],
    feature_set: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    compute_centered = "cs_centered" in feature_set
    compute_rank = "cs_rank" in feature_set
    feature_names: list[str] = []
    frames: list[pd.DataFrame] = []
    for base_feature in _BASE_FEATURES:
        values = frame.xs(base_feature, axis=1, level=1, drop_level=True)
        if isinstance(values, pd.Series):
            values = values.to_frame()
        values = values.reindex(columns=pd.Index(assets, name="asset"))
        if compute_centered:
            centered = _cross_sectional_center(values)
            name = f"cs_centered_{base_feature}"
            feature_names.append(name)
            frames.append(_attach_feature(centered, name))
        if compute_rank:
            ranked = _cross_sectional_rank(values)
            name = f"cs_rank_{base_feature}"
            feature_names.append(name)
            frames.append(_attach_feature(ranked, name))
    combined = pd.concat(frames, axis=1)
    combined.columns = combined.columns.set_names(["asset", "feature"])
    combined = reindex_asset_features(combined, assets, feature_names)
    return combined, feature_names


def _cross_sectional_center(values: pd.DataFrame) -> pd.DataFrame:
    mean = values.mean(axis=1, skipna=True)
    return values.sub(mean, axis=0)


def _cross_sectional_rank(values: pd.DataFrame) -> pd.DataFrame:
    ranks = values.rank(axis=1, method="average", na_option="keep")
    counts = values.notna().sum(axis=1).replace(0, np.nan)
    return (ranks - 0.5).div(counts, axis=0)


def _attach_feature(values: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [values.columns, [feature_name]], names=["asset", "feature"]
    )
    return pd.DataFrame(
        values.to_numpy(dtype=float),
        index=values.index,
        columns=columns,
    )
