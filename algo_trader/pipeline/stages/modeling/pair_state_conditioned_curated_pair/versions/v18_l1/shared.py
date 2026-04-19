from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.config_support import coerce_mapping
from algo_trader.pipeline.stages.modeling.curated_pair_support import (
    make_curated_pair_seed_builder,
)
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    IndexRelativeMeasurementConfig,
    IndexRelativeMeasurementCoordinates,
    PairSpec,
    build_custom_relative_config,
    make_coordinate_builder,
    make_custom_group_builder,
    project_basket_mean,
)

_CURATED_PAIR_SPECS: tuple[PairSpec, ...] = (
    (
        "curated_pair_ibch20_vs_ibde40",
        ("IBCH20",),
        ("IBDE40",),
    ),
)


@dataclass(frozen=True)
class PairStateConditionedCuratedPairConfig(IndexRelativeMeasurementConfig):
    state_window: int = 4


def build_pair_state_conditioned_curated_pair_config(
    raw: object,
) -> PairStateConditionedCuratedPairConfig:
    label = "model.params.pair_state_conditioned_curated_pair"
    values = coerce_mapping(raw, label=label)
    base = build_custom_relative_config(
        raw,
        label=label,
        relative_weight_key="range_obs_weight",
    )
    state_window = int(values.get("state_window", 4))
    if state_window < 1:
        raise ConfigError(
            "pair_state_conditioned_curated_pair.state_window must be >= 1"
        )
    return PairStateConditionedCuratedPairConfig(
        enabled=base.enabled,
        df=base.df,
        weights=base.weights,
        mad_floor=base.mad_floor,
        eps=base.eps,
        state_window=state_window,
    )


def build_pair_state_conditioned_range_mask(
    *,
    observed: torch.Tensor,
    coordinates: IndexRelativeMeasurementCoordinates,
    time_mask: torch.BoolTensor | None,
    state_window: int,
) -> torch.BoolTensor:
    coordinate_obs = project_basket_mean(
        loc=observed[:, coordinates.index_mask],
        basis=coordinates.basis,
    )
    pair_mask = torch.tensor(
        [
            name.startswith("curated_pair_")
            for name in coordinates.coordinate_names
        ],
        device=coordinate_obs.device,
        dtype=torch.bool,
    )
    if not bool(pair_mask.any()):
        return cast(
            torch.BoolTensor,
            torch.zeros(
                (int(observed.shape[0]),),
                device=coordinate_obs.device,
                dtype=torch.bool,
            ),
        )
    pair_series = coordinate_obs[:, pair_mask][:, 0].detach().cpu().tolist()
    active_mask = _active_rows(time_mask, len(pair_series))
    return cast(
        torch.BoolTensor,
        torch.tensor(
            _range_mask_from_series(
                pair_series=pair_series,
                active_mask=active_mask,
                state_window=state_window,
            ),
            device=coordinate_obs.device,
            dtype=torch.bool,
        ),
    )


def _active_rows(
    time_mask: torch.BoolTensor | None,
    length: int,
) -> list[bool]:
    if time_mask is None:
        return [True] * length
    return [bool(value) for value in time_mask.detach().cpu().tolist()]


def _range_mask_from_series(
    *,
    pair_series: list[float],
    active_mask: list[bool],
    state_window: int,
) -> list[bool]:
    history: list[float] = []
    trend_history: list[float] = []
    output: list[bool] = []
    for value, is_active in zip(pair_series, active_mask, strict=True):
        if not is_active:
            output.append(False)
            continue
        if not history:
            output.append(False)
            history.append(float(value))
            continue
        recent = history[-state_window:]
        trend = sum(recent) / float(len(recent))
        trend_history.append(abs(trend))
        threshold = _median(trend_history)
        output.append(abs(trend) <= threshold or history[-1] == 0.0)
        history.append(float(value))
    return output


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2 == 1:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


build_pair_state_conditioned_curated_pair_coordinates = make_coordinate_builder(
    make_curated_pair_seed_builder(
        pair_specs=_CURATED_PAIR_SPECS,
        fallback_prefix="curated_pair",
    )
)
build_pair_state_conditioned_curated_pair_observation_groups = (
    make_custom_group_builder(
        relative_group_name="pair_state_conditioned_curated_pair_obs",
        relative_names=lambda coordinate_names: frozenset(
            name for name in coordinate_names if name.startswith("curated_pair_")
        ),
        residual_group_name="pair_state_conditioned_curated_pair_residual_obs",
    )
)

__all__ = [
    "PairStateConditionedCuratedPairConfig",
    "build_pair_state_conditioned_curated_pair_config",
    "build_pair_state_conditioned_curated_pair_coordinates",
    "build_pair_state_conditioned_curated_pair_observation_groups",
    "build_pair_state_conditioned_range_mask",
    "state_window_from_config",
]


def state_window_from_config(config: IndexRelativeMeasurementConfig) -> int:
    if isinstance(config, PairStateConditionedCuratedPairConfig):
        return config.state_window
    return 4
