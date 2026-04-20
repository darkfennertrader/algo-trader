from __future__ import annotations

from typing import cast

import torch

from algo_trader.pipeline.stages.modeling.config_support import coerce_mapping
from algo_trader.pipeline.stages.modeling.curated_pair_support import (
    build_curated_pair_seed_entries,
)
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    BasketObservationGroup,
    IndexRelativeMeasurementConfig,
    IndexRelativeMeasurementWeights,
    PairSpec,
    SeedEntry,
    make_coordinate_builder,
    make_named_group_builder,
)
from algo_trader.pipeline.stages.modeling.vector_support import (
    VectorBuildConfig,
    equal_weight_vector,
)

_ANCHOR_PAIR_SPECS: tuple[PairSpec, ...] = (
    (
        "anchor_pair_ibch20_vs_ibde40",
        ("IBCH20",),
        ("IBDE40",),
    ),
)


def build_hierarchical_index_experts_config(
    raw: object,
) -> IndexRelativeMeasurementConfig:
    label = "model.params.hierarchical_index_experts"
    values = coerce_mapping(raw, label=label)
    if not values:
        return _default_config()
    obs_weight = float(values.get("obs_weight", 0.06))
    broad_weight = float(values.get("broad_obs_weight", 0.08))
    anchor_weight = float(values.get("anchor_pair_obs_weight", 0.16))
    residual_weight = float(values.get("residual_obs_weight", 0.03))
    normalized = _normalize_expert_weights(
        obs_weight=obs_weight,
        broad_weight=broad_weight,
        anchor_weight=anchor_weight,
        residual_weight=residual_weight,
    )
    return IndexRelativeMeasurementConfig(
        enabled=bool(values.get("enabled", True)),
        df=float(values.get("df", 8.0)),
        weights=normalized,
        mad_floor=float(values.get("mad_floor", 1e-4)),
        eps=float(values.get("eps", 1e-6)),
    )


def _default_config() -> IndexRelativeMeasurementConfig:
    return IndexRelativeMeasurementConfig(
        enabled=True,
        df=8.0,
        weights=_normalize_expert_weights(
            obs_weight=0.06,
            broad_weight=0.08,
            anchor_weight=0.16,
            residual_weight=0.03,
        ),
        mad_floor=1e-4,
        eps=1e-6,
    )


def _normalize_expert_weights(
    *,
    obs_weight: float,
    broad_weight: float,
    anchor_weight: float,
    residual_weight: float,
) -> IndexRelativeMeasurementWeights:
    total = broad_weight + anchor_weight + residual_weight
    if total <= 0.0:
        broad_share = anchor_share = residual_share = obs_weight / 3.0
    else:
        scale = obs_weight / total
        broad_share = broad_weight * scale
        anchor_share = anchor_weight * scale
        residual_share = residual_weight * scale
    return IndexRelativeMeasurementWeights(
        obs_weight=obs_weight,
        level_obs_weight=broad_share,
        relative_obs_weight=anchor_share,
        residual_obs_weight=residual_share,
    )


def _build_seed_entries(
    index_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    vector_config = VectorBuildConfig(device=device, dtype=dtype, strict=False)
    broad_entry = (
        "broad_index_equal_weight",
        equal_weight_vector(
            index_names,
            index_names,
            config=vector_config,
        ),
    )
    anchor_entries = build_curated_pair_seed_entries(
        index_names=index_names,
        pair_specs=_ANCHOR_PAIR_SPECS,
        fallback_prefix="anchor_pair",
        device=device,
        dtype=dtype,
    )
    return (broad_entry,) + anchor_entries


def build_hierarchical_index_experts_observation_groups(
    config: IndexRelativeMeasurementConfig,
    coordinate_names: tuple[str, ...],
    device: torch.device,
) -> tuple[BasketObservationGroup, ...]:
    return _group_builder(config, coordinate_names, device)


build_hierarchical_index_experts_coordinates = make_coordinate_builder(
    _build_seed_entries
)
_group_builder = make_named_group_builder(
    level_group_name="hierarchical_index_broad_obs",
    level_names=frozenset({"broad_index_equal_weight"}),
    relative_group_name="hierarchical_index_anchor_obs",
    relative_names=lambda coordinate_names: frozenset(
        name for name in coordinate_names if name.startswith("anchor_pair_")
    ),
    residual_group_name="hierarchical_index_residual_obs",
)

__all__ = [
    "IndexRelativeMeasurementConfig",
    "build_hierarchical_index_experts_config",
    "build_hierarchical_index_experts_coordinates",
    "build_hierarchical_index_experts_observation_groups",
]
