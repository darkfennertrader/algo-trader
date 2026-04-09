from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import PanelDataset, PreprocessSpec, SimulationConfig
from .cv_groups import build_equal_groups, make_outer_folds
from .feature_panel_data import FeaturePanelData
from .runner_helpers import build_group_by_index, resolve_outer_test_group_ids


@dataclass(frozen=True)
class SimulationCVStructure:
    warmup_idx: np.ndarray
    groups: list[np.ndarray]
    outer_ids: list[int]
    outer_folds: list
    group_by_index: np.ndarray


@dataclass(frozen=True)
class SimulationContext:
    panel: FeaturePanelData
    y: Any
    timestamps: Sequence[Any]
    preprocess_spec: PreprocessSpec
    cv: SimulationCVStructure
    feature_names: Sequence[str]
    assets: Sequence[str]

    @property
    def X(self) -> Any:
        return self.panel.X

    @property
    def M(self) -> Any:
        return self.panel.M

    @property
    def X_global(self) -> Any:
        return self.panel.X_global

    @property
    def M_global(self) -> Any:
        return self.panel.M_global

    @property
    def global_feature_names(self) -> Sequence[str]:
        return self.panel.global_feature_names


def build_simulation_context(
    config: SimulationConfig,
    dataset: PanelDataset,
) -> SimulationContext:
    preprocess_spec = _resolve_preprocess_spec(
        config.preprocessing,
        dataset,
        config.flags.use_feature_names_for_scaling,
    )
    X, M, y = dataset.data, dataset.missing_mask, dataset.targets
    warmup_idx, groups = build_equal_groups(
        T=int(X.shape[0]),
        warmup_len=config.cv.window.warmup_len,
        group_len=config.cv.window.group_len,
    )
    if not groups:
        raise SimulationError("No groups available; check cv settings")
    outer_ids = resolve_outer_test_group_ids(
        config.outer.test_group_ids,
        config.outer.last_n,
        len(groups),
    )
    outer_folds = make_outer_folds(
        warmup_idx=warmup_idx,
        groups=groups,
        outer_test_group_ids=outer_ids,
        exclude_warmup=config.cv.exclude_warmup,
    )
    return SimulationContext(
        panel=FeaturePanelData(
            X=X,
            M=M,
            X_global=dataset.global_data,
            M_global=dataset.global_missing_mask,
            global_feature_names=list(dataset.global_features),
        ),
        y=y,
        timestamps=list(dataset.dates),
        preprocess_spec=preprocess_spec,
        cv=SimulationCVStructure(
            warmup_idx=warmup_idx,
            groups=groups,
            outer_ids=outer_ids,
            outer_folds=outer_folds,
            group_by_index=build_group_by_index(
                groups=groups,
                total_len=int(X.shape[0]),
            ),
        ),
        feature_names=list(dataset.features),
        assets=list(dataset.assets),
    )


def _resolve_preprocess_spec(
    spec: PreprocessSpec,
    dataset: PanelDataset,
    use_feature_names_for_scaling: bool,
) -> PreprocessSpec:
    if not use_feature_names_for_scaling:
        return spec
    if spec.scaling.inputs.feature_names is not None:
        return spec
    return replace(
        spec,
        scaling=replace(
            spec.scaling,
            inputs=replace(
                spec.scaling.inputs,
                feature_names=list(dataset.features),
            ),
        ),
    )
