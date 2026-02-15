from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class DataPaths:
    tensor_path: str | None = None
    targets_path: str | None = None
    missing_mask_path: str | None = None
    timestamps_path: str | None = None
    assets_path: str | None = None
    features_path: str | None = None


@dataclass(frozen=True)
class DataSelection:
    start_date: str | None = None
    end_date: str | None = None
    frequency: str = "daily"
    asset_subset: list[str] | None = None
    feature_subset: list[str] | None = None


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    paths: DataPaths = field(default_factory=DataPaths)
    selection: DataSelection = field(default_factory=DataSelection)
    dataset_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CVWindow:
    warmup_len: int
    group_len: int


@dataclass(frozen=True)
class CVLeakage:
    horizon: int
    embargo_len: int


@dataclass(frozen=True)
class CPCVParams:
    q: int
    max_inner_combos: int | None
    seed: int


@dataclass(frozen=True)
class CVParams:
    window: CVWindow
    leakage: CVLeakage
    cpcv: CPCVParams
    include_warmup_in_inner_train: bool = True


@dataclass(frozen=True)
class CleaningSpec:
    min_usable_ratio: float
    min_variance: float
    max_abs_corr: float
    corr_subsample: int | None


@dataclass(frozen=True)
class ScalingSpec:
    mad_eps: float
    impute_missing_to_zero: bool
    feature_names: list[str] | None = None
    append_mask_as_features: bool = False


@dataclass(frozen=True)
class PreprocessSpec:
    cleaning: CleaningSpec
    scaling: ScalingSpec


@dataclass(frozen=True)
class OuterFold:
    k_test: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_group_ids: list[int]


@dataclass(frozen=True)
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_group_ids: tuple[int, ...]


@dataclass(frozen=True)
class FeatureCleaningState:
    feature_idx: np.ndarray
    usable_ratio: np.ndarray
    variance: np.ndarray
    dropped_low_usable: np.ndarray
    dropped_low_var: np.ndarray
    dropped_duplicates: np.ndarray
    duplicate_pairs: list[tuple[int, int, float]]


@dataclass(frozen=True)
class RobustScalerState:
    feature_idx: np.ndarray
    shift: torch.Tensor
    scale: torch.Tensor
    mad_eps: float


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    guide_name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    trainer_name: str = "svi_trainer"
    num_steps: int = 2_000
    learning_rate: float = 1e-3
    batch_size: int | None = None
    num_elbo_particles: int = 1


@dataclass(frozen=True)
class TuningConfig:
    param_space: Mapping[str, Any] = field(default_factory=dict)
    num_samples: int = 1
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictiveConfig:
    num_samples_inner: int = 100
    num_samples_outer: int = 100


@dataclass(frozen=True)
class ScoringConfig:
    spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AllocationConfig:
    spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostConfig:
    spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelingSpec:
    model: ModelConfig
    training: TrainingConfig
    tuning: TuningConfig


@dataclass(frozen=True)
class EvaluationSpec:
    scoring: ScoringConfig
    predictive: PredictiveConfig
    allocation: AllocationConfig
    cost: CostConfig


@dataclass(frozen=True)
class OuterConfig:
    test_group_ids: list[int] | None = None
    last_n: int | None = None


@dataclass(frozen=True)
class SimulationFlags:
    use_feature_names_for_scaling: bool = True
    use_gpu: bool = False
    simulation_mode: Literal["dry_run", "stub", "full"] = "full"
    stop_after: (
        Literal["inputs", "cv", "inner", "outer", "results"] | None
    ) = None


@dataclass(frozen=True)
class SimulationConfig:
    data: DataConfig
    cv: CVParams
    preprocessing: PreprocessSpec
    modeling: ModelingSpec
    evaluation: EvaluationSpec
    outer: OuterConfig
    flags: SimulationFlags
