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
class DataConfig:
    simulation_output_path: str | None = None
    paths: DataPaths = field(default_factory=DataPaths)
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
    exclude_warmup: bool = False


@dataclass(frozen=True)
class CleaningSpec:
    min_usable_ratio: float
    min_variance: float
    max_abs_corr: float
    corr_subsample: int | None


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
    purged_idx: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    embargoed_idx: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )


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
class BreakoutScaleState:
    positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    p: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    denom: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    const: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )
    var_floor: float = 0.0


@dataclass(frozen=True)
class WinsorState:
    positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    lower: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    upper: torch.Tensor = field(default_factory=lambda: torch.zeros(0))


@dataclass(frozen=True)
class RobustScalerState:
    feature_idx: np.ndarray
    shift: torch.Tensor
    scale: torch.Tensor
    mad_eps: float
    breakout: BreakoutScaleState = field(default_factory=BreakoutScaleState)
    winsor: WinsorState = field(default_factory=WinsorState)
    near_constant_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.bool)
    )


@dataclass(frozen=True)
class GuardrailSpec:
    abs_eps: float = 1e-6
    rel_eps: float = 1e-3
    rel_offset: float = 1e-8


@dataclass(frozen=True)
class ClipSpec:
    min_value: float = -10.0
    max_value: float = 10.0
    max_abs_fail: float = 50.0


@dataclass(frozen=True)
class WinsorSpec:
    lower_q: float = 0.005
    upper_q: float = 0.995


@dataclass(frozen=True)
class ScalingInputSpec:
    impute_missing_to_zero: bool = True
    feature_names: list[str] | None = None
    append_mask_as_features: bool = False


@dataclass(frozen=True)
class ScalingSpec:
    mad_eps: float
    breakout_var_floor: float = 1e-3
    scale_floor: float = 1e-3
    guardrail: GuardrailSpec = field(default_factory=GuardrailSpec)
    clip: ClipSpec = field(default_factory=ClipSpec)
    winsor: WinsorSpec = field(default_factory=WinsorSpec)
    inputs: ScalingInputSpec = field(default_factory=ScalingInputSpec)


@dataclass(frozen=True)
class PreprocessSpec:
    cleaning: CleaningSpec
    scaling: ScalingSpec


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    guide_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    guide_params: Mapping[str, Any] = field(default_factory=dict)
    prebuild: "ModelPrebuildConfig | None" = None


@dataclass(frozen=True)
class ModelPrebuildConfig:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True


LogProbScaleMode = Literal["none", "num_obs"]


@dataclass(frozen=True)
class TrainingSVIConfig:
    num_steps: int = 2_000
    learning_rate: float = 1e-3
    tbptt_window_len: int | None = None
    tbptt_burn_in_len: int = 0
    grad_accum_steps: int = 1
    num_elbo_particles: int = 1
    log_every: int | None = 100


@dataclass(frozen=True)
class TrainingConfig:
    svi: TrainingSVIConfig = field(default_factory=TrainingSVIConfig)
    target_normalization: bool = False
    log_prob_scaling: bool = False


TuningParamType = Literal["float", "int", "categorical", "bool"]
TuningTransform = Literal["linear", "log", "log10", "none"]


@dataclass(frozen=True)
class TuningParamSpec:
    path: str
    param_type: TuningParamType
    bounds: tuple[float, float] | None = None
    values: tuple[Any, ...] | None = None
    transform: TuningTransform = "none"
    when: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class TuningAggregateConfig:
    method: Literal["mean", "median", "mean_minus_std"] = "mean"
    penalty: float = 0.5


@dataclass(frozen=True)
class TuningResourcesConfig:
    cpu: float | None = None
    gpu: float | None = None


@dataclass(frozen=True)
class TuningRayConfig:
    address: str | None = None
    resources: TuningResourcesConfig = field(
        default_factory=TuningResourcesConfig
    )


@dataclass(frozen=True)
class TuningConfig:
    space: tuple[TuningParamSpec, ...] = field(default_factory=tuple)
    num_samples: int = 1
    seed: int = 0
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    engine: Literal["local", "ray"] = "local"
    aggregate: TuningAggregateConfig = field(
        default_factory=TuningAggregateConfig
    )
    ray: TuningRayConfig = field(default_factory=TuningRayConfig)


@dataclass(frozen=True)
class ModelSelectionESBand:
    c: float = 1.0
    min_keep: int = 1
    max_keep: int = 10


@dataclass(frozen=True)
class ModelSelectionBootstrap:
    num_samples: int = 500
    seed: int = 123


@dataclass(frozen=True)
class ModelSelectionTail:
    alpha: float = 0.1


@dataclass(frozen=True)
class ModelSelectionBatching:
    candidates: int = 1
    splits: int = 1


@dataclass(frozen=True)
class ModelSelectionComplexity:
    method: Literal["random"] = "random"
    seed: int = 123


@dataclass(frozen=True)
class FanChartsConfig:
    enable: bool = False
    assets_mode: Literal["all", "list"] = "all"
    assets: tuple[str, ...] = field(default_factory=tuple)
    quantiles: tuple[float, ...] = (
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
    )
    coverage_levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class DiagnosticsConfig:
    fan_charts: FanChartsConfig = field(default_factory=FanChartsConfig)


@dataclass(frozen=True)
class ModelSelectionConfig:
    enable: bool = False
    phase_name: str = "post_tune_model_selection"
    es_band: ModelSelectionESBand = field(
        default_factory=ModelSelectionESBand
    )
    bootstrap: ModelSelectionBootstrap = field(
        default_factory=ModelSelectionBootstrap
    )
    tail: ModelSelectionTail = field(default_factory=ModelSelectionTail)
    batching: ModelSelectionBatching = field(
        default_factory=ModelSelectionBatching
    )
    complexity: ModelSelectionComplexity = field(
        default_factory=ModelSelectionComplexity
    )


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: int
    params: Mapping[str, Any]


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
    model_selection: ModelSelectionConfig
    diagnostics: DiagnosticsConfig


@dataclass(frozen=True)
class OuterConfig:
    test_group_ids: list[int] | None = None
    last_n: int | None = None


@dataclass(frozen=True)
class SimulationFlags:
    use_feature_names_for_scaling: bool = True
    use_gpu: bool = False
    smoke_test_enabled: bool = False
    smoke_test_debug: bool = False
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
