from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain.simulation import (
    Allocator,
    ModelFitter,
    PnLCalculator,
    Predictor,
    Scorer,
)


@dataclass(frozen=True)
class SimulationHooks:
    fit_model: ModelFitter
    predict: Predictor
    score: Scorer
    allocate: Allocator
    compute_pnl: PnLCalculator


@dataclass(frozen=True)
class _SVIParams:
    steps: int
    learning_rate: float
    num_elbo_particles: int
    log_every: int | None
    grad_accum_steps: int


@dataclass(frozen=True)
class _TBPTTParams:
    window_len: int | None
    burn_in_len: int


@dataclass(frozen=True)
class _TBPTTInputs:
    X_asset_obs: torch.Tensor
    X_global_obs: torch.Tensor | None
    y_obs: torch.Tensor
    valid_mask: torch.BoolTensor
    window_len: int


@dataclass(frozen=True)
class _FitInputs:
    X_train: torch.Tensor
    X_train_global: torch.Tensor | None
    y_train: torch.Tensor


@dataclass(frozen=True)
class _OnlineFilteringParams:
    steps_per_observation: int


@dataclass(frozen=True)
class _TrainingParams:
    method: str
    svi: _SVIParams
    tbptt: _TBPTTParams
    online_filtering: _OnlineFilteringParams
    log_prob_scaling: bool
    target_normalization: bool


@dataclass(frozen=True)
class _TargetNormState:
    center: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class _TrainingArtifacts:
    norm_state: _TargetNormState | None
    posterior_summary: Mapping[str, Any] | None
    training_diagnostics: Mapping[str, Any] | None
    filtering_state: Mapping[str, Any] | None = None
    structural_posterior_means: Mapping[str, Any] | None = None
    param_store_state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _DebugConfig:
    enabled: bool
    output_dir: str | None
