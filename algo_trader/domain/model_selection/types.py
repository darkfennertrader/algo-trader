from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DataPaths:
    tensor_path: str | None = None
    targets_path: str | None = None
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
    """
    Configuration related to the panel data and period selection.

    Assumes:
      - Core data is a tensor of shape (T, A, F):
          T: time
          A: assets
          F: features
    """

    dataset_name: str
    paths: DataPaths = field(default_factory=DataPaths)
    selection: DataSelection = field(default_factory=DataSelection)
    dataset_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CVSampling:
    max_splits: int | None = None
    shuffle_splits: bool = False
    random_state: int | None = None


@dataclass(frozen=True)
class CVGuards:
    min_train_size: int | None = None
    min_block_size: int | None = None


@dataclass(frozen=True)
class CVConfig:
    """Configuration for cross-validation (e.g., CPCV)."""

    cv_name: str = "cpcv"
    n_blocks: int = 5
    test_block_size: int = 1
    embargo_size: int = 0
    purge_size: int = 0
    guards: CVGuards = field(default_factory=CVGuards)
    sampling: CVSampling = field(default_factory=CVSampling)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Pyro model + guide."""

    model_name: str
    guide_name: str
    latent_dim: int = 16
    hidden_dim: int = 64


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    trainer_name: str = "svi_trainer"
    num_steps: int = 2_000
    learning_rate: float = 1e-3
    batch_size: int | None = None
    num_elbo_particles: int = 1


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for evaluation metrics."""

    metric_names: list[str]


@dataclass(frozen=True)
class ExperimentConfig:
    """High-level configuration wrapper used by model selection."""

    data: DataConfig
    cv: CVConfig
    model: ModelConfig
    training: TrainingConfig
    metrics: MetricConfig
    use_gpu: bool = False
