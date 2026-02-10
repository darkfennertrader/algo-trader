from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.domain.model_selection import (
    BaseCVSplitter,
    BaseTrainer,
    CVConfig,
    ExperimentConfig,
    PanelDataset,
)
from .registry import ModelSelectionRegistries
from .types import Batch, GuideFn, MetricFn, ModelFn


@dataclass(frozen=True)
class ModelSpec:
    model_fn: ModelFn
    guide_fn: GuideFn
    device: str


@dataclass
class Experiment:
    dataset: PanelDataset
    cv_splitter: BaseCVSplitter
    trainer: BaseTrainer
    metric_fns: Mapping[str, MetricFn]
    cv_config: CVConfig

    def run(self) -> Mapping[str, float]:
        splits: list[tuple[list[int], list[int]]] = list(
            self.cv_splitter.split(self.dataset.dates)
        )
        fold_metrics: list[Mapping[str, float]] = []
        for train_idx, val_idx in splits:
            train_batch = cast(
                Batch, self.dataset.slice_by_indices(train_idx)
            )
            val_batch = cast(Batch, self.dataset.slice_by_indices(val_idx))
            self.trainer.fit(train_batch)
            fold_metrics.append(self.trainer.evaluate(val_batch, self.metric_fns))
        return _aggregate_metrics(fold_metrics)


def build_experiment(
    config: ExperimentConfig,
    *,
    registries: ModelSelectionRegistries,
) -> Experiment:
    device = _resolve_device(config.use_gpu)
    dataset = registries.datasets.build(
        config.data.dataset_name, config=config.data, device=device
    )
    cv_splitter = registries.cv_splitters.build(config.cv.cv_name, config=config.cv)
    model_fn = registries.models.build(
        config.model.model_name, config=config.model, device=device
    )
    guide_fn = registries.guides.build(
        config.model.guide_name, config=config.model, device=device
    )
    trainer = registries.trainers.build(
        config.training.trainer_name,
        model_spec=ModelSpec(model_fn=model_fn, guide_fn=guide_fn, device=device),
        training_config=config.training,
        device=device,
    )
    metric_fns = {
        name: registries.metrics.build(name, config={})
        for name in config.metrics.metric_names
    }
    return Experiment(
        dataset=dataset,
        cv_splitter=cv_splitter,
        trainer=trainer,
        metric_fns=metric_fns,
        cv_config=config.cv,
    )


def _resolve_device(use_gpu: bool) -> str:
    if use_gpu and not torch.cuda.is_available():
        raise ConfigError("use_gpu is true but CUDA is not available")
    return "cuda" if use_gpu else "cpu"


def _aggregate_metrics(
    fold_metrics: list[Mapping[str, float]]
) -> Mapping[str, float]:
    if not fold_metrics:
        return {}
    totals: dict[str, float] = {}
    for metrics in fold_metrics:
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    count = float(len(fold_metrics))
    return {name: total / count for name, total in totals.items()}
