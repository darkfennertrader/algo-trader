from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from algo_trader.domain import ModelSelectionError
from algo_trader.domain.model_selection import (
    BaseCVSplitter,
    BaseTrainer,
    CVConfig,
    DataConfig,
    ModelConfig,
    PanelDataset,
    Registry,
    TrainingConfig,
)
from algo_trader.infrastructure.data import load_panel_tensor_dataset
from .cv import CombinatorialPurgedCV
from .types import Batch, GuideFn, MetricFn, ModelFn


@dataclass(frozen=True)
class ModelSelectionRegistries:
    datasets: Registry[PanelDataset]
    cv_splitters: Registry[BaseCVSplitter]
    models: Registry[ModelFn]
    guides: Registry[GuideFn]
    trainers: Registry[BaseTrainer]
    metrics: Registry[MetricFn]


def default_registries() -> ModelSelectionRegistries:
    datasets = Registry[PanelDataset]()
    datasets.register("tensor_bundle")(_build_tensor_bundle_dataset)

    cv_splitters = Registry[BaseCVSplitter]()
    cv_splitters.register("cpcv")(_build_cpcv_splitter)

    models = Registry[ModelFn]()
    models.register("generic_model")(_build_generic_model)

    guides = Registry[GuideFn]()
    guides.register("generic_guide")(_build_generic_guide)

    trainers = Registry[BaseTrainer]()
    trainers.register("svi_trainer")(_build_svi_trainer)

    metrics = Registry[MetricFn]()
    metrics.register("dummy_metric")(_build_dummy_metric)

    return ModelSelectionRegistries(
        datasets=datasets,
        cv_splitters=cv_splitters,
        models=models,
        guides=guides,
        trainers=trainers,
        metrics=metrics,
    )


class SVITrainer:
    def __init__(
        self,
        model_spec: object,
        training_config: TrainingConfig,
        device: str,
    ) -> None:
        self.model_spec = model_spec
        self.training_config = training_config
        self.device = device
        self._is_trained = False

    def fit(
        self,
        train_batch: Batch,
        training_config: TrainingConfig | None = None,
    ) -> object:
        _ = train_batch, training_config
        self._is_trained = True
        return self

    def evaluate(
        self,
        val_batch: Batch,
        metrics: Mapping[str, MetricFn],
    ) -> Mapping[str, float]:
        if not self._is_trained:
            raise ModelSelectionError("Trainer must be fit() before evaluate().")
        results: dict[str, float] = {}
        for name, metric_fn in metrics.items():
            results[name] = float(metric_fn(self, val_batch))
        return results


def _build_tensor_bundle_dataset(
    config: DataConfig, device: str
) -> PanelDataset:
    if config.paths.tensor_path is None:
        raise ModelSelectionError("data.paths.tensor_path is required")
    return load_panel_tensor_dataset(
        paths=config.paths,
        device=device,
    ).select_period_and_subsets(config)


def _build_cpcv_splitter(config: CVConfig) -> BaseCVSplitter:
    return CombinatorialPurgedCV(config)


def _build_generic_model(config: ModelConfig, device: str) -> ModelFn:
    _ = config, device

    def model_fn(batch: Batch) -> None:
        _ = batch

    return model_fn


def _build_generic_guide(config: ModelConfig, device: str) -> GuideFn:
    _ = config, device

    def guide_fn(batch: Batch) -> None:
        _ = batch

    return guide_fn


def _build_svi_trainer(
    model_spec: object, training_config: TrainingConfig, device: str
) -> BaseTrainer:
    return SVITrainer(model_spec, training_config, device)


def _build_dummy_metric(config: Mapping[str, object]) -> MetricFn:
    _ = config

    def metric_fn(trainer: BaseTrainer, val_batch: Batch) -> float:
        _ = trainer, val_batch
        return 0.0

    return metric_fn
