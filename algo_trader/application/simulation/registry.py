from __future__ import annotations

from dataclasses import dataclass

from algo_trader.domain.simulation import Registry
from algo_trader.infrastructure.data import (
    load_feature_store_panel_dataset,
    load_panel_tensor_dataset,
)


@dataclass(frozen=True)
class SimulationRegistries:
    datasets: Registry


def default_registries() -> SimulationRegistries:
    datasets = Registry()

    @datasets.register("tensor_bundle")
    def _build_tensor_bundle(*, config, device: str):
        return load_panel_tensor_dataset(paths=config.paths, device=device)

    @datasets.register("feature_store_panel")
    def _build_feature_store_panel(*, config, device: str):
        return load_feature_store_panel_dataset(config=config, device=device)

    return SimulationRegistries(datasets=datasets)
