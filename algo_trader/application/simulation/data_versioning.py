from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from algo_trader.domain import SimulationError
from algo_trader.infrastructure.data.versioning import (
    resolve_feature_store_version_label,
    resolve_root_dir,
)


@dataclass(frozen=True)
class DatasetVersionContext:
    feature_store: Path
    data_lake: Path
    version_label: str


def resolve_dataset_version_context(
    dataset_params: Mapping[str, Any],
) -> DatasetVersionContext:
    feature_store = resolve_root_dir(
        dataset_params,
        key="feature_store",
        env_name="FEATURE_STORE_SOURCE",
        error_type=SimulationError,
    )
    data_lake = resolve_root_dir(
        dataset_params,
        key="data_lake",
        env_name="DATA_LAKE_SOURCE",
        error_type=SimulationError,
    )
    version_label = resolve_feature_store_version_label(
        feature_store,
        data_lake,
        error_type=SimulationError,
        feature_error="No feature store versions found",
        lake_error="No data lake versions found",
    )
    return DatasetVersionContext(
        feature_store=feature_store,
        data_lake=data_lake,
        version_label=version_label,
    )
