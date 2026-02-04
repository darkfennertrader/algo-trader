from __future__ import annotations

from pathlib import Path

from algo_trader.domain import DataSourceError
from algo_trader.infrastructure import ensure_directory, require_env


def resolve_data_lake() -> Path:
    data_lake = Path(require_env("DATA_LAKE_SOURCE")).expanduser()
    if not data_lake.exists():
        raise DataSourceError(
            "DATA_LAKE_SOURCE does not exist",
            context={"path": str(data_lake)},
        )
    if not data_lake.is_dir():
        raise DataSourceError(
            "DATA_LAKE_SOURCE must be a directory",
            context={"path": str(data_lake)},
        )
    return data_lake


def resolve_feature_store() -> Path:
    feature_store = Path(require_env("FEATURE_STORE_SOURCE")).expanduser()
    ensure_directory(
        feature_store,
        error_type=DataSourceError,
        invalid_message="FEATURE_STORE_SOURCE must be a directory",
        create_message="FEATURE_STORE_SOURCE cannot be created",
    )
    return feature_store
