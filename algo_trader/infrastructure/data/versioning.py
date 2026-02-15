from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Type

from algo_trader.domain import AlgoTraderError
from algo_trader.infrastructure import require_env, resolve_latest_week_dir

logger = logging.getLogger(__name__)


def resolve_root_dir(
    dataset_params: Mapping[str, Any],
    *,
    key: str,
    env_name: str,
    error_type: Type[AlgoTraderError],
) -> Path:
    raw = dataset_params.get(key)
    if raw is None:
        raw = require_env(env_name)
    root = Path(str(raw)).expanduser()
    if not root.exists():
        raise error_type(
            f"{env_name} does not exist",
            context={"path": str(root)},
        )
    if not root.is_dir():
        raise error_type(
            f"{env_name} must be a directory",
            context={"path": str(root)},
        )
    return root


def resolve_feature_store_version_label(
    feature_store: Path,
    data_lake: Path,
    *,
    error_type: Type[AlgoTraderError],
    feature_error: str,
    lake_error: str,
) -> str:
    feature_dir = resolve_latest_week_dir(
        feature_store,
        error_type=error_type,
        error_message=feature_error,
    )
    data_lake_dir = resolve_latest_week_dir(
        data_lake,
        error_type=error_type,
        error_message=lake_error,
    )
    if feature_dir.name != data_lake_dir.name:
        logger.warning(
            "Latest feature store version %s differs from data lake %s",
            feature_dir.name,
            data_lake_dir.name,
        )
    return feature_dir.name
