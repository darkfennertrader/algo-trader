from __future__ import annotations

import logging
import time
from pathlib import Path

from algo_trader.infrastructure import ensure_directory, log_boundary, require_env
from algo_trader.domain import DataSourceError
from .config import FredRequestConfig
from .exporter import ExogenousCsvExporter
from .factory import build_exogenous_provider

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "fred_config.yml"
)

logger = logging.getLogger(__name__)


def _run_context(config_path: Path | None) -> dict[str, str]:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    return {"config_path": str(resolved_path)}


@log_boundary("exogenous.run", context=_run_context)
def run(config_path: Path | None = None) -> list[Path]:
    config = FredRequestConfig.load(config_path or DEFAULT_CONFIG_PATH)
    provider = build_exogenous_provider(config)
    output_root = _resolve_output_root()
    exporter = ExogenousCsvExporter(
        output_root=output_root, provider_name=provider.name()
    )
    start_time = time.time()
    output_paths: list[Path] = []
    for series in config.series:
        frame = provider.fetch_series(
            series=series,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        output_paths.append(exporter.export_series(series=series, frame=frame))
    logger.info(
        "Exogenous download completed provider=%s series=%s duration=%.2fs",
        provider.name(),
        len(config.series),
        time.time() - start_time,
    )
    return output_paths


def _resolve_output_root() -> Path:
    output_root = Path(require_env("EXOGENOUS_FEATURES_SOURCE")).expanduser()
    ensure_directory(
        output_root,
        error_type=DataSourceError,
        invalid_message="EXOGENOUS_FEATURES_SOURCE must be a directory",
        create_message="EXOGENOUS_FEATURES_SOURCE cannot be created",
        context={"path": str(output_root)},
    )
    return output_root
