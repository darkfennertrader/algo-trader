from __future__ import annotations

from pathlib import Path
from typing import Callable

from algo_trader.domain import ConfigError, EnvVarError
from algo_trader.domain.market_data import HistoricalDataExporter
from algo_trader.infrastructure import optional_env, require_env
from algo_trader.infrastructure.exporters import CsvHistoricalDataExporter
from .config import HistoricalRequestConfig

DEFAULT_EXPORTER = "csv"


def _resolve_exporter_name(_request_config: HistoricalRequestConfig) -> str:
    env_value = optional_env("HISTORICAL_DATA_EXPORTER")
    if env_value:
        return env_value
    return DEFAULT_EXPORTER


def _normalize_data_source(raw_value: str) -> Path:
    cleaned = raw_value.strip().strip("'").strip('"')
    if not cleaned:
        raise EnvVarError(
            "DATA_SOURCE must be set in .env",
            context={"env_var": "DATA_SOURCE"},
        )
    return Path(cleaned).expanduser().resolve()


def _load_output_root() -> Path:
    raw_value = require_env("DATA_SOURCE")
    return _normalize_data_source(raw_value)


def _build_csv_exporter(
    request_config: HistoricalRequestConfig,
) -> HistoricalDataExporter:
    output_root = _load_output_root()
    year, month = request_config.resolve_export_month()
    return CsvHistoricalDataExporter(
        output_root=output_root,
        year=year,
        month=month,
    )


ExporterBuilder = Callable[[HistoricalRequestConfig], HistoricalDataExporter]

_EXPORTER_BUILDERS: dict[str, ExporterBuilder] = {
    "csv": _build_csv_exporter,
}


def build_historical_data_exporter(
    request_config: HistoricalRequestConfig,
) -> HistoricalDataExporter:
    exporter_name = _resolve_exporter_name(request_config).lower()
    builder = _EXPORTER_BUILDERS.get(exporter_name)
    if builder is not None:
        return builder(request_config)
    raise ConfigError(
        f"Unsupported historical data exporter '{exporter_name}'",
        context={"exporter": exporter_name},
    )
