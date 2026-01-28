from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import pandas as pd
import torch

from algo_trader.application.historical import (
    DEFAULT_CONFIG_PATH,
    HistoricalRequestConfig,
)
from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.domain.market_data import TickerConfig
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputNames,
    OutputPaths,
    OutputWriter,
    build_weekly_output_paths,
    ensure_directory,
    format_run_at,
    log_boundary,
    require_env,
)
from algo_trader.infrastructure.paths import format_tilde_path
from algo_trader.infrastructure.data import (
    ReturnType,
    ReturnsSource,
    ReturnsSourceConfig,
    symbol_directory,
)

YearMonth = tuple[int, int]

logger = logging.getLogger(__name__)

_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")
_OUTPUT_NAMES = OutputNames(
    output_name="returns.csv",
    metadata_name="returns_meta.json",
)
_RETURN_TENSOR_NAME = "return_tensor.pt"
_CHECK_AVERAGE_NAME = "check_average.json"
_TENSOR_METADATA_NAME = "tensor_metadata.json"
_TENSOR_SCALE = 1_000_000
_TENSOR_TIMESTAMP_UNIT = "epoch_hours"
_TENSOR_TIMEZONE = "UTC"
_TENSOR_VALUE_DTYPE = "int64"
_NANOSECONDS_PER_HOUR = 3_600_000_000_000


@dataclass(frozen=True)
class MetadataContext:
    returns: pd.DataFrame
    assets: list[str]
    return_type: ReturnType
    source_dir: Path
    destination_dir: Path
    monthly_avg_close_by_asset: dict[str, dict[str, Decimal]] = field(
        default_factory=dict
    )
    tensor_info: ReturnTensorInfo | None = None


@dataclass(frozen=True)
class ReturnTensorBundle:
    values: torch.Tensor
    timestamps: torch.Tensor
    missing_mask: torch.Tensor
    assets: list[str]


@dataclass(frozen=True)
class ReturnTensorInfo:
    path: Path
    assets: list[str]
    scale: int
    timestamp_unit: str
    timezone: str
    dtype: str
    missing_mask: bool = True


@dataclass(frozen=True)
class RunConfig:
    assets: list[str]
    data_source: Path
    data_lake: Path
    start: YearMonth | None
    end: YearMonth | None
    return_type: ReturnType


@dataclass(frozen=True)
class RunRequest:
    config_path: Path | None
    start: str | None
    end: str | None
    return_type: str
    assets: str | None


@dataclass(frozen=True)
class ReturnData:
    returns: pd.DataFrame
    monthly_avg_close_by_asset: dict[str, dict[str, Decimal]]


def _run_context(
    config_path: Path | None,
    start: str | None,
    end: str | None,
    return_type: str,
    assets: str | None,
) -> dict[str, str]:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    return {
        "config_path": str(resolved_path),
        "start": start or "",
        "end": end or "",
        "return_type": return_type,
        "assets": assets or "",
    }


@log_boundary("data_cleaning.run", context=_run_context)
def run(
    *,
    config_path: Path | None,
    start: str | None,
    end: str | None,
    return_type: str,
    assets: str | None,
) -> Path:
    request = _build_run_request(
        config_path, start, end, return_type, assets
    )
    config = _resolve_run_config(request)
    logger.info("Using %s assets for data_cleaning", len(config.assets))

    return_data = _load_return_data(config)
    output_paths = _prepare_output_paths(config.data_lake, date.today())
    writer = _build_output_writer()
    output_path = _write_returns(
        return_data.returns, output_paths.output_path, writer
    )
    tensor_info = _write_return_tensor(
        return_data.returns, output_paths.output_dir
    )
    metadata_context = MetadataContext(
        returns=return_data.returns,
        assets=config.assets,
        return_type=config.return_type,
        source_dir=config.data_source,
        destination_dir=output_paths.output_dir,
        monthly_avg_close_by_asset=return_data.monthly_avg_close_by_asset,
        tensor_info=tensor_info,
    )
    _write_check_average(
        metadata_context,
        output_dir=output_paths.output_dir,
        writer=writer,
    )
    _write_tensor_metadata(
        metadata_context,
        output_dir=output_paths.output_dir,
        writer=writer,
    )
    _write_metadata(
        metadata_context,
        metadata_path=output_paths.metadata_path,
        writer=writer,
    )
    logger.info(
        "Saved returns CSV path=%s rows=%s assets=%s",
        output_path,
        len(return_data.returns),
        len(return_data.returns.columns),
    )
    return output_path


def _build_run_request(
    config_path: Path | None,
    start: str | None,
    end: str | None,
    return_type: str,
    assets: str | None,
) -> RunRequest:
    return RunRequest(config_path, start, end, return_type, assets)


def _resolve_run_config(request: RunRequest) -> RunConfig:
    asset_list = _resolve_asset_list(
        request.config_path, request.assets
    )
    if not asset_list:
        raise ConfigError("No assets available for data_cleaning")
    data_source, data_lake = _resolve_data_paths()
    start_month, end_month = _parse_month_range(
        request.start, request.end
    )
    return_type_value = _normalize_return_type(request.return_type)
    return RunConfig(
        assets=asset_list,
        data_source=data_source,
        data_lake=data_lake,
        start=start_month,
        end=end_month,
        return_type=return_type_value,
    )


def _load_return_data(config: RunConfig) -> ReturnData:
    returns = _load_returns(
        base_dir=config.data_source,
        assets=config.assets,
        return_type=config.return_type,
        start=config.start,
        end=config.end,
    )
    monthly_avg_close_by_asset = _load_monthly_avg_closes(
        base_dir=config.data_source,
        assets=config.assets,
        return_type=config.return_type,
        start=config.start,
        end=config.end,
    )
    return ReturnData(
        returns=returns,
        monthly_avg_close_by_asset=monthly_avg_close_by_asset,
    )


def _resolve_asset_list(
    config_path: Path | None, assets: str | None
) -> list[str]:
    assets_override = _parse_assets(assets)
    tickers: Iterable[TickerConfig] = []
    if assets_override is None:
        config = HistoricalRequestConfig.load(
            config_path or DEFAULT_CONFIG_PATH
        )
        tickers = config.tickers
    return _resolve_assets(assets_override, tickers)


def _parse_assets(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    assets = [item.strip() for item in raw.split(",") if item.strip()]
    if not assets:
        raise ConfigError("assets must contain at least one symbol")
    return _dedupe_assets(assets)


def _resolve_assets(
    assets_override: list[str] | None,
    tickers: Iterable[TickerConfig],
) -> list[str]:
    if assets_override is not None:
        return _dedupe_assets(assets_override)
    assets = [symbol_directory(ticker) for ticker in tickers]
    return _dedupe_assets(assets)


def _dedupe_assets(assets: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for asset in assets:
        normalized = asset.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _parse_month(value: str | None, field: str) -> YearMonth | None:
    if value is None or not value.strip():
        return None
    match = _MONTH_PATTERN.match(value.strip())
    if not match:
        raise ConfigError(
            f"{field} must be in YYYY-MM format (received '{value}')"
        )
    return int(match.group("year")), int(match.group("month"))


def _parse_month_range(
    start: str | None, end: str | None
) -> tuple[YearMonth | None, YearMonth | None]:
    start_month = _parse_month(start, "start")
    end_month = _parse_month(end, "end")
    _validate_month_window(start_month, end_month)
    return start_month, end_month


def _validate_month_window(
    start: YearMonth | None, end: YearMonth | None
) -> None:
    if start and end and start > end:
        raise ConfigError("start must be before or equal to end")


def _resolve_data_paths() -> tuple[Path, Path]:
    data_source = Path(require_env("DATA_SOURCE")).expanduser()
    data_lake = Path(require_env("DATA_LAKE_SOURCE")).expanduser()
    _validate_directory(data_source, "DATA_SOURCE")
    _validate_directory(data_lake, "DATA_LAKE_SOURCE")
    if not data_source.exists():
        raise DataSourceError(
            "DATA_SOURCE does not exist",
            context={"path": str(data_source)},
        )
    return data_source, data_lake


def _validate_directory(path: Path, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise DataSourceError(
            f"{label} must be a directory",
            context={"path": str(path)},
        )


def _load_returns(
    *,
    base_dir: Path,
    assets: list[str],
    return_type: ReturnType,
    start: YearMonth | None,
    end: YearMonth | None,
) -> pd.DataFrame:
    source = _build_returns_source(
        base_dir=base_dir,
        assets=assets,
        return_type=return_type,
        start=start,
        end=end,
    )
    return source.get_returns_frame()


def _load_monthly_avg_closes(
    *,
    base_dir: Path,
    assets: list[str],
    return_type: ReturnType,
    start: YearMonth | None,
    end: YearMonth | None,
) -> dict[str, dict[str, Decimal]]:
    source = _build_returns_source(
        base_dir=base_dir,
        assets=assets,
        return_type=return_type,
        start=start,
        end=end,
    )
    daily_prices = source.get_daily_price_series()
    return _monthly_average_closes(daily_prices)


def _build_returns_source(
    *,
    base_dir: Path,
    assets: list[str],
    return_type: ReturnType,
    start: YearMonth | None,
    end: YearMonth | None,
) -> ReturnsSource:
    return ReturnsSource(
        ReturnsSourceConfig(
            base_dir=base_dir,
            assets=assets,
            return_type=return_type,
            start=start,
            end=end,
        )
    )


def _prepare_output_paths(data_lake: Path, run_date: date) -> OutputPaths:
    output_paths = build_weekly_output_paths(
        data_lake, run_date, _OUTPUT_NAMES
    )
    ensure_directory(
        output_paths.output_dir,
        error_type=DataProcessingError,
        invalid_message="Data cleaning output path must be a directory",
        create_message="Failed to prepare data cleaning output directory",
    )
    return output_paths


def _build_output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write returns CSV",
        ),
        metadata_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write returns metadata",
        ),
    )


def _write_returns(
    returns: pd.DataFrame,
    output_path: Path,
    writer: OutputWriter,
) -> Path:
    writer.write_frame(returns, output_path)
    return output_path


def _write_metadata(
    context: MetadataContext,
    *,
    metadata_path: Path,
    writer: OutputWriter,
) -> Path:
    metadata = _build_metadata(context)
    writer.write_metadata(metadata, metadata_path)
    return metadata_path


def _build_metadata(
    context: MetadataContext,
) -> OrderedDict[str, object]:
    index = _normalize_index_dates(context.returns.index)
    start_date = index[0] if index else ""
    end_date = index[-1] if index else ""
    missing_by_asset: dict[str, dict[str, object]] = {}
    zero_returns_by_asset: dict[str, dict[str, object]] = {}
    for asset in context.assets:
        if asset not in context.returns.columns:
            missing_by_asset[asset] = {
                "missing_dates": index,
                "missing_count": len(index),
            }
            zero_returns_by_asset[asset] = {
                "zero_dates": [],
                "zero_count": 0,
            }
            continue
        series = context.returns[asset]
        missing_mask = series.isna()
        missing_dates = [
            date_str
            for date_str, is_missing in zip(index, missing_mask, strict=False)
            if is_missing
        ]
        missing_by_asset[asset] = {
            "missing_dates": missing_dates,
            "missing_count": len(missing_dates),
        }
        zero_mask = series.eq(0) & series.notna()
        zero_dates = [
            date_str
            for date_str, is_zero in zip(index, zero_mask, strict=False)
            if is_zero
        ]
        zero_returns_by_asset[asset] = {
            "zero_dates": zero_dates,
            "zero_count": len(zero_dates),
        }
    return OrderedDict(
        [
            ("assets", context.assets),
            ("return_type", context.return_type),
            ("start_date", start_date),
            ("end_date", end_date),
            ("missing_by_asset", missing_by_asset),
            ("zero_returns_by_asset", zero_returns_by_asset),
            ("run_at", format_run_at(datetime.now(timezone.utc))),
            ("source", format_tilde_path(context.source_dir)),
            ("destination", format_tilde_path(context.destination_dir)),
        ]
    )


def _write_check_average(
    context: MetadataContext,
    *,
    output_dir: Path,
    writer: OutputWriter,
) -> Path:
    payload = _build_check_average(context)
    path = output_dir / _CHECK_AVERAGE_NAME
    writer.write_metadata(payload, path)
    return path


def _build_check_average(
    context: MetadataContext,
) -> OrderedDict[str, object]:
    return OrderedDict(
        [
            (
                "monthly_avg_close_by_asset",
                context.monthly_avg_close_by_asset,
            ),
            ("run_at", format_run_at(datetime.now(timezone.utc))),
        ]
    )


def _write_tensor_metadata(
    context: MetadataContext,
    *,
    output_dir: Path,
    writer: OutputWriter,
) -> Path:
    payload = _build_tensor_metadata(context)
    path = output_dir / _TENSOR_METADATA_NAME
    writer.write_metadata(payload, path)
    return path


def _build_tensor_metadata(
    context: MetadataContext,
) -> OrderedDict[str, object]:
    return OrderedDict(
        [
            ("tensor", _tensor_metadata(context)),
            ("run_at", format_run_at(datetime.now(timezone.utc))),
        ]
    )


def _monthly_average_closes(
    daily_prices: dict[str, pd.Series],
) -> dict[str, dict[str, Decimal]]:
    monthly_averages: dict[str, dict[str, Decimal]] = {}
    for asset, series in daily_prices.items():
        if series.empty:
            monthly_averages[asset] = {}
            continue
        cleaned = series.dropna()
        if cleaned.empty:
            monthly_averages[asset] = {}
            continue
        monthly = cleaned.resample("MS").mean()
        monthly_averages[asset] = {}
        for timestamp, value in monthly.items():
            if pd.notna(value):
                month_key = cast(pd.Timestamp, timestamp).strftime("%Y-%m")
                monthly_averages[asset][month_key] = _to_decimal(value)
    return monthly_averages


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _normalize_index_dates(index: pd.Index) -> list[str]:
    if index.empty:
        return []
    if isinstance(index, pd.DatetimeIndex):
        return [item.strftime("%Y-%m-%d") for item in index]
    return [str(item) for item in index]


def _normalize_return_type(raw: str) -> ReturnType:
    normalized = raw.strip().lower()
    if normalized not in {"simple", "log"}:
        raise ConfigError(
            f"return_type must be 'simple' or 'log' (received '{raw}')"
        )
    return cast(ReturnType, normalized)


def _year_week(run_date: date) -> tuple[int, int]:
    iso = run_date.isocalendar()
    return iso.year, iso.week


def _write_return_tensor(
    returns: pd.DataFrame, output_dir: Path
) -> ReturnTensorInfo:
    bundle = _build_return_tensor_bundle(returns, scale=_TENSOR_SCALE)
    tensor_path = output_dir / _RETURN_TENSOR_NAME
    _write_tensor_bundle(bundle, tensor_path)
    return ReturnTensorInfo(
        path=tensor_path,
        assets=bundle.assets,
        scale=_TENSOR_SCALE,
        timestamp_unit=_TENSOR_TIMESTAMP_UNIT,
        timezone=_TENSOR_TIMEZONE,
        dtype=_TENSOR_VALUE_DTYPE,
        missing_mask=True,
    )


def _build_return_tensor_bundle(
    returns: pd.DataFrame, *, scale: int
) -> ReturnTensorBundle:
    if returns.empty:
        raise DataProcessingError(
            "Returns frame is empty",
            context={"rows": "0"},
        )
    index = _require_utc_hourly_index(returns.index)
    assets = [str(asset) for asset in returns.columns]
    values = returns.to_numpy(dtype=float)
    missing_mask = np.isnan(values)
    safe_values = np.where(missing_mask, 0.0, values)
    scaled = np.rint(safe_values * scale).astype("int64")
    timestamps = _timestamps_to_epoch_hours(index)
    return ReturnTensorBundle(
        values=torch.as_tensor(scaled, dtype=torch.int64),
        timestamps=torch.as_tensor(timestamps, dtype=torch.int64),
        missing_mask=torch.as_tensor(missing_mask, dtype=torch.bool),
        assets=assets,
    )


def _require_utc_hourly_index(
    index: pd.Index,
) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            "Returns index must be datetime",
            context={"index_type": type(index).__name__},
        )
    if index.tz is None:
        raise DataProcessingError(
            "Returns index must be timezone-aware",
            context={"timezone": _TENSOR_TIMEZONE},
        )
    if str(index.tz) != _TENSOR_TIMEZONE:
        raise DataProcessingError(
            "Returns index must be UTC",
            context={"timezone": str(index.tz)},
        )
    if index.hasnans:
        raise DataProcessingError(
            "Returns index contains NaT values",
            context={"timezone": str(index.tz)},
        )
    if (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise DataProcessingError(
            "Returns index must be hourly",
            context={"timezone": str(index.tz)},
        )
    return index


def _timestamps_to_epoch_hours(
    index: pd.DatetimeIndex,
) -> np.ndarray:
    epoch_ns = index.view("int64")
    return (epoch_ns // _NANOSECONDS_PER_HOUR).astype("int64")


def _write_tensor_bundle(
    bundle: ReturnTensorBundle, path: Path
) -> None:
    payload = {
        "values": bundle.values,
        "timestamps": bundle.timestamps,
        "missing_mask": bundle.missing_mask,
    }
    try:
        torch.save(payload, path)
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write return tensor",
            context={"path": str(path)},
        ) from exc


def _tensor_metadata(
    context: MetadataContext,
) -> dict[str, object] | None:
    if context.tensor_info is None:
        return None
    rows, cols = context.returns.shape
    return {
        "path": format_tilde_path(context.tensor_info.path),
        "assets": context.tensor_info.assets,
        "scale": context.tensor_info.scale,
        "dtype": context.tensor_info.dtype,
        "timestamp_unit": context.tensor_info.timestamp_unit,
        "timezone": context.tensor_info.timezone,
        "missing_mask": context.tensor_info.missing_mask,
        "values_shape": [rows, cols],
        "timestamps_shape": [rows],
        "missing_mask_shape": [rows, cols],
    }
