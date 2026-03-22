from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from algo_trader.application.data_io import read_indexed_csv
from algo_trader.application.data_sources import resolve_data_lake
from algo_trader.application.exogenous.config import (
    FredCleaningOutputConfig,
    FredFillPolicyConfig,
    FredRequestConfig,
    FredSeriesConfig,
)
from algo_trader.domain import DataProcessingError, DataSourceError
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputWriter,
    ensure_directory,
    format_run_at,
    format_tilde_path,
    log_boundary,
    require_env,
    resolve_latest_week_dir,
)
from algo_trader.infrastructure.data import (
    require_utc_hourly_index,
    timestamps_to_epoch_hours,
    write_tensor_bundle,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "fred_config.yml"
)
_OUTPUT_TIMEZONE = "UTC"


@dataclass(frozen=True)
class RunRequest:
    config_path: Path | None = None


@dataclass(frozen=True)
class ExogenousOutputPaths:
    output_dir: Path
    output_path: Path
    metadata_path: Path
    tensor_path: Path


@dataclass(frozen=True)
class SeriesAlignmentStats:
    raw_rows: int
    missing_count_full: int
    missing_count_trimmed: int = 0
    first_valid_position: int | None = None
    drop_reason: str | None = None


@dataclass(frozen=True)
class SeriesAlignmentResult:
    feature_name: str
    config: FredSeriesConfig
    source_path: Path
    aligned: pd.Series
    stats: SeriesAlignmentStats

    @property
    def raw_rows(self) -> int:
        return self.stats.raw_rows

    @property
    def missing_count_full(self) -> int:
        return self.stats.missing_count_full

    @property
    def missing_count_trimmed(self) -> int:
        return self.stats.missing_count_trimmed

    @property
    def first_valid_position(self) -> int | None:
        return self.stats.first_valid_position

    @property
    def drop_reason(self) -> str | None:
        return self.stats.drop_reason


@dataclass(frozen=True)
class MetadataBuildContext:
    config: FredRequestConfig
    raw_root: Path
    returns_path: Path
    outputs: ExogenousOutputPaths


def _run_context(request: RunRequest) -> dict[str, str]:
    resolved_path = request.config_path or DEFAULT_CONFIG_PATH
    return {"config_path": str(resolved_path)}


@log_boundary("exogenous_cleaning.run", context=_run_context)
def run(*, request: RunRequest) -> Path:
    config = FredRequestConfig.load(request.config_path or DEFAULT_CONFIG_PATH)
    raw_root = _resolve_raw_root()
    version_dir = _resolve_version_dir()
    returns_path = version_dir / config.cleaning.calendar_index_file
    calendar_index = _load_calendar_index(returns_path)
    alignments = _align_series(
        config=config,
        raw_root=raw_root,
        calendar_index=calendar_index,
    )
    cleaned_frame, kept, dropped = _build_cleaned_frame(
        alignments=alignments,
    )
    outputs = _prepare_output_paths(version_dir, config.cleaning.output)
    writer = _build_output_writer()
    writer.write_frame(cleaned_frame, outputs.output_path)
    _write_tensor(cleaned_frame, outputs.tensor_path)
    metadata = _build_metadata(
        cleaned_frame=cleaned_frame,
        kept=kept,
        dropped=dropped,
        context=MetadataBuildContext(
            config=config,
            raw_root=raw_root,
            returns_path=returns_path,
            outputs=outputs,
        ),
    )
    writer.write_metadata(metadata, outputs.metadata_path)
    logger.info(
        "Saved cleaned exogenous data path=%s rows=%s features=%s",
        outputs.output_path,
        len(cleaned_frame),
        len(cleaned_frame.columns),
    )
    return outputs.output_path


def _resolve_raw_root() -> Path:
    raw_root = Path(require_env("EXOGENOUS_FEATURES_SOURCE")).expanduser()
    if not raw_root.exists():
        raise DataSourceError(
            "EXOGENOUS_FEATURES_SOURCE does not exist",
            context={"path": str(raw_root)},
        )
    if not raw_root.is_dir():
        raise DataSourceError(
            "EXOGENOUS_FEATURES_SOURCE must be a directory",
            context={"path": str(raw_root)},
        )
    return raw_root


def _resolve_version_dir() -> Path:
    data_lake = resolve_data_lake()
    return resolve_latest_week_dir(
        data_lake,
        error_type=DataSourceError,
        error_message="No YYYY-WW data directories found",
    )


def _load_calendar_index(returns_path: Path) -> pd.DatetimeIndex:
    returns_frame = read_indexed_csv(
        returns_path,
        missing_message="returns.csv not found for exogenous_cleaning",
        read_message="Failed to read returns.csv for exogenous_cleaning",
    )
    return require_utc_hourly_index(
        returns_frame.index,
        label="returns",
        timezone=_OUTPUT_TIMEZONE,
    )


def _align_series(
    *,
    config: FredRequestConfig,
    raw_root: Path,
    calendar_index: pd.DatetimeIndex,
) -> list[SeriesAlignmentResult]:
    results: list[SeriesAlignmentResult] = []
    for series in config.series:
        results.append(
            _align_single_series(
                config=config,
                raw_root=raw_root,
                calendar_index=calendar_index,
                series=series,
            )
        )
    return results


def _align_single_series(
    *,
    config: FredRequestConfig,
    raw_root: Path,
    calendar_index: pd.DatetimeIndex,
    series: FredSeriesConfig,
) -> SeriesAlignmentResult:
    source_path = _raw_series_path(raw_root, config.provider, series)
    feature_name = _feature_name(config.provider, series)
    try:
        raw_frame = _read_raw_series(source_path)
    except FileNotFoundError:
        aligned = _empty_aligned_series(calendar_index, feature_name)
        return SeriesAlignmentResult(
            feature_name=feature_name,
            config=series,
            source_path=source_path,
            aligned=aligned,
            stats=SeriesAlignmentStats(
                raw_rows=0,
                missing_count_full=len(aligned),
                first_valid_position=None,
                drop_reason="raw_series_not_found",
            ),
        )
    if raw_frame.empty:
        aligned = _empty_aligned_series(calendar_index, feature_name)
        return SeriesAlignmentResult(
            feature_name=feature_name,
            config=series,
            source_path=source_path,
            aligned=aligned,
            stats=SeriesAlignmentStats(
                raw_rows=0,
                missing_count_full=len(aligned),
                first_valid_position=None,
                drop_reason="raw_series_empty",
            ),
        )
    aligned = _align_raw_frame(
        raw_frame=raw_frame,
        calendar_index=calendar_index,
        provider=config.provider,
        series=series,
        fill_policy=config.cleaning.fill_policy,
    )
    missing_count = int(aligned.isna().sum())
    first_valid = _first_valid_position(aligned)
    return SeriesAlignmentResult(
        feature_name=feature_name,
        config=series,
        source_path=source_path,
        aligned=aligned,
        stats=SeriesAlignmentStats(
            raw_rows=len(raw_frame),
            missing_count_full=missing_count,
            first_valid_position=first_valid,
        ),
    )


def _raw_series_path(
    raw_root: Path, provider: str, series: FredSeriesConfig
) -> Path:
    return raw_root / provider / series.dir_name / f"{series.series_id}.csv"


def _feature_name(provider: str, series: FredSeriesConfig) -> str:
    family = series.family_key or series.dir_name.replace("/", "_")
    name = series.alias or series.series_id
    return f"{provider}__{family}__{name}"


def _read_raw_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataSourceError(
            "Failed to read raw exogenous CSV",
            context={"path": str(path)},
        ) from exc
    if "date" not in frame.columns or "value" not in frame.columns:
        raise DataSourceError(
            "Raw exogenous CSV must contain date and value columns",
            context={"path": str(path)},
        )
    payload = frame[["date", "value"]].copy()
    payload["date"] = pd.to_datetime(payload["date"], errors="raise")
    payload["date"] = payload["date"].dt.normalize()
    payload["value"] = pd.to_numeric(payload["value"], errors="coerce")
    payload = payload.dropna(subset=["date", "value"])
    payload = payload.drop_duplicates(subset=["date"], keep="last")
    return payload.sort_values("date").reset_index(drop=True)


def _empty_aligned_series(
    calendar_index: pd.DatetimeIndex, feature_name: str
) -> pd.Series:
    return pd.Series(
        np.nan,
        index=calendar_index,
        dtype=float,
        name=feature_name,
    )


def _align_raw_frame(
    *,
    raw_frame: pd.DataFrame,
    calendar_index: pd.DatetimeIndex,
    provider: str,
    series: FredSeriesConfig,
    fill_policy: FredFillPolicyConfig,
) -> pd.Series:
    calendar = pd.DataFrame(
        {
            "calendar_date": calendar_index.tz_localize(None).normalize(),
        }
    )
    source = raw_frame.rename(columns={"date": "source_date"})
    aligned = pd.merge_asof(
        calendar,
        source,
        left_on="calendar_date",
        right_on="source_date",
        direction="backward",
    )
    if series.release_lag_weeks:
        aligned[["source_date", "value"]] = aligned[
            ["source_date", "value"]
        ].shift(series.release_lag_weeks)
    max_age_weeks = _max_ffill_weeks(series, fill_policy)
    invalid = _invalid_fill_mask(aligned, max_age_weeks)
    values = aligned["value"].where(~invalid)
    return pd.Series(
        values.to_numpy(dtype=float),
        index=calendar_index,
        dtype=float,
        name=_feature_name(provider, series),
    )


def _max_ffill_weeks(
    series: FredSeriesConfig, fill_policy: FredFillPolicyConfig
) -> int:
    if series.frequency == "m":
        return fill_policy.monthly_max_ffill_weeks
    return fill_policy.weekly_max_ffill_weeks


def _invalid_fill_mask(aligned: pd.DataFrame, max_age_weeks: int) -> pd.Series:
    source_dates = aligned["source_date"]
    age_days = (aligned["calendar_date"] - source_dates).dt.days
    age_weeks = age_days.floordiv(7)
    return source_dates.isna() | age_weeks.gt(max_age_weeks)


def _first_valid_position(series: pd.Series) -> int | None:
    valid = series.notna().to_numpy()
    if not valid.any():
        return None
    return int(valid.argmax())


def _build_cleaned_frame(
    *,
    alignments: Sequence[SeriesAlignmentResult],
) -> tuple[pd.DataFrame, list[SeriesAlignmentResult], list[SeriesAlignmentResult]]:
    kept = [
        replace(
            alignment,
            stats=replace(
                alignment.stats,
                missing_count_trimmed=int(alignment.aligned.isna().sum()),
            ),
        )
        for alignment in alignments
    ]
    dropped: list[SeriesAlignmentResult] = []
    if not kept:
        raise DataProcessingError(
            "No exogenous features available after cleaning",
        )
    frame = pd.concat([item.aligned for item in kept], axis=1)
    if frame.empty:
        raise DataProcessingError(
            "No exogenous features available after cleaning",
        )
    return frame, kept, dropped


def _prepare_output_paths(
    version_dir: Path, output_config: FredCleaningOutputConfig
) -> ExogenousOutputPaths:
    output_dir = version_dir / output_config.subdir
    ensure_directory(
        output_dir,
        error_type=DataProcessingError,
        invalid_message="Exogenous cleaning output path must be a directory",
        create_message="Failed to prepare exogenous cleaning output directory",
    )
    return ExogenousOutputPaths(
        output_dir=output_dir,
        output_path=output_dir / output_config.cleaned_csv,
        metadata_path=output_dir / output_config.metadata_json,
        tensor_path=output_dir / output_config.tensor_pt,
    )


def _build_output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write cleaned exogenous CSV",
        ),
        metadata_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write cleaned exogenous metadata",
        ),
    )


def _write_tensor(frame: pd.DataFrame, tensor_path: Path) -> None:
    index = require_utc_hourly_index(
        frame.index,
        label="exogenous_cleaned",
        timezone=_OUTPUT_TIMEZONE,
    )
    values = np.array(frame.to_numpy(dtype=float), copy=True)
    timestamps = timestamps_to_epoch_hours(index)
    payload = {
        "values": torch.as_tensor(values, dtype=torch.float64),
        "timestamps": torch.as_tensor(timestamps, dtype=torch.int64),
    }
    try:
        torch.save(payload, tensor_path)
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write cleaned exogenous tensor",
            context={"path": str(tensor_path)},
        ) from exc


def _build_metadata(
    *,
    cleaned_frame: pd.DataFrame,
    kept: Sequence[SeriesAlignmentResult],
    dropped: Sequence[SeriesAlignmentResult],
    context: MetadataBuildContext,
) -> dict[str, object]:
    version_dir = context.outputs.output_dir.parent
    return {
        "provider": context.config.provider,
        "version_label": version_dir.name,
        "run_at": format_run_at(datetime.now(timezone.utc)),
        "source_root": format_tilde_path(context.raw_root),
        "calendar_source": format_tilde_path(context.returns_path),
        "destination": format_tilde_path(context.outputs.output_path),
        "start_date": cleaned_frame.index[0].isoformat(),
        "end_date": cleaned_frame.index[-1].isoformat(),
        "rows": len(cleaned_frame),
        "features": len(cleaned_frame.columns),
        "feature_names": list(cleaned_frame.columns),
        "kept_features": [_series_metadata(item) for item in kept],
        "dropped_features": [_dropped_series_metadata(item) for item in dropped],
        "tensor": format_tilde_path(context.outputs.tensor_path),
    }


def _series_metadata(alignment: SeriesAlignmentResult) -> dict[str, object]:
    return {
        "feature_name": alignment.feature_name,
        "series_id": alignment.config.series_id,
        "family_key": alignment.config.family_key,
        "alias": alignment.config.alias,
        "currency": alignment.config.currency,
        "priority": alignment.config.priority,
        "future_role": alignment.config.future_role,
        "frequency": alignment.config.frequency,
        "release_lag_weeks": alignment.config.release_lag_weeks,
        "source_path": format_tilde_path(alignment.source_path),
        "raw_rows": alignment.raw_rows,
        "missing_count_before_trim": alignment.missing_count_full,
        "missing_count_after_trim": alignment.missing_count_trimmed,
    }


def _dropped_series_metadata(alignment: SeriesAlignmentResult) -> dict[str, object]:
    payload = _series_metadata(alignment)
    payload["drop_reason"] = alignment.drop_reason
    return payload
