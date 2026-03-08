from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from algo_trader.domain import ExportError
from algo_trader.infrastructure import ErrorPolicy, ensure_directory, write_csv
from .config import FredSeriesConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExogenousCsvExporter:
    output_root: Path
    provider_name: str

    def export_series(
        self, *, series: FredSeriesConfig, frame: pd.DataFrame
    ) -> Path:
        provider_dir = self.output_root / self.provider_name / series.dir_name
        ensure_directory(
            provider_dir,
            error_type=ExportError,
            invalid_message="Exogenous output path must be a directory",
            create_message="Failed to prepare exogenous output directory",
            context={"path": str(provider_dir)},
        )
        file_path = provider_dir / f"{series.series_id}.csv"
        payload = _normalize_frame(frame)
        write_csv(
            payload,
            file_path,
            error_policy=ErrorPolicy(
                error_type=ExportError,
                message="Failed to export exogenous CSV",
                context={
                    "series_id": series.series_id,
                    "path": str(file_path),
                },
            ),
            include_index=False,
        )
        logger.info(
            "Saved exogenous CSV series_id=%s rows=%s path=%s",
            series.series_id,
            len(payload),
            file_path,
        )
        return file_path


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns or "value" not in frame.columns:
        raise ExportError(
            "Exogenous frame must contain date and value columns",
            context={"columns": ",".join([str(item) for item in frame.columns])},
        )
    payload = frame[["date", "value"]].copy()
    payload["date"] = pd.to_datetime(
        payload["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    payload["value"] = pd.to_numeric(payload["value"], errors="coerce")
    return payload.sort_values(by="date").reset_index(drop=True)
