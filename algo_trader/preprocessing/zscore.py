from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError
from .validation import normalize_datetime_index, validate_no_unknown_params

MissingPolicy = Literal["drop", "zero"]


@dataclass(frozen=True)
class ZScorePreprocessorConfig:
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    missing: MissingPolicy


@dataclass(frozen=True)
class ZScoreResult:
    mean: pd.Series
    std: pd.Series
    missing_mask: np.ndarray
    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None


class ZScorePreprocessor:
    def __init__(self) -> None:
        self._result: ZScoreResult | None = None

    def process(
        self, data: pd.DataFrame, *, params: Mapping[str, str]
    ) -> pd.DataFrame:
        config = _parse_config(params)
        normalized = normalize_datetime_index(
            data, label="input", preprocessor_name="zscore"
        )
        filtered = _filter_by_date(
            normalized, start_date=config.start_date, end_date=config.end_date
        )
        cleaned, missing_mask = _handle_missing_with_mask(
            filtered, config.missing
        )
        zscored, mean, std = _zscore_with_stats(cleaned)
        start_timestamp, end_timestamp = _timestamp_range(cleaned.index)
        self._result = ZScoreResult(
            mean=mean,
            std=std,
            missing_mask=missing_mask,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        return zscored

    def result(self) -> ZScoreResult:
        if self._result is None:
            raise DataProcessingError(
                "Z-score artifacts are not available before processing",
                context={"preprocessor": "zscore"},
            )
        return self._result


def _parse_config(params: Mapping[str, str]) -> ZScorePreprocessorConfig:
    validate_no_unknown_params(
        params, allowed={"start_date", "end_date", "missing"}
    )
    start_date = _parse_date(params.get("start_date"))
    end_date = _parse_date(params.get("end_date"))
    if start_date and end_date and start_date > end_date:
        raise ConfigError(
            "start_date must be before or equal to end_date",
            context={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
    missing = _parse_missing(params.get("missing"))
    return ZScorePreprocessorConfig(
        start_date=start_date, end_date=end_date, missing=missing
    )


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Date must be in YYYY-MM-DD format",
            context={"value": value},
        ) from exc
    if pd.isna(parsed):
        raise ConfigError(
            "Date must be in YYYY-MM-DD format",
            context={"value": value},
        )
    return pd.Timestamp(parsed)


def _parse_missing(value: str | None) -> MissingPolicy:
    if value is None:
        return "zero"
    normalized = value.strip().lower()
    if normalized == "drop":
        return "drop"
    if normalized == "zero":
        return "zero"
    raise ConfigError(
        "missing must be either 'drop' or 'zero'",
        context={"value": value},
    )


def _filter_by_date(
    frame: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if start_date is None and end_date is None:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    if start_date is not None:
        mask &= frame.index >= start_date
    if end_date is not None:
        mask &= frame.index <= end_date
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise DataProcessingError(
            "No data available after applying date filter",
            context={
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
            },
        )
    return filtered


def _handle_missing(
    frame: pd.DataFrame, missing: MissingPolicy
) -> pd.DataFrame:
    if missing == "drop":
        cleaned = frame.dropna(axis=0, how="any")
        if cleaned.empty:
            raise DataProcessingError(
                "No data available after dropping missing values",
                context={"missing": "drop"},
            )
        return cleaned
    return frame.fillna(0)


def _handle_missing_with_mask(
    frame: pd.DataFrame, missing: MissingPolicy
) -> tuple[pd.DataFrame, np.ndarray]:
    if missing == "drop":
        cleaned = _handle_missing(frame, missing)
        missing_mask = np.zeros(cleaned.shape, dtype=bool)
        return cleaned, missing_mask
    missing_mask = frame.isna().to_numpy()
    cleaned = _handle_missing(frame, missing)
    return cleaned, missing_mask


def _zscore_with_stats(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0)
    zero_std = std[std == 0]
    if not zero_std.empty:
        raise DataProcessingError(
            "Z-score requires non-zero standard deviation",
            context={"columns": ", ".join(zero_std.index)},
        )
    return frame.sub(mean, axis=1).div(std, axis=1), mean, std


def _timestamp_range(
    index: pd.Index,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            "Z-score index must be datetime",
            context={"index_type": type(index).__name__},
        )
    if index.empty:
        return None, None
    return index[0], index[-1]
