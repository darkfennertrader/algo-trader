from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError, DataProcessingError


def normalize_datetime_index(
    frame: pd.DataFrame, *, label: str, preprocessor_name: str
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DataProcessingError(
            "Preprocessor output must be a pandas DataFrame",
            context={"preprocessor": preprocessor_name, "label": label},
        )
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    except (ValueError, TypeError) as exc:
        raise DataProcessingError(
            "Datetime index is invalid",
            context={"label": label, "preprocessor": preprocessor_name},
        ) from exc
    missing_mask = np.asarray(index.isna())
    if missing_mask.any():
        raise DataProcessingError(
            "Datetime index contains invalid timestamps",
            context={"label": label, "preprocessor": preprocessor_name},
        )
    normalized = frame.copy()
    normalized.index = pd.DatetimeIndex(index)
    return normalized


def validate_no_unknown_params(
    params: Mapping[str, str], *, allowed: set[str]
) -> None:
    unknown = sorted(key for key in params if key not in allowed)
    if unknown:
        raise ConfigError(
            "Unknown preprocessor params",
            context={"unknown": ", ".join(unknown)},
        )
