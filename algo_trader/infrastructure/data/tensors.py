from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from algo_trader.domain import DataProcessingError

_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_UNIT_TO_NANOSECONDS = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
}


def require_utc_hourly_index(
    index: pd.Index, *, label: str, timezone: str
) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataProcessingError(
            f"{label} index must be datetime",
            context={"index_type": type(index).__name__},
        )
    if index.tz is None:
        raise DataProcessingError(
            f"{label} index must be timezone-aware",
            context={"timezone": timezone},
        )
    if str(index.tz) != timezone:
        raise DataProcessingError(
            f"{label} index must be {timezone}",
            context={"timezone": str(index.tz)},
        )
    if index.hasnans:
        raise DataProcessingError(
            f"{label} index contains NaT values",
            context={"timezone": str(index.tz)},
        )
    if (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise DataProcessingError(
            f"{label} index must be hourly",
            context={"timezone": str(index.tz)},
        )
    return index


def timestamps_to_epoch_hours(index: pd.DatetimeIndex) -> np.ndarray:
    epoch = index.view("int64")
    unit = getattr(index.dtype, "unit", "ns") or "ns"
    multiplier = _UNIT_TO_NANOSECONDS.get(unit)
    if multiplier is None:
        raise DataProcessingError(
            "Unsupported datetime unit",
            context={"unit": str(unit)},
        )
    if multiplier != 1:
        epoch = epoch * multiplier
    return (epoch // _NANOSECONDS_PER_HOUR).astype("int64")


def write_tensor_bundle(
    path: Path,
    *,
    values: torch.Tensor,
    timestamps: torch.Tensor,
    missing_mask: torch.Tensor,
    error_message: str,
) -> None:
    payload = {
        "values": values,
        "timestamps": timestamps,
        "missing_mask": missing_mask,
    }
    try:
        torch.save(payload, path)
    except Exception as exc:
        raise DataProcessingError(
            error_message,
            context={"path": str(path)},
        ) from exc
