from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import SimulationError


def normalize_timestamps(values: Sequence[Any]) -> list[int | str]:
    normalized: list[int | str] = []
    for item in values:
        if isinstance(item, (int, np.integer)):
            normalized.append(int(item))
            continue
        if hasattr(item, "isoformat"):
            normalized.append(str(item.isoformat()))
            continue
        normalized.append(str(item))
    return normalized


def format_timestamp_dates(values: Sequence[Any]) -> list[str]:
    if not values:
        return []
    if isinstance(values[0], (int, np.integer)):
        return _format_epoch_hour_dates(values)
    stamps = pd.to_datetime(list(values), utc=True)
    return [stamp.strftime("%Y-%m-%d") for stamp in stamps]


def format_timestamp_date(value: Any) -> str:
    return format_timestamp_dates([value])[0]


def _format_epoch_hour_dates(values: Sequence[Any]) -> list[str]:
    epoch_hours = np.asarray(values, dtype=np.int64)
    stamps = pd.to_datetime(epoch_hours * 3600, unit="s", utc=True)
    return [stamp.strftime("%Y-%m-%d") for stamp in stamps]


def write_json_file(
    path: Path,
    payload: Mapping[str, Any],
    *,
    message: str,
) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        raise SimulationError(message, context={"path": str(path)}) from exc
