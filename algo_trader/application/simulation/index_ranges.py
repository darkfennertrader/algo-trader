from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from algo_trader.domain import SimulationError


def indices_to_ranges(indices: Sequence[int] | np.ndarray) -> list[list[int]]:
    array = np.asarray(indices, dtype=int)
    if array.size == 0:
        return []
    sorted_unique = [int(value) for value in np.unique(array).tolist()]
    starts = [sorted_unique[0]]
    ends: list[int] = []
    prev = sorted_unique[0]
    for item in sorted_unique[1:]:
        current = int(item)
        if current != prev + 1:
            ends.append(prev)
            starts.append(current)
        prev = current
    ends.append(prev)
    return [[start, end] for start, end in zip(starts, ends)]


def ranges_to_indices(ranges: Any, *, field: str) -> np.ndarray:
    parsed = _validate_ranges(ranges, field=field)
    if not parsed:
        return np.array([], dtype=int)
    expanded = [np.arange(start, end + 1, dtype=int) for start, end in parsed]
    if not expanded:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(expanded))


def decode_indices_field(
    payload: Mapping[str, Any],
    *,
    idx_key: str,
    ranges_key: str,
    field: str,
) -> np.ndarray:
    idx_values = payload.get(idx_key)
    if idx_values is not None:
        return _validate_indices(idx_values, field=field)
    ranges = payload.get(ranges_key)
    if ranges is not None:
        return ranges_to_indices(ranges, field=field)
    raise SimulationError(
        "Missing index field",
        context={"field": field, "expected": f"{idx_key} or {ranges_key}"},
    )


def _validate_indices(indices: Any, *, field: str) -> np.ndarray:
    if not isinstance(indices, list):
        raise SimulationError(
            "Index field must be a list",
            context={"field": field},
        )
    output: list[int] = []
    for value in indices:
        if not isinstance(value, int):
            raise SimulationError(
                "Index field contains non-integer value",
                context={"field": field},
            )
        output.append(int(value))
    return np.unique(np.asarray(output, dtype=int))


def _validate_ranges(ranges: Any, *, field: str) -> list[tuple[int, int]]:
    if not isinstance(ranges, list):
        raise SimulationError(
            "Range field must be a list",
            context={"field": field},
        )
    parsed: list[tuple[int, int]] = []
    for item in ranges:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
        ):
            raise SimulationError(
                "Range item must be [start, end] integers",
                context={"field": field},
            )
        start = int(item[0])
        end = int(item[1])
        if start > end:
            raise SimulationError(
                "Range start must be <= end",
                context={"field": field, "start": str(start), "end": str(end)},
            )
        parsed.append((start, end))
    return parsed
