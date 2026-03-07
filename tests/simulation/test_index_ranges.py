import numpy as np
import pytest

from algo_trader.application.simulation.index_ranges import (
    decode_indices_field,
    indices_to_ranges,
    ranges_to_indices,
)
from algo_trader.domain import SimulationError


def test_indices_to_ranges_merges_contiguous_values() -> None:
    values = np.array([9, 1, 2, 3, 5, 7, 10, 4], dtype=int)
    assert indices_to_ranges(values) == [[1, 5], [7, 7], [9, 10]]


def test_ranges_to_indices_expands_ranges() -> None:
    result = ranges_to_indices([[1, 5], [7, 7], [9, 10]], field="train")
    assert result.tolist() == [1, 2, 3, 4, 5, 7, 9, 10]


def test_decode_indices_field_prefers_legacy_idx() -> None:
    payload = {"train_idx": [3, 1, 2], "train_ranges": [[9, 11]]}
    result = decode_indices_field(
        payload,
        idx_key="train_idx",
        ranges_key="train_ranges",
        field="train",
    )
    assert result.tolist() == [1, 2, 3]


def test_decode_indices_field_supports_ranges() -> None:
    payload = {"train_ranges": [[1, 3], [7, 7]]}
    result = decode_indices_field(
        payload,
        idx_key="train_idx",
        ranges_key="train_ranges",
        field="train",
    )
    assert result.tolist() == [1, 2, 3, 7]


def test_decode_indices_field_raises_when_missing() -> None:
    with pytest.raises(SimulationError, match="Missing index field"):
        decode_indices_field(
            {},
            idx_key="train_idx",
            ranges_key="train_ranges",
            field="train",
        )
