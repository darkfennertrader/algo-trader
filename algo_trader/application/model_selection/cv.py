from __future__ import annotations

from itertools import chain, combinations
import random
from typing import Sequence

from algo_trader.domain import ModelSelectionError
from algo_trader.domain.model_selection import (
    BaseCVSplitter,
    CVConfig,
    CVSampling,
)


class CombinatorialPurgedCV(BaseCVSplitter):
    def __init__(self, config: CVConfig) -> None:
        _validate_cv_config(config)
        self.config = config

    def split(
        self, dates: Sequence[object]
    ) -> list[tuple[list[int], list[int]]]:
        n_samples = len(dates)
        if n_samples == 0:
            return []
        if n_samples < self.config.n_blocks:
            raise ModelSelectionError(
                "Number of samples must be >= n_blocks",
                context={
                    "samples": str(n_samples),
                    "n_blocks": str(self.config.n_blocks),
                },
            )
        blocks = _make_blocks(n_samples, self.config.n_blocks)
        _validate_block_sizes(blocks, self.config)
        _validate_min_train_size(blocks, self.config)
        combos = list(
            combinations(range(len(blocks)), self.config.test_block_size)
        )
        combos = _select_combinations(combos, self.config.sampling)
        splits: list[tuple[list[int], list[int]]] = []
        all_indices = set(range(n_samples))
        for combo in combos:
            test_blocks = [blocks[idx] for idx in combo]
            test_indices = _build_test_indices(test_blocks)
            train_indices = _build_train_indices(
                candidate_train=all_indices.difference(test_indices),
                test_blocks=test_blocks,
                n_samples=n_samples,
                purge_window=self.config.purge_size,
                embargo_size=self.config.embargo_size,
            )
            if not train_indices:
                raise ModelSelectionError(
                    "Training indices empty after purging/embargo",
                    context={
                        "combo": str(combo),
                        "purge_size": str(self.config.purge_size),
                        "embargo_size": str(self.config.embargo_size),
                    },
                )
            _ensure_min_train_size(
                train_indices=train_indices,
                min_train_size=self.config.guards.min_train_size,
                combo=combo,
            )
            splits.append((sorted(train_indices), test_indices))
        return splits


def _validate_cv_config(config: CVConfig) -> None:
    if config.n_blocks <= 0:
        raise ModelSelectionError("n_blocks must be positive")
    if config.test_block_size <= 0:
        raise ModelSelectionError("test_block_size must be positive")
    if config.test_block_size > config.n_blocks:
        raise ModelSelectionError(
            "test_block_size must be <= n_blocks",
            context={
                "test_block_size": str(config.test_block_size),
                "n_blocks": str(config.n_blocks),
            },
        )
    if config.embargo_size < 0 or config.purge_size < 0:
        raise ModelSelectionError(
            "purge_size and embargo_size must be non-negative"
        )
    if (
        config.guards.min_train_size is not None
        and config.guards.min_train_size <= 0
    ):
        raise ModelSelectionError("min_train_size must be positive")
    if (
        config.guards.min_block_size is not None
        and config.guards.min_block_size <= 0
    ):
        raise ModelSelectionError("min_block_size must be positive")
    _validate_sampling(config.sampling)


def _validate_sampling(sampling: CVSampling) -> None:
    if sampling.max_splits is not None and sampling.max_splits <= 0:
        raise ModelSelectionError("max_splits must be positive")


def _make_blocks(n_samples: int, n_blocks: int) -> list[list[int]]:
    base_size = n_samples // n_blocks
    remainder = n_samples % n_blocks
    sizes = [base_size] * n_blocks
    for idx in range(remainder):
        sizes[idx] += 1
    blocks: list[list[int]] = []
    start = 0
    for size in sizes:
        end = start + size
        blocks.append(list(range(start, end)))
        start = end
    return blocks


def _validate_block_sizes(
    blocks: Sequence[Sequence[int]], config: CVConfig
) -> None:
    if config.guards.min_block_size is None:
        return
    min_block_size = min(len(block) for block in blocks)
    if min_block_size < config.guards.min_block_size:
        raise ModelSelectionError(
            "Minimum block size below min_block_size",
            context={
                "min_block_size": str(min_block_size),
                "required_min_block_size": str(config.guards.min_block_size),
            },
        )


def _validate_min_train_size(
    blocks: Sequence[Sequence[int]], config: CVConfig
) -> None:
    if config.guards.min_train_size is None:
        return
    block_sizes = [len(block) for block in blocks]
    max_train_size = _max_train_size(block_sizes, config.test_block_size)
    if config.guards.min_train_size > max_train_size:
        raise ModelSelectionError(
            "min_train_size larger than max possible training size",
            context={
                "min_train_size": str(config.guards.min_train_size),
                "max_train_size": str(max_train_size),
            },
        )


def _max_train_size(block_sizes: Sequence[int], test_block_size: int) -> int:
    if not block_sizes or test_block_size <= 0:
        return 0
    smallest_test = sorted(block_sizes)[:test_block_size]
    return sum(block_sizes) - sum(smallest_test)


def _ensure_min_train_size(
    *,
    train_indices: set[int],
    min_train_size: int | None,
    combo: tuple[int, ...],
) -> None:
    if min_train_size is None:
        return
    train_size = len(train_indices)
    if train_size < min_train_size:
        raise ModelSelectionError(
            "Training indices below min_train_size",
            context={
                "combo": str(combo),
                "train_size": str(train_size),
                "min_train_size": str(min_train_size),
            },
        )


def _select_combinations(
    combos: list[tuple[int, ...]], sampling: CVSampling
) -> list[tuple[int, ...]]:
    if sampling.shuffle_splits or sampling.max_splits is not None:
        combos = list(combos)
    if sampling.max_splits is not None and len(combos) > sampling.max_splits:
        if sampling.shuffle_splits:
            rng = _random_for_sampling(sampling)
            return rng.sample(combos, sampling.max_splits)
        return combos[: sampling.max_splits]
    if sampling.shuffle_splits:
        rng = _random_for_sampling(sampling)
        rng.shuffle(combos)
    return combos


def _random_for_sampling(sampling: CVSampling) -> random.Random:
    seed = sampling.random_state
    return random.Random(seed)


def _build_test_indices(test_blocks: Sequence[Sequence[int]]) -> list[int]:
    return sorted(set(chain.from_iterable(test_blocks)))


def _build_train_indices(
    *,
    candidate_train: set[int],
    test_blocks: Sequence[Sequence[int]],
    n_samples: int,
    purge_window: int,
    embargo_size: int,
) -> set[int]:
    train_indices = set(candidate_train)
    if purge_window > 0:
        for block in test_blocks:
            if not block:
                continue
            _purge_block(train_indices, block, n_samples, purge_window)
    if embargo_size > 0:
        _apply_embargo(train_indices, test_blocks, n_samples, embargo_size)
    return train_indices


def _purge_block(
    train_indices: set[int],
    block: Sequence[int],
    n_samples: int,
    purge_window: int,
) -> None:
    block_start = block[0]
    block_end = block[-1]
    purge_start = max(0, block_start - purge_window)
    purge_end = min(n_samples - 1, block_end + purge_window)
    for idx in range(purge_start, purge_end + 1):
        train_indices.discard(idx)


def _apply_embargo(
    train_indices: set[int],
    test_blocks: Sequence[Sequence[int]],
    n_samples: int,
    embargo_size: int,
) -> None:
    last_test_index = _last_test_index(test_blocks)
    if last_test_index is None:
        return
    emb_start = last_test_index + 1
    emb_end = min(n_samples - 1, last_test_index + embargo_size)
    for idx in range(emb_start, emb_end + 1):
        train_indices.discard(idx)


def _last_test_index(test_blocks: Sequence[Sequence[int]]) -> int | None:
    last_indices = [block[-1] for block in test_blocks if block]
    if not last_indices:
        return None
    return max(last_indices)
