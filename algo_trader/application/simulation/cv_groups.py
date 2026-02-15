from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import itertools
import numpy as np

from algo_trader.domain.simulation import CPCVSplit, CVParams, OuterFold


def build_equal_groups(
    T: int, warmup_len: int, group_len: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if warmup_len < 0 or warmup_len >= T:
        raise ValueError("warmup_len must be in [0, T)")
    if group_len <= 0:
        raise ValueError("group_len must be positive")

    warmup_idx = np.arange(0, warmup_len, dtype=int)

    usable_start = warmup_len
    usable_len = T - usable_start
    remainder = usable_len % group_len
    aligned_start = usable_start + remainder
    group_count = (T - aligned_start) // group_len

    groups: List[np.ndarray] = []
    for group_id in range(group_count):
        start = aligned_start + group_id * group_len
        end = start + group_len
        groups.append(np.arange(start, end, dtype=int))

    return warmup_idx, groups


def make_outer_folds(
    warmup_idx: np.ndarray,
    groups: List[np.ndarray],
    outer_test_group_ids: Sequence[int],
) -> List[OuterFold]:
    folds: List[OuterFold] = []
    for k in outer_test_group_ids:
        if k < 0 or k >= len(groups):
            raise ValueError("outer_test_group_id out of range")
        test_idx = groups[k]
        prior_groups = groups[:k]
        if prior_groups:
            train_idx = np.concatenate([warmup_idx] + prior_groups)
        else:
            train_idx = warmup_idx.copy()
        folds.append(
            OuterFold(
                k_test=k,
                train_idx=np.unique(train_idx),
                test_idx=np.unique(test_idx),
                inner_group_ids=list(range(0, k)),
            )
        )
    return folds


def _combo_balance_objective(
    counts: Dict[int, int],
    combo: Tuple[int, ...],
    target: float,
) -> Tuple[float, float, int]:
    prospective = dict(counts)
    for group_id in combo:
        prospective[group_id] += 1
    vals = np.array(list(prospective.values()), dtype=float)
    imbalance_range = float(vals.max() - vals.min())
    sq_dev = float(np.sum((vals - target) ** 2))
    max_count = int(vals.max())
    return (imbalance_range, sq_dev, max_count)


def sample_cpcv_combos(
    group_ids: Sequence[int],
    q: int,
    max_combos: int | None,
    seed: int,
) -> List[Tuple[int, ...]]:
    group_ids = list(group_ids)
    all_combos = list(itertools.combinations(group_ids, q))

    if (max_combos is None) or (max_combos >= len(all_combos)):
        return all_combos

    return _select_balanced_combos(all_combos, group_ids, q, max_combos, seed)


def _select_balanced_combos(
    all_combos: List[Tuple[int, ...]],
    group_ids: List[int],
    q: int,
    max_combos: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    n_groups = len(group_ids)
    target = (max_combos * q) / max(1, n_groups)
    counts = {gid: 0 for gid in group_ids}
    remaining = all_combos.copy()
    chosen: List[Tuple[int, ...]] = []
    rng = np.random.default_rng(seed)

    for _ in range(max_combos):
        pick_i = _pick_best_combo(counts, remaining, target, rng)
        combo = remaining.pop(pick_i)
        chosen.append(combo)
        for group_id in combo:
            counts[group_id] += 1
        if not remaining:
            break

    return chosen


def _pick_best_combo(
    counts: Dict[int, int],
    remaining: List[Tuple[int, ...]],
    target: float,
    rng: np.random.Generator,
) -> int:
    best_obj = None
    best_idxs: List[int] = []

    for i, combo in enumerate(remaining):
        obj = _combo_balance_objective(counts, combo, target)
        if (best_obj is None) or (obj < best_obj):
            best_obj = obj
            best_idxs = [i]
        elif obj == best_obj:
            best_idxs.append(i)

    if len(best_idxs) == 1:
        return best_idxs[0]
    return int(rng.choice(best_idxs))


def _overlaps_label_interval_vec(
    train_t: np.ndarray,
    test_start: int,
    test_end: int,
    horizon: int,
) -> np.ndarray:
    return (train_t < (test_end + horizon)) & ((train_t + horizon) > test_start)


def _apply_purge_and_embargo(
    train_idx: np.ndarray,
    test_blocks: Sequence[np.ndarray],
    horizon: int,
    embargo_len: int,
) -> np.ndarray:
    exclude = np.zeros(train_idx.shape[0], dtype=bool)
    for block in test_blocks:
        blk_start = int(block[0])
        blk_end = int(block[-1]) + 1

        exclude |= _overlaps_label_interval_vec(
            train_idx, blk_start, blk_end, horizon
        )

        emb_start = blk_end
        emb_end = blk_end + embargo_len
        exclude |= (train_idx >= emb_start) & (train_idx < emb_end)

    return train_idx[~exclude]


def _build_train_indices(
    inner_group_ids: Sequence[int],
    combo: Tuple[int, ...],
    group_map: dict[int, np.ndarray],
) -> np.ndarray | None:
    train_group_ids = [group_id for group_id in inner_group_ids if group_id not in combo]
    if not train_group_ids:
        return None
    return np.sort(
        np.concatenate([np.sort(group_map[g]) for g in train_group_ids])
    )


def make_cpcv_splits(
    warmup_idx: np.ndarray,
    groups: List[np.ndarray],
    inner_group_ids: Sequence[int],
    params: CVParams,
) -> List[CPCVSplit]:
    if params.leakage.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if params.leakage.embargo_len < 0:
        raise ValueError("embargo_len must be >= 0")

    inner_group_ids = list(inner_group_ids)
    group_map = {gid: groups[gid] for gid in inner_group_ids}

    combos = sample_cpcv_combos(
        group_ids=inner_group_ids,
        q=params.cpcv.q,
        max_combos=params.cpcv.max_inner_combos,
        seed=params.cpcv.seed,
    )

    splits: List[CPCVSplit] = []

    for combo in combos:
        test_blocks = [np.sort(group_map[g]) for g in combo]
        test_idx = np.sort(np.concatenate(test_blocks))

        train_idx = _build_train_indices(inner_group_ids, combo, group_map)
        if train_idx is None:
            continue

        train_idx = _apply_purge_and_embargo(
            train_idx=train_idx,
            test_blocks=test_blocks,
            horizon=params.leakage.horizon,
            embargo_len=params.leakage.embargo_len,
        )

        if params.include_warmup_in_inner_train and warmup_idx.size > 0:
            train_idx = np.concatenate([warmup_idx, train_idx])

        train_idx = np.unique(train_idx)
        test_idx = np.unique(test_idx)

        if train_idx.size == 0 or test_idx.size == 0:
            continue

        splits.append(
            CPCVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                test_group_ids=tuple(combo),
            )
        )

    return splits
