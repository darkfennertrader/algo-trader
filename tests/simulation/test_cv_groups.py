import numpy as np

from algo_trader.application.simulation.cv_groups import (
    build_equal_groups,
    make_cpcv_splits,
)
from algo_trader.domain.simulation import CPCVParams, CVLeakage, CVParams, CVWindow


def test_build_equal_groups_truncates() -> None:
    warmup_idx, groups = build_equal_groups(T=10, warmup_len=2, group_len=3)
    assert warmup_idx.tolist() == [0, 1]
    assert len(groups) == 2
    assert groups[0].tolist() == [2, 3, 4]
    assert groups[1].tolist() == [5, 6, 7]


def test_cpcv_purge_and_embargo() -> None:
    warmup_idx, groups = build_equal_groups(T=8, warmup_len=0, group_len=2)
    params = CVParams(
        window=CVWindow(warmup_len=0, group_len=2),
        leakage=CVLeakage(horizon=1, embargo_len=1),
        cpcv=CPCVParams(q=1, max_inner_combos=None, seed=1),
        include_warmup_in_inner_train=False,
    )
    splits = make_cpcv_splits(
        warmup_idx=warmup_idx,
        groups=groups,
        inner_group_ids=[0, 1, 2],
        params=params,
    )
    target = next(split for split in splits if split.test_group_ids == (1,))
    assert target.test_idx.tolist() == [2, 3]
    assert target.train_idx.tolist() == [0, 1, 5]
