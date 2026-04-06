from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tqdm import tqdm

from algo_trader.domain.simulation import OuterFold

from ..io_utils import format_timestamp_date


@dataclass
class WalkforwardProgress:
    _bar: tqdm

    def update(self, outer_k: int, timestamp: Any) -> None:
        self._bar.set_postfix_str(
            f"outer={outer_k} date={format_timestamp_date(timestamp)}"
        )
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


@dataclass
class SeedStudyProgress:
    _bar: tqdm

    def show(self) -> None:
        self._bar.refresh()

    def update(self, seed: int) -> None:
        self._bar.set_postfix_str(f"seed={seed}")
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


def build_walkforward_progress(
    outer_folds: Sequence[OuterFold],
) -> WalkforwardProgress | None:
    total_weeks = sum(int(len(fold.test_idx)) for fold in outer_folds)
    if total_weeks <= 0:
        return None
    return WalkforwardProgress(
        _bar=tqdm(
            total=total_weeks,
            desc="walkforward",
            dynamic_ncols=True,
            delay=1.0,
        )
    )


def build_seed_stability_progress(
    total_seeds: int,
) -> SeedStudyProgress | None:
    if total_seeds <= 0:
        return None
    progress = SeedStudyProgress(
        _bar=tqdm(
            total=total_seeds,
            desc="seed_stability",
            dynamic_ncols=True,
        )
    )
    progress.show()
    return progress
