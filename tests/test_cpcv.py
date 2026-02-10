import pytest

from algo_trader.application.model_selection import CombinatorialPurgedCV
from algo_trader.domain import ModelSelectionError
from algo_trader.domain.model_selection import CVConfig, CVGuards


def test_cpcv_min_block_size_guard() -> None:
    config = CVConfig(
        n_blocks=4, test_block_size=1, guards=CVGuards(min_block_size=3)
    )
    splitter = CombinatorialPurgedCV(config)
    with pytest.raises(ModelSelectionError):
        splitter.split(list(range(10)))


def test_cpcv_min_train_size_impossible() -> None:
    config = CVConfig(
        n_blocks=5, test_block_size=2, guards=CVGuards(min_train_size=7)
    )
    splitter = CombinatorialPurgedCV(config)
    with pytest.raises(ModelSelectionError):
        splitter.split(list(range(10)))


def test_cpcv_empty_train_after_purge_raises() -> None:
    config = CVConfig(n_blocks=3, test_block_size=1, purge_size=2)
    splitter = CombinatorialPurgedCV(config)
    with pytest.raises(ModelSelectionError):
        splitter.split(list(range(6)))
