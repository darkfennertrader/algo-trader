from pathlib import Path

import numpy as np

from algo_trader.application.simulation.split_timeline_plot import (
    write_splits_timeline_plot,
)
from algo_trader.domain.simulation import CPCVSplit


def test_write_splits_timeline_plot_creates_png(tmp_path: Path) -> None:
    splits = [
        CPCVSplit(
            train_idx=np.arange(10, 20, dtype=int),
            test_idx=np.arange(0, 5, dtype=int),
            test_group_ids=(0,),
            purged_idx=np.array([5], dtype=int),
            embargoed_idx=np.arange(5, 8, dtype=int),
        ),
        CPCVSplit(
            train_idx=np.concatenate(
                [
                    np.arange(0, 5, dtype=int),
                    np.arange(13, 20, dtype=int),
                ]
            ),
            test_idx=np.arange(8, 13, dtype=int),
            test_group_ids=(1,),
            purged_idx=np.array([], dtype=int),
            embargoed_idx=np.array([], dtype=int),
        ),
    ]
    output_path = tmp_path / "splits_timeline.png"
    write_splits_timeline_plot(
        output_path=output_path,
        splits=splits,
        warmup_idx=np.arange(0, 3, dtype=int),
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
