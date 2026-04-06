from __future__ import annotations

from pathlib import Path

from algo_trader.application.simulation.downstream_plots import (
    write_downstream_plots,
)
from tests.simulation.helpers import (
    write_log_weekly_data_source_metadata,
)


def test_write_downstream_plots_creates_downstream_charts(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "simulation_run"
    walkforward_dir = base_dir / "walkforward"
    walkforward_dir.mkdir(parents=True, exist_ok=True)
    write_log_weekly_data_source_metadata(base_dir)
    (walkforward_dir / "stitched_returns.csv").write_text(
        "\n".join(
            [
                "timestamp,primary,equal_weight",
                "2024-01-05,0.01,0.005",
                "2024-01-12,0.02,0.004",
            ]
        ),
        encoding="utf-8",
    )

    write_downstream_plots(base_dir=base_dir, dataset_params={})

    assert (
        walkforward_dir / "plots" / "cumulative_returns.png"
    ).exists()
    assert (
        walkforward_dir / "plots" / "underwater_drawdown.png"
    ).exists()
