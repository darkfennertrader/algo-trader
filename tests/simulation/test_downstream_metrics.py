from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from algo_trader.application.simulation.data_source_metadata import (
    DataSourceMetadata,
    load_data_source_metadata,
    write_data_source_metadata,
)
from algo_trader.application.simulation.downstream_metrics import (
    write_downstream_metrics,
)
from tests.simulation.helpers import (
    write_log_weekly_data_source_metadata,
)


def test_write_downstream_metrics_creates_summary_files(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "simulation_run"
    portfolio_dir = base_dir / "portfolios" / "primary"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "stitched_returns.csv").write_text(
        "timestamp,primary\n2024-01-05,0.01\n",
        encoding="utf-8",
    )
    write_log_weekly_data_source_metadata(base_dir)
    pd.DataFrame(
        {
            "timestamp": ["2024-01-05", "2024-01-12"],
            "outer_k": [0, 0],
            "gross_return": [0.0200, 0.0300],
            "net_return": [0.0190, 0.0280],
            "cost": [0.0010, 0.0020],
            "turnover": [0.0000, 0.2000],
        }
    ).to_csv(portfolio_dir / "weekly_returns.csv", index=False)

    write_downstream_metrics(base_dir=base_dir, dataset_params={})

    summary_csv = pd.read_csv(base_dir / "metrics" / "summary.csv")
    summary_json = json.loads(
        (base_dir / "metrics" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    primary_json = json.loads(
        (
            base_dir
            / "metrics"
            / "by_portfolio"
            / "primary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary_csv["portfolio_name"].tolist() == ["primary"]
    assert summary_json["metadata"]["return_type"] == "log"
    assert summary_json["metadata"]["return_frequency"] == "weekly"
    assert primary_json["initial_capital"] == 100.0
    assert primary_json["n_periods"] == 2
    assert primary_json["total_cost"] == 0.003
    assert primary_json["mean_turnover"] == 0.1
    assert primary_json["max_drawdown"] == 0.0
    expected_wealth = 100.0 * math.exp(0.019 + 0.028)
    assert math.isclose(primary_json["final_net_wealth"], expected_wealth)


def test_write_data_source_metadata_writes_resolved_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = DataSourceMetadata(
        version_label="2026-14",
        return_type="log",
        return_frequency="weekly",
        data_lake_dir="/tmp/data_lake/2026-14",
    )

    monkeypatch.setattr(
        "algo_trader.application.simulation.data_source_metadata._resolve_data_source_metadata",
        lambda _: expected,
    )

    metadata = write_data_source_metadata(
        base_dir=tmp_path / "simulation_run",
        dataset_params={"feature_store": "unused"},
    )

    assert metadata == expected
    loaded = load_data_source_metadata(
        base_dir=tmp_path / "simulation_run",
        dataset_params={},
    )
    assert loaded == expected
