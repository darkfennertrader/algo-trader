from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from algo_trader.application.simulation.walkforward import seed_stability


def test_build_seed_config_payload_overrides_seed_and_output() -> None:
    config = SimpleNamespace(
        walkforward=SimpleNamespace(
            num_seeds=3,
            seeds=(7, 19, 43),
            max_parallel_seeds_per_gpu=2,
        ),
        data=SimpleNamespace(
            simulation_output_path="validated_model",
            portfolio_output_path="portfolio/herc",
            dataset_params={},
        ),
        flags=SimpleNamespace(
            execution_mode="walkforward",
            use_gpu=True,
            simulation_mode="full",
            smoke_test_enabled=False,
            smoke_test_debug=False,
            use_feature_names_for_scaling=True,
        ),
    )

    build_payload = getattr(
        seed_stability,
        "_build_seed_config_payload",
    )
    payload = build_payload(
        config=config,
        output_name="seed_19",
        seed=19,
    )

    assert payload["walkforward"]["num_seeds"] == 1
    assert payload["walkforward"]["seeds"] == [19]
    assert payload["walkforward"]["max_parallel_seeds_per_gpu"] == 1
    assert payload["data"]["simulation_output_path"] == "seed_19"
    assert "portfolio_output_path" not in payload["data"]
    assert "paths" not in payload["data"]


def test_stage_seed_task_copies_inputs_and_best_configs(tmp_path: Path) -> None:
    source_dir = tmp_path / "validated_model"
    (source_dir / "inputs").mkdir(parents=True)
    (source_dir / "outer").mkdir(parents=True)
    (source_dir / "inner" / "outer_40").mkdir(parents=True)
    (source_dir / "inputs" / "panel_tensor.pt").write_text(
        "panel", encoding="utf-8"
    )
    (source_dir / "inputs" / "data_source.json").write_text(
        "{}", encoding="utf-8"
    )
    (source_dir / "outer" / "best_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (source_dir / "inner" / "outer_40" / "best_config.json").write_text(
        "{}", encoding="utf-8"
    )
    task = seed_stability.SeedStudyTask(
        seed=7,
        output_label="validated_model/walkforward/seed_stability/seed_7",
        output_dir=tmp_path / "validated_model" / "walkforward" / "seed_stability" / "seed_7",
        config_path=tmp_path / "validated_model" / "walkforward" / "seed_stability" / "seed_7" / "simulation.seed.yml",
        log_path=tmp_path / "validated_model" / "walkforward" / "seed_stability" / "seed_7" / "seed_run.log",
    )

    stage_seed_task = getattr(seed_stability, "_stage_seed_task")
    stage_seed_task(
        source_dir=source_dir,
        task=task,
    )

    assert (task.output_dir / "inputs" / "panel_tensor.pt").exists()
    assert (task.output_dir / "inputs" / "data_source.json").exists()
    assert (task.output_dir / "outer" / "best_config.json").exists()
    assert (
        task.output_dir / "inner" / "outer_40" / "best_config.json"
    ).exists()


def test_write_seed_stability_outputs_aggregates_per_seed_metrics(
    tmp_path: Path,
) -> None:
    results = []
    for seed, annualized_return in ((7, 0.1), (19, 0.2)):
        seed_dir = tmp_path / f"seed_{seed}"
        metrics_dir = seed_dir / "metrics"
        metrics_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "portfolio_name": "primary",
                    "annualized_return_geometric": annualized_return,
                    "sharpe": 1.0 + annualized_return,
                    "max_drawdown": 0.1,
                    "total_turnover": 2.0,
                    "mean_turnover": 0.5,
                }
            ]
        ).to_csv(metrics_dir / "summary.csv", index=False)
        results.append(
            seed_stability.SeedStudyResult(
                seed=seed,
                output_dir=seed_dir,
                log_path=seed_dir / "seed_run.log",
            )
        )

    write_outputs = getattr(seed_stability, "_write_seed_stability_outputs")
    summary = write_outputs(
        study_dir=tmp_path / "walkforward" / "seed_stability",
        source_dir=tmp_path / "validated_model",
        results=tuple(results),
    )

    summary_csv = tmp_path / "walkforward" / "seed_stability" / "summary.csv"
    dispersion_csv = tmp_path / "walkforward" / "seed_stability" / "dispersion.csv"
    assert summary["seed_stability"]["seed_count"] == 2
    assert summary_csv.exists()
    assert dispersion_csv.exists()
    summary_frame = pd.read_csv(summary_csv)
    assert summary_frame["seed"].tolist() == [7, 19]


def test_resolve_study_dir_uses_portfolio_output_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIMULATION_SOURCE", str(tmp_path))
    config = SimpleNamespace(
        data=SimpleNamespace(
            portfolio_output_path="portfolio/herc",
            dataset_params={},
        )
    )

    resolve_study_dir = getattr(seed_stability, "_resolve_study_dir")
    study_dir = resolve_study_dir(
        config=config,
        source_dir=tmp_path / "validated_model",
    )

    assert study_dir == tmp_path / "portfolio" / "herc" / "seed_stability"
