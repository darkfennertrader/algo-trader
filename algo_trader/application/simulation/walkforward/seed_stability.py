from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import logging
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence, cast

import pandas as pd

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import SimulationConfig

from ..artifacts import resolve_saved_study_dir, resolve_simulation_output_dir
from ..config import config_to_input_dict
from ..io_utils import write_json_file
from ..seed_stability_common import (
    SeedStudyResult,
    SeedStudyTask,
    load_seed_summary,
    prepare_seed_task,
    run_logged_seed_task,
    run_seed_tasks_from_config,
    seed_name,
)
from .pathing import (
    resolve_portfolio_base_dir,
    seed_stability_dir,
)
logger = logging.getLogger(__name__)


def run_seed_stability_study(
    *,
    config: SimulationConfig,
) -> Mapping[str, Any]:
    _validate_seed_stability_config(config)
    source_dir = _resolve_source_dir(config)
    study_dir = _resolve_study_dir(config=config, source_dir=source_dir)
    tasks = _build_seed_tasks(config=config, study_dir=study_dir)
    _stage_seed_tasks(source_dir=source_dir, tasks=tasks)
    results = _run_seed_tasks(config=config, tasks=tasks)
    aggregate = _write_seed_stability_outputs(
        study_dir=study_dir,
        source_dir=source_dir,
        results=results,
    )
    return aggregate


def _validate_seed_stability_config(config: SimulationConfig) -> None:
    if config.flags.execution_mode != "walkforward":
        raise ConfigError(
            "walkforward seed study requires execution.mode=walkforward"
        )
    if config.walkforward.num_seeds <= 1:
        raise ConfigError(
            "walkforward seed study requires walkforward.num_seeds > 1"
        )


def _resolve_source_dir(config: SimulationConfig) -> Path:
    source_root = resolve_simulation_output_dir(
        simulation_output_path=config.data.simulation_output_path,
        dataset_params=config.data.dataset_params,
    )
    if not source_root.exists():
        raise SimulationError(
            "Seed-stability source experiment directory is missing",
            context={"path": str(source_root)},
        )
    source_dir = resolve_saved_study_dir(source_root)
    _validate_source_inputs(source_dir)
    _validate_source_best_configs(source_dir)
    return source_dir


def _resolve_study_dir(
    *,
    config: SimulationConfig,
    source_dir: Path,
) -> Path:
    portfolio_dir = resolve_portfolio_base_dir(
        source_dir=source_dir,
        portfolio_output_path=config.data.portfolio_output_path,
        dataset_params=config.data.dataset_params,
    )
    return seed_stability_dir(portfolio_dir)


def _validate_source_inputs(source_dir: Path) -> None:
    inputs_dir = source_dir / "inputs"
    if not inputs_dir.exists():
        raise SimulationError(
            "Seed-stability source inputs directory is missing",
            context={"path": str(inputs_dir)},
        )
    panel_tensor_path = inputs_dir / "panel_tensor.pt"
    if not panel_tensor_path.exists():
        raise SimulationError(
            "Seed-stability source panel tensor is missing",
            context={"path": str(panel_tensor_path)},
        )


def _validate_source_best_configs(source_dir: Path) -> None:
    global_best_path = source_dir / "outer" / "best_config.json"
    inner_best_paths = tuple(source_dir.glob("inner/outer_*/best_config.json"))
    if global_best_path.exists() or inner_best_paths:
        return
    raise SimulationError(
        "Seed-stability source experiment has no saved best config artifacts",
        context={"path": str(source_dir)},
    )


def _build_seed_tasks(
    *,
    config: SimulationConfig,
    study_dir: Path,
) -> tuple[SeedStudyTask, ...]:
    return tuple(
        _build_seed_task(
            config=config,
            study_dir=study_dir,
            seed=seed,
        )
        for seed in config.walkforward.seeds
    )


def _build_seed_task(
    *,
    config: SimulationConfig,
    study_dir: Path,
    seed: int,
) -> SeedStudyTask:
    payload = _build_seed_config_payload(
        config=config,
        output_name=task_seed_output_name(seed),
        seed=seed,
    )
    return prepare_seed_task(
        study_dir=study_dir,
        output_name=task_seed_output_name(seed),
        seed=seed,
        payload=payload,
        write_message="Failed to write seed-stability simulation config",
    )


def task_seed_output_name(seed: int) -> str:
    return seed_name(seed)


def _build_seed_config_payload(
    *,
    config: Any,
    output_name: str,
    seed: int,
) -> dict[str, Any]:
    payload = _config_payload_dict(config)
    walkforward = _dict_field(payload, "walkforward")
    walkforward["num_seeds"] = 1
    walkforward["seeds"] = [seed]
    walkforward["max_parallel_seeds_per_gpu"] = 1
    data = _dict_field(payload, "data")
    data["simulation_output_path"] = output_name
    data.pop("portfolio_output_path", None)
    payload["walkforward"] = walkforward
    payload["data"] = data
    return payload


def _config_payload_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return cast(
            dict[str, Any],
            config_to_input_dict(cast(SimulationConfig, config)),
        )
    return {
        "walkforward": _namespace_to_dict(config.walkforward),
        "data": _namespace_to_dict(config.data),
        "flags": _namespace_to_dict(config.flags),
    }


def _namespace_to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return cast(dict[str, Any], asdict(cast(Any, value)))
    mapping = getattr(value, "__dict__", None)
    if isinstance(mapping, Mapping):
        return {
            str(key): _namespace_to_dict(item)
            if _is_namespace_like(item)
            else item
            for key, item in mapping.items()
        }
    raise ConfigError("Seed-stability config payload is not serializable")


def _is_namespace_like(value: Any) -> bool:
    return is_dataclass(value) or hasattr(value, "__dict__")


def _dict_field(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Seed-stability payload missing mapping field '{key}'")
    return dict(value)


def _stage_seed_tasks(
    *,
    source_dir: Path,
    tasks: Sequence[SeedStudyTask],
) -> None:
    for task in tasks:
        _stage_seed_task(source_dir=source_dir, task=task)


def _stage_seed_task(
    *,
    source_dir: Path,
    task: SeedStudyTask,
) -> None:
    _copy_directory(
        source=source_dir / "inputs",
        target=task.output_dir / "inputs",
        message="Failed to stage seed-stability inputs",
    )
    _copy_optional_file(
        source=source_dir / "outer" / "best_config.json",
        target=task.output_dir / "outer" / "best_config.json",
        required=False,
        message="Failed to stage global best config",
    )
    for source_path in source_dir.glob("inner/outer_*/best_config.json"):
        relative_path = source_path.relative_to(source_dir)
        _copy_optional_file(
            source=source_path,
            target=task.output_dir / relative_path,
            required=True,
            message="Failed to stage per-outer best config",
        )


def _copy_directory(
    *,
    source: Path,
    target: Path,
    message: str,
) -> None:
    if not source.exists():
        raise SimulationError(message, context={"path": str(source)})
    try:
        shutil.copytree(source, target, dirs_exist_ok=True)
    except Exception as exc:
        raise SimulationError(
            message,
            context={"source": str(source), "target": str(target)},
        ) from exc


def _copy_optional_file(
    *,
    source: Path,
    target: Path,
    required: bool,
    message: str,
) -> None:
    if not source.exists():
        if required:
            raise SimulationError(message, context={"path": str(source)})
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
    except Exception as exc:
        raise SimulationError(
            message,
            context={"source": str(source), "target": str(target)},
        ) from exc


def _run_seed_tasks(
    *,
    config: SimulationConfig,
    tasks: Sequence[SeedStudyTask],
) -> tuple[SeedStudyResult, ...]:
    if not tasks:
        return ()
    return run_seed_tasks_from_config(
        config=config,
        tasks=tasks,
        parallel_error_message=(
            "walkforward seed parallelization requires at least one CUDA device"
        ),
        task_runner=_run_seed_task,
    )


def _run_seed_task(
    task: SeedStudyTask,
    gpu_id: int | None,
) -> SeedStudyResult:
    return run_logged_seed_task(
        task=task,
        gpu_id=gpu_id,
        failure_message="Seed-stability walkforward seed run failed",
        simulation_root=task.output_dir.parent,
    )


def _write_seed_stability_outputs(
    *,
    study_dir: Path,
    source_dir: Path,
    results: Sequence[SeedStudyResult],
) -> Mapping[str, Any]:
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = _build_seed_summary_frame(results)
    dispersion_frame = _build_dispersion_frame(summary_frame)
    summary_path = study_dir / "summary.csv"
    dispersion_path = study_dir / "dispersion.csv"
    _write_csv(summary_path, summary_frame, "Failed to write seed summary")
    _write_csv(
        dispersion_path,
        dispersion_frame,
        "Failed to write seed dispersion summary",
    )
    payload = {
        "source_experiment_dir": str(source_dir),
        "seed_count": len(results),
        "seeds": [result.seed for result in results],
        "summary_csv": str(summary_path),
        "dispersion_csv": str(dispersion_path),
    }
    write_json_file(
        path=study_dir / "summary.json",
        payload=payload,
        message="Failed to write seed stability summary",
    )
    return {
        "seed_stability": payload,
    }


def _build_seed_summary_frame(
    results: Sequence[SeedStudyResult],
) -> pd.DataFrame:
    frames = [
        _load_seed_summary(result)
        for result in results
    ]
    return pd.concat(frames, ignore_index=True)


def _load_seed_summary(result: SeedStudyResult) -> pd.DataFrame:
    walkforward_path = (
        result.output_dir / "walkforward" / "metrics" / "summary.csv"
    )
    direct_path = result.output_dir / "metrics" / "summary.csv"
    path = (
        walkforward_path
        if walkforward_path.exists()
        else direct_path
    )
    return load_seed_summary(
        result=result,
        path=path,
        message="Failed to read seed downstream metrics summary",
    )


def _build_dispersion_frame(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "annualized_return_geometric",
        "sharpe",
        "max_drawdown",
        "total_turnover",
        "mean_turnover",
    ]
    grouped = summary.groupby("portfolio_name", sort=True)[metric_columns]
    aggregated = grouped.agg(["mean", "std", "min", "max"])
    column_index = cast(Any, aggregated.columns)
    aggregated.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in column_index.to_flat_index()
    ]
    return aggregated.reset_index()


def _write_csv(path: Path, frame: pd.DataFrame, message: str) -> None:
    try:
        frame.to_csv(path, index=False)
    except Exception as exc:
        raise SimulationError(message, context={"path": str(path)}) from exc
