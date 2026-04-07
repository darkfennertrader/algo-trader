from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, is_dataclass
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence, cast

import pandas as pd
import torch
import yaml

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import SimulationConfig

from ..artifacts import resolve_simulation_output_dir
from ..config import config_to_input_dict
from ..io_utils import write_json_file
from .pathing import (
    resolve_portfolio_base_dir,
    seed_stability_dir,
    walkforward_dir,
)
from .progress import SeedStudyProgress, build_seed_stability_progress

logger = logging.getLogger(__name__)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class SeedStudyTask:
    seed: int
    output_label: str
    output_dir: Path
    config_path: Path
    log_path: Path


@dataclass(frozen=True)
class SeedStudyResult:
    seed: int
    output_dir: Path
    log_path: Path


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
    source_dir = resolve_simulation_output_dir(
        simulation_output_path=config.data.simulation_output_path,
        dataset_params=config.data.dataset_params,
    )
    if not source_dir.exists():
        raise SimulationError(
            "Seed-stability source experiment directory is missing",
            context={"path": str(source_dir)},
        )
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
    output_dir = study_dir / task_seed_output_name(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "simulation.seed.yml"
    payload = _build_seed_config_payload(
        config=config,
        output_name=task_seed_output_name(seed),
        seed=seed,
    )
    _write_seed_config(config_path=config_path, payload=payload)
    return SeedStudyTask(
        seed=seed,
        output_label=str(output_dir),
        output_dir=output_dir,
        config_path=config_path,
        log_path=output_dir / "seed_run.log",
    )


def task_seed_output_name(seed: int) -> str:
    return f"seed_{seed}"


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


def _write_seed_config(
    *,
    config_path: Path,
    payload: Mapping[str, Any],
) -> None:
    try:
        config_path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False),
            encoding="utf-8",
        )
    except Exception as exc:
        raise SimulationError(
            "Failed to write seed-stability simulation config",
            context={"path": str(config_path)},
        ) from exc


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
    progress = build_seed_stability_progress(len(tasks))
    try:
        if _max_concurrent_seed_runs(config) <= 1:
            return _run_seed_tasks_serial(
                config=config,
                tasks=tasks,
                progress=progress,
            )
        return _run_seed_tasks_parallel(
            config=config,
            tasks=tasks,
            progress=progress,
        )
    finally:
        if progress is not None:
            progress.close()


def _run_seed_tasks_serial(
    *,
    config: SimulationConfig,
    tasks: Sequence[SeedStudyTask],
    progress: SeedStudyProgress | None,
) -> tuple[SeedStudyResult, ...]:
    results: list[SeedStudyResult] = []
    gpu_id = _default_gpu_id(config)
    for task in tasks:
        result = _run_seed_task(task=task, gpu_id=gpu_id)
        results.append(result)
        _update_seed_progress(progress=progress, seed=result.seed)
    return tuple(results)


def _default_gpu_id(config: SimulationConfig) -> int | None:
    if not config.flags.use_gpu:
        return None
    return 0


def _run_seed_tasks_parallel(
    *,
    config: SimulationConfig,
    tasks: Sequence[SeedStudyTask],
    progress: SeedStudyProgress | None,
) -> tuple[SeedStudyResult, ...]:
    gpu_slots = _build_gpu_slots(config)
    max_workers = min(len(tasks), len(gpu_slots))
    if max_workers <= 1:
        return _run_seed_tasks_serial(
            config=config,
            tasks=tasks,
            progress=progress,
        )
    pending = list(tasks)
    active: dict[Any, int] = {}
    results: list[SeedStudyResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for gpu_id in gpu_slots[:max_workers]:
            if not pending:
                break
            task = pending.pop(0)
            future = executor.submit(_run_seed_task, task=task, gpu_id=gpu_id)
            active[future] = gpu_id
        while active:
            future = next(iter(_completed_futures(active)))
            gpu_id = active.pop(future)
            result = future.result()
            results.append(result)
            _update_seed_progress(progress=progress, seed=result.seed)
            if pending:
                task = pending.pop(0)
                active[
                    executor.submit(_run_seed_task, task=task, gpu_id=gpu_id)
                ] = gpu_id
    return tuple(sorted(results, key=lambda item: item.seed))


def _update_seed_progress(
    *,
    progress: SeedStudyProgress | None,
    seed: int,
) -> None:
    if progress is not None:
        progress.update(seed)


def _completed_futures(
    active: Mapping[Any, int],
) -> Iterable[Any]:
    done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
    return done


def _build_gpu_slots(config: SimulationConfig) -> tuple[int, ...]:
    if not config.flags.use_gpu:
        raise ConfigError(
            "walkforward seed parallelization requires use_gpu=true"
        )
    gpu_count = int(torch.cuda.device_count())
    if gpu_count <= 0:
        raise ConfigError(
            "walkforward seed parallelization requires at least one CUDA device"
        )
    per_gpu = config.walkforward.max_parallel_seeds_per_gpu
    return tuple(
        gpu_id
        for gpu_id in range(gpu_count)
        for _ in range(per_gpu)
    )


def _max_concurrent_seed_runs(config: SimulationConfig) -> int:
    if not config.flags.use_gpu:
        return 1
    gpu_count = int(torch.cuda.device_count())
    if gpu_count <= 0:
        return 1
    return gpu_count * config.walkforward.max_parallel_seeds_per_gpu


def _run_seed_task(
    *,
    task: SeedStudyTask,
    gpu_id: int | None,
) -> SeedStudyResult:
    logger.info(
        "seed_stability.start seed=%s gpu=%s output=%s",
        task.seed,
        gpu_id,
        task.output_dir,
    )
    env = _build_seed_env(gpu_id=gpu_id, simulation_root=task.output_dir.parent)
    command = _build_seed_command(task.config_path)
    with task.log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            check=False,
            cwd=_WORKSPACE_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise SimulationError(
            "Seed-stability walkforward seed run failed",
            context={
                "seed": str(task.seed),
                "log_path": str(task.log_path),
                "returncode": str(completed.returncode),
            },
        )
    logger.info(
        "seed_stability.complete seed=%s gpu=%s output=%s",
        task.seed,
        gpu_id,
        task.output_dir,
    )
    return SeedStudyResult(
        seed=task.seed,
        output_dir=task.output_dir,
        log_path=task.log_path,
    )


def _build_seed_env(
    *,
    gpu_id: int | None,
    simulation_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env["SIMULATION_SOURCE"] = str(simulation_root)
    if gpu_id is None:
        return env
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _build_seed_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "algo_trader.cli.main",
        "simulation",
        "--simulation-config",
        str(config_path),
    ]


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
    path = walkforward_dir(result.output_dir) / "metrics" / "summary.csv"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SimulationError(
            "Failed to read seed downstream metrics summary",
            context={"seed": str(result.seed), "path": str(path)},
        ) from exc
    frame.insert(0, "seed", result.seed)
    return frame


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
