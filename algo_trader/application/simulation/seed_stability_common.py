from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import torch
import yaml

from algo_trader.domain import ConfigError, SimulationError
from .walkforward.progress import build_seed_stability_progress

logger = logging.getLogger(__name__)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SeedStudyTask:
    seed: int
    output_dir: Path
    config_path: Path
    log_path: Path
    output_label: str | None = None


@dataclass(frozen=True)
class SeedStudyResult:
    seed: int
    output_dir: Path
    log_path: Path


def create_seed_task(
    *,
    seed: int,
    output_dir: Path,
    config_path: Path,
) -> SeedStudyTask:
    return SeedStudyTask(
        seed=seed,
        output_label=str(output_dir),
        output_dir=output_dir,
        config_path=config_path,
        log_path=output_dir / "seed_run.log",
    )


def seed_name(seed: int) -> str:
    return f"seed_{seed}"


def write_seed_config(
    *,
    config_path: Path,
    payload: Mapping[str, Any],
    message: str,
) -> None:
    try:
        config_path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False),
            encoding="utf-8",
        )
    except Exception as exc:
        raise SimulationError(message, context={"path": str(config_path)}) from exc


def prepare_seed_task(
    *,
    study_dir: Path,
    output_name: str,
    seed: int,
    payload: Mapping[str, Any],
    write_message: str,
) -> SeedStudyTask:
    output_dir = study_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "simulation.seed.yml"
    write_seed_config(
        config_path=config_path,
        payload=payload,
        message=write_message,
    )
    return create_seed_task(
        seed=seed,
        output_dir=output_dir,
        config_path=config_path,
    )


def build_gpu_slots(
    *,
    use_gpu: bool,
    per_gpu: int,
    parallel_error_message: str,
) -> tuple[int | None, ...]:
    if not use_gpu:
        return (None,)
    gpu_count = int(torch.cuda.device_count())
    if gpu_count <= 0:
        raise ConfigError(parallel_error_message)
    return tuple(
        gpu_id
        for gpu_id in range(gpu_count)
        for _ in range(per_gpu)
    )


def run_seed_tasks(
    *,
    tasks: Sequence[SeedStudyTask],
    gpu_slots: Sequence[int | None],
    progress: Any,
    task_runner: Callable[[SeedStudyTask, int | None], SeedStudyResult],
) -> tuple[SeedStudyResult, ...]:
    if len(gpu_slots) <= 1:
        return _run_seed_tasks_serial(
            tasks=tasks,
            progress=progress,
            gpu_id=gpu_slots[0] if gpu_slots else None,
            task_runner=task_runner,
        )
    return _run_seed_tasks_parallel(
        tasks=tasks,
        gpu_slots=gpu_slots,
        progress=progress,
        task_runner=task_runner,
    )


def run_seed_tasks_with_progress(
    *,
    tasks: Sequence[SeedStudyTask],
    gpu_slots: Sequence[int | None],
    progress: Any,
    task_runner: Callable[[SeedStudyTask, int | None], SeedStudyResult],
) -> tuple[SeedStudyResult, ...]:
    try:
        return run_seed_tasks(
            tasks=tasks,
            gpu_slots=gpu_slots,
            progress=progress,
            task_runner=task_runner,
        )
    finally:
        if progress is not None:
            progress.close()


def run_seed_tasks_from_config(
    *,
    config: Any,
    tasks: Sequence[SeedStudyTask],
    parallel_error_message: str,
    task_runner: Callable[[SeedStudyTask, int | None], SeedStudyResult],
) -> tuple[SeedStudyResult, ...]:
    gpu_slots = build_gpu_slots(
        use_gpu=config.flags.use_gpu,
        per_gpu=config.walkforward.max_parallel_seeds_per_gpu,
        parallel_error_message=parallel_error_message,
    )
    return run_seed_tasks_with_progress(
        tasks=tasks,
        gpu_slots=gpu_slots,
        progress=build_seed_stability_progress(len(tasks)),
        task_runner=task_runner,
    )


def _run_seed_tasks_serial(
    *,
    tasks: Sequence[SeedStudyTask],
    progress: Any,
    gpu_id: int | None,
    task_runner: Callable[[SeedStudyTask, int | None], SeedStudyResult],
) -> tuple[SeedStudyResult, ...]:
    results: list[SeedStudyResult] = []
    for task in tasks:
        result = task_runner(task, gpu_id)
        results.append(result)
        _update_seed_progress(progress=progress, seed=result.seed)
    return tuple(results)


def _run_seed_tasks_parallel(
    *,
    tasks: Sequence[SeedStudyTask],
    gpu_slots: Sequence[int | None],
    progress: Any,
    task_runner: Callable[[SeedStudyTask, int | None], SeedStudyResult],
) -> tuple[SeedStudyResult, ...]:
    max_workers = min(len(tasks), len(gpu_slots))
    pending = list(tasks)
    active: dict[Any, int | None] = {}
    results: list[SeedStudyResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for gpu_id in gpu_slots[:max_workers]:
            if not pending:
                break
            task = pending.pop(0)
            future = executor.submit(task_runner, task, gpu_id)
            active[future] = gpu_id
        while active:
            future = next(iter(_completed_futures(active)))
            gpu_id = active.pop(future)
            result = future.result()
            results.append(result)
            _update_seed_progress(progress=progress, seed=result.seed)
            if pending:
                task = pending.pop(0)
                active[executor.submit(task_runner, task, gpu_id)] = gpu_id
    return tuple(sorted(results, key=lambda item: item.seed))


def _completed_futures(active: Mapping[Any, int | None]) -> Iterable[Any]:
    done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
    return done


def _update_seed_progress(*, progress: Any, seed: int) -> None:
    if progress is not None:
        progress.update(seed)


def run_logged_seed_task(
    *,
    task: SeedStudyTask,
    gpu_id: int | None,
    failure_message: str,
    simulation_root: Path | None = None,
) -> SeedStudyResult:
    logger.info(
        "seed_stability.start seed=%s gpu=%s output=%s",
        task.seed,
        gpu_id,
        task.output_dir,
    )
    env = build_seed_env(gpu_id=gpu_id, simulation_root=simulation_root)
    command = build_seed_command(task.config_path)
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
            failure_message,
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


def build_seed_env(
    *,
    gpu_id: int | None,
    simulation_root: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    if simulation_root is not None:
        env["SIMULATION_SOURCE"] = str(simulation_root)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def build_seed_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "algo_trader.cli.main",
        "simulation",
        "--simulation-config",
        str(config_path),
    ]


def load_seed_summary(
    *,
    result: SeedStudyResult,
    path: Path,
    message: str,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SimulationError(
            message,
            context={"seed": str(result.seed), "path": str(path)},
        ) from exc
    frame.insert(0, "seed", result.seed)
    return frame
