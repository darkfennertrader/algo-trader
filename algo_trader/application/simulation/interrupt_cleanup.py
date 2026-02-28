from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
from subprocess import CalledProcessError, run
import time

import torch

from .tune_runner import shutdown_ray_for_tuning

logger = logging.getLogger(__name__)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_STOPPED_SIMULATION_STATE = "T"
_KILL_WAIT_SECONDS = 0.2


def cleanup_before_simulation_run(
    *,
    use_ray: bool,
    ray_address: str | None,
    use_gpu: bool,
) -> None:
    _cleanup_stopped_simulation_processes()
    if use_ray and ray_address is None:
        _stop_local_ray_cluster()
    if use_gpu:
        _clear_cuda_memory()


def cleanup_after_simulation_run(
    *,
    use_ray: bool,
    ray_address: str | None,
    use_gpu: bool,
    interrupted: bool,
) -> None:
    if use_ray:
        _safe_shutdown_ray_runtime()
        if ray_address is None:
            _stop_local_ray_cluster()
    _cleanup_stopped_simulation_processes()
    if use_gpu:
        _clear_cuda_memory()
    if interrupted:
        logger.info("Completed post-interrupt simulation cleanup")


def cleanup_after_keyboard_interrupt(
    *,
    use_ray: bool,
    ray_address: str | None,
    use_gpu: bool,
) -> None:
    cleanup_after_simulation_run(
        use_ray=use_ray,
        ray_address=ray_address,
        use_gpu=use_gpu,
        interrupted=True,
    )


def _safe_shutdown_ray_runtime() -> None:
    try:
        shutdown_ray_for_tuning()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to shutdown Ray runtime cleanly: %s", exc)


def _stop_local_ray_cluster() -> None:
    if _run_ray_command(["ray", "stop"]):
        logger.info("Stopped local Ray cluster after interruption")
        return
    if _run_ray_command(["ray", "stop", "--force"]):
        logger.warning(
            "Forced local Ray cluster shutdown after interruption"
        )
        return
    logger.warning("Unable to stop local Ray cluster after interruption")


def _run_ray_command(command: list[str]) -> bool:
    try:
        run(command, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        logger.warning("Ray CLI not found while running command: %s", command)
        return False
    except CalledProcessError as exc:
        logger.debug(
            "Ray command failed command=%s stderr=%s",
            command,
            exc.stderr.strip(),
        )
        return False


def _cleanup_stopped_simulation_processes() -> None:
    pids = _list_stopped_simulation_pids(os.getpid())
    if not pids:
        return
    _signal_processes(pids, signal.SIGTERM)
    time.sleep(_KILL_WAIT_SECONDS)
    remaining = _alive_processes(pids)
    if remaining:
        _signal_processes(remaining, signal.SIGKILL)
    logger.info("Cleaned stopped simulation processes count=%s", len(pids))


def _list_stopped_simulation_pids(current_pid: int) -> list[int]:
    try:
        completed = run(
            ["ps", "-eo", "pid=,stat=,cmd="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, CalledProcessError):
        return []
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        parsed = _parse_ps_line(line)
        if parsed is None:
            continue
        pid, state, command = parsed
        if pid == current_pid:
            continue
        if _STOPPED_SIMULATION_STATE not in state:
            continue
        if _is_workspace_simulation_command(command):
            pids.append(pid)
    return pids


def _parse_ps_line(line: str) -> tuple[int, str, str] | None:
    parts = line.strip().split(None, 2)
    if len(parts) != 3:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        return None
    return pid, parts[1], parts[2]


def _is_workspace_simulation_command(command: str) -> bool:
    root = str(_WORKSPACE_ROOT)
    return root in command and "algotrader simulation" in command


def _signal_processes(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger.warning("No permission to signal process pid=%s", pid)


def _alive_processes(pids: list[int]) -> list[int]:
    alive: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            alive.append(pid)
    return alive


def _clear_cuda_memory() -> None:
    if not torch.cuda.is_available():
        return
    device_count = int(torch.cuda.device_count())
    for device_id in range(device_count):
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    logger.info(
        "Cleared CUDA cache after interruption for %s device(s)",
        device_count,
    )
