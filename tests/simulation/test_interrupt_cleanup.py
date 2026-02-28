from __future__ import annotations

from pytest import LogCaptureFixture, MonkeyPatch

from algo_trader.application.simulation import interrupt_cleanup


def test_cleanup_interrupt_local_ray_and_gpu(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        interrupt_cleanup,
        "_safe_shutdown_ray_runtime",
        lambda: calls.append("shutdown"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_stop_local_ray_cluster",
        lambda: calls.append("stop_local"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_clear_cuda_memory",
        lambda: calls.append("clear_cuda"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_cleanup_stopped_simulation_processes",
        lambda: calls.append("cleanup_stopped"),
    )

    interrupt_cleanup.cleanup_after_keyboard_interrupt(
        use_ray=True,
        ray_address=None,
        use_gpu=True,
    )

    assert calls == [
        "shutdown",
        "stop_local",
        "cleanup_stopped",
        "clear_cuda",
    ]


def test_cleanup_interrupt_remote_ray_skips_local_stop(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        interrupt_cleanup,
        "_safe_shutdown_ray_runtime",
        lambda: calls.append("shutdown"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_stop_local_ray_cluster",
        lambda: calls.append("stop_local"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_clear_cuda_memory",
        lambda: calls.append("clear_cuda"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_cleanup_stopped_simulation_processes",
        lambda: calls.append("cleanup_stopped"),
    )

    interrupt_cleanup.cleanup_after_keyboard_interrupt(
        use_ray=True,
        ray_address="auto",
        use_gpu=False,
    )

    assert calls == ["shutdown", "cleanup_stopped"]


def test_cleanup_before_run_local_ray_and_gpu(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        interrupt_cleanup,
        "_cleanup_stopped_simulation_processes",
        lambda: calls.append("cleanup_stopped"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_stop_local_ray_cluster",
        lambda: calls.append("stop_local"),
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_clear_cuda_memory",
        lambda: calls.append("clear_cuda"),
    )

    interrupt_cleanup.cleanup_before_simulation_run(
        use_ray=True,
        ray_address=None,
        use_gpu=True,
    )

    assert calls == ["cleanup_stopped", "stop_local", "clear_cuda"]


def test_cleanup_interrupt_fallback_to_force_ray_stop(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    attempts: list[list[str]] = []

    def fake_run_ray_command(command: list[str]) -> bool:
        attempts.append(command)
        return command == ["ray", "stop", "--force"]

    monkeypatch.setattr(
        interrupt_cleanup, "_run_ray_command", fake_run_ray_command
    )
    monkeypatch.setattr(
        interrupt_cleanup, "_safe_shutdown_ray_runtime", lambda: None
    )
    monkeypatch.setattr(
        interrupt_cleanup, "_clear_cuda_memory", lambda: None
    )
    monkeypatch.setattr(
        interrupt_cleanup,
        "_cleanup_stopped_simulation_processes",
        lambda: None,
    )

    with caplog.at_level("WARNING"):
        interrupt_cleanup.cleanup_after_keyboard_interrupt(
            use_ray=True,
            ray_address=None,
            use_gpu=False,
        )

    assert attempts == [["ray", "stop"], ["ray", "stop", "--force"]]
    assert "Forced local Ray cluster shutdown" in caplog.text
