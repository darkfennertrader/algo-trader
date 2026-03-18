from pathlib import Path

import pytest

from pytest import CaptureFixture, MonkeyPatch

from algo_trader.cli import main_module
from algo_trader.domain import ConfigError
from algo_trader.main import main


def test_main_help(capsys: CaptureFixture[str]) -> None:
    """Ensure the CLI help text renders."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "historical" in captured.out


def test_main_historical_invokes_runner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Ensure the historical command dispatches to the runner with config."""
    config_path = tmp_path / "tickers.yml"
    recorded: dict[str, Path | None] = {"config_path": None}

    def fake_run(config_path: Path | None = None) -> None:
        recorded["config_path"] = config_path

    monkeypatch.setattr(main_module, "run", fake_run)

    exit_code = main(["--config", str(config_path), "historical"])

    assert exit_code == 0
    assert recorded["config_path"] == config_path


def test_main_exogenous_invokes_runner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    recorded: dict[str, Path | None] = {"config_path": None}

    def fake_run(config_path: Path | None = None) -> None:
        recorded["config_path"] = config_path

    monkeypatch.setattr(main_module, "run_exogenous", fake_run)

    exit_code = main(["--config", str(config_path), "exogenous"])

    assert exit_code == 0
    assert recorded["config_path"] == config_path


def test_main_exogenous_cleaning_invokes_runner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    recorded: dict[str, Path | None] = {"config_path": None}

    def fake_run(*, request) -> None:  # type: ignore[no-untyped-def]
        recorded["config_path"] = request.config_path

    monkeypatch.setattr(main_module, "run_exogenous_cleaning", fake_run)

    exit_code = main(["--config", str(config_path), "exogenous_cleaning"])

    assert exit_code == 0
    assert recorded["config_path"] == config_path


def test_main_exogenous_feature_engineering_invokes_runner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    recorded: dict[str, Path | str | None] = {
        "config_path": None,
        "start_date": None,
        "end_date": None,
    }

    def fake_run(*, request) -> None:  # type: ignore[no-untyped-def]
        recorded["config_path"] = request.config_path
        recorded["start_date"] = request.start_date
        recorded["end_date"] = request.end_date

    monkeypatch.setattr(main_module, "run_exogenous_feature_engineering", fake_run)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "exogenous_feature_engineering",
            "--start-date",
            "2024-01-15",
            "--end-date",
            "2024-01-31",
        ]
    )

    assert exit_code == 0
    assert recorded["config_path"] == config_path
    assert recorded["start_date"] == "2024-01-15"
    assert recorded["end_date"] == "2024-01-31"


def test_main_handles_domain_error(
    caplog: pytest.LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    def fake_run(config_path: Path | None = None) -> None:
        raise ConfigError("Bad config", context={"field": "tickers"})

    monkeypatch.setattr(main_module, "run", fake_run)

    with caplog.at_level("ERROR"):
        exit_code = main(["historical"])

    assert exit_code == 1
    assert "Bad config" in caplog.text


def test_main_handles_keyboard_interrupt(monkeypatch: MonkeyPatch) -> None:
    def fake_run(config_path: Path | None = None) -> None:
        _ = config_path
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_module, "run", fake_run)

    exit_code = main(["historical"])

    assert exit_code == 130
