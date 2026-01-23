import pytest

from pytest import CaptureFixture

from algo_trader.main import main


def test_main_help(capsys: CaptureFixture[str]) -> None:
    """Ensure the CLI help text renders."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "historical" in captured.out
