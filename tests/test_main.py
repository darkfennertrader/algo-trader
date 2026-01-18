from pytest import CaptureFixture

from algo_trader.main import main


def test_main_prints_greeting(capsys: CaptureFixture[str]) -> None:
    """Ensure the CLI entrypoint prints the expected greeting."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello from Algo-Trader!"
