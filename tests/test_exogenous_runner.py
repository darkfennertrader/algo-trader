from __future__ import annotations

from pathlib import Path

import pandas as pd
from pytest import MonkeyPatch

from algo_trader.application.exogenous import runner as exogenous_runner


def test_exogenous_runner_exports_series(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2020-01-01"\n'
        'end_date: "2020-01-31"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "market_risk"\n'
        '    units: "lin"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n',
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    monkeypatch.setenv("EXOGENOUS_FEATURES_SOURCE", str(output_root))

    class _FakeProvider:
        def name(self) -> str:
            return "fred"

        def fetch_series(self, *, series, start_date, end_date):  # type: ignore[no-untyped-def]
            _ = (series, start_date, end_date)
            return pd.DataFrame(
                {
                    "date": ["2020-01-03", "2020-01-10"],
                    "value": [14.0, 15.0],
                }
            )

    monkeypatch.setattr(
        exogenous_runner,
        "build_exogenous_provider",
        lambda config: _FakeProvider(),
    )

    output_paths = exogenous_runner.run(config_path=config_path)

    expected_path = output_root / "fred" / "market_risk" / "VIXCLS.csv"
    assert output_paths == [expected_path]
    assert expected_path.exists()
    lines = expected_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,value"
    assert lines[1].startswith("2020-01-03")
