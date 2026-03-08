from __future__ import annotations

import json
import urllib.parse
from types import SimpleNamespace
from typing import Any

import pandas as pd
from pytest import MonkeyPatch

from algo_trader.providers.exogenous.fred import provider as fred_provider_module
from algo_trader.providers.exogenous.fred.provider import FredSeriesProvider


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        _ = (exc_type, exc, tb)
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_fred_provider_fetches_and_filters_inclusive(
    monkeypatch: MonkeyPatch,
) -> None:
    captured_url: dict[str, str] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        _ = timeout
        captured_url["url"] = request.full_url
        return _FakeHttpResponse(
            {
                "observations": [
                    {"date": "2019-12-31", "value": "10.0"},
                    {"date": "2020-01-01", "value": "11.0"},
                    {"date": "2020-12-31", "value": "."},
                    {"date": "2021-01-01", "value": "12.0"},
                ]
            }
        )

    monkeypatch.setattr(
        fred_provider_module.urllib.request, "urlopen", fake_urlopen
    )
    provider = FredSeriesProvider(api_key="demo")
    series = SimpleNamespace(
        series_id="VIXCLS",
        units="lin",
        frequency="w",
        aggregation_method="eop",
    )

    frame = provider.fetch_series(
        series=series,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    parsed = urllib.parse.urlparse(captured_url["url"])
    query = urllib.parse.parse_qs(parsed.query)
    assert query["series_id"] == ["VIXCLS"]
    assert query["observation_start"] == ["2020-01-01"]
    assert query["observation_end"] == ["2020-12-31"]
    assert query["frequency"] == ["w"]
    assert query["aggregation_method"] == ["eop"]
    assert query["units"] == ["lin"]
    assert list(frame["date"]) == ["2020-01-01", "2020-12-31"]
    assert frame["value"].iloc[0] == 11.0
    assert pd.isna(frame["value"].iloc[1])

