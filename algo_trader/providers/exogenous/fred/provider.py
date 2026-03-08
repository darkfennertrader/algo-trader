from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import pandas as pd

from algo_trader.domain import ProviderError


class FredSeriesLike(Protocol):
    @property
    def series_id(self) -> str: ...

    @property
    def units(self) -> str | None: ...

    @property
    def frequency(self) -> str | None: ...

    @property
    def aggregation_method(self) -> str | None: ...

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 2.0)
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class FredSeriesProvider:
    api_key: str

    def name(self) -> str:
        return "fred"

    def fetch_series(
        self,
        *,
        series: FredSeriesLike,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        params = _build_params(
            api_key=self.api_key,
            series=series,
            start_date=start_date,
            end_date=end_date,
        )
        payload = _request_payload(params=params, series_id=series.series_id)
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise ProviderError(
                "FRED response missing observations",
                context={"series_id": series.series_id},
            )
        frame = _observations_to_frame(observations, series_id=series.series_id)
        mask = (frame["date"] >= start_date) & (frame["date"] <= end_date)
        return frame.loc[mask].reset_index(drop=True)


def _build_params(
    *,
    api_key: str,
    series: FredSeriesLike,
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series.series_id,
        "observation_start": start_date,
        "observation_end": end_date,
    }
    if series.units:
        params["units"] = series.units
    if series.frequency:
        params["frequency"] = series.frequency
    if series.aggregation_method:
        params["aggregation_method"] = series.aggregation_method
    return params


def _request_payload(
    *, params: Mapping[str, str], series_id: str
) -> Mapping[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{_BASE_URL}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "algo-trader/0.1"},
        method="GET",
    )
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw_body = response.read().decode("utf-8")
            data = json.loads(raw_body)
            if not isinstance(data, Mapping):
                raise ProviderError(
                    "FRED response must be a JSON object",
                    context={"series_id": series_id},
                )
            error_message = data.get("error_message")
            error_code = data.get("error_code")
            if isinstance(error_message, str):
                raise ProviderError(
                    f"FRED API error: {error_message}",
                    context={
                        "series_id": series_id,
                        "error_code": str(error_code) if error_code else "",
                    },
                )
            return data
        except urllib.error.HTTPError as exc:
            if _should_retry_http(exc.code, attempt):
                _sleep_before_retry(attempt)
                continue
            raise ProviderError(
                "FRED HTTP request failed",
                context={"series_id": series_id, "status": str(exc.code)},
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < _MAX_ATTEMPTS:
                _sleep_before_retry(attempt)
                continue
            raise ProviderError(
                "FRED request failed",
                context={"series_id": series_id},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "FRED response is not valid JSON",
                context={"series_id": series_id},
            ) from exc
    raise ProviderError(
        "FRED request failed after retries",
        context={"series_id": series_id},
    )


def _should_retry_http(status: int, attempt: int) -> bool:
    return status in _RETRY_STATUS_CODES and attempt < _MAX_ATTEMPTS


def _sleep_before_retry(attempt: int) -> None:
    if attempt > len(_BACKOFF_SECONDS):
        return
    time.sleep(_BACKOFF_SECONDS[attempt - 1])


def _observations_to_frame(
    observations: list[Any], *, series_id: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        raw_date = str(item.get("date", "")).strip()
        if not raw_date:
            continue
        raw_value = str(item.get("value", "")).strip()
        value: float | None = None
        if raw_value and raw_value != ".":
            try:
                value = float(raw_value)
            except ValueError:
                value = None
        rows.append({"date": raw_date, "value": value, "series_id": series_id})
    return pd.DataFrame(rows, columns=["date", "value", "series_id"])
