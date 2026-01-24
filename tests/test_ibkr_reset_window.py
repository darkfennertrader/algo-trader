from datetime import datetime
from zoneinfo import ZoneInfo

from algo_trader.application.historical import (
    is_daily_reset_window,
    weekend_reset_window,
)

EUROPE_TZ = ZoneInfo("Europe/Paris")
ET_TZ = ZoneInfo("America/New_York")


def test_daily_reset_window_matches_sunday() -> None:
    now_local = datetime(2024, 1, 7, 6, 30, tzinfo=EUROPE_TZ)
    assert is_daily_reset_window(now_local)


def test_daily_reset_window_excludes_saturday() -> None:
    now_local = datetime(2024, 1, 6, 6, 30, tzinfo=EUROPE_TZ)
    assert not is_daily_reset_window(now_local)


def test_daily_reset_window_end_is_exclusive() -> None:
    now_local = datetime(2024, 1, 8, 7, 45, tzinfo=EUROPE_TZ)
    assert not is_daily_reset_window(now_local)


def test_weekend_reset_window_friday_night() -> None:
    now_et = datetime(2024, 1, 5, 23, 30, tzinfo=ET_TZ)
    window = weekend_reset_window(now_et)
    assert window is not None
    start_et, end_et = window
    assert start_et == datetime(2024, 1, 5, 23, 0, tzinfo=ET_TZ)
    assert end_et == datetime(2024, 1, 6, 3, 0, tzinfo=ET_TZ)


def test_weekend_reset_window_saturday_early() -> None:
    now_et = datetime(2024, 1, 6, 2, 30, tzinfo=ET_TZ)
    window = weekend_reset_window(now_et)
    assert window is not None
    start_et, end_et = window
    assert start_et == datetime(2024, 1, 5, 23, 0, tzinfo=ET_TZ)
    assert end_et == datetime(2024, 1, 6, 3, 0, tzinfo=ET_TZ)
