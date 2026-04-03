from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from algo_trader.application.research.index_measurement import (
    IndexMeasurementStudyConfig,
    default_output_dir,
    run_index_measurement_study,
)
from algo_trader.application.research.index_representation.types import IndexPosteriorDataset


def test_default_output_dir_uses_env_root() -> None:
    path = default_output_dir("demo")
    assert path.name == "demo"
    assert "index_measurement" in str(path)


def test_index_measurement_study_outputs_expected_tables(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    result = run_index_measurement_study(
        dataset=dataset,
        config=IndexMeasurementStudyConfig(output_dir=tmp_path, min_history=8, window_size=12),
    )
    assert set(result.us_block.weight_summary["component"]) == {"IBUS30", "IBUST100"}
    assert set(result.europe_block.weight_summary["component"]) == {
        "IBDE40",
        "IBFR40",
        "IBNL25",
        "IBES35",
    }
    assert "cross_us_europe" in set(result.residual_map.summary["group"])
    assert set(result.basket_proxy.summary["basket"]) == {
        "us_index",
        "europe_index",
        "us_minus_europe",
        "index_equal_weight",
    }


def _synthetic_dataset() -> IndexPosteriorDataset:
    periods = 24
    index = pd.date_range("2020-01-03", periods=periods, freq="W-FRI", tz="UTC")
    names = _asset_names()
    steps = np.linspace(-1.0, 1.0, num=periods)
    series = _series_by_name(steps)
    truth = _truth_frame(
        index=index,
        names=names,
        columns=[series[name] for name in names],
    )
    base = truth.to_numpy(dtype=float)
    samples = tuple(_sample_draws(row, len(names)) for row in base)
    return IndexPosteriorDataset(
        timestamps=index,
        asset_names=names,
        truth=truth,
        samples=samples,
    )


def _asset_names() -> tuple[str, ...]:
    return (
        "IBUS30",
        "IBUS500",
        "IBUST100",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBCH20",
    )


def _europe_components(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    de40 = 0.015 * steps + 0.002 * np.sin(np.arange(len(steps)) * 0.8)
    fr40 = 0.014 * steps + 0.002 * np.cos(np.arange(len(steps)) * 0.7)
    nl25 = 0.013 * steps + 0.0015 * np.sin(np.arange(len(steps)) * 0.6)
    es35 = 0.012 * steps + 0.0012 * np.cos(np.arange(len(steps)) * 0.5)
    return de40, fr40, nl25, es35


def _us_components(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    us30 = 0.02 * steps + 0.003 * np.sin(np.arange(len(steps)))
    ust100 = 0.03 * steps + 0.004 * np.cos(np.arange(len(steps)))
    us500 = 0.6 * us30 + 0.4 * ust100 + 0.001 * np.sin(np.arange(len(steps)) * 0.5)
    return us30, us500, ust100


def _series_by_name(steps: np.ndarray) -> dict[str, np.ndarray]:
    us30, us500, ust100 = _us_components(steps)
    de40, fr40, nl25, es35 = _europe_components(steps)
    eu50 = 0.35 * de40 + 0.30 * fr40 + 0.20 * nl25 + 0.15 * es35
    return {
        "IBUS30": us30,
        "IBUS500": us500,
        "IBUST100": ust100,
        "IBDE40": de40,
        "IBES35": es35,
        "IBEU50": eu50,
        "IBFR40": fr40,
        "IBGB100": 0.011 * steps + 0.001 * np.sin(np.arange(len(steps)) * 0.4),
        "IBNL25": nl25,
        "IBCH20": 0.01 * steps + 0.001 * np.cos(np.arange(len(steps)) * 0.3),
    }


def _truth_frame(
    *,
    index: pd.DatetimeIndex,
    names: Sequence[str],
    columns: Sequence[np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        np.column_stack(columns),
        index=index,
        columns=pd.Index(list(names), dtype="object"),
    )


def _sample_draws(row: np.ndarray, width: int) -> np.ndarray:
    shifts = np.linspace(-1.0, 1.0, num=9)
    return np.vstack(
        [row + 0.002 * np.sin(np.arange(width) + shift) for shift in shifts]
    )
