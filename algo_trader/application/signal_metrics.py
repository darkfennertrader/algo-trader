from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

_CALIBRATION_BIN_COUNT = 10


def top_k_count(asset_count: int) -> int:
    if asset_count <= 1:
        return 1
    return min(max(1, ceil(asset_count * 0.1)), asset_count - 1)


def top_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    order = list(np.argsort(-values, kind="stable"))
    return np.asarray(order[:top_k], dtype=int)


def rest_indices(asset_count: int, selected_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(asset_count, dtype=bool)
    mask[selected_idx] = False
    return np.flatnonzero(mask)


def pearson_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size < 2 or rhs.size < 2:
        return float("nan")
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return float("nan")
    return float(pd.Series(lhs).corr(pd.Series(rhs), method="pearson"))


def spearman_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(pd.Series(lhs).corr(pd.Series(rhs), method="spearman"))


def mean_spread(
    values: np.ndarray,
    selected_idx: np.ndarray,
    other_idx: np.ndarray,
) -> float:
    if selected_idx.size == 0 or other_idx.size == 0:
        return float("nan")
    return float(np.mean(values[selected_idx]) - np.mean(values[other_idx]))


def hit_rate(values: np.ndarray, selected_idx: np.ndarray) -> float:
    benchmark = float(np.median(values))
    return float(np.mean(values[selected_idx] > benchmark))


def brier_score(
    predicted_positive: np.ndarray,
    actual_positive: np.ndarray,
) -> float:
    actual = actual_positive.astype(float)
    return float(np.mean((predicted_positive - actual) ** 2))


def build_calibration_frame(
    predicted_positive: np.ndarray,
    actual_positive: np.ndarray,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, _CALIBRATION_BIN_COUNT + 1)
    frame = pd.DataFrame(
        {
            "predicted_positive_probability": predicted_positive,
            "actual_positive": actual_positive.astype(float),
        }
    )
    frame["bucket"] = pd.cut(
        frame["predicted_positive_probability"],
        bins=bins.tolist(),
        include_lowest=True,
    )
    grouped = frame.groupby("bucket", observed=False)
    return grouped.agg(
        count=("actual_positive", "size"),
        mean_predicted=("predicted_positive_probability", "mean"),
        realized_positive_rate=("actual_positive", "mean"),
    )


def calibration_rmse(
    predicted_positive: np.ndarray,
    actual_positive: np.ndarray,
) -> float:
    calibration = build_calibration_frame(predicted_positive, actual_positive)
    counts = calibration["count"].to_numpy(dtype=float)
    valid = counts > 0.0
    if not np.any(valid):
        return float("nan")
    predicted = calibration.loc[valid, "mean_predicted"].to_numpy(float)
    realized = calibration.loc[valid, "realized_positive_rate"].to_numpy(float)
    error = (predicted - realized) ** 2
    return float(np.sqrt(np.average(error, weights=counts[valid])))


def nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def nanstd(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.std(valid))


def positive_fraction(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid > 0.0))
