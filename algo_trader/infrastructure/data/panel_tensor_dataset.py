from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from algo_trader.domain import ConfigError, DataSourceError
from algo_trader.domain.simulation import DataConfig, DataPaths
from algo_trader.infrastructure.data.tensors import timestamps_to_epoch_hours
from .tensor_bundle_io import load_tensor_bundle, require_tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanelTensorDataset:
    data: torch.Tensor
    targets: torch.Tensor
    missing_mask: torch.Tensor
    dates: Sequence[Any]
    assets: Sequence[str]
    features: Sequence[str]
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ConfigError(
                "data tensor must have shape (T, A, F)",
                context={"shape": str(tuple(self.data.shape))},
            )
        if self.targets.ndim != 2:
            raise ConfigError(
                "targets tensor must have shape (T, A)",
                context={"shape": str(tuple(self.targets.shape))},
            )
        if self.data.shape[:2] != self.targets.shape:
            raise ConfigError(
                "targets tensor must align with data (T, A)",
                context={
                    "data_shape": str(tuple(self.data.shape)),
                    "targets_shape": str(tuple(self.targets.shape)),
                },
            )
        if self.missing_mask.shape != self.data.shape:
            raise ConfigError(
                "missing_mask must align with data (T, A, F)",
                context={
                    "data_shape": str(tuple(self.data.shape)),
                    "mask_shape": str(tuple(self.missing_mask.shape)),
                },
            )
        if self.missing_mask.dtype != torch.bool:
            raise ConfigError(
                "missing_mask must be a boolean tensor",
                context={"dtype": str(self.missing_mask.dtype)},
            )
        if len(self.dates) != self.data.shape[0]:
            raise ConfigError(
                "dates length must match tensor T dimension",
                context={
                    "dates": str(len(self.dates)),
                    "rows": str(self.data.shape[0]),
                },
            )
        if len(self.assets) != self.data.shape[1]:
            raise ConfigError(
                "assets length must match tensor A dimension",
                context={
                    "assets": str(len(self.assets)),
                    "columns": str(self.data.shape[1]),
                },
            )
        if len(self.features) != self.data.shape[2]:
            raise ConfigError(
                "features length must match tensor F dimension",
                context={
                    "features": str(len(self.features)),
                    "columns": str(self.data.shape[2]),
                },
            )

    def select_period_and_subsets(self, config: DataConfig) -> "PanelTensorDataset":
        selection = config.selection
        time_idx = _select_time_indices(
            self.dates, selection.start_date, selection.end_date
        )
        asset_idx, assets = _select_labels(
            self.assets, selection.asset_subset, label="asset"
        )
        feature_idx, features = _select_labels(
            self.features, selection.feature_subset, label="feature"
        )
        data = self.data
        targets = self.targets
        missing_mask = self.missing_mask
        dates = list(self.dates)
        if time_idx is not None:
            data = data.index_select(
                dim=0, index=_to_index_tensor(time_idx, data.device)
            )
            targets = targets.index_select(
                dim=0, index=_to_index_tensor(time_idx, targets.device)
            )
            missing_mask = missing_mask.index_select(
                dim=0, index=_to_index_tensor(time_idx, missing_mask.device)
            )
            dates = [dates[idx] for idx in time_idx]
        if asset_idx is not None:
            data = data.index_select(
                dim=1, index=_to_index_tensor(asset_idx, data.device)
            )
            targets = targets.index_select(
                dim=1, index=_to_index_tensor(asset_idx, targets.device)
            )
            missing_mask = missing_mask.index_select(
                dim=1, index=_to_index_tensor(asset_idx, missing_mask.device)
            )
        if feature_idx is not None:
            data = data.index_select(
                dim=2, index=_to_index_tensor(feature_idx, data.device)
            )
            missing_mask = missing_mask.index_select(
                dim=2, index=_to_index_tensor(feature_idx, missing_mask.device)
            )
        return PanelTensorDataset(
            data=data,
            targets=targets,
            missing_mask=missing_mask,
            dates=dates,
            assets=assets,
            features=features,
            device=self.device,
        )

    def slice_by_indices(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        return {
            "x": self.data[indices],
            "y": self.targets[indices],
        }


def load_panel_tensor_dataset(
    *,
    paths: DataPaths,
    device: str,
) -> PanelTensorDataset:
    if paths.tensor_path is None:
        raise DataSourceError("tensor_path is required")
    tensor_path = Path(paths.tensor_path).expanduser()
    payload = load_tensor_bundle(
        tensor_path, error_message="Failed to load tensor bundle"
    )
    values = require_tensor(payload.get("values"), label="values")
    if values.ndim != 3:
        raise ConfigError(
            "tensor values must have shape (T, A, F)",
            context={"shape": str(tuple(values.shape))},
        )
    timestamps = _load_timestamps(
        _optional_path(paths.timestamps_path),
        payload=payload,
        count=values.shape[0],
    )
    targets = _load_targets(_optional_path(paths.targets_path), values=values)
    assets = _load_labels(
        _optional_path(paths.assets_path),
        default_prefix="Asset",
        count=values.shape[1],
    )
    features = _load_labels(
        _optional_path(paths.features_path),
        default_prefix="Feature",
        count=values.shape[2],
    )
    data = values.to(device)
    missing_mask = _load_missing_mask(
        _optional_path(paths.missing_mask_path), payload=payload, values=values
    )
    return PanelTensorDataset(
        data=data,
        targets=targets.to(device),
        missing_mask=missing_mask.to(device),
        dates=timestamps,
        assets=assets,
        features=features,
        device=device,
    )


def _load_targets(
    path: Path | None, *, values: torch.Tensor
) -> torch.Tensor:
    if path is None:
        logger.warning("No targets_path provided; using zeros tensor.")
        return torch.zeros(
            (values.shape[0], values.shape[1]), dtype=values.dtype
        )
    try:
        targets = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise DataSourceError(
            "Failed to load targets tensor",
            context={"path": str(path)},
        ) from exc
    if not isinstance(targets, torch.Tensor):
        raise DataSourceError(
            "Targets payload must be a tensor",
            context={"path": str(path)},
        )
    if targets.shape[:2] != values.shape[:2]:
        raise ConfigError(
            "Targets tensor must align with data (T, A)",
            context={
                "targets_shape": str(tuple(targets.shape)),
                "data_shape": str(tuple(values.shape)),
            },
        )
    return targets


def _load_missing_mask(
    path: Path | None,
    *,
    payload: dict[str, object],
    values: torch.Tensor,
) -> torch.Tensor:
    if path is None:
        raw = payload.get("missing_mask")
        if raw is None:
            return torch.isnan(values)
        mask = require_tensor(raw, label="missing_mask")
    else:
        try:
            mask = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise DataSourceError(
                "Failed to load missing_mask tensor",
                context={"path": str(path)},
            ) from exc
        if not isinstance(mask, torch.Tensor):
            raise DataSourceError(
                "Missing mask payload must be a tensor",
                context={"path": str(path)},
            )
    if mask.shape != values.shape:
        raise ConfigError(
            "Missing mask must align with values (T, A, F)",
            context={
                "mask_shape": str(tuple(mask.shape)),
                "values_shape": str(tuple(values.shape)),
            },
        )
    return mask.to(dtype=torch.bool)


def _load_timestamps(
    path: Path | None,
    *,
    payload: dict[str, object],
    count: int,
) -> list[Any]:
    if path is None:
        raw = payload.get("timestamps")
        return _coerce_timestamps(raw, count=count)
    try:
        if path.suffix.lower() == ".json":
            return _load_json_values(path)
        loaded = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise DataSourceError(
            "Failed to load timestamps",
            context={"path": str(path)},
        ) from exc
    return _coerce_timestamps(loaded, count=count)


def _coerce_timestamps(raw: object, *, count: int) -> list[Any]:
    if isinstance(raw, torch.Tensor):
        timestamps = raw.cpu().tolist()
    elif isinstance(raw, list):
        timestamps = raw
    elif raw is None:
        timestamps = list(range(count))
    else:
        raise DataSourceError(
            "Unsupported timestamps payload",
            context={"type": type(raw).__name__},
        )
    if len(timestamps) != count:
        raise ConfigError(
            "Timestamps length must match tensor T dimension",
            context={"timestamps": str(len(timestamps)), "rows": str(count)},
        )
    return list(timestamps)


def _load_labels(
    path: Path | None, *, default_prefix: str, count: int
) -> list[str]:
    if path is None:
        return [f"{default_prefix}_{idx}" for idx in range(count)]
    values = _load_json_values(path)
    labels = [str(value) for value in values]
    if len(labels) != count:
        raise ConfigError(
            "Label count must match tensor dimension",
            context={"labels": str(len(labels)), "expected": str(count)},
        )
    return labels


def _load_json_values(path: Path) -> list[Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataSourceError(
            "Failed to read JSON values",
            context={"path": str(path)},
        ) from exc
    if not isinstance(raw, list):
        raise DataSourceError(
            "JSON payload must be a list",
            context={"path": str(path)},
        )
    return raw


def _optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _select_labels(
    labels: Sequence[str], subset: Sequence[str] | None, *, label: str
) -> tuple[np.ndarray | None, list[str]]:
    if not subset:
        return None, list(labels)
    idx: list[int] = []
    selected: list[str] = []
    for name in subset:
        if name not in labels:
            raise ConfigError(f"Unknown {label} in subset: {name}")
        pos = labels.index(name)
        if pos not in idx:
            idx.append(pos)
            selected.append(name)
    return np.array(idx, dtype=int), selected


def _select_time_indices(
    dates: Sequence[Any], start_date: str | None, end_date: str | None
) -> np.ndarray | None:
    if start_date is None and end_date is None:
        return None
    if not dates:
        return np.array([], dtype=int)
    mode = _infer_date_mode(dates)
    start_val = _coerce_date_value(start_date, mode) if start_date else None
    end_val = _coerce_date_value(end_date, mode) if end_date else None
    mask = _build_time_mask(dates, mode, start_val, end_val)
    return np.flatnonzero(mask)


def _infer_date_mode(dates: Sequence[Any]) -> str:
    first = dates[0]
    if isinstance(first, (int, np.integer)):
        return "epoch_hours"
    return "timestamp"


def _coerce_date_value(value: str | None, mode: str) -> int | pd.Timestamp:
    if value is None:
        raise ConfigError("start_date/end_date must be strings when provided")
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if mode == "epoch_hours":
        epoch = timestamps_to_epoch_hours(pd.DatetimeIndex([ts]))
        return int(epoch[0])
    return ts


def _build_time_mask(
    dates: Sequence[Any],
    mode: str,
    start_val: int | pd.Timestamp | None,
    end_val: int | pd.Timestamp | None,
) -> np.ndarray:
    if mode == "epoch_hours":
        values = np.array([int(item) for item in dates], dtype="int64")
        return _mask_between(values, start_val, end_val)
    parsed = pd.to_datetime(list(dates))
    if parsed.tz is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return _mask_between(parsed.to_numpy(), start_val, end_val)


def _mask_between(
    values: np.ndarray,
    start_val: int | pd.Timestamp | None,
    end_val: int | pd.Timestamp | None,
) -> np.ndarray:
    mask = np.ones(len(values), dtype=bool)
    if start_val is not None:
        mask &= values >= start_val
    if end_val is not None:
        mask &= values <= end_val
    return mask


def _to_index_tensor(indices: Iterable[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(list(indices), dtype=torch.long, device=device)
