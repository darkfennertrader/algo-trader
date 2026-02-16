from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from algo_trader.domain import ConfigError, DataSourceError
from algo_trader.domain.simulation import DataPaths
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
    targets = _load_targets(
        _optional_path(paths.targets_path),
        payload=payload,
        values=values,
    )
    assets = _load_labels(
        _optional_path(paths.assets_path),
        payload=payload,
        payload_key="assets",
        default_prefix="Asset",
        count=values.shape[1],
    )
    features = _load_labels(
        _optional_path(paths.features_path),
        payload=payload,
        payload_key="features",
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
    path: Path | None,
    *,
    payload: dict[str, object],
    values: torch.Tensor,
) -> torch.Tensor:
    if path is None:
        raw = payload.get("targets")
        if raw is None:
            logger.warning("No targets found; using zeros tensor.")
            return torch.zeros(
                (values.shape[0], values.shape[1]), dtype=values.dtype
            )
        targets = require_tensor(raw, label="targets")
    else:
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
    path: Path | None,
    *,
    payload: dict[str, object],
    payload_key: str,
    default_prefix: str,
    count: int,
) -> list[str]:
    if path is None:
        raw = payload.get(payload_key)
        if raw is not None:
            return _coerce_labels(raw, count=count, label=payload_key)
        return [f"{default_prefix}_{idx}" for idx in range(count)]
    values = _load_json_values(path)
    labels = [str(value) for value in values]
    if len(labels) != count:
        raise ConfigError(
            "Label count must match tensor dimension",
            context={"labels": str(len(labels)), "expected": str(count)},
        )
    return labels


def _coerce_labels(
    raw: object, *, count: int, label: str
) -> list[str]:
    if isinstance(raw, torch.Tensor):
        values = raw.cpu().tolist()
    elif isinstance(raw, list):
        values = raw
    else:
        raise DataSourceError(
            f"Unsupported {label} payload",
            context={"type": type(raw).__name__},
        )
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
