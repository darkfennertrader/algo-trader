from __future__ import annotations
# pylint: disable=duplicate-code

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


@dataclass(frozen=True, kw_only=True)
class PanelTensorDataset:  # pylint: disable=too-many-instance-attributes
    data: torch.Tensor
    targets: torch.Tensor
    missing_mask: torch.Tensor
    global_data: torch.Tensor | None = None
    global_missing_mask: torch.Tensor | None = None
    dates: Sequence[Any]
    assets: Sequence[str]
    features: Sequence[str]
    global_features: Sequence[str] = ()
    device: str = "cpu"

    def __post_init__(self) -> None:
        _validate_asset_block(self.data, self.targets, self.missing_mask)
        _validate_global_block(
            self.data,
            self.global_data,
            self.global_missing_mask,
            self.global_features,
        )
        _validate_axis_labels(
            self.data, self.dates, self.assets, self.features
        )

    def slice_by_indices(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {
            "x": self.data[indices],
            "y": self.targets[indices],
        }
        if self.global_data is not None:
            payload["x_global"] = self.global_data[indices]
        return payload


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
    global_data = _load_optional_global_values(payload=payload, count=values.shape[0])
    global_missing_mask = _load_optional_global_missing_mask(
        payload=payload,
        global_data=global_data,
    )
    global_features = _load_optional_global_features(
        payload=payload,
        global_data=global_data,
    )
    return PanelTensorDataset(
        data=data,
        targets=targets.to(device),
        missing_mask=missing_mask.to(device),
        global_data=(
            None if global_data is None else global_data.to(device)
        ),
        global_missing_mask=(
            None
            if global_missing_mask is None
            else global_missing_mask.to(device)
        ),
        dates=timestamps,
        assets=assets,
        features=features,
        global_features=global_features,
        device=device,
    )


def build_synthetic_panel_dataset(
    *,
    T: int,
    A: int,
    F: int,
    seed: int,
    device: str,
) -> PanelTensorDataset:
    if T <= 0 or A <= 0 or F <= 0:
        raise ConfigError(
            "Synthetic tensor sizes must be positive",
            context={"T": str(T), "A": str(A), "F": str(F)},
        )
    gen = torch.Generator().manual_seed(seed)
    data = torch.randn((T, A, F), generator=gen, dtype=torch.float32)
    targets = torch.randn((T, A), generator=gen, dtype=torch.float32)
    missing_mask = torch.zeros((T, A, F), dtype=torch.bool)
    dates = list(range(T))
    assets = [f"Asset{i}" for i in range(A)]
    features = [f"Feature{i}" for i in range(F)]
    return PanelTensorDataset(
        data=data.to(device),
        targets=targets.to(device),
        missing_mask=missing_mask.to(device),
        global_data=None,
        global_missing_mask=None,
        dates=dates,
        assets=assets,
        features=features,
        global_features=(),
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


def _validate_asset_block(
    data: torch.Tensor, targets: torch.Tensor, missing_mask: torch.Tensor
) -> None:
    if data.ndim != 3:
        raise ConfigError(
            "data tensor must have shape (T, A, F)",
            context={"shape": str(tuple(data.shape))},
        )
    if targets.ndim != 2:
        raise ConfigError(
            "targets tensor must have shape (T, A)",
            context={"shape": str(tuple(targets.shape))},
        )
    if data.shape[:2] != targets.shape:
        raise ConfigError(
            "targets tensor must align with data (T, A)",
            context={
                "data_shape": str(tuple(data.shape)),
                "targets_shape": str(tuple(targets.shape)),
            },
        )
    if missing_mask.shape != data.shape:
        raise ConfigError(
            "missing_mask must align with data (T, A, F)",
            context={
                "data_shape": str(tuple(data.shape)),
                "mask_shape": str(tuple(missing_mask.shape)),
            },
        )
    if missing_mask.dtype != torch.bool:
        raise ConfigError(
            "missing_mask must be a boolean tensor",
            context={"dtype": str(missing_mask.dtype)},
        )


def _validate_global_block(
    data: torch.Tensor,
    global_data: torch.Tensor | None,
    global_missing_mask: torch.Tensor | None,
    global_features: Sequence[str],
) -> None:
    if global_data is not None:
        if global_data.ndim != 2:
            raise ConfigError(
                "global_data tensor must have shape (T, G)",
                context={"shape": str(tuple(global_data.shape))},
            )
        if global_data.shape[0] != data.shape[0]:
            raise ConfigError(
                "global_data must align with data on T",
                context={
                    "data_shape": str(tuple(data.shape)),
                    "global_shape": str(tuple(global_data.shape)),
                },
            )
    if global_missing_mask is not None:
        if global_data is None:
            raise ConfigError(
                "global_missing_mask requires global_data",
                context={"mask_shape": str(tuple(global_missing_mask.shape))},
            )
        if global_missing_mask.shape != global_data.shape:
            raise ConfigError(
                "global_missing_mask must align with global_data (T, G)",
                context={
                    "global_shape": str(tuple(global_data.shape)),
                    "mask_shape": str(tuple(global_missing_mask.shape)),
                },
            )
        if global_missing_mask.dtype != torch.bool:
            raise ConfigError(
                "global_missing_mask must be a boolean tensor",
                context={"dtype": str(global_missing_mask.dtype)},
            )
    if global_data is not None and len(global_features) != global_data.shape[1]:
        raise ConfigError(
            "global_features length must match tensor G dimension",
            context={
                "global_features": str(len(global_features)),
                "columns": str(global_data.shape[1]),
            },
        )
    if global_data is None and len(global_features) != 0:
        raise ConfigError(
            "global_features requires global_data",
            context={"count": str(len(global_features))},
        )


def _validate_axis_labels(
    data: torch.Tensor,
    dates: Sequence[Any],
    assets: Sequence[str],
    features: Sequence[str],
) -> None:
    if len(dates) != data.shape[0]:
        raise ConfigError(
            "dates length must match tensor T dimension",
            context={"dates": str(len(dates)), "rows": str(data.shape[0])},
        )
    if len(assets) != data.shape[1]:
        raise ConfigError(
            "assets length must match tensor A dimension",
            context={"assets": str(len(assets)), "columns": str(data.shape[1])},
        )
    if len(features) != data.shape[2]:
        raise ConfigError(
            "features length must match tensor F dimension",
            context={
                "features": str(len(features)),
                "columns": str(data.shape[2]),
            },
        )


def _load_optional_global_values(
    *, payload: dict[str, object], count: int
) -> torch.Tensor | None:
    raw = payload.get("global_values")
    if raw is None:
        return None
    values = require_tensor(raw, label="global_values")
    if values.ndim != 2:
        raise ConfigError(
            "global_values must have shape (T, G)",
            context={"shape": str(tuple(values.shape))},
        )
    if values.shape[0] != count:
        raise ConfigError(
            "global_values must align with data on T",
            context={
                "global_shape": str(tuple(values.shape)),
                "rows": str(count),
            },
        )
    return values


def _load_optional_global_missing_mask(
    *, payload: dict[str, object], global_data: torch.Tensor | None
) -> torch.Tensor | None:
    raw = payload.get("global_missing_mask")
    if raw is None:
        if global_data is None:
            return None
        return torch.isnan(global_data)
    if global_data is None:
        raise ConfigError(
            "global_missing_mask requires global_values",
            context={"label": "global_missing_mask"},
        )
    mask = require_tensor(raw, label="global_missing_mask")
    if mask.shape != global_data.shape:
        raise ConfigError(
            "global_missing_mask must align with global_values (T, G)",
            context={
                "mask_shape": str(tuple(mask.shape)),
                "values_shape": str(tuple(global_data.shape)),
            },
        )
    return mask.to(dtype=torch.bool)


def _load_optional_global_features(
    *, payload: dict[str, object], global_data: torch.Tensor | None
) -> list[str]:
    raw = payload.get("global_features")
    if raw is None:
        if global_data is None:
            return []
        return [f"GlobalFeature{i}" for i in range(global_data.shape[1])]
    if not isinstance(raw, list):
        raise DataSourceError("global_features must be a list")
    names = [str(name) for name in raw]
    if global_data is not None and len(names) != global_data.shape[1]:
        raise ConfigError(
            "global_features length must match global_values G dimension",
            context={
                "global_features": str(len(names)),
                "columns": str(global_data.shape[1]),
            },
        )
    return names


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
