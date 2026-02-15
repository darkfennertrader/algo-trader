from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain import ConfigError, DataSourceError
from algo_trader.domain.simulation import DataConfig
from .panel_tensor_dataset import PanelTensorDataset
from .tensor_bundle_io import load_tensor_bundle, require_tensor
from .versioning import resolve_feature_store_version_label, resolve_root_dir


def load_feature_store_panel_dataset(
    *, config: DataConfig, device: str
) -> PanelTensorDataset:
    params = _normalize_params(config.dataset_params)
    feature_store = _resolve_root_dir(
        params, key="feature_store", env_name="FEATURE_STORE_SOURCE"
    )
    data_lake = _resolve_root_dir(
        params, key="data_lake", env_name="DATA_LAKE_SOURCE"
    )
    version_label = _resolve_version_label(
        params, feature_store / "features", data_lake
    )
    group_names = _resolve_group_names(
        params, feature_store / "features" / version_label
    )
    feature_bundle = _load_feature_groups(
        feature_store / "features" / version_label, group_names, params
    )
    target_bundle = _load_targets(
        data_lake / version_label, params, n_assets=feature_bundle.n_assets
    )
    aligned = _align_features_and_targets(
        feature_bundle, target_bundle, target_shift=params.target_shift
    )
    assets = _load_assets(data_lake / version_label, aligned.n_assets)
    return PanelTensorDataset(
        data=aligned.features.to(device),
        targets=aligned.targets.to(device),
        missing_mask=aligned.missing_mask.to(device),
        dates=aligned.timestamps,
        assets=assets,
        features=aligned.feature_names,
        device=device,
    )


@dataclass(frozen=True)
class _FeatureParams:
    feature_store: str | None
    data_lake: str | None
    version_label: str | None
    groups: Sequence[str] | None
    target_shift: int
    target_scale: int
    prefix_feature_names: bool


def _normalize_params(raw: Mapping[str, Any]) -> _FeatureParams:
    target_shift = _coerce_int(raw.get("target_shift", 1), label="target_shift")
    if target_shift < 0:
        raise ConfigError("target_shift must be >= 0")
    target_scale = _coerce_int(
        raw.get("target_scale", 1_000_000), label="target_scale"
    )
    if target_scale <= 0:
        raise ConfigError("target_scale must be positive")
    prefix_feature_names = bool(raw.get("prefix_feature_names", True))
    groups = _coerce_group_list(raw.get("groups"))
    return _FeatureParams(
        feature_store=_coerce_str(raw.get("feature_store")),
        data_lake=_coerce_str(raw.get("data_lake")),
        version_label=_coerce_str(raw.get("version_label")),
        groups=groups,
        target_shift=target_shift,
        target_scale=target_scale,
        prefix_feature_names=prefix_feature_names,
    )


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _coerce_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ConfigError(f"{label} must be an int")
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{label} must be an int") from exc
    raise ConfigError(f"{label} must be an int")


def _coerce_group_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return None
        return [value.strip()]
    if not isinstance(value, Sequence):
        raise ConfigError("groups must be a list of strings or 'all'")
    groups: list[str] = []
    for item in value:
        name = str(item).strip()
        if name:
            groups.append(name)
    return groups or None


def _resolve_root_dir(
    params: _FeatureParams, *, key: str, env_name: str
) -> Path:
    return resolve_root_dir(
        {
            key: getattr(params, key),
        },
        key=key,
        env_name=env_name,
        error_type=DataSourceError,
    )


def _resolve_version_label(
    params: _FeatureParams, feature_root: Path, data_lake: Path
) -> str:
    if params.version_label:
        return params.version_label
    return resolve_feature_store_version_label(
        feature_root,
        data_lake,
        error_type=DataSourceError,
        feature_error="No feature store versions found",
        lake_error="No data lake versions found",
    )


def _resolve_group_names(
    params: _FeatureParams, feature_version_dir: Path
) -> list[str]:
    if params.groups:
        return list(params.groups)
    groups = [
        entry.name
        for entry in feature_version_dir.iterdir()
        if entry.is_dir()
    ]
    if not groups:
        raise DataSourceError(
            "No feature groups found",
            context={"path": str(feature_version_dir)},
        )
    return sorted(groups)


@dataclass(frozen=True)
class _FeatureGroupBundle:
    values: torch.Tensor
    missing_mask: torch.Tensor
    timestamps: torch.Tensor
    feature_names: list[str]
    n_assets: int


def _load_feature_groups(
    root: Path, groups: Sequence[str], params: _FeatureParams
) -> _FeatureGroupBundle:
    values_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    feature_names: list[str] = []
    timestamps: torch.Tensor | None = None
    n_assets: int | None = None
    for group in groups:
        group_dir = root / group
        bundle = _load_feature_group(group_dir, group, params)
        if timestamps is None:
            timestamps = bundle.timestamps
            n_assets = bundle.values.shape[1]
        else:
            if not torch.equal(bundle.timestamps, timestamps):
                raise DataSourceError(
                    "Feature timestamps differ across groups",
                    context={"group": group, "path": str(group_dir)},
                )
            if bundle.values.shape[1] != n_assets:
                raise DataSourceError(
                    "Feature asset counts differ across groups",
                    context={"group": group, "path": str(group_dir)},
                )
        values_list.append(bundle.values)
        mask_list.append(bundle.missing_mask)
        feature_names.extend(bundle.feature_names)
    if timestamps is None or n_assets is None:
        raise DataSourceError(
            "No feature tensors loaded",
            context={"root": str(root)},
        )
    values = torch.cat(values_list, dim=2)
    missing_mask = torch.cat(mask_list, dim=2)
    return _FeatureGroupBundle(
        values=values,
        missing_mask=missing_mask,
        timestamps=timestamps,
        feature_names=feature_names,
        n_assets=n_assets,
    )


@dataclass(frozen=True)
class _SingleGroupBundle:
    values: torch.Tensor
    missing_mask: torch.Tensor
    timestamps: torch.Tensor
    feature_names: list[str]


def _load_feature_group(
    group_dir: Path, group: str, params: _FeatureParams
) -> _SingleGroupBundle:
    tensor_path = group_dir / "features_tensor.pt"
    metadata_path = group_dir / "metadata.json"
    payload = load_tensor_bundle(
        tensor_path, error_message="Failed to load feature tensor"
    )
    values = require_tensor(payload.get("values"), label="values")
    timestamps = require_tensor(payload.get("timestamps"), label="timestamps")
    missing_mask = require_tensor(payload.get("missing_mask"), label="missing_mask")
    if values.ndim != 3:
        raise DataSourceError(
            "Feature tensor must be [T, A, F]",
            context={"path": str(tensor_path)},
        )
    if missing_mask.shape != values.shape:
        raise DataSourceError(
            "Feature missing_mask must match values shape",
            context={"path": str(tensor_path)},
        )
    metadata = _load_json(metadata_path)
    raw_names = metadata.get("feature_names")
    if not isinstance(raw_names, list):
        raise DataSourceError(
            "Feature metadata missing feature_names",
            context={"path": str(metadata_path)},
        )
    names = [str(name) for name in raw_names]
    if len(names) != values.shape[2]:
        raise DataSourceError(
            "Feature name count does not match tensor F",
            context={"path": str(metadata_path)},
        )
    if params.prefix_feature_names:
        names = [f"{group}::{name}" for name in names]
    return _SingleGroupBundle(
        values=values,
        missing_mask=missing_mask.to(dtype=torch.bool),
        timestamps=timestamps,
        feature_names=names,
    )


@dataclass(frozen=True)
class _TargetBundle:
    values: torch.Tensor
    timestamps: torch.Tensor
    n_assets: int


def _load_targets(
    data_lake_dir: Path, params: _FeatureParams, n_assets: int
) -> _TargetBundle:
    tensor_path = data_lake_dir / "return_tensor.pt"
    payload = load_tensor_bundle(
        tensor_path, error_message="Failed to load return tensor"
    )
    values = require_tensor(payload.get("values"), label="values")
    timestamps = require_tensor(payload.get("timestamps"), label="timestamps")
    if values.ndim != 2:
        raise DataSourceError(
            "Return tensor must be [T, A]",
            context={"path": str(tensor_path)},
        )
    if values.shape[1] != n_assets:
        raise DataSourceError(
            "Return tensor asset count does not match features",
            context={"path": str(tensor_path)},
        )
    scaled = values.to(dtype=torch.float64) / float(params.target_scale)
    return _TargetBundle(values=scaled, timestamps=timestamps, n_assets=n_assets)


@dataclass(frozen=True)
class _AlignedBundle:
    features: torch.Tensor
    missing_mask: torch.Tensor
    targets: torch.Tensor
    timestamps: list[int]
    feature_names: list[str]
    n_assets: int


def _align_features_and_targets(
    features: _FeatureGroupBundle,
    targets: _TargetBundle,
    *,
    target_shift: int,
) -> _AlignedBundle:
    feature_ts = _to_numpy_ints(features.timestamps)
    target_ts = _to_numpy_ints(targets.timestamps)
    if feature_ts.size == 0 or target_ts.size == 0:
        raise DataSourceError("Empty timestamps in features or targets")
    common = _intersection_indices(feature_ts, target_ts)
    if common.feature_idx.size == 0:
        raise DataSourceError("No overlapping timestamps between X and y")
    X = features.values.index_select(
        dim=0, index=_to_tensor(common.feature_idx, features.values.device)
    )
    M = features.missing_mask.index_select(
        dim=0, index=_to_tensor(common.feature_idx, features.missing_mask.device)
    )
    y = targets.values.index_select(
        dim=0, index=_to_tensor(common.target_idx, targets.values.device)
    )
    timestamps = feature_ts[common.feature_idx].tolist()
    if target_shift > 0:
        if target_shift >= len(timestamps):
            raise ConfigError("target_shift is too large for available data")
        X = X[:-target_shift]
        M = M[:-target_shift]
        y = y[target_shift:]
        timestamps = timestamps[:-target_shift]
    return _AlignedBundle(
        features=X,
        missing_mask=M,
        targets=y,
        timestamps=timestamps,
        feature_names=features.feature_names,
        n_assets=features.n_assets,
    )


@dataclass(frozen=True)
class _Intersection:
    feature_idx: np.ndarray
    target_idx: np.ndarray


def _intersection_indices(
    feature_ts: np.ndarray, target_ts: np.ndarray
) -> _Intersection:
    target_index = {int(ts): idx for idx, ts in enumerate(target_ts)}
    mask = np.isin(feature_ts, target_ts)
    feature_idx = np.flatnonzero(mask)
    target_idx = np.array(
        [target_index[int(ts)] for ts in feature_ts[feature_idx]],
        dtype=int,
    )
    return _Intersection(feature_idx=feature_idx, target_idx=target_idx)


def _to_tensor(indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _to_numpy_ints(values: torch.Tensor) -> np.ndarray:
    if values.ndim != 1:
        raise DataSourceError("Timestamps must be 1D")
    return values.detach().cpu().numpy().astype("int64")


def _load_assets(data_lake_dir: Path, n_assets: int) -> list[str]:
    for filename in ("tensor_metadata.json", "returns_meta.json"):
        path = data_lake_dir / filename
        if path.exists():
            payload = _load_json(path)
            assets = _extract_assets(payload)
            if assets:
                if len(assets) != n_assets:
                    raise DataSourceError(
                        "Asset count mismatch between returns metadata and features",
                        context={"path": str(path)},
                    )
                return assets
    return [f"Asset_{idx}" for idx in range(n_assets)]


def _extract_assets(payload: Mapping[str, Any]) -> list[str]:
    tensor = payload.get("tensor")
    if isinstance(tensor, Mapping):
        assets = tensor.get("assets")
        if isinstance(assets, list):
            return [str(asset) for asset in assets]
    assets = payload.get("assets")
    if isinstance(assets, list):
        return [str(asset) for asset in assets]
    return []


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataSourceError(
            "Failed to read JSON metadata",
            context={"path": str(path)},
        ) from exc
    if not isinstance(raw, Mapping):
        raise DataSourceError(
            "Metadata JSON must be an object",
            context={"path": str(path)},
        )
    return raw
