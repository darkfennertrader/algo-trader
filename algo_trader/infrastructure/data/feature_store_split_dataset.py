from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import torch

from algo_trader.domain import ConfigError, DataSourceError
from algo_trader.domain.simulation import DataConfig

from .feature_store_panel_dataset import (
    _FeatureParams,
    _TargetBundle,
    _coerce_group_list,
    _coerce_int,
    _coerce_str,
    _is_feature_group_dir,
    _load_assets,
    _load_json,
    _load_targets,
    _load_feature_group,
    _resolve_root_dir,
    _resolve_version_label,
    _to_numpy_ints,
    _to_tensor,
)
from .panel_tensor_dataset import PanelTensorDataset
from .tensor_bundle_io import load_tensor_bundle, require_tensor

_EXOGENOUS_NAMESPACE = "exogenous"


@dataclass(frozen=True)
class _SplitFeatureParams:  # pylint: disable=too-many-instance-attributes
    feature_store: str | None
    data_lake: str | None
    groups: Sequence[str] | None
    target_shift: int
    target_scale: int
    prefix_feature_names: bool
    include_exogenous_asset: bool
    include_exogenous_global: bool


@dataclass(frozen=True)
class _SingleAssetGroupBundle:
    values: torch.Tensor
    missing_mask: torch.Tensor
    timestamps: torch.Tensor
    feature_names: list[str]
    n_assets: int


@dataclass(frozen=True)
class _AssetFeatureBundle:
    values: torch.Tensor
    missing_mask: torch.Tensor
    timestamps: torch.Tensor
    feature_names: list[str]
    n_assets: int


@dataclass(frozen=True)
class _GlobalFeatureBundle:
    values: torch.Tensor
    missing_mask: torch.Tensor
    timestamps: torch.Tensor
    feature_names: list[str]


@dataclass(frozen=True)
class _AlignedSplitBundle:  # pylint: disable=too-many-instance-attributes
    asset_features: torch.Tensor
    asset_missing_mask: torch.Tensor
    global_features: torch.Tensor | None
    global_missing_mask: torch.Tensor | None
    targets: torch.Tensor
    timestamps: list[int]
    asset_feature_names: list[str]
    global_feature_names: list[str]
    n_assets: int


def load_feature_store_split_dataset(
    *, config: DataConfig, device: str
) -> PanelTensorDataset:
    params = _normalize_params(config)
    feature_store = _resolve_root_dir(
        cast(_FeatureParams, params),
        key="feature_store",
        env_name="FEATURE_STORE_SOURCE",
    )
    data_lake = _resolve_root_dir(
        cast(_FeatureParams, params),
        key="data_lake",
        env_name="DATA_LAKE_SOURCE",
    )
    version_label = _resolve_version_label(feature_store, data_lake)
    version_root = feature_store / version_label
    asset_groups = _resolve_standard_group_names(params, version_root)
    asset_bundle = _load_asset_feature_groups(version_root, asset_groups, params)
    target_bundle = _load_targets(
        data_lake / version_label,
        cast(_FeatureParams, params),
        n_assets=asset_bundle.n_assets,
    )
    global_bundle = _load_global_feature_group(version_root, params)
    aligned = _align_split_inputs(
        asset_features=asset_bundle,
        global_features=global_bundle,
        targets=target_bundle,
        target_shift=params.target_shift,
    )
    assets = _load_assets(data_lake / version_label, aligned.n_assets)
    return PanelTensorDataset(
        data=aligned.asset_features.to(device),
        targets=aligned.targets.to(device),
        missing_mask=aligned.asset_missing_mask.to(device),
        global_data=(
            None
            if aligned.global_features is None
            else aligned.global_features.to(device)
        ),
        global_missing_mask=(
            None
            if aligned.global_missing_mask is None
            else aligned.global_missing_mask.to(device)
        ),
        dates=aligned.timestamps,
        assets=assets,
        features=aligned.asset_feature_names,
        global_features=aligned.global_feature_names,
        device=device,
    )


def _normalize_params(config: DataConfig) -> _SplitFeatureParams:
    raw = config.dataset_params
    target_shift = _coerce_int(raw.get("target_shift", 1), label="target_shift")
    if target_shift < 0:
        raise ConfigError("target_shift must be >= 0")
    target_scale = _coerce_int(
        raw.get("target_scale", 1_000_000), label="target_scale"
    )
    if target_scale <= 0:
        raise ConfigError("target_scale must be positive")
    return _SplitFeatureParams(
        feature_store=_coerce_str(raw.get("feature_store")),
        data_lake=_coerce_str(raw.get("data_lake")),
        groups=_coerce_group_list(raw.get("groups")),
        target_shift=target_shift,
        target_scale=target_scale,
        prefix_feature_names=bool(raw.get("prefix_feature_names", True)),
        include_exogenous_asset=bool(raw.get("include_exogenous_asset", True)),
        include_exogenous_global=bool(raw.get("include_exogenous_global", True)),
    )


def _resolve_standard_group_names(
    params: _SplitFeatureParams, feature_version_dir: Path
) -> list[str]:
    if params.groups:
        return [group for group in params.groups if group != _EXOGENOUS_NAMESPACE]
    groups = [
        entry.name
        for entry in feature_version_dir.iterdir()
        if entry.name != _EXOGENOUS_NAMESPACE and _is_feature_group_dir(entry)
    ]
    if not groups and not _has_exogenous_asset_group(feature_version_dir, params):
        raise DataSourceError(
            "No feature groups found",
            context={"path": str(feature_version_dir)},
        )
    return sorted(groups)


def _has_exogenous_asset_group(
    feature_version_dir: Path, params: _SplitFeatureParams
) -> bool:
    if not params.include_exogenous_asset:
        return False
    return (feature_version_dir / _EXOGENOUS_NAMESPACE / "asset").is_dir()


def _load_asset_feature_groups(
    root: Path, groups: Sequence[str], params: _SplitFeatureParams
) -> _AssetFeatureBundle:
    bundles = [_load_asset_group(root / group, group, params) for group in groups]
    if params.include_exogenous_asset:
        exogenous_asset_dir = root / _EXOGENOUS_NAMESPACE / "asset"
        if exogenous_asset_dir.is_dir():
            bundles.append(
                _load_asset_group(
                    exogenous_asset_dir,
                    f"{_EXOGENOUS_NAMESPACE}::asset",
                    params,
                )
            )
    if not bundles:
        raise DataSourceError(
            "No asset-level feature groups found",
            context={"root": str(root)},
        )
    common_ts = _intersect_many([_to_numpy_ints(bundle.timestamps) for bundle in bundles])
    if common_ts.size == 0:
        raise DataSourceError("No overlapping timestamps across asset feature groups")
    values_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    feature_names: list[str] = []
    n_assets: int | None = None
    common_idx = torch.as_tensor(common_ts, dtype=torch.int64)
    for bundle in bundles:
        if n_assets is None:
            n_assets = bundle.n_assets
        elif bundle.n_assets != n_assets:
            raise DataSourceError("Asset counts differ across feature groups")
        indices = _lookup_indices(bundle.timestamps, common_idx)
        values_list.append(bundle.values.index_select(dim=0, index=indices))
        mask_list.append(bundle.missing_mask.index_select(dim=0, index=indices))
        feature_names.extend(bundle.feature_names)
    if n_assets is None:
        raise DataSourceError("No asset feature tensors loaded")
    timestamps = torch.as_tensor(common_ts, dtype=torch.int64)
    return _AssetFeatureBundle(
        values=torch.cat(values_list, dim=2),
        missing_mask=torch.cat(mask_list, dim=2),
        timestamps=timestamps,
        feature_names=feature_names,
        n_assets=n_assets,
    )


def _load_asset_group(
    group_dir: Path, group: str, params: _SplitFeatureParams
) -> _SingleAssetGroupBundle:
    loaded = _load_feature_group(group_dir, group, cast(_FeatureParams, params))
    return _SingleAssetGroupBundle(
        values=loaded.values,
        missing_mask=loaded.missing_mask,
        timestamps=loaded.timestamps,
        feature_names=loaded.feature_names,
        n_assets=loaded.values.shape[1],
    )


def _load_global_feature_group(
    root: Path, params: _SplitFeatureParams
) -> _GlobalFeatureBundle | None:
    if not params.include_exogenous_global:
        return None
    group_dir = root / _EXOGENOUS_NAMESPACE / "global"
    if not group_dir.is_dir():
        return None
    tensor_path = group_dir / "features_tensor.pt"
    metadata_path = group_dir / "metadata.json"
    payload = load_tensor_bundle(
        tensor_path, error_message="Failed to load exogenous global tensor"
    )
    values = require_tensor(payload.get("values"), label="values")
    timestamps = require_tensor(payload.get("timestamps"), label="timestamps")
    missing_mask = require_tensor(payload.get("missing_mask"), label="missing_mask")
    if values.ndim != 2:
        raise DataSourceError(
            "Global feature tensor must be [T, G]",
            context={"path": str(tensor_path)},
        )
    if missing_mask.shape != values.shape:
        raise DataSourceError(
            "Global feature missing_mask must match values shape",
            context={"path": str(tensor_path)},
        )
    metadata = _load_json(metadata_path)
    raw_names = metadata.get("feature_names")
    if not isinstance(raw_names, list):
        raise DataSourceError(
            "Global feature metadata missing feature_names",
            context={"path": str(metadata_path)},
        )
    feature_names = [str(name) for name in raw_names]
    if len(feature_names) != values.shape[1]:
        raise DataSourceError(
            "Global feature name count does not match tensor G",
            context={"path": str(metadata_path)},
        )
    return _GlobalFeatureBundle(
        values=values,
        missing_mask=missing_mask.to(dtype=torch.bool),
        timestamps=timestamps,
        feature_names=feature_names,
    )


def _align_split_inputs(
    *,
    asset_features: _AssetFeatureBundle,
    global_features: _GlobalFeatureBundle | None,
    targets: _TargetBundle,
    target_shift: int,
) -> _AlignedSplitBundle:
    base_common = _intersect_many(
        [
            _to_numpy_ints(asset_features.timestamps),
            _to_numpy_ints(targets.timestamps),
            *(
                [_to_numpy_ints(global_features.timestamps)]
                if global_features is not None
                else []
            ),
        ]
    )
    if base_common.size == 0:
        raise DataSourceError("No overlapping timestamps across asset/global features and targets")
    asset_idx = _lookup_indices(
        asset_features.timestamps, torch.as_tensor(base_common, dtype=torch.int64)
    )
    target_idx = _lookup_indices(
        targets.timestamps, torch.as_tensor(base_common, dtype=torch.int64)
    )
    X_asset = asset_features.values.index_select(dim=0, index=asset_idx)
    M_asset = asset_features.missing_mask.index_select(dim=0, index=asset_idx)
    y = targets.values.index_select(dim=0, index=target_idx)
    X_global: torch.Tensor | None = None
    M_global: torch.Tensor | None = None
    if global_features is not None:
        global_idx = _lookup_indices(
            global_features.timestamps,
            torch.as_tensor(base_common, dtype=torch.int64),
        )
        X_global = global_features.values.index_select(dim=0, index=global_idx)
        M_global = global_features.missing_mask.index_select(dim=0, index=global_idx)
    timestamps = base_common.tolist()
    if target_shift > 0:
        if target_shift >= len(timestamps):
            raise ConfigError("target_shift is too large for available data")
        X_asset = X_asset[:-target_shift]
        M_asset = M_asset[:-target_shift]
        y = y[target_shift:]
        timestamps = timestamps[:-target_shift]
        if X_global is not None and M_global is not None:
            X_global = X_global[:-target_shift]
            M_global = M_global[:-target_shift]
    return _AlignedSplitBundle(
        asset_features=X_asset,
        asset_missing_mask=M_asset,
        global_features=X_global,
        global_missing_mask=M_global,
        targets=y,
        timestamps=timestamps,
        asset_feature_names=asset_features.feature_names,
        global_feature_names=(
            [] if global_features is None else global_features.feature_names
        ),
        n_assets=asset_features.n_assets,
    )


def _lookup_indices(timestamps: torch.Tensor, common: torch.Tensor) -> torch.Tensor:
    index_map = {
        int(value): idx for idx, value in enumerate(_to_numpy_ints(timestamps))
    }
    return _to_tensor(
        np.array([index_map[int(value)] for value in common.tolist()], dtype=int),
        timestamps.device,
    )


def _intersect_many(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype="int64")
    common = arrays[0]
    for array in arrays[1:]:
        common = np.intersect1d(common, array, assume_unique=False)
    return common.astype("int64", copy=False)
