from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np
import torch

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import CPCVSplit, FeatureCleaningState
from algo_trader.infrastructure import ensure_directory, require_env
from algo_trader.infrastructure.data.versioning import (
    resolve_feature_store_version_label,
    resolve_root_dir,
)



@dataclass(frozen=True)
class SimulationInputs:
    X: torch.Tensor
    M: torch.Tensor
    y: torch.Tensor
    timestamps: Sequence[Any]
    assets: Sequence[str]
    features: Sequence[str]


class SimulationArtifacts:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._inputs_dir = self._base_dir / "inputs"
        self._preprocess_dir = self._base_dir / "preprocessing"
        self._inner_dir = self._base_dir / "inner"
        self._outer_dir = self._base_dir / "outer"
        self._results_dir = self._base_dir / "results"
        for path in (
            self._inputs_dir,
            self._preprocess_dir,
            self._inner_dir,
            self._outer_dir,
            self._results_dir,
        ):
            _ensure_dir(path, message="Failed to create simulation output")

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def write_inputs(self, *, inputs: SimulationInputs) -> None:
        payload = {
            "values": inputs.X.detach().cpu(),
            "missing_mask": inputs.M.detach().cpu(),
            "targets": inputs.y.detach().cpu(),
            "timestamps": _normalize_timestamps(inputs.timestamps),
            "assets": list(inputs.assets),
            "features": list(inputs.features),
        }
        path = self._inputs_dir / "panel_tensor.pt"
        _save_torch(payload, path, message="Failed to write simulation inputs")
        self._write_inputs_csv(inputs)

    def _write_inputs_csv(self, inputs: SimulationInputs) -> None:
        feature_frame = _build_feature_frame(inputs)
        target_frame = _build_target_frame(inputs)
        mask_frame = _build_mask_frame(inputs)
        _write_csv(
            self._inputs_dir / "features.csv",
            feature_frame,
            message="Failed to write features CSV",
        )
        _write_csv(
            self._inputs_dir / "targets.csv",
            target_frame,
            message="Failed to write targets CSV",
        )
        _write_csv(
            self._inputs_dir / "missing_mask.csv",
            mask_frame,
            message="Failed to write missing mask CSV",
        )

    def write_cv_structure(
        self,
        *,
        warmup_idx: np.ndarray,
        groups: Sequence[np.ndarray],
        outer_ids: Sequence[int],
        outer_folds: Sequence[Mapping[str, Any]],
    ) -> None:
        payload = {
            "warmup_idx": warmup_idx.tolist(),
            "groups": [group.tolist() for group in groups],
            "outer_test_group_ids": list(outer_ids),
            "outer_folds": [_to_serializable(item) for item in outer_folds],
        }
        _write_json(
            self._results_dir / "cv_structure.json",
            payload,
            message="Failed to write CV structure",
        )
    def write_cleaning_state(
        self, *, outer_k: int, cleaning: FeatureCleaningState
    ) -> None:
        target_dir = self._preprocess_dir / f"outer_{outer_k}"
        _ensure_dir(target_dir, message="Failed to create preprocessing output")
        payload = asdict(cleaning)
        path = target_dir / "cleaning_state.pt"
        _save_torch(payload, path, message="Failed to write cleaning state")

    def write_inner(
        self,
        *,
        outer_k: int,
        inner_splits: Sequence[CPCVSplit],
        best_config: Mapping[str, Any],
    ) -> None:
        target_dir = self._inner_dir / f"outer_{outer_k}"
        _ensure_dir(target_dir, message="Failed to create inner output")
        splits_payload = [
            {
                "train_idx": split.train_idx.tolist(),
                "test_idx": split.test_idx.tolist(),
                "test_group_ids": list(split.test_group_ids),
            }
            for split in inner_splits
        ]
        _write_json(
            target_dir / "splits.json",
            splits_payload,
            message="Failed to write inner splits",
        )
        _write_json(
            target_dir / "best_config.json",
            _to_serializable(best_config),
            message="Failed to write best config",
        )

    def write_outer_result(
        self, *, outer_k: int, result: Mapping[str, Any]
    ) -> None:
        target_dir = self._outer_dir / f"outer_{outer_k}"
        _ensure_dir(target_dir, message="Failed to create outer output")
        payload = _to_serializable(result)
        _write_json(
            target_dir / "result.json",
            payload,
            message="Failed to write outer result",
        )

    def write_results(self, summary: Mapping[str, Any]) -> None:
        payload = _to_serializable(summary)
        _write_json(
            self._results_dir / "summary.json",
            payload,
            message="Failed to write simulation summary",
        )


def resolve_simulation_output_dir(
    *,
    dataset_name: str,
    dataset_params: Mapping[str, Any],
) -> Path:
    root = Path(require_env("SIMULATION_SOURCE")).expanduser()
    _ensure_dir(root, message="Failed to create SIMULATION_SOURCE")
    version_label = _resolve_version_label(dataset_name, dataset_params)
    base_dir = root / version_label
    _ensure_dir(base_dir, message="Failed to create simulation version dir")
    return base_dir


def _resolve_version_label(
    dataset_name: str, dataset_params: Mapping[str, Any]
) -> str:
    raw = dataset_params.get("version_label")
    if raw:
        return str(raw)
    if dataset_name != "feature_store_panel":
        return "unversioned"
    feature_store = resolve_root_dir(
        dataset_params,
        key="feature_store",
        env_name="FEATURE_STORE_SOURCE",
        error_type=SimulationError,
    )
    data_lake = resolve_root_dir(
        dataset_params,
        key="data_lake",
        env_name="DATA_LAKE_SOURCE",
        error_type=SimulationError,
    )
    return resolve_feature_store_version_label(
        feature_store / "features",
        data_lake,
        error_type=SimulationError,
        feature_error="No feature store versions found",
        lake_error="No data lake versions found",
    )


def _normalize_timestamps(values: Sequence[Any]) -> list[int | str]:
    normalized: list[int | str] = []
    for item in values:
        if isinstance(item, (int, np.integer)):
            normalized.append(int(item))
        elif hasattr(item, "isoformat"):
            normalized.append(str(item.isoformat()))
        else:
            normalized.append(str(item))
    return normalized


def _format_timestamp_strings(values: Sequence[Any]) -> list[str]:
    if not values:
        return []
    first = values[0]
    if isinstance(first, (int, np.integer)):
        epoch_hours = np.asarray(list(values), dtype="int64")
        stamps = pd.to_datetime(epoch_hours * 3600, unit="s", utc=True)
    else:
        stamps = pd.to_datetime(list(values), utc=True)
    return [
        stamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        for stamp in stamps.to_pydatetime()
    ]


def _build_feature_frame(inputs: SimulationInputs) -> pd.DataFrame:
    values = inputs.X.detach().cpu().numpy()
    assets = list(inputs.assets)
    features = list(inputs.features)
    columns = pd.MultiIndex.from_product(
        [assets, features], names=["asset", "feature"]
    )
    flat = values.reshape(values.shape[0], len(assets) * len(features))
    index = pd.Index(
        _format_timestamp_strings(inputs.timestamps), name="timestamp"
    )
    return pd.DataFrame(flat, index=index, columns=columns)


def _build_target_frame(inputs: SimulationInputs) -> pd.DataFrame:
    values = inputs.y.detach().cpu().numpy()
    index = pd.Index(
        _format_timestamp_strings(inputs.timestamps), name="timestamp"
    )
    return pd.DataFrame(values, index=index, columns=list(inputs.assets))


def _build_mask_frame(inputs: SimulationInputs) -> pd.DataFrame:
    values = inputs.M.detach().cpu().numpy()
    assets = list(inputs.assets)
    features = list(inputs.features)
    columns = pd.MultiIndex.from_product(
        [assets, features], names=["asset", "feature"]
    )
    flat = values.reshape(values.shape[0], len(assets) * len(features))
    index = pd.Index(
        _format_timestamp_strings(inputs.timestamps), name="timestamp"
    )
    return pd.DataFrame(flat, index=index, columns=columns)


def _to_serializable(value: Any) -> Any:
    result: Any
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
        result = _to_serializable(result)
    elif isinstance(value, Mapping):
        result = {str(k): _to_serializable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_to_serializable(item) for item in value]
    elif isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, torch.Tensor):
        result = value.detach().cpu().tolist()
    elif isinstance(value, (int, float, str, bool)) or value is None:
        result = value
    else:
        result = str(value)
    return result


def _write_json(path: Path, payload: Any, *, message: str) -> None:
    try:
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        raise SimulationError(message, context={"path": str(path)}) from exc


def _write_csv(path: Path, frame: pd.DataFrame, *, message: str) -> None:
    try:
        frame.to_csv(path)
    except Exception as exc:
        raise SimulationError(message, context={"path": str(path)}) from exc


def _save_torch(payload: Any, path: Path, *, message: str) -> None:
    try:
        torch.save(payload, path)
    except Exception as exc:
        raise SimulationError(message, context={"path": str(path)}) from exc


def _ensure_dir(path: Path, *, message: str) -> None:
    ensure_directory(
        path,
        error_type=SimulationError,
        invalid_message="Simulation output path must be a directory",
        create_message=message,
    )
