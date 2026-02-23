from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyro
import torch

from algo_trader.domain import ConfigError, DataSourceError, InferenceError
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputNames,
    OutputPaths,
    OutputWriter,
    ensure_directory,
    format_run_at,
    format_tilde_path,
    log_boundary,
    move_tensor_to_device,
    require_env,
    resolve_torch_device,
    resolve_latest_week_dir,
)
from algo_trader.pipeline import modeling
from algo_trader.preprocessing import normalize_datetime_index
from ..data_io import read_indexed_csv
from ..pipeline_utils import parse_pipeline_name

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_NAMES = OutputNames(
    output_name="params.csv",
    metadata_name="metadata.json",
)


@dataclass(frozen=True)
class InferenceOptions:
    steps: int = 200
    learning_rate: Decimal = Decimal("0.01")
    seed: int | None = None


@dataclass(frozen=True)
class DataSelection:
    preprocessor_name: str = "identity"
    pipeline: str = "debug"
    input_path: Path | None = None


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    guide_name: str
    options: InferenceOptions
    data: DataSelection


@dataclass(frozen=True)
class InputState:
    frame: pd.DataFrame
    input_path: Path
    version_label: str


@dataclass(frozen=True)
class InferenceResult:
    params: Mapping[str, torch.Tensor]
    final_loss: Decimal


@dataclass(frozen=True)
class MetadataContext:
    config: InferenceConfig
    input_state: InputState
    output_path: Path
    param_shapes: Mapping[str, list[int]]
    final_loss: Decimal


def _run_context(
    model_name: str,
    guide_name: str,
    options: InferenceOptions | None = None,
    data: DataSelection | None = None,
) -> Mapping[str, str]:
    resolved = options or InferenceOptions()
    return {
        "model": model_name,
        "guide": guide_name,
        "steps": str(resolved.steps),
    }


@log_boundary("modeling.run", context=_run_context)
def run(
    *,
    model_name: str,
    guide_name: str,
    options: InferenceOptions | None = None,
    data: DataSelection | None = None,
) -> Path:
    config = _build_config(
        model_name=model_name,
        guide_name=guide_name,
        options=options,
        data=data,
    )
    input_state = _load_input_state(config)
    model = _resolve_model(config.model_name)
    guide = _resolve_guide(config.guide_name)
    result = _run_inference(
        model=model,
        guide=guide,
        data=_frame_to_tensor(input_state.frame),
        options=config.options,
    )
    output_paths = _write_outputs(config, input_state, result)
    logger.info(
        "Saved inference output path=%s params=%s metadata=%s",
        output_paths.output_path,
        len(result.params),
        output_paths.metadata_path,
    )
    return output_paths.output_path


def _build_config(
    *,
    model_name: str,
    guide_name: str,
    options: InferenceOptions | None,
    data: DataSelection | None,
) -> InferenceConfig:
    normalized_options = _normalize_options(options)
    normalized_data = _normalize_data_selection(data)
    return InferenceConfig(
        model_name=_normalize_name(model_name, label="model"),
        guide_name=_normalize_name(guide_name, label="guide"),
        options=normalized_options,
        data=normalized_data,
    )


def _normalize_options(
    options: InferenceOptions | None,
) -> InferenceOptions:
    resolved = options or InferenceOptions()
    if resolved.steps <= 0:
        raise ConfigError(
            "steps must be a positive integer",
            context={"steps": str(resolved.steps)},
        )
    learning_rate = _coerce_decimal(resolved.learning_rate)
    if learning_rate <= Decimal("0"):
        raise ConfigError(
            "learning_rate must be positive",
            context={"learning_rate": str(learning_rate)},
        )
    if resolved.seed is not None and resolved.seed < 0:
        raise ConfigError(
            "seed must be a non-negative integer",
            context={"seed": str(resolved.seed)},
        )
    return InferenceOptions(
        steps=resolved.steps,
        learning_rate=learning_rate,
        seed=resolved.seed,
    )


def _normalize_data_selection(
    selection: DataSelection | None,
) -> DataSelection:
    resolved = selection or DataSelection()
    pipeline = parse_pipeline_name(resolved.pipeline)
    preprocessor_name = _normalize_name(
        resolved.preprocessor_name, label="preprocessor"
    )
    return DataSelection(
        preprocessor_name=preprocessor_name,
        pipeline=pipeline,
        input_path=resolved.input_path,
    )


def _load_input_state(config: InferenceConfig) -> InputState:
    if config.data.input_path is not None:
        path = Path(config.data.input_path).expanduser()
        return _load_input_from_path(path, format_run_at(_now()))
    return _load_input_from_feature_store(
        preprocessor_name=config.data.preprocessor_name,
        pipeline=config.data.pipeline,
    )


def _load_input_from_path(path: Path, version_label: str) -> InputState:
    if not path.exists():
        raise DataSourceError(
            "Input path does not exist",
            context={"path": str(path)},
        )
    frame = _load_frame(path)
    return InputState(
        frame=frame,
        input_path=path,
        version_label=version_label,
    )


def _load_input_from_feature_store(
    *, preprocessor_name: str, pipeline: str
) -> InputState:
    feature_store = _resolve_feature_store()
    base_dir = feature_store / preprocessor_name
    if pipeline:
        base_dir = base_dir / pipeline
    if not base_dir.exists():
        raise DataSourceError(
            "Prepared data directory not found",
            context={"path": str(base_dir)},
        )
    if not base_dir.is_dir():
        raise DataSourceError(
            "Prepared data path must be a directory",
            context={"path": str(base_dir)},
        )
    latest_dir = resolve_latest_week_dir(
        base_dir,
        error_type=DataSourceError,
        error_message="No prepared data directories found",
    )
    input_name = _input_name_for_preprocessor(preprocessor_name)
    input_path = latest_dir / input_name
    frame = _load_frame(input_path)
    return InputState(
        frame=frame,
        input_path=input_path,
        version_label=latest_dir.name,
    )


def _input_name_for_preprocessor(preprocessor_name: str) -> str:
    if preprocessor_name == "pca":
        return "factors.csv"
    return "processed.csv"


def _load_frame(path: Path) -> pd.DataFrame:
    frame = read_indexed_csv(
        path,
        missing_message="Prepared data file not found",
        read_message="Failed to read prepared data CSV",
    )
    frame = normalize_datetime_index(
        frame, label="input", preprocessor_name="modeling"
    )
    if frame.empty:
        raise InferenceError(
            "Prepared data is empty",
            context={"path": str(path)},
        )
    if frame.shape[1] < 1:
        raise InferenceError(
            "Prepared data must have at least one column",
            context={"path": str(path)},
        )
    if frame.isna().any().any():
        raise InferenceError(
            "Prepared data contains missing values",
            context={"path": str(path)},
        )
    try:
        numeric = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise InferenceError(
            "Prepared data must be numeric",
            context={"path": str(path)},
        ) from exc
    return numeric


def _resolve_model(
    name: str, params: Mapping[str, Any] | None = None
) -> modeling.PyroModel:
    registry = modeling.default_model_registry()
    return registry.get(name, params)


def _resolve_guide(
    name: str, params: Mapping[str, Any] | None = None
) -> modeling.PyroGuide:
    registry = modeling.default_guide_registry()
    return registry.get(name, params)


def _run_inference(
    *,
    model: modeling.PyroModel,
    guide: modeling.PyroGuide,
    data: torch.Tensor,
    options: InferenceOptions,
) -> InferenceResult:
    device = resolve_torch_device()
    data = move_tensor_to_device(data, device)
    logger.info("Using torch device=%s", device)
    if options.seed is not None:
        pyro.set_rng_seed(options.seed)
    pyro.clear_param_store()
    # pylint: disable=no-member
    optimizer = pyro.optim.Adam({"lr": float(options.learning_rate)})  # type: ignore
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())  # type: ignore
    final_loss = Decimal("0")
    log_interval = _log_interval(options.steps)
    batch = modeling.ModelBatch(X=None, y=data, M=None)
    for step in range(options.steps):
        loss = svi.step(batch)
        final_loss = Decimal(str(loss))
        if (step + 1) % log_interval == 0 or step == 0:
            logger.info(
                "SVI step=%s loss=%.4f",
                step + 1,
                float(final_loss),
            )
    params = {
        name: value.detach().cpu()
        for name, value in pyro.get_param_store().items()
    }
    return InferenceResult(params=params, final_loss=final_loss)


def _log_interval(steps: int) -> int:
    if steps <= 10:
        return 1
    return max(1, steps // 5)


def _frame_to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    values = frame.to_numpy(dtype=float, copy=True)
    return torch.as_tensor(values, dtype=torch.float32)


def _write_outputs(
    config: InferenceConfig,
    input_state: InputState,
    result: InferenceResult,
) -> OutputPaths:
    output_paths = _prepare_output_paths(
        model_name=config.model_name,
        guide_name=config.guide_name,
        pipeline=config.data.pipeline,
        version_label=input_state.version_label,
    )
    writer = _output_writer()
    params_frame, param_shapes = _params_frame(result.params)
    writer.write_frame(params_frame, output_paths.output_path)
    metadata = _build_metadata(
        MetadataContext(
            config=config,
            input_state=input_state,
            output_path=output_paths.output_path,
            param_shapes=param_shapes,
            final_loss=result.final_loss,
        )
    )
    writer.write_metadata(metadata, output_paths.metadata_path)
    return output_paths


def _prepare_output_paths(
    *,
    model_name: str,
    guide_name: str,
    pipeline: str,
    version_label: str,
) -> OutputPaths:
    model_store = _resolve_model_store()
    output_dir = model_store / model_name / guide_name
    if pipeline:
        output_dir = output_dir / pipeline
    output_dir = output_dir / version_label
    ensure_directory(
        output_dir,
        error_type=InferenceError,
        invalid_message="Model store output path must be a directory",
        create_message="Failed to prepare model store output directory",
    )
    return OutputPaths(
        output_dir=output_dir,
        output_path=output_dir / _DEFAULT_OUTPUT_NAMES.output_name,
        metadata_path=output_dir / _DEFAULT_OUTPUT_NAMES.metadata_name,
    )


def _output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=InferenceError,
            message="Failed to write inference parameters",
        ),
        metadata_policy=ErrorPolicy(
            error_type=InferenceError,
            message="Failed to write inference metadata",
        ),
    )


def _params_frame(
    params: Mapping[str, torch.Tensor],
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    rows: list[dict[str, Decimal | int | str]] = []
    shapes: dict[str, list[int]] = {}
    for name, tensor in params.items():
        shapes[name] = list(tensor.shape)
        flat = tensor.reshape(-1).tolist()
        for index, value in enumerate(flat):
            rows.append(
                {
                    "param": name,
                    "index": index,
                    "value": _coerce_decimal(value),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, shapes


def _build_metadata(context: MetadataContext) -> dict[str, object]:
    return {
        "run_at": format_run_at(_now()),
        "model": context.config.model_name,
        "guide": context.config.guide_name,
        "steps": context.config.options.steps,
        "learning_rate": context.config.options.learning_rate,
        "seed": context.config.options.seed,
        "preprocessor": context.config.data.preprocessor_name,
        "pipeline": context.config.data.pipeline,
        "rows": len(context.input_state.frame),
        "columns": len(context.input_state.frame.columns),
        "final_loss": context.final_loss,
        "param_shapes": context.param_shapes,
        "source": format_tilde_path(context.input_state.input_path),
        "destination": format_tilde_path(context.output_path),
    }


def _resolve_feature_store() -> Path:
    feature_store = Path(require_env("FEATURE_STORE_SOURCE")).expanduser()
    if not feature_store.exists():
        raise DataSourceError(
            "FEATURE_STORE_SOURCE does not exist",
            context={"path": str(feature_store)},
        )
    if not feature_store.is_dir():
        raise DataSourceError(
            "FEATURE_STORE_SOURCE must be a directory",
            context={"path": str(feature_store)},
        )
    return feature_store


def _resolve_model_store() -> Path:
    model_store = Path(require_env("MODEL_STORE_SOURCE")).expanduser()
    ensure_directory(
        model_store,
        error_type=InferenceError,
        invalid_message="MODEL_STORE_SOURCE must be a directory",
        create_message="MODEL_STORE_SOURCE cannot be created",
    )
    return model_store


def _normalize_name(raw: str, *, label: str) -> str:
    normalized = raw.strip().lower()
    if not normalized:
        raise ConfigError(f"{label} name must not be empty")
    return normalized


def _coerce_decimal(value: Decimal | float | int) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _now() -> datetime:
    return datetime.now(timezone.utc)
