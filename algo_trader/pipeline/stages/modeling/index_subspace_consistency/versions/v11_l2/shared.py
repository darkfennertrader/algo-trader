from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import Any, Mapping, cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.shared import (
    build_index_basis_coordinates,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)

_SPREAD_DIM = 4


@dataclass(frozen=True)
class IndexSubspacePriorScaleConfig:
    spread_scale_center: float = 0.30
    spread_scale_log_scale: float = 0.12
    correlation_concentration: float = 30.0


@dataclass(frozen=True)
class IndexSubspaceConfig:
    enabled: bool = True
    spread_df: float = 8.0
    obs_weight: float = 0.18
    prior_scales: IndexSubspacePriorScaleConfig = IndexSubspacePriorScaleConfig()
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexSubspacePosteriorMeans:
    spread_scale: torch.Tensor
    spread_corr_cholesky: torch.Tensor


@dataclass(frozen=True)
class IndexSubspaceCoordinates:
    basis: torch.Tensor
    index_mask: torch.BoolTensor


def build_index_subspace_config(raw: object) -> IndexSubspaceConfig:
    values = _coerce_mapping(raw, label="model.params.index_subspace_consistency")
    if not values:
        return IndexSubspaceConfig()
    base = IndexSubspaceConfig()
    scale_values = _coerce_mapping(
        values.get("prior_scales"),
        label="model.params.index_subspace_consistency.prior_scales",
    )
    return IndexSubspaceConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        spread_df=float(values.get("spread_df", base.spread_df)),
        obs_weight=float(values.get("obs_weight", base.obs_weight)),
        prior_scales=IndexSubspacePriorScaleConfig(
            spread_scale_center=float(
                scale_values.get(
                    "spread_scale_center",
                    base.prior_scales.spread_scale_center,
                )
            ),
            spread_scale_log_scale=float(
                scale_values.get(
                    "spread_scale_log_scale",
                    base.prior_scales.spread_scale_log_scale,
                )
            ),
            correlation_concentration=float(
                scale_values.get(
                    "correlation_concentration",
                    base.prior_scales.correlation_concentration,
                )
            ),
        ),
        eps=float(values.get("eps", base.eps)),
    )


def build_index_subspace_coordinates(
    *,
    assets: RuntimeAssetMetadata,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexSubspaceCoordinates:
    basis_coordinates = build_index_basis_coordinates(
        assets=assets,
        device=device,
        dtype=dtype,
    )
    basis = basis_coordinates.spread_matrix
    index_mask = cast(torch.BoolTensor, basis.abs().sum(dim=-1) > 0.0)
    return IndexSubspaceCoordinates(basis=basis, index_mask=index_mask)


def project_subspace_mean(*, loc: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ta,aj->tj", loc, basis)


def project_subspace_covariance(
    *,
    cov_factor: torch.Tensor,
    cov_diag: torch.Tensor,
    basis: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    transformed_factor = torch.einsum("tak,aj->tjk", cov_factor, basis)
    low_rank_cov = transformed_factor @ transformed_factor.transpose(-1, -2)
    diagonal_cov = torch.einsum("ai,ta,aj->tij", basis, cov_diag, basis)
    jitter = torch.eye(
        _SPREAD_DIM,
        device=cov_factor.device,
        dtype=cov_factor.dtype,
    ).unsqueeze(0) * float(eps)
    return low_rank_cov + diagonal_cov + jitter


def spread_scale_tril_from_covariance(
    *,
    subspace_covariance: torch.Tensor,
    spread_scale: torch.Tensor,
    spread_corr_cholesky: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    base_std = (
        torch.diagonal(subspace_covariance, dim1=-2, dim2=-1)
        .clamp_min(float(eps))
        .sqrt()
    )
    scaled_std = base_std * spread_scale.unsqueeze(0)
    return torch.diag_embed(scaled_std) @ spread_corr_cholesky.unsqueeze(0)


def subspace_time_mask(
    *,
    time_mask: torch.BoolTensor | None,
    index_mask: torch.BoolTensor,
) -> torch.BoolTensor | None:
    if time_mask is None:
        return None
    return time_mask


def initial_index_subspace_posterior_means(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexSubspacePosteriorMeans:
    return IndexSubspacePosteriorMeans(
        spread_scale=torch.full(
            (_SPREAD_DIM,),
            0.30,
            device=device,
            dtype=dtype,
        ),
        spread_corr_cholesky=torch.eye(_SPREAD_DIM, device=device, dtype=dtype),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = [
    "IndexSubspaceConfig",
    "IndexSubspaceCoordinates",
    "IndexSubspacePosteriorMeans",
    "IndexSubspacePriorScaleConfig",
    "build_index_subspace_config",
    "build_index_subspace_coordinates",
    "initial_index_subspace_posterior_means",
    "project_subspace_covariance",
    "project_subspace_mean",
    "spread_scale_tril_from_covariance",
    "subspace_time_mask",
]
