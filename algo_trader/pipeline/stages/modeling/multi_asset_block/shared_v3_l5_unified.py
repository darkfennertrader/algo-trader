from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.runtime_support import require_tensor_entry

from .shared_v3_l1_unified import (
    CovarianceLoadings,
    MeanTensorMeans,
    RuntimeAssetMetadata,
    StructuralTensorMeans,
    build_index_group_exposure,
)

_BASE_STATE_COUNT = 4
_INDEX_GROUP_INSERT_AT = 3


def v3_l5_state_count(*, group_count: int) -> int:
    return _BASE_STATE_COUNT + max(group_count, 0)


def v3_l5_group_state_slice(*, group_count: int) -> slice:
    upper = _INDEX_GROUP_INSERT_AT + max(group_count, 0)
    return slice(_INDEX_GROUP_INSERT_AT, upper)


def v3_l5_commodity_state_index(*, group_count: int) -> int:
    return _INDEX_GROUP_INSERT_AT + max(group_count, 0)


def coerce_v3_l5_state_tensor(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    group_count: int,
) -> torch.Tensor:
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    state_count = v3_l5_state_count(group_count=group_count)
    if tensor.numel() == 1:
        return tensor.expand(state_count)
    if tensor.numel() == state_count:
        return tensor
    if tensor.numel() == _BASE_STATE_COUNT:
        if group_count < 1:
            return tensor
        prefix = tensor[:_INDEX_GROUP_INSERT_AT]
        suffix = tensor[_INDEX_GROUP_INSERT_AT:]
        group_pad = torch.zeros(group_count, device=device, dtype=dtype)
        return torch.cat([prefix, group_pad, suffix], dim=0)
    raise ConfigError(
        "v3_l5_unified filtering state expects 1, 4, or 4 + group_count latent values"
    )


def build_dynamic_index_group_block(
    *,
    assets: RuntimeAssetMetadata,
    group_scale: torch.Tensor,
    group_state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    group_count = int(assets.index_group_count)
    asset_count = int(assets.class_ids.shape[0])
    if group_count < 1 or int(group_scale.numel()) < 1:
        if group_state.ndim == 1:
            return torch.zeros((asset_count, 0), device=device, dtype=dtype)
        if group_state.ndim == 2:
            sample_count = int(group_state.shape[0])
            return torch.zeros((sample_count, asset_count, 0), device=device, dtype=dtype)
        raise ConfigError("v3_l5_unified group state must be rank 1 or 2")
    exposure = build_index_group_exposure(
        group_ids=assets.index_group_ids,
        group_count=group_count,
        device=device,
        dtype=dtype,
    )
    base_scale = group_scale.to(device=device, dtype=dtype)
    if group_state.ndim == 1:
        scaled = base_scale * torch.exp(
            0.5 * group_state.to(device=device, dtype=dtype)
        )
        return exposure * scaled.unsqueeze(0)
    if group_state.ndim == 2:
        scaled = base_scale.unsqueeze(0) * torch.exp(
            0.5 * group_state.to(device=device, dtype=dtype)
        )
        return exposure.unsqueeze(0) * scaled.unsqueeze(1)
    raise ConfigError("v3_l5_unified group state must be rank 1 or 2")


@dataclass(frozen=True)
class RegimePosteriorMeansV3L5:
    s_u_fx_broad_mean: torch.Tensor
    s_u_fx_cross_mean: torch.Tensor
    s_u_index_mean: torch.Tensor
    s_u_index_group_mean: torch.Tensor
    s_u_commodity_mean: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            [
                self.s_u_fx_broad_mean.reshape(1),
                self.s_u_fx_cross_mean.reshape(1),
                self.s_u_index_mean.reshape(1),
                self.s_u_index_group_mean.reshape(-1),
                self.s_u_commodity_mean.reshape(1),
            ]
        )


@dataclass(frozen=True)
class StructuralPosteriorMeansV3L5:
    tensors: StructuralTensorMeans
    regime: RegimePosteriorMeansV3L5

    @property
    def alpha(self) -> torch.Tensor:
        return self.tensors.mean.alpha

    @property
    def sigma_idio(self) -> torch.Tensor:
        return self.tensors.mean.sigma_idio

    @property
    def w(self) -> torch.Tensor:
        return self.tensors.mean.w

    @property
    def beta(self) -> torch.Tensor:
        return self.tensors.mean.beta

    @property
    def B_global(self) -> torch.Tensor:
        return self.tensors.loadings.B_global

    @property
    def B_fx_broad(self) -> torch.Tensor:
        return self.tensors.loadings.B_fx_broad

    @property
    def B_fx_cross(self) -> torch.Tensor:
        return self.tensors.loadings.B_fx_cross

    @property
    def B_index(self) -> torch.Tensor:
        return self.tensors.loadings.B_index

    @property
    def B_index_static(self) -> torch.Tensor:
        return self.tensors.loadings.B_index_static

    @property
    def index_group_scale(self) -> torch.Tensor:
        return self.tensors.loadings.index_group_scale

    @property
    def B_commodity(self) -> torch.Tensor:
        return self.tensors.loadings.B_commodity

    @property
    def s_u_fx_broad_mean(self) -> torch.Tensor:
        return self.regime.s_u_fx_broad_mean

    @property
    def s_u_fx_cross_mean(self) -> torch.Tensor:
        return self.regime.s_u_fx_cross_mean

    @property
    def s_u_index_mean(self) -> torch.Tensor:
        return self.regime.s_u_index_mean

    @property
    def s_u_index_group_mean(self) -> torch.Tensor:
        return self.regime.s_u_index_group_mean

    @property
    def s_u_commodity_mean(self) -> torch.Tensor:
        return self.regime.s_u_commodity_mean

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "alpha": self.alpha.detach(),
            "sigma_idio": self.sigma_idio.detach(),
            "w": self.w.detach(),
            "beta": self.beta.detach(),
            "B_global": self.B_global.detach(),
            "B_fx_broad": self.B_fx_broad.detach(),
            "B_fx_cross": self.B_fx_cross.detach(),
            "B_index": self.B_index.detach(),
            "B_index_static": self.B_index_static.detach(),
            "index_group_scale": self.index_group_scale.detach(),
            "B_commodity": self.B_commodity.detach(),
            "s_u_fx_broad_mean": self.s_u_fx_broad_mean.detach(),
            "s_u_fx_cross_mean": self.s_u_fx_cross_mean.detach(),
            "s_u_index_mean": self.s_u_index_mean.detach(),
            "s_u_index_group_mean": self.s_u_index_group_mean.detach(),
            "s_u_commodity_mean": self.s_u_commodity_mean.detach(),
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeansV3L5":
        tensor_keys = (
            "alpha",
            "sigma_idio",
            "w",
            "beta",
            "B_global",
            "B_fx_broad",
            "B_fx_cross",
            "B_index",
            "B_index_static",
            "index_group_scale",
            "B_commodity",
            "s_u_fx_broad_mean",
            "s_u_fx_cross_mean",
            "s_u_index_mean",
            "s_u_index_group_mean",
            "s_u_commodity_mean",
        )
        values = {
            key: _require_tensor(payload, key).detach() for key in tensor_keys
        }
        return cls(
            tensors=StructuralTensorMeans(
                mean=MeanTensorMeans(
                    alpha=values["alpha"],
                    sigma_idio=values["sigma_idio"],
                    w=values["w"],
                    beta=values["beta"],
                ),
                loadings=CovarianceLoadings(
                    B_global=values["B_global"],
                    B_fx_broad=values["B_fx_broad"],
                    B_fx_cross=values["B_fx_cross"],
                    B_index=values["B_index"],
                    B_index_static=values["B_index_static"],
                    index_group_scale=values["index_group_scale"],
                    B_commodity=values["B_commodity"],
                ),
            ),
            regime=RegimePosteriorMeansV3L5(
                s_u_fx_broad_mean=values["s_u_fx_broad_mean"],
                s_u_fx_cross_mean=values["s_u_fx_cross_mean"],
                s_u_index_mean=values["s_u_index_mean"],
                s_u_index_group_mean=values["s_u_index_group_mean"],
                s_u_commodity_mean=values["s_u_commodity_mean"],
            ),
        )


def _require_tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    return require_tensor_entry(payload, key)


__all__ = [
    "RegimePosteriorMeansV3L5",
    "StructuralPosteriorMeansV3L5",
    "build_dynamic_index_group_block",
    "coerce_v3_l5_state_tensor",
    "v3_l5_commodity_state_index",
    "v3_l5_group_state_slice",
    "v3_l5_state_count",
]
