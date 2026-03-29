from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import cast

import torch
from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms import AffineCoupling
from pyro.nn import DenseNN

from .shared_v3_l1_unified import INDEX_CLASS_ID, RuntimeAssetMetadata
from .shared_v3_l6_unified import (
    RegimePosteriorMeansV3L6 as RegimePosteriorMeansV3L10,
    StructuralPosteriorMeansV3L6 as StructuralPosteriorMeansV3L10,
    build_dynamic_us_europe_spread_block,
    coerce_v3_l6_state_tensor,
    v3_l6_commodity_state_index,
    v3_l6_spread_state_index,
    v3_l6_state_count,
    v3_l6_state_site_names,
)

INDEX_FLOW_MODULE_NAME = "multi_asset_block_v3_l10_index_flow"


def v3_l10_state_count() -> int:
    return v3_l6_state_count()


def v3_l10_region_state_index() -> int:
    return v3_l6_spread_state_index()


def v3_l10_commodity_state_index() -> int:
    return v3_l6_commodity_state_index()


def v3_l10_state_site_names() -> tuple[str, ...]:
    return v3_l6_state_site_names()


def coerce_v3_l10_state_tensor(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return coerce_v3_l6_state_tensor(value, device=device, dtype=dtype)


def build_dynamic_us_europe_region_block(
    *,
    assets: RuntimeAssetMetadata,
    region_state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return build_dynamic_us_europe_spread_block(
        assets=assets,
        spread_state=region_state,
        device=device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class IndexFlowConfig:
    enabled: bool = True
    hidden_dim: int = 16
    log_scale_min_clip: float = -0.4
    log_scale_max_clip: float = 0.4


class IndexBlockAffineCouplingFlow(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        *,
        index_indices: torch.LongTensor,
        hidden_dim: int,
        log_scale_min_clip: float,
        log_scale_max_clip: float,
    ) -> None:
        super().__init__(cache_size=1)
        self.register_buffer("_index_indices_buffer", index_indices.clone())
        index_dim = int(index_indices.numel())
        split_dim = max(1, index_dim // 2)
        transformed_dim = index_dim - split_dim
        hypernet = DenseNN(
            split_dim,
            [hidden_dim],
            param_dims=[transformed_dim, transformed_dim],
            nonlinearity=torch.nn.Tanh(),
        )
        _initialize_identity(hypernet)
        self._inner = AffineCoupling(
            split_dim,
            hypernet,
            log_scale_min_clip=log_scale_min_clip,
            log_scale_max_clip=log_scale_max_clip,
        )

    @property
    def event_dim(self) -> int:
        return 1

    @property
    def sign(self) -> int:
        return 1

    @property
    def _index_indices(self) -> torch.LongTensor:
        return cast(torch.LongTensor, self._buffers["_index_indices_buffer"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        index_block = x.index_select(-1, self._index_indices)
        transformed = cast(torch.Tensor, self._inner(index_block))
        return _replace_index_block(x, self._index_indices, transformed)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        index_block = y.index_select(-1, self._index_indices)
        inverted = cast(torch.Tensor, self._inner.inv(index_block))
        return _replace_index_block(y, self._index_indices, inverted)

    def log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x_index = x.index_select(-1, self._index_indices)
        y_index = y.index_select(-1, self._index_indices)
        return self._inner.log_abs_det_jacobian(x_index, y_index)


def build_index_block_affine_flow(
    *,
    assets: RuntimeAssetMetadata,
    config: IndexFlowConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexBlockAffineCouplingFlow | None:
    if not config.enabled:
        return None
    index_indices = cast(
        torch.LongTensor,
        index_asset_indices(assets).to(device=device, dtype=torch.long),
    )
    if int(index_indices.numel()) < 2:
        return None
    flow = IndexBlockAffineCouplingFlow(
        index_indices=index_indices,
        hidden_dim=int(config.hidden_dim),
        log_scale_min_clip=float(config.log_scale_min_clip),
        log_scale_max_clip=float(config.log_scale_max_clip),
    )
    return flow.to(device=device, dtype=dtype)


def index_asset_indices(assets: RuntimeAssetMetadata) -> torch.LongTensor:
    active = (assets.class_ids == INDEX_CLASS_ID).nonzero(as_tuple=False).reshape(-1)
    return cast(torch.LongTensor, active.to(dtype=torch.long))


def _initialize_identity(network: DenseNN) -> None:
    final_layer = network.layers[-1]
    if isinstance(final_layer, torch.nn.Linear):
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)


def _replace_index_block(
    value: torch.Tensor,
    index_indices: torch.LongTensor,
    replacement: torch.Tensor,
) -> torch.Tensor:
    result = value.clone()
    result[..., index_indices] = replacement
    return result


__all__ = [
    "INDEX_FLOW_MODULE_NAME",
    "IndexBlockAffineCouplingFlow",
    "IndexFlowConfig",
    "RegimePosteriorMeansV3L10",
    "StructuralPosteriorMeansV3L10",
    "build_dynamic_us_europe_region_block",
    "build_index_block_affine_flow",
    "coerce_v3_l10_state_tensor",
    "index_asset_indices",
    "v3_l10_commodity_state_index",
    "v3_l10_region_state_index",
    "v3_l10_state_count",
    "v3_l10_state_site_names",
]
