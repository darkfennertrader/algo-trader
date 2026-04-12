from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.config_support import coerce_mapping
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide, PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.runtime_support import (
    sample_time_observations,
)

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .guide_v3_l1_unified import MultiAssetBlockGuideV3L1UnifiedOnlineFiltering
from .predict_v3_l1_unified import predict_multi_asset_block_v3_l1_unified
from .shared_v3_l1_unified import (
    CovarianceLoadings,
    FX_CLASS_ID,
    INDEX_CLASS_ID,
    COMMODITY_CLASS_ID,
    FactorCountConfig,
    MeanTensorMeans,
    PanelDimensions,
    RuntimeAssetMetadata,
    StructuralTensorMeans,
    V3L1UnifiedRuntimeBatch,
    asset_class_mask,
    build_factor_count_config,
    build_index_group_block,
    build_panel_dimensions,
    coerce_four_state_tensor,
)

_STATE_COUNT = 4


@dataclass(frozen=True)
class MeanPriors:
    alpha_scale: float = 0.03
    sigma_fx_scale: float = 0.03
    sigma_index_scale: float = 0.06
    sigma_commodity_scale: float = 0.04
    beta_scale: float = 0.05
    tau0_scale: float = 0.05


@dataclass(frozen=True)
class FactorScalePriors:
    global_b_scale: float = 0.08
    fx_broad_b_scale: float = 0.10
    fx_cross_b_scale: float = 0.06
    index_b_scale: float = 0.08
    index_static_b_scale: float = 0.10
    index_group_scale: float = 0.0
    commodity_b_scale: float = 0.08


@dataclass(frozen=True)
class FactorPriors:
    counts: FactorCountConfig = field(default_factory=FactorCountConfig)
    scales: FactorScalePriors = field(default_factory=FactorScalePriors)


@dataclass(frozen=True)
class RegimeBlockPriors:
    phi: float = 0.97
    s_u_scale: float = 0.03


@dataclass(frozen=True)
class RegimePriors:
    fx_broad: RegimeBlockPriors = field(
        default_factory=lambda: RegimeBlockPriors(phi=0.95, s_u_scale=0.03)
    )
    fx_cross: RegimeBlockPriors = field(
        default_factory=lambda: RegimeBlockPriors(phi=0.985, s_u_scale=0.01)
    )
    index: RegimeBlockPriors = field(
        default_factory=lambda: RegimeBlockPriors(phi=0.97, s_u_scale=0.02)
    )
    commodity: RegimeBlockPriors = field(
        default_factory=lambda: RegimeBlockPriors(phi=0.97, s_u_scale=0.02)
    )
    eps: float = 1e-6


@dataclass(frozen=True)
class V3L1UnifiedModelPriors:
    mean: MeanPriors = field(default_factory=MeanPriors)
    factors: FactorPriors = field(default_factory=FactorPriors)
    regime: RegimePriors = field(default_factory=RegimePriors)


@dataclass(frozen=True)
class _ModelContext:
    batch: V3L1UnifiedRuntimeBatch
    device: torch.device
    dtype: torch.dtype
    priors: V3L1UnifiedModelPriors
    shape: PanelDimensions


@dataclass(frozen=True)
class _RegimeScales:
    values: torch.Tensor


@dataclass
class MultiAssetBlockModelV3L1UnifiedOnlineFiltering(PyroModel):
    priors: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_context(build_v3_l1_unified_runtime_batch(batch), self.priors)
        structural = _sample_structural_sites(context)
        regime_scales = _sample_regime_scales(context)
        regime_path = _sample_regime_path(context, regime_scales)
        obs_dist = _build_observation_distribution(context, structural, regime_path)
        _sample_observations(context, obs_dist)

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        summaries = getattr(guide, "structural_predictive_summaries", None)
        if not callable(summaries):
            summaries = getattr(guide, "structural_posterior_means", None)
        if not callable(summaries):
            return None
        return predict_multi_asset_block_v3_l1_unified(
            model=self,
            guide=cast(MultiAssetBlockGuideV3L1UnifiedOnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("multi_asset_block_model_v3_l1_unified_online_filtering")
def build_multi_asset_block_model_v3_l1_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    return MultiAssetBlockModelV3L1UnifiedOnlineFiltering(
        priors=_build_model_priors(params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V3L1UnifiedModelPriors:
    values = coerce_mapping(params, label="model.params")
    if not values:
        return V3L1UnifiedModelPriors()
    extra = set(values) - {"mean", "factors", "regime"}
    if extra:
        raise ConfigError(
            "Unknown multi_asset_block_model_v3_l1_unified_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return V3L1UnifiedModelPriors(
        mean=_build_mean_priors(values.get("mean")),
        factors=_build_factor_priors(values.get("factors")),
        regime=_build_regime_priors(values.get("regime")),
    )


def _build_mean_priors(raw: object) -> MeanPriors:
    values = coerce_mapping(raw, label="model.params.mean")
    if not values:
        return MeanPriors()
    base = MeanPriors()
    return MeanPriors(
        alpha_scale=float(values.get("alpha_scale", base.alpha_scale)),
        sigma_fx_scale=float(values.get("sigma_fx_scale", base.sigma_fx_scale)),
        sigma_index_scale=float(values.get("sigma_index_scale", base.sigma_index_scale)),
        sigma_commodity_scale=float(
            values.get("sigma_commodity_scale", base.sigma_commodity_scale)
        ),
        beta_scale=float(values.get("beta_scale", base.beta_scale)),
        tau0_scale=float(values.get("tau0_scale", base.tau0_scale)),
    )


def _build_factor_priors(raw: object) -> FactorPriors:
    values = coerce_mapping(raw, label="model.params.factors")
    if not values:
        return FactorPriors()
    base = FactorPriors()
    return FactorPriors(
        counts=build_factor_count_config(values, base=base.counts),
        scales=FactorScalePriors(
            global_b_scale=float(
                values.get("global_b_scale", base.scales.global_b_scale)
            ),
            fx_broad_b_scale=float(
                values.get(
                    "fx_broad_b_scale", base.scales.fx_broad_b_scale
                )
            ),
            fx_cross_b_scale=float(
                values.get(
                    "fx_cross_b_scale", base.scales.fx_cross_b_scale
                )
            ),
            index_b_scale=float(
                values.get("index_b_scale", base.scales.index_b_scale)
            ),
            index_static_b_scale=float(
                values.get(
                    "index_static_b_scale",
                    base.scales.index_static_b_scale,
                )
            ),
            index_group_scale=float(
                values.get(
                    "index_group_scale",
                    base.scales.index_group_scale,
                )
            ),
            commodity_b_scale=float(
                values.get(
                    "commodity_b_scale", base.scales.commodity_b_scale
                )
            ),
        ),
    )


def _build_regime_priors(raw: object) -> RegimePriors:
    values = coerce_mapping(raw, label="model.params.regime")
    if not values:
        return RegimePriors()
    return RegimePriors(
        fx_broad=_build_regime_block(values.get("fx_broad"), RegimePriors().fx_broad),
        fx_cross=_build_regime_block(values.get("fx_cross"), RegimePriors().fx_cross),
        index=_build_regime_block(values.get("index"), RegimePriors().index),
        commodity=_build_regime_block(values.get("commodity"), RegimePriors().commodity),
        eps=float(values.get("eps", RegimePriors().eps)),
    )


def _build_regime_block(raw: object, base: RegimeBlockPriors) -> RegimeBlockPriors:
    values = coerce_mapping(raw, label="model.params.regime.block")
    if not values:
        return base
    return RegimeBlockPriors(
        phi=float(values.get("phi", base.phi)),
        s_u_scale=float(values.get("s_u_scale", base.s_u_scale)),
    )


def _build_context(
    batch: V3L1UnifiedRuntimeBatch, priors: V3L1UnifiedModelPriors
) -> _ModelContext:
    asset_tensor = batch.X_asset
    return _ModelContext(
        batch=batch,
        device=asset_tensor.device,
        dtype=asset_tensor.dtype,
        priors=priors,
        shape=build_panel_dimensions(batch),
    )


def _sample_structural_sites(context: _ModelContext) -> StructuralTensorMeans:
    alpha = _sample_alpha(context)
    sigma_idio = _sample_sigma_idio(context)
    w = _sample_w(context)
    beta = _sample_beta(context)
    return StructuralTensorMeans(
        mean=MeanTensorMeans(
            alpha=alpha,
            sigma_idio=sigma_idio,
            w=w,
            beta=beta,
        ),
        loadings=CovarianceLoadings(
            B_global=_sample_loading(
                context,
                "B_global",
                context.priors.factors.counts.global_factor_count,
                context.priors.factors.scales.global_b_scale,
            ),
            B_fx_broad=_sample_loading(
                context,
                "B_fx_broad",
                context.priors.factors.counts.fx_broad_factor_count,
                context.priors.factors.scales.fx_broad_b_scale,
            ),
            B_fx_cross=_sample_loading(
                context,
                "B_fx_cross",
                context.priors.factors.counts.fx_cross_factor_count,
                context.priors.factors.scales.fx_cross_b_scale,
            ),
            B_index=_sample_loading(
                context,
                "B_index",
                context.priors.factors.counts.index_factor_count,
                context.priors.factors.scales.index_b_scale,
            ),
            B_index_static=_sample_loading(
                context,
                "B_index_static",
                context.priors.factors.counts.index_static_factor_count,
                context.priors.factors.scales.index_static_b_scale,
            ),
            index_group_scale=_sample_index_group_scale(context),
            B_commodity=_sample_loading(
                context,
                "B_commodity",
                context.priors.factors.counts.commodity_factor_count,
                context.priors.factors.scales.commodity_b_scale,
            ),
        ),
    )


def _sample_alpha(context: _ModelContext) -> torch.Tensor:
    with pyro.plate("asset", context.shape.A, dim=-2):
        return pyro.sample(
            "alpha",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                torch.tensor(context.priors.mean.alpha_scale, device=context.device, dtype=context.dtype),
            ),
        ).squeeze(-1)


def _sample_sigma_idio(context: _ModelContext) -> torch.Tensor:
    scale = _sigma_prior_scale(context)
    with pyro.plate("asset_sigma", context.shape.A, dim=-1):
        return pyro.sample(
            "sigma_idio",
            dist.LogNormal(scale.log(), torch.full_like(scale, 0.35)),
        )


def _sigma_prior_scale(context: _ModelContext) -> torch.Tensor:
    class_ids = context.batch.assets.class_ids
    scale = torch.full(
        (context.shape.A,),
        context.priors.mean.sigma_index_scale,
        device=context.device,
        dtype=context.dtype,
    )
    scale = torch.where(
        class_ids == FX_CLASS_ID,
        torch.full_like(scale, context.priors.mean.sigma_fx_scale),
        scale,
    )
    return torch.where(
        class_ids == COMMODITY_CLASS_ID,
        torch.full_like(scale, context.priors.mean.sigma_commodity_scale),
        scale,
    )


def _sample_w(context: _ModelContext) -> torch.Tensor:
    tau0 = pyro.sample(
        "tau0",
        dist.HalfNormal(
            torch.tensor(context.priors.mean.tau0_scale, device=context.device, dtype=context.dtype)
        ),
    )
    with pyro.plate("asset_w", context.shape.A, dim=-2):
        with pyro.plate("feature_w", context.shape.F, dim=-1):
            return pyro.sample(
                "w",
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    tau0,
                ),
            )


def _sample_beta(context: _ModelContext) -> torch.Tensor:
    with pyro.plate("asset_beta", context.shape.A, dim=-2):
        with pyro.plate("global_beta", context.shape.G, dim=-1):
            return pyro.sample(
                "beta",
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    torch.tensor(context.priors.mean.beta_scale, device=context.device, dtype=context.dtype),
                ),
            )


def _sample_loading(
    context: _ModelContext, name: str, factor_count: int, scale: float
) -> torch.Tensor:
    if factor_count < 1:
        return torch.zeros(
            (context.shape.A, 0),
            device=context.device,
            dtype=context.dtype,
        )
    with pyro.plate(f"{name}_asset", context.shape.A, dim=-2):
        with pyro.plate(f"{name}_k", factor_count, dim=-1):
            return pyro.sample(
                name,
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    torch.tensor(scale, device=context.device, dtype=context.dtype),
                ),
            )


def _sample_index_group_scale(context: _ModelContext) -> torch.Tensor:
    group_count = int(context.batch.assets.index_group_count)
    scale = float(context.priors.factors.scales.index_group_scale)
    if group_count < 1 or scale <= 0.0:
        return torch.zeros(0, device=context.device, dtype=context.dtype)
    with pyro.plate("index_group", group_count, dim=-1):
        return pyro.sample(
            "index_group_scale",
            dist.HalfNormal(
                torch.full(
                    (group_count,),
                    scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )


def _sample_regime_scales(context: _ModelContext) -> _RegimeScales:
    values = torch.stack(
        [
            pyro.sample("s_u_fx_broad", dist.HalfNormal(torch.tensor(context.priors.regime.fx_broad.s_u_scale, device=context.device, dtype=context.dtype))),
            pyro.sample("s_u_fx_cross", dist.HalfNormal(torch.tensor(context.priors.regime.fx_cross.s_u_scale, device=context.device, dtype=context.dtype))),
            pyro.sample("s_u_index", dist.HalfNormal(torch.tensor(context.priors.regime.index.s_u_scale, device=context.device, dtype=context.dtype))),
            pyro.sample("s_u_commodity", dist.HalfNormal(torch.tensor(context.priors.regime.commodity.s_u_scale, device=context.device, dtype=context.dtype))),
        ]
    )
    return _RegimeScales(values=values)


def _sample_regime_path(
    context: _ModelContext, regime_scales: _RegimeScales
) -> torch.Tensor:
    phi = _phi_vector(context)
    previous = _initial_regime_loc(context)
    states = []
    for index in range(context.shape.T):
        mean = phi * previous
        state = pyro.sample(
            f"h_{index + 1}",
            dist.Normal(mean, regime_scales.values).to_event(1),
        )
        states.append(state)
        previous = state
    return torch.stack(states, dim=0)


def _phi_vector(context: _ModelContext) -> torch.Tensor:
    return torch.tensor(
        [
            context.priors.regime.fx_broad.phi,
            context.priors.regime.fx_cross.phi,
            context.priors.regime.index.phi,
            context.priors.regime.commodity.phi,
        ],
        device=context.device,
        dtype=context.dtype,
    )


def _initial_regime_loc(context: _ModelContext) -> torch.Tensor:
    if context.batch.filtering_state is None:
        return torch.zeros(_STATE_COUNT, device=context.device, dtype=context.dtype)
    return coerce_four_state_tensor(
        context.batch.filtering_state.h_loc,
        device=context.device,
        dtype=context.dtype,
    )


def _build_observation_distribution(
    context: _ModelContext,
    structural: StructuralTensorMeans,
    regime_path: torch.Tensor,
) -> dist.LowRankMultivariateNormal:
    loc = _mean_path(context, structural)
    cov_factor = _cov_factor_path(context, structural, regime_path)
    cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=cov_factor,
        cov_diag=cov_diag,
    )


def _mean_path(
    context: _ModelContext, structural: StructuralTensorMeans
) -> torch.Tensor:
    asset_term = (
        context.batch.X_asset * structural.mean.w.unsqueeze(0)
    ).sum(dim=-1)
    global_term = torch.einsum(
        "tg,ag->ta", context.batch.X_global, structural.mean.beta
    )
    return structural.mean.alpha.unsqueeze(0) + asset_term + global_term


def _cov_factor_path(
    context: _ModelContext,
    structural: StructuralTensorMeans,
    regime_path: torch.Tensor,
) -> torch.Tensor:
    dtype = context.dtype
    class_ids = context.batch.assets.class_ids
    fx_mask = asset_class_mask(
        class_ids, class_id=FX_CLASS_ID, dtype=dtype
    ).unsqueeze(-1)
    index_mask = asset_class_mask(
        class_ids, class_id=INDEX_CLASS_ID, dtype=dtype
    ).unsqueeze(-1)
    commodity_mask = asset_class_mask(
        class_ids, class_id=COMMODITY_CLASS_ID, dtype=dtype
    ).unsqueeze(-1)
    loadings = structural.loadings
    fx_broad = loadings.B_fx_broad.unsqueeze(0) * fx_mask.unsqueeze(0) * torch.exp(0.5 * regime_path[:, 0]).view(-1, 1, 1)
    fx_cross = loadings.B_fx_cross.unsqueeze(0) * fx_mask.unsqueeze(0) * torch.exp(0.5 * regime_path[:, 1]).view(-1, 1, 1)
    index_static = loadings.B_index_static.unsqueeze(0) * index_mask.unsqueeze(0)
    index_group_block = _index_group_block(
        assets=context.batch.assets,
        loadings=loadings,
        device=context.device,
        dtype=context.dtype,
    )
    index_block = loadings.B_index.unsqueeze(0) * index_mask.unsqueeze(0) * torch.exp(0.5 * regime_path[:, 2]).view(-1, 1, 1)
    commodity_block = loadings.B_commodity.unsqueeze(0) * commodity_mask.unsqueeze(0) * torch.exp(0.5 * regime_path[:, 3]).view(-1, 1, 1)
    return torch.cat(
        [
            loadings.B_global.unsqueeze(0).expand(context.shape.T, -1, -1),
            fx_broad,
            fx_cross,
            index_static.expand(context.shape.T, -1, -1),
            index_group_block.expand(context.shape.T, -1, -1),
            index_block,
            commodity_block,
        ],
        dim=-1,
    )


def _index_group_block(
    *,
    assets: RuntimeAssetMetadata,
    loadings: CovarianceLoadings,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return build_index_group_block(
        assets=assets,
        group_scale=loadings.index_group_scale,
        device=device,
        dtype=dtype,
    )


def _sample_observations(
    context: _ModelContext, obs_dist: dist.LowRankMultivariateNormal
) -> None:
    sample_time_observations(
        time_count=context.shape.T,
        obs_dist=obs_dist,
        y_obs=context.batch.observations.y_obs,
        time_mask=context.batch.observations.time_mask,
        obs_scale=context.batch.observations.obs_scale,
    )
