from __future__ import annotations

from importlib import import_module

from .registry_core import (
    GuideRegistry,
    ModelRegistry,
    PredictorRegistry,
    _GUIDE_REGISTRY,
    _MODEL_REGISTRY,
    _PREDICTOR_REGISTRY,
)

_MODEL_MODULES = (
    "algo_trader.pipeline.stages.modeling.test_model",
    "algo_trader.pipeline.stages.modeling.factor.model_l10",
    "algo_trader.pipeline.stages.modeling.factor.model_l11",
    "algo_trader.pipeline.stages.modeling.factor.model_l12",
    "algo_trader.pipeline.stages.modeling.factor.model_l13",
    "algo_trader.pipeline.stages.modeling.factor.model_l14",
    "algo_trader.pipeline.stages.modeling.factor.model_l15",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l1",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l2",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l3",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l4",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l5",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l6",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l2_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l3_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l4_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l5_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l6_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l7_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l8_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l9_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10a_clean_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10b_clean_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10c_clean_unified",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.model",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.model",
    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.model",
    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.model",
    "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.model",
    "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.model",
    "algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.model",
    "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l1.model",
    "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.model",
    "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.model",
    "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l2.model",
    "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l1.model",
    "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l2.model",
    "algo_trader.pipeline.stages.modeling.equity_fx_measurement.versions.v12_l1.model",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l2.model",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l3.model",
    "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.model",
    "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l2.model",
)

_GUIDE_MODULES = (
    "algo_trader.pipeline.stages.modeling.test_guide",
    "algo_trader.pipeline.stages.modeling.factor.guide_l10",
    "algo_trader.pipeline.stages.modeling.factor.guide_l11",
    "algo_trader.pipeline.stages.modeling.factor.guide_l12",
    "algo_trader.pipeline.stages.modeling.factor.guide_l13",
    "algo_trader.pipeline.stages.modeling.factor.guide_l14",
    "algo_trader.pipeline.stages.modeling.factor.guide_l15",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l1",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l2",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l3",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l4",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l5",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l6",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l2_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l3_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l4_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l5_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l6_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l7_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l8_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l9_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l10_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l10a_clean_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l10b_clean_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l10c_clean_unified",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.guide",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.guide",
    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.guide",
    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.guide",
    "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.guide",
    "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.guide",
    "algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.guide",
    "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l1.guide",
    "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.guide",
    "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.guide",
    "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l2.guide",
    "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l1.guide",
    "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l2.guide",
    "algo_trader.pipeline.stages.modeling.equity_fx_measurement.versions.v12_l1.guide",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l2.guide",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l3.guide",
    "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.guide",
    "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l2.guide",
)

_PREDICTOR_MODULES = (
    "algo_trader.pipeline.stages.modeling.factor.predict_l10",
    "algo_trader.pipeline.stages.modeling.factor.predict_l11",
    "algo_trader.pipeline.stages.modeling.factor.predict_l12",
    "algo_trader.pipeline.stages.modeling.factor.predict_l13",
    "algo_trader.pipeline.stages.modeling.factor.predict_l14",
    "algo_trader.pipeline.stages.modeling.factor.predict_l15",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.predict_v2_l1",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.predict_v2_l2",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.predict_v2_l3",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.predict_v2_l4",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.predict_v2_l5",
    "algo_trader.pipeline.stages.modeling.fx_currency_factor.predict_v2_l6",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l1_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l2_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l3_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l4_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l5_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l6_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l7_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l8_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l9_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l10_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l10a_clean_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l10b_clean_unified",
    "algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l10c_clean_unified",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.predict",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.predict",
    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.predict",
    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.predict",
    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.predict",
    "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.predict",
    "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.predict",
    "algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.predict",
    "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l1.predict",
    "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.predict",
    "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.predict",
    "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l2.predict",
    "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l1.predict",
    "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l2.predict",
    "algo_trader.pipeline.stages.modeling.equity_fx_measurement.versions.v12_l1.predict",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.predict",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l2.predict",
    "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l3.predict",
    "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.predict",
    "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l2.predict",
)


def _import_registry_modules(module_names: tuple[str, ...]) -> None:
    for module_name in module_names:
        import_module(module_name)


def default_model_registry() -> ModelRegistry:
    _import_registry_modules(_MODEL_MODULES)
    return _MODEL_REGISTRY


def default_guide_registry() -> GuideRegistry:
    _import_registry_modules(_GUIDE_MODULES)
    return _GUIDE_REGISTRY


def default_predictor_registry() -> PredictorRegistry:
    _import_registry_modules(_PREDICTOR_MODULES)
    return _PREDICTOR_REGISTRY
