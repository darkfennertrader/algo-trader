from __future__ import annotations

from .registry_core import (
    GuideRegistry,
    ModelRegistry,
    PredictorRegistry,
    _GUIDE_REGISTRY,
    _MODEL_REGISTRY,
    _PREDICTOR_REGISTRY,
)


def default_model_registry() -> ModelRegistry:
    # Add new model modules here so decorators execute on registry creation.
    from . import test_model  # pylint: disable=import-outside-toplevel
    from .factor import model_l10  # pylint: disable=import-outside-toplevel
    from .factor import model_l11  # pylint: disable=import-outside-toplevel
    from .factor import model_l12  # pylint: disable=import-outside-toplevel
    from .factor import model_l13  # pylint: disable=import-outside-toplevel
    from .factor import model_l14  # pylint: disable=import-outside-toplevel
    from .factor import model_l15  # pylint: disable=import-outside-toplevel
    from .fx_currency_factor import model_v2_l1  # pylint: disable=import-outside-toplevel
    from .fx_currency_factor import model_v2_l2  # pylint: disable=import-outside-toplevel

    return _MODEL_REGISTRY


def default_guide_registry() -> GuideRegistry:
    # Add new guide modules here so decorators execute on registry creation.
    from . import test_guide  # pylint: disable=import-outside-toplevel
    from .factor import guide_l10  # pylint: disable=import-outside-toplevel
    from .factor import guide_l11  # pylint: disable=import-outside-toplevel
    from .factor import guide_l12  # pylint: disable=import-outside-toplevel
    from .factor import guide_l13  # pylint: disable=import-outside-toplevel
    from .factor import guide_l14  # pylint: disable=import-outside-toplevel
    from .factor import guide_l15  # pylint: disable=import-outside-toplevel
    from .fx_currency_factor import guide_v2_l1  # pylint: disable=import-outside-toplevel
    from .fx_currency_factor import guide_v2_l2  # pylint: disable=import-outside-toplevel

    return _GUIDE_REGISTRY


def default_predictor_registry() -> PredictorRegistry:
    # Add new predictor modules here so decorators execute on registry creation.
    from .factor import predict_l10  # pylint: disable=import-outside-toplevel
    from .factor import predict_l11  # pylint: disable=import-outside-toplevel
    from .factor import predict_l12  # pylint: disable=import-outside-toplevel
    from .factor import predict_l13  # pylint: disable=import-outside-toplevel
    from .factor import predict_l14  # pylint: disable=import-outside-toplevel
    from .factor import predict_l15  # pylint: disable=import-outside-toplevel
    from .fx_currency_factor import predict_v2_l1  # pylint: disable=import-outside-toplevel
    from .fx_currency_factor import predict_v2_l2  # pylint: disable=import-outside-toplevel

    return _PREDICTOR_REGISTRY
