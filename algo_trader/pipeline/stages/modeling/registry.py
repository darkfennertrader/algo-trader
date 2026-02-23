from __future__ import annotations

from .registry_core import (
    GuideRegistry,
    ModelRegistry,
    _GUIDE_REGISTRY,
    _MODEL_REGISTRY,
)


def default_model_registry() -> ModelRegistry:
    # Add new model modules here so decorators execute on registry creation.
    from . import test_model  # pylint: disable=import-outside-toplevel
    from .factor import model_v1  # pylint: disable=import-outside-toplevel

    return _MODEL_REGISTRY


def default_guide_registry() -> GuideRegistry:
    # Add new guide modules here so decorators execute on registry creation.
    from . import test_guide  # pylint: disable=import-outside-toplevel
    from .factor import guide_v1  # pylint: disable=import-outside-toplevel

    return _GUIDE_REGISTRY
