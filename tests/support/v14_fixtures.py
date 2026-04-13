from __future__ import annotations

import torch

from algo_trader.pipeline.stages import modeling

V14_TEST_ASSET_NAMES = (
    "EUR.USD",
    "IBUS30",
    "IBUS500",
    "IBUST100",
    "IBDE40",
    "IBES35",
    "IBEU50",
    "IBFR40",
    "IBGB100",
    "IBNL25",
    "IBCH20",
    "XAU.USD",
)


def build_v14_runtime_batch(*, with_targets: bool) -> modeling.ModelBatch:
    targets = None
    if with_targets:
        targets = torch.zeros((2, len(V14_TEST_ASSET_NAMES)), dtype=torch.float32)
    return modeling.ModelBatch(
        X_asset=torch.zeros((2, len(V14_TEST_ASSET_NAMES), 4), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=targets,
        asset_names=V14_TEST_ASSET_NAMES,
    )


__all__ = ["V14_TEST_ASSET_NAMES", "build_v14_runtime_batch"]
