from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np


def expand_param_space(param_space: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not param_space:
        return [{}]
    keys = list(param_space.keys())
    values: list[Sequence[Any]] = []
    for key in keys:
        raw = param_space[key]
        if isinstance(raw, (list, tuple)):
            values.append(list(raw))
        else:
            values.append([raw])
    configs = [dict(zip(keys, combo, strict=False)) for combo in product(*values)]
    return configs or [{}]


def select_candidates(
    configs: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if len(configs) <= num_samples:
        return configs
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(configs), size=num_samples, replace=False)
    return [configs[int(i)] for i in indices]
