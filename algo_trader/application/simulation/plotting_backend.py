from __future__ import annotations

from typing import Any

from algo_trader.domain import SimulationError


def require_pyplot() -> Any:
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover
        raise SimulationError(
            "Plotting dependencies missing for simulation diagnostics"
        ) from exc
    return plt


def require_pyplot_and_seaborn() -> tuple[Any, Any]:
    plt = require_pyplot()
    try:
        import seaborn as sns  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover
        raise SimulationError(
            "Simulation diagnostics require seaborn"
        ) from exc
    return plt, sns
