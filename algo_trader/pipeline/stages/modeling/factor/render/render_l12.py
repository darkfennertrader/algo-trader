from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.pipeline.stages.modeling.factor.guide_l12 import (
    FactorGuideL12OnlineFiltering,
    FilteringState,
    Level12GuideConfig,
)
from algo_trader.pipeline.stages.modeling.factor.model_l12 import (
    FactorModelL12OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch


# Run from repo root:
# uv run python -m algo_trader.pipeline.stages.modeling.factor.render.render_l12


def main() -> None:
    output_dir = _render_output_dir()
    model_path = _render_model_graph(output_dir)
    guide_path = _render_guide_graph(output_dir)
    predict_path = _render_predict_graph(output_dir)
    print(f"Saved model graph to {model_path}")
    print(f"Saved guide graph to {guide_path}")
    print(f"Saved predictive graph to {predict_path}")


def _render_output_dir() -> Path:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _render_model_graph(output_dir: Path) -> Path:
    batch = _render_batch()
    model = FactorModelL12OnlineFiltering()
    output_path = output_dir / "model_l12.png"
    pyro.render_model(
        model,
        model_args=(batch,),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_guide_graph(output_dir: Path) -> Path:
    pyro.clear_param_store()
    batch = _render_batch()
    guide = FactorGuideL12OnlineFiltering(config=Level12GuideConfig())
    output_path = output_dir / "guide_l12.png"
    pyro.render_model(
        guide,
        model_args=(batch,),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_predict_graph(output_dir: Path) -> Path:
    graph = graphviz.Digraph("predict_l12", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")

    graph.node("state", "state\nfiltering_state\npredictive scale summaries")
    graph.node("batch", "prediction batch\nX_asset[t]\nX_global[t]")
    graph.node("init", "sample h_t\nfrom FilteringState")
    graph.node(
        "ar1",
        "class AR(1) transition\nsample h_{t+1,c}\npropagate P_{t+1|t,c}",
    )
    graph.node(
        "scale",
        "sample v[t+1,c]\ncompute u[t+1,a]\ngather class-level volatility",
    )
    graph.node("mean", "build mu[t+1]\nalpha + X_asset w + X_global beta")
    graph.node("obs", "LowRankMVN\nsample y[t+1]")
    graph.node("out", "outputs\nsamples / mean / covariance")

    graph.edge("state", "init")
    graph.edge("batch", "mean")
    graph.edge("init", "ar1")
    graph.edge("state", "scale")
    graph.edge("ar1", "scale")
    graph.edge("ar1", "obs")
    graph.edge("scale", "obs")
    graph.edge("mean", "obs")
    graph.edge("obs", "out")

    output_path = output_dir / "predict_l12.png"
    graph.render(outfile=str(output_path), cleanup=True)
    return output_path


def _render_batch() -> ModelBatch:
    return ModelBatch(
        X_asset=torch.zeros((2, 2, 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, 2), dtype=torch.float32),
        asset_names=("AUD.CAD", "IBUS500"),
        filtering_state=FilteringState(
            h_loc=torch.zeros(3, dtype=torch.float32),
            h_scale=torch.full((3,), 0.25, dtype=torch.float32),
            steps_seen=52,
        ),
    )


if __name__ == "__main__":
    main()
