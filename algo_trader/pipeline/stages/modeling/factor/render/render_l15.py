from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.infrastructure.data import symbol_directory
from algo_trader.pipeline.stages.modeling.factor.guide_l15 import (
    FactorGuideL15OnlineFiltering,
    FilteringState,
    Level15GuideConfig,
)
from algo_trader.pipeline.stages.modeling.factor.model_l15 import (
    FactorModelL15OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch


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
    model = FactorModelL15OnlineFiltering()
    output_path = output_dir / "model_l15.png"
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
    guide = FactorGuideL15OnlineFiltering(config=Level15GuideConfig())
    output_path = output_dir / "guide_l15.png"
    pyro.render_model(
        guide,
        model_args=(batch,),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_predict_graph(output_dir: Path) -> Path:
    graph = graphviz.Digraph("predict_l15", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")

    graph.node("state", "state\nFX filtering_state\npredictive scale summaries")
    graph.node("batch", "prediction batch\nFX X_asset[t]\nX_global[t]")
    graph.node("gain", "gain-style FX filter\nh_loc = prior + gain * innovation")
    graph.node("ar1", "shared FX AR(1)\nsample h_{t+1}\npropagate P_{t+1|t}")
    graph.node("scale", "asset loadings lambda_h[a]\nshared v[t+1]\ncompute u[t+1,a]")
    graph.node("mean", "build mu[t+1]\nalpha + X_asset w + X_global beta")
    graph.node("obs", "LowRankMVN\nsample FX y[t+1]")
    graph.node("out", "outputs\nsamples / mean / covariance")

    graph.edge("state", "gain")
    graph.edge("batch", "gain")
    graph.edge("gain", "ar1")
    graph.edge("ar1", "scale")
    graph.edge("state", "scale")
    graph.edge("batch", "mean")
    graph.edge("scale", "obs")
    graph.edge("mean", "obs")
    graph.edge("obs", "out")

    output_path = output_dir / "predict_l15.png"
    graph.render(outfile=str(output_path), cleanup=True)
    return output_path


def _render_batch() -> ModelBatch:
    asset_names = _active_fx_render_names()
    return ModelBatch(
        X_asset=torch.zeros((2, 2, 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, 2), dtype=torch.float32),
        asset_names=asset_names,
        filtering_state=FilteringState(
            h_loc=torch.tensor(0.0, dtype=torch.float32),
            h_scale=torch.tensor(0.25, dtype=torch.float32),
            steps_seen=52,
        ),
    )


def _active_fx_render_names() -> tuple[str, str]:
    config_path = Path(__file__).resolve().parents[6] / "config" / "tickers.yml"
    config = HistoricalRequestConfig.load(config_path)
    active_fx = tuple(symbol_directory(ticker) for ticker in config.tickers[:2])
    if len(active_fx) != 2:
        raise ValueError("L15 render requires at least two active FX tickers")
    return active_fx


if __name__ == "__main__":
    main()
