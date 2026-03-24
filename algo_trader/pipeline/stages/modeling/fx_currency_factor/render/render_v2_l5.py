from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.infrastructure.data import symbol_directory
from algo_trader.pipeline.stages.modeling.fx_currency_factor.guide_v2_l5 import (
    FXCurrencyFactorGuideV2L5OnlineFiltering,
    FilteringState,
    V2L5GuideConfig,
)
from algo_trader.pipeline.stages.modeling.fx_currency_factor.model_v2_l5 import (
    FXCurrencyFactorModelV2L5OnlineFiltering,
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
    model = FXCurrencyFactorModelV2L5OnlineFiltering()
    output_path = output_dir / "model_v2_l5.png"
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
    guide = FXCurrencyFactorGuideV2L5OnlineFiltering(config=V2L5GuideConfig())
    output_path = output_dir / "guide_v2_l5.png"
    pyro.render_model(
        guide,
        model_args=(batch,),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_predict_graph(output_dir: Path) -> Path:
    graph = graphviz.Digraph("predict_v2_l5", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")

    graph.node("state", "state\nscalar FX filtering_state\npair summaries + currency centers")
    graph.node("batch", "prediction batch\nFX pairs\nX_asset[t]\nX_global[t]")
    graph.node("exposure", "build P[a,c]\nanchor one currency")
    graph.node("gain", "gain-style filter\nh_loc = prior + gain * innovation")
    graph.node("ar1", "shared FX AR(1)\nsample h_{t+1}\npropagate P_{t+1|t}")
    graph.node("center", "soft mean hierarchy\nalpha ~ P alpha_currency\nw ~ P theta_currency")
    graph.node("macro", "currency macro block\nm_currency = X_global Gamma")
    graph.node("cov", "currency covariance block\nB_pair = P B_currency")
    graph.node("idio", "hierarchical static nugget\nsigma0 + |P| sigma_currency + delta_pair")
    graph.node("scale", "shared heavy-tail shock\nu[t+1] = exp(h - 0.5P) * v")
    graph.node("mean", "pair mean\nalpha + X_asset w + P m_currency")
    graph.node("obs", "LowRankMVN over pairs\ncommon block scales with u\nidio block stays static")
    graph.node("out", "outputs\nsamples / mean / covariance")

    graph.edge("batch", "exposure")
    graph.edge("state", "gain")
    graph.edge("batch", "gain")
    graph.edge("gain", "ar1")
    graph.edge("exposure", "center")
    graph.edge("batch", "center")
    graph.edge("batch", "macro")
    graph.edge("exposure", "macro")
    graph.edge("exposure", "cov")
    graph.edge("exposure", "idio")
    graph.edge("ar1", "scale")
    graph.edge("center", "mean")
    graph.edge("macro", "mean")
    graph.edge("cov", "obs")
    graph.edge("idio", "obs")
    graph.edge("scale", "obs")
    graph.edge("mean", "obs")
    graph.edge("obs", "out")

    output_path = output_dir / "predict_v2_l5.png"
    graph.render(outfile=str(output_path), cleanup=True)
    return output_path


def _render_batch() -> ModelBatch:
    asset_names = _active_fx_render_names()
    return ModelBatch(
        X_asset=torch.zeros((2, len(asset_names), 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, len(asset_names)), dtype=torch.float32),
        asset_names=asset_names,
        filtering_state=FilteringState(
            h_loc=torch.tensor(0.0, dtype=torch.float32),
            h_scale=torch.tensor(0.25, dtype=torch.float32),
            steps_seen=52,
        ),
    )


def _active_fx_render_names() -> tuple[str, ...]:
    config_path = Path(__file__).resolve().parents[6] / "config" / "tickers.yml"
    config = HistoricalRequestConfig.load(config_path)
    active_fx = tuple(symbol_directory(ticker) for ticker in config.tickers)
    usd_pairs = tuple(name for name in active_fx if ".USD" in name or "USD." in name)
    selected = usd_pairs[:3]
    if len(selected) < 3:
        selected = active_fx[:3]
    if len(selected) < 2:
        raise ValueError("V2 L5 render requires at least two active FX tickers")
    return selected


if __name__ == "__main__":
    main()
