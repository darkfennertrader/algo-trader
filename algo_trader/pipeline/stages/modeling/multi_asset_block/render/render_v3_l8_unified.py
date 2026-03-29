from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.pipeline.stages.modeling.protocols import ModelBatch
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    FilteringState,
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l8_unified import (
    MultiAssetBlockGuideV3L8UnifiedOnlineFiltering,
    _build_guide_config,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l8_unified import (
    V3L8UnifiedModelPriors,
    _build_context,
    _sample_regime_path,
    _sample_regime_scales,
    _sample_structural_sites,
    build_multi_asset_block_model_v3_l8_unified_online_filtering,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.v3_l8_defaults import (
    guide_default_params_v3_l8,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _render_model_graph(output_dir)
    structural_path = _render_structural_graph(output_dir)
    regime_path = _render_regime_graph(output_dir)
    guide_path = _render_guide_graph(output_dir)
    predict_path = _render_predict_graph(output_dir)
    print(f"Saved model graph to {model_path}")
    print(f"Saved structural graph to {structural_path}")
    print(f"Saved regime graph to {regime_path}")
    print(f"Saved guide graph to {guide_path}")
    print(f"Saved predictive graph to {predict_path}")


def _render_model_graph(output_dir: Path) -> Path:
    output_path = output_dir / "model_v3_l8_unified.png"
    pyro.render_model(
        build_multi_asset_block_model_v3_l8_unified_online_filtering({}),
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_structural_graph(output_dir: Path) -> Path:
    output_path = output_dir / "model_v3_l8_unified_structural.png"
    pyro.render_model(
        _structural_render_model,
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_regime_graph(output_dir: Path) -> Path:
    output_path = output_dir / "model_v3_l8_unified_regime.png"
    pyro.render_model(
        _regime_render_model,
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_guide_graph(output_dir: Path) -> Path:
    pyro.clear_param_store()
    output_path = output_dir / "guide_v3_l8_unified.png"
    pyro.render_model(
        MultiAssetBlockGuideV3L8UnifiedOnlineFiltering(
            config=_build_guide_config(guide_default_params_v3_l8())
        ),
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_predict_graph(output_dir: Path) -> Path:
    graph = graphviz.Digraph("predict_v3_l8_unified", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")
    graph.node("state", "state\nbroad + regional state")
    graph.node("batch", "prediction batch\nmixed universe")
    graph.node("classify", "classify assets\nFX / index / commodity")
    graph.node("carrier", "fixed regional carrier\nUS minus Europe")
    graph.node("residual", "small static residual factor\nindex-only learned nugget")
    graph.node("ar1", "state rollout\nbroad AR(1) states")
    graph.node("tail", "regional shocks\nheavy-tailed innovation")
    graph.node(
        "cov",
        "covariance blocks\nglobal + FX + broad index +\nregional spread + small residual index factor + commodity",
    )
    graph.node("mean", "mean path\nalpha + X_asset w + X_global beta")
    graph.node("obs", "LowRankMVN over full universe")
    graph.node("out", "outputs\nsamples / mean / covariance")
    graph.edge("batch", "classify")
    graph.edge("classify", "carrier")
    graph.edge("classify", "residual")
    graph.edge("state", "ar1")
    graph.edge("state", "tail")
    graph.edge("batch", "mean")
    graph.edge("classify", "cov")
    graph.edge("carrier", "cov")
    graph.edge("residual", "cov")
    graph.edge("ar1", "cov")
    graph.edge("tail", "cov")
    graph.edge("mean", "obs")
    graph.edge("cov", "obs")
    graph.edge("obs", "out")
    output_path = output_dir / "predict_v3_l8_unified.png"
    graph.render(outfile=str(output_path), cleanup=True)
    return output_path


def _structural_render_model(batch: ModelBatch) -> None:
    context = _build_context(
        build_v3_l1_unified_runtime_batch(batch), V3L8UnifiedModelPriors()
    )
    _sample_structural_sites(context)


def _regime_render_model(batch: ModelBatch) -> None:
    context = _build_context(
        build_v3_l1_unified_runtime_batch(batch), V3L8UnifiedModelPriors()
    )
    regime_scales = _sample_regime_scales(context)
    _sample_regime_path(context, regime_scales)


def _render_batch() -> ModelBatch:
    return ModelBatch(
        X_asset=torch.zeros((2, 4, 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, 4), dtype=torch.float32),
        asset_names=("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
        filtering_state=FilteringState(
            h_loc=torch.zeros(5, dtype=torch.float32),
            h_scale=torch.full((5,), 0.15, dtype=torch.float32),
            steps_seen=26,
        ),
    )


if __name__ == "__main__":
    main()
