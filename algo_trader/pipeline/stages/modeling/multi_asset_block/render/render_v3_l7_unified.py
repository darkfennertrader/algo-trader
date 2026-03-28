from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.pipeline.stages.modeling.protocols import ModelBatch
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    FilteringState,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l7_unified import (
    MultiAssetBlockGuideV3L7UnifiedOnlineFiltering,
    _build_guide_config,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l7_unified import (
    build_multi_asset_block_model_v3_l7_unified_online_filtering,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.v3_l7_defaults import (
    guide_default_params_v3_l7,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _render_model_graph(output_dir)
    guide_path = _render_guide_graph(output_dir)
    predict_path = _render_predict_graph(output_dir)
    print(f"Saved model graph to {model_path}")
    print(f"Saved guide graph to {guide_path}")
    print(f"Saved predictive graph to {predict_path}")


def _render_model_graph(output_dir: Path) -> Path:
    output_path = output_dir / "model_v3_l7_unified.png"
    pyro.render_model(
        build_multi_asset_block_model_v3_l7_unified_online_filtering({}),
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_guide_graph(output_dir: Path) -> Path:
    pyro.clear_param_store()
    output_path = output_dir / "guide_v3_l7_unified.png"
    pyro.render_model(
        MultiAssetBlockGuideV3L7UnifiedOnlineFiltering(
            config=_build_guide_config(guide_default_params_v3_l7())
        ),
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_predict_graph(output_dir: Path) -> Path:
    graph = graphviz.Digraph("predict_v3_l7_unified", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")
    graph.node("state", "state\nbroad + two named spread states")
    graph.node("batch", "prediction batch\nmixed universe")
    graph.node("classify", "classify assets\nFX / index / commodity")
    graph.node("carrier1", "fixed spread carrier\nUS minus Europe")
    graph.node("carrier2", "fixed spread carrier\ncontinental Europe vs UK/CH")
    graph.node("ar1", "state rollout\nbroad AR(1) states")
    graph.node("tail", "spread shocks\ntwo heavy-tailed innovations")
    graph.node(
        "cov",
        "covariance blocks\nglobal + FX + broad index +\nUS-Europe spread + Europe-vs-UK/CH spread + commodity",
    )
    graph.node("mean", "mean path\nalpha + X_asset w + X_global beta")
    graph.node("obs", "LowRankMVN over full universe")
    graph.node("out", "outputs\nsamples / mean / covariance")
    graph.edge("batch", "classify")
    graph.edge("classify", "carrier1")
    graph.edge("classify", "carrier2")
    graph.edge("state", "ar1")
    graph.edge("state", "tail")
    graph.edge("batch", "mean")
    graph.edge("classify", "cov")
    graph.edge("carrier1", "cov")
    graph.edge("carrier2", "cov")
    graph.edge("ar1", "cov")
    graph.edge("tail", "cov")
    graph.edge("mean", "obs")
    graph.edge("cov", "obs")
    graph.edge("obs", "out")
    output_path = output_dir / "predict_v3_l7_unified.png"
    graph.render(outfile=str(output_path), cleanup=True)
    return output_path


def _render_batch() -> ModelBatch:
    return ModelBatch(
        X_asset=torch.zeros((2, 4, 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, 4), dtype=torch.float32),
        asset_names=("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
        filtering_state=FilteringState(
            h_loc=torch.zeros(6, dtype=torch.float32),
            h_scale=torch.full((6,), 0.15, dtype=torch.float32),
            steps_seen=26,
        ),
    )


if __name__ == "__main__":
    main()
