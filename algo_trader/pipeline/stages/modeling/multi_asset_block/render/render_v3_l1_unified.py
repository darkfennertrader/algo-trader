from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    FilteringState,
    MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    V3L1UnifiedGuideConfig,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified import (
    MultiAssetBlockModelV3L1UnifiedOnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch


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
    output_path = output_dir / "model_v3_l1_unified.png"
    pyro.render_model(
        MultiAssetBlockModelV3L1UnifiedOnlineFiltering(),
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_guide_graph(output_dir: Path) -> Path:
    pyro.clear_param_store()
    output_path = output_dir / "guide_v3_l1_unified.png"
    pyro.render_model(
        MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(
            config=V3L1UnifiedGuideConfig()
        ),
        model_args=(_render_batch(),),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


def _render_predict_graph(output_dir: Path) -> Path:
    graph = graphviz.Digraph("predict_v3_l1_unified", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")
    graph.node("state", "state\nfour-state filtering_state\nFX broad/cross\nindex\ncommodity")
    graph.node("batch", "prediction batch\nmixed universe\nX_asset[t]\nX_global[t]")
    graph.node("classify", "classify assets\nFX / index / commodity")
    graph.node("ar1", "state rollout\nAR(1) per block")
    graph.node("mean", "mean path\nalpha + X_asset w + X_global beta")
    graph.node("cov", "covariance blocks\nstatic global + dynamic\nFX/index/commodity")
    graph.node("obs", "LowRankMVN over full universe")
    graph.node("out", "outputs\nsamples / mean / covariance")
    graph.edge("batch", "classify")
    graph.edge("state", "ar1")
    graph.edge("batch", "mean")
    graph.edge("classify", "cov")
    graph.edge("ar1", "cov")
    graph.edge("mean", "obs")
    graph.edge("cov", "obs")
    graph.edge("obs", "out")
    output_path = output_dir / "predict_v3_l1_unified.png"
    graph.render(outfile=str(output_path), cleanup=True)
    return output_path


def _render_batch() -> ModelBatch:
    return ModelBatch(
        X_asset=torch.zeros((2, 3, 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, 3), dtype=torch.float32),
        asset_names=("EUR.USD", "IBUS500", "XAUUSD"),
        filtering_state=FilteringState(
            h_loc=torch.zeros(4, dtype=torch.float32),
            h_scale=torch.full((4,), 0.15, dtype=torch.float32),
            steps_seen=26,
        ),
    )


if __name__ == "__main__":
    main()
