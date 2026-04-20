from __future__ import annotations

from pathlib import Path

from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.render_support import (
    RenderSpec,
    build_v3_l1_based_split,
    run_family,
)


def main() -> None:
    run_family(
        "Render model/guide/predict graphs for hierarchical_index_experts versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v19_l1": {
            "version": "v19_l1",
            "batch": {
                "asset_names": (
                    "EUR.USD",
                    "IBUS30",
                    "IBUST100",
                    "IBDE40",
                    "IBES35",
                    "IBFR40",
                    "IBGB100",
                    "IBNL25",
                    "IBCH20",
                    "XAU.USD",
                ),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v19_l1",
                "nodes": (
                    ("baseline", "shared unified latent backbone"),
                    ("family", "v19_l1 hierarchical_index_experts"),
                    ("raw", "head 1\nraw returns for all assets"),
                    ("broad", "expert 1\nbroad reduced-index view"),
                    ("anchor", "expert 2\nIBCH20 minus IBDE40"),
                    ("residual", "expert 3\nresidual index cleanup"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "broad"),
                    ("family", "anchor"),
                    ("family", "residual"),
                    ("raw", "out"),
                    ("broad", "out"),
                    ("anchor", "out"),
                    ("residual", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v19_l1",
                "nodes": (
                    ("source", "broad index lines stayed weak,\nbut reduced-universe pair research\nfound one durable anchor pair"),
                    ("family", "v19_l1 hierarchical_index_experts"),
                    ("thesis", "keep a broad index head,\nbut bias it toward the durable anchor\ninstead of supervising all pairs equally"),
                    ("goal", "reuse the broad reduced-index view\nwithout diluting the cleanest pair signal"),
                    ("criteria", "improve reduced-universe index signal\nwithout collapsing to a one-pair model"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.hierarchical_index_experts.versions.v19_l1.model",
            "model_builder_attr": "build_hierarchical_index_experts_model_v19_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.hierarchical_index_experts.versions.v19_l1.guide",
            "guide_builder_attr": "build_hierarchical_index_experts_guide_v19_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.hierarchical_index_experts.versions.v19_l1.model",
                "V19L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
