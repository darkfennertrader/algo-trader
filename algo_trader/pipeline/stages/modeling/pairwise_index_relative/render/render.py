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
        "Render model/guide/predict graphs for pairwise_index_relative versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v16_l1": {
            "version": "v16_l1",
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
                "name": "predict_v16_l1",
                "nodes": (
                    ("baseline", "shared unified latent backbone"),
                    ("family", "v16_l1 pairwise_index_relative"),
                    ("raw", "head 1\nraw returns for all assets"),
                    ("pairwise", "head 2\ncurated pairwise index spreads"),
                    ("groups", "pairwise / residual\nauxiliary groups"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "pairwise"),
                    ("pairwise", "groups"),
                    ("raw", "out"),
                    ("groups", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v16_l1",
                "nodes": (
                    ("source", "reduced-universe studies showed\npairwise index ordering remains weak"),
                    ("family", "v16_l1 pairwise_index_relative"),
                    ("thesis", "explicitly supervise which index\nshould beat which other index weekly"),
                    ("goal", "keep the raw-return posterior\nbut add a closer monetization object"),
                    ("criteria", "improve index-slice posterior signal\non the cleaned reduced universe"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.pairwise_index_relative.versions.v16_l1.model",
            "model_builder_attr": "build_pairwise_index_relative_model_v16_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.pairwise_index_relative.versions.v16_l1.guide",
            "guide_builder_attr": "build_pairwise_index_relative_guide_v16_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.pairwise_index_relative.versions.v16_l1.model",
                "V16L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
