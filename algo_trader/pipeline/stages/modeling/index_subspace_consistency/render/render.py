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
        "Render model/guide/predict graphs for index_subspace_consistency versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v11_l1": {
            "version": "v11_l1",
            "batch": {
                "asset_names": (
                    "EUR.USD",
                    "IBUS30",
                    "IBUS500",
                    "IBUST100",
                    "IBDE40",
                    "IBES35",
                    "IBEU50",
                    "IBFR40",
                    "IBGB100",
                    "IBNL25",
                    "IBCH20",
                    "XAU.USD",
                ),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v11_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 raw-space backbone"),
                    ("family", "new v11 dual-view family"),
                    ("raw", "raw index space\nkept in main likelihood"),
                    ("subspace", "auxiliary subspace\n1 global + 4 spread coordinates"),
                    ("global", "global_level\nStudent-t consistency site"),
                    ("spread", "spread block\nsmall shrunk joint Student-t block"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "subspace"),
                    ("subspace", "global"),
                    ("subspace", "spread"),
                    ("raw", "out"),
                    ("global", "out"),
                    ("spread", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v11_l1",
                "nodes": (
                    ("source", "derived from v4 + transformed-space studies"),
                    ("family", "v11_l1 index-subspace consistency"),
                    ("thesis", "raw-space index model is broadly right,\nbut spread subspace needs explicit regularization"),
                    ("goal", "keep v4_l1 raw-space realism\nand add spread-space discipline"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l1.model",
            "model_builder_attr": "build_index_subspace_consistency_model_v11_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l1.guide",
            "guide_builder_attr": "build_index_subspace_consistency_guide_v11_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l1.model",
                "V11L1ModelPriors",
                ("base",),
            ),
        },
        "v11_l2": {
            "version": "v11_l2",
            "batch": {
                "asset_names": (
                    "EUR.USD",
                    "IBUS30",
                    "IBUS500",
                    "IBUST100",
                    "IBDE40",
                    "IBES35",
                    "IBEU50",
                    "IBFR40",
                    "IBGB100",
                    "IBNL25",
                    "IBCH20",
                    "XAU.USD",
                ),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v11_l2",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 raw-space backbone"),
                    ("family", "new v11 spread-only follow-up"),
                    ("raw", "raw index space\nkept in main likelihood"),
                    ("subspace", "auxiliary spread subspace\n4 spread coordinates only"),
                    ("spread", "spread block\nweaker shrunk joint Student-t block"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "subspace"),
                    ("subspace", "spread"),
                    ("raw", "out"),
                    ("spread", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v11_l2",
                "nodes": (
                    ("source", "derived from v11_l1 result"),
                    ("family", "v11_l2 spread-only consistency"),
                    ("thesis", "keep raw-space model\nbut weaken dual-view supervision"),
                    ("goal", "drop global auxiliary term\nand regularize only spread subspace"),
                    ("criteria", "must preserve basket gains\nand improve calibration / residual dependence"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l2.model",
            "model_builder_attr": "build_index_subspace_consistency_model_v11_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l2.guide",
            "guide_builder_attr": "build_index_subspace_consistency_guide_v11_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.index_subspace_consistency.versions.v11_l2.model",
                "V11L2ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
