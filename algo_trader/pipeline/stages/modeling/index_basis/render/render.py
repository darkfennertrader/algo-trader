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
        "Render model/guide/predict graphs for index_basis versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v8_l1": {
            "version": "v8_l1",
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
                "name": "predict_v8_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v8 index-basis family"),
                    ("basis", "index basis\n1 global + 4 spread coordinates"),
                    ("global", "global_level\nstandalone Student-t"),
                    ("spread", "spread block\nsmall joint 4D structure"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "basis"),
                    ("basis", "global"),
                    ("basis", "spread"),
                    ("global", "out"),
                    ("spread", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v8_l1",
                "nodes": (
                    ("study", "derived from index_representation_study"),
                    ("family", "v8 index-basis family"),
                    ("representation", "representation-first index block"),
                    ("goal", "test whether transformed basis\nbeats more dependence-only tweaks"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("study", "family"),
                    ("family", "representation"),
                    ("representation", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.model",
            "model_builder_attr": "build_index_basis_model_v8_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.guide",
            "guide_builder_attr": "build_index_basis_guide_v8_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.index_basis.versions.v8_l1.model",
                "V8L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
