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
        "Render model/guide/predict graphs for multi_output_index_relative versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v15_l1": {
            "version": "v15_l1",
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
                "name": "predict_v15_l1",
                "nodes": (
                    ("baseline", "shared unified latent backbone"),
                    ("family", "v15_l1 multi_output_index_relative"),
                    ("raw", "head 1\nraw returns for all assets"),
                    ("relative", "head 2\nindex-relative coordinates"),
                    ("groups", "level / relative / residual\nauxiliary groups"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "relative"),
                    ("relative", "groups"),
                    ("raw", "out"),
                    ("groups", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v15_l1",
                "nodes": (
                    ("source", "Family 14 improved index ordering\nbut not enough monetization"),
                    ("family", "v15_l1 multi_output_index_relative"),
                    ("thesis", "one backbone should support two tasks:\nraw returns and index-relative discrimination"),
                    ("goal", "keep the trusted raw-return posterior\nwhile explicitly supervising the index-relative slice"),
                    ("criteria", "improve index-slice posterior signal\nwithout using trading metrics for promotion"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.multi_output_index_relative.versions.v15_l1.model",
            "model_builder_attr": "build_multi_output_index_relative_model_v15_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.multi_output_index_relative.versions.v15_l1.guide",
            "guide_builder_attr": "build_multi_output_index_relative_guide_v15_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.multi_output_index_relative.versions.v15_l1.model",
                "V15L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
