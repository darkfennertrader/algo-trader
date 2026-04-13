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
        "Render model/guide/predict graphs for index_relative_measurement versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v14_l1": {
            "version": "v14_l1",
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
                "name": "predict_v14_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 raw-space backbone"),
                    ("family", "new v14_l1 index-relative-measurement family"),
                    ("nonindex", "raw non-index observations\nFX + commodities"),
                    ("index", "full-rank index-relative basis"),
                    ("groups", "level / relative / residual\nmeasurement groups"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "nonindex"),
                    ("family", "index"),
                    ("index", "groups"),
                    ("nonindex", "out"),
                    ("groups", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v14_l1",
                "nodes": (
                    ("source", "Family 13 exhausted on 11 years"),
                    ("family", "v14_l1 index_relative_measurement"),
                    ("thesis", "index weakness is a measurement-geometry problem,\nnot only a missing weak regularizer"),
                    ("goal", "keep the unified backbone,\nbut observe the index block in relative coordinates"),
                    ("criteria", "improve index-slice posterior signal\nwithout using trading metrics for promotion"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.model",
            "model_builder_attr": "build_index_relative_measurement_model_v14_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.guide",
            "guide_builder_attr": "build_index_relative_measurement_guide_v14_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.model",
                "V14L1ModelPriors",
                ("base",),
            ),
        }
        ,
        "v14_l2": {
            "version": "v14_l2",
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
                "name": "predict_v14_l2",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 raw-space backbone"),
                    ("family", "v14_l2 per-index relative measurement"),
                    ("nonindex", "raw non-index observations\nFX + commodities"),
                    ("index", "level + per-index relative basis"),
                    ("groups", "level / per-index relative / residual\ngroups"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "nonindex"),
                    ("family", "index"),
                    ("index", "groups"),
                    ("nonindex", "out"),
                    ("groups", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v14_l2",
                "nodes": (
                    ("source", "v14_l1 improved index calibration\nbut not monetization"),
                    ("family", "v14_l2 index_relative_measurement"),
                    ("thesis", "supervise the actual per-index relative problem,\nnot only coarse regional relative coordinates"),
                    ("goal", "improve top-k spread / hit rate\nwithin the index slice"),
                    ("criteria", "better index-slice posterior signal\nwithout using trading metrics for promotion"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l2.model",
            "model_builder_attr": "build_index_relative_measurement_model_v14_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l2.guide",
            "guide_builder_attr": "build_index_relative_measurement_guide_v14_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l2.model",
                "V14L2ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
