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
        "Render model/guide/predict graphs for basket_consistency versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v13_l1": {
            "version": "v13_l1",
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
                "name": "predict_v13_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 raw-space backbone"),
                    ("family", "new v13 basket-consistency family"),
                    ("raw", "raw index space\nkept in main likelihood"),
                    ("basket", "auxiliary basket space\n4 decision baskets only"),
                    ("whiten", "training-split basket whitening\nMAD + shrunk covariance"),
                    ("aux", "weak diagonal Student-t\nconsistency regularizer"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "basket"),
                    ("basket", "whiten"),
                    ("whiten", "aux"),
                    ("raw", "out"),
                    ("aux", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v13_l1",
                "nodes": (
                    ("source", "derived from repeated basket misspecification"),
                    ("family", "v13_l1 basket consistency"),
                    ("thesis", "raw-space v4_l1 is broadly right,\nbut decision-basket space needs weak regularization"),
                    ("goal", "target only us/europe/spread/equal-weight baskets\nwithout rebuilding the index block"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model",
            "model_builder_attr": "build_basket_consistency_model_v13_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide",
            "guide_builder_attr": "build_basket_consistency_guide_v13_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model",
                "V13L1ModelPriors",
                ("base",),
            ),
        },
        "v13_l2": {
            "version": "v13_l2",
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
                "name": "predict_v13_l2",
                "nodes": (
                    ("baseline", "baseline\ntrusted v13_l1 raw-space backbone"),
                    ("family", "narrow v13_l2 follow-up"),
                    ("basket", "same 4 whitened decision baskets"),
                    ("split", "split auxiliary likelihood\nlevel baskets + spread basket"),
                    ("weight", "upweight spread basket\nrelative to level baskets"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "basket"),
                    ("basket", "split"),
                    ("split", "weight"),
                    ("weight", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v13_l2",
                "nodes": (
                    ("source", "posterior-signal slice diagnostics"),
                    ("family", "v13_l2 basket consistency"),
                    ("thesis", "commodities retain signal,\nindices remain the drag"),
                    ("goal", "de-emphasize level baskets and\nstress us-minus-europe spread consistency"),
                    ("criteria", "improve index slice signal\nwithout breaking the raw-space backbone"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l2.model",
            "model_builder_attr": "build_basket_consistency_model_v13_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l2.guide",
            "guide_builder_attr": "build_basket_consistency_guide_v13_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l2.model",
                "V13L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
