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
        "Render model/guide/predict graphs for hybrid_measurement versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v9_l1": {
            "version": "v9_l1",
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
                "name": "predict_v9_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v9 hybrid-measurement family"),
                    ("latent", "latent primitive states\nus_broad / us_style / euro_core /\niberia / uk_ch"),
                    ("anchor", "soft anchored measurement matrix\nH = H0 + Delta"),
                    ("composite", "composite rows\nIBUS500 / IBEU50 tight residuals"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "latent"),
                    ("family", "anchor"),
                    ("anchor", "composite"),
                    ("latent", "out"),
                    ("composite", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v9_l1",
                "nodes": (
                    ("study", "derived from index_measurement_study"),
                    ("family", "v9 hybrid-measurement family"),
                    ("thesis", "some indices are composites,\nnot primitive peers"),
                    ("goal", "soft measurement structure,\nnot hard basis swap"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("study", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l1.model",
            "model_builder_attr": "build_hybrid_measurement_model_v9_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l1.guide",
            "guide_builder_attr": "build_hybrid_measurement_guide_v9_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l1.model",
                "V9L1ModelPriors",
                ("base",),
            ),
        },
        "v9_l2": {
            "version": "v9_l2",
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
                "name": "predict_v9_l2",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v9_l2 contrast-state branch"),
                    ("latent", "contrast states only\nus_style / euro_periphery /\nuk_ch_vs_euro"),
                    ("orth", "orthogonalized measurement block\nzero broad-level competition"),
                    ("anchor", "anchored H with strong shrinkage\ncomposite rows nearly fixed"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "latent"),
                    ("family", "orth"),
                    ("orth", "anchor"),
                    ("latent", "out"),
                    ("anchor", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v9_l2",
                "nodes": (
                    ("study", "derived from hybrid_measurement_postmortem"),
                    ("family", "v9_l2 contrast-state hybrid branch"),
                    ("thesis", "shared global channel carries level;\nlocal states carry only contrasts"),
                    ("goal", "remove measurement-state over-dominance\nwithout hard basis replacement"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("study", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.model",
            "model_builder_attr": "build_hybrid_measurement_model_v9_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.guide",
            "guide_builder_attr": "build_hybrid_measurement_guide_v9_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.model",
                "V9L2ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
