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
        "Render model/guide/predict graphs for observable_state_dependence versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v7_l1": {
            "version": "v7_l1",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v7_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v7 observable-state dependence family"),
                    ("index", "index dependence only"),
                    ("state", "observed state gate\nfrom X_global + index features"),
                    ("scale", "broad + regional\nfactor-row scaling"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "state"),
                    ("state", "scale"),
                    ("scale", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v7_l1",
                "nodes": (
                    ("family", "v7 observable-state dependence family"),
                    ("study", "post-v6 evaluation"),
                    ("overlay", "index-only observed-state adapter\ncondition dependence on observed features"),
                    ("goal", "test whether observable market state\nexplains remaining index misspecification"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("study", "family"),
                    ("family", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.model",
            "model_builder_attr": "build_observable_state_dependence_model_v7_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.guide",
            "guide_builder_attr": "build_observable_state_dependence_guide_v7_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.model",
                "V7L1ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.observable_state_dependence.versions.v7_l1.model",
                    "_sample_observable_state_coefficients",
                    ("observable_state_dependence",),
                ),
            ),
        },
    }


if __name__ == "__main__":
    main()
