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
        "Render model/guide/predict graphs for pair_state_conditioned_curated_pair versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v18_l1": {
            "version": "v18_l1",
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
                "name": "predict_v18_l1",
                "nodes": (
                    ("baseline", "shared unified latent backbone"),
                    ("family", "v18_l1 pair_state_conditioned_curated_pair"),
                    ("raw", "head 1\nraw returns for all assets"),
                    ("pair", "head 2\nIBCH20 minus IBDE40"),
                    ("state", "lagged ex-ante state\nrange weeks only"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "pair"),
                    ("pair", "state"),
                    ("raw", "out"),
                    ("state", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v18_l1",
                "nodes": (
                    ("source", "pair-state study found\nIBCH20 minus IBDE40 in range\nas the cleanest ex-ante object"),
                    ("family", "v18_l1 pair_state_conditioned_curated_pair"),
                    ("thesis", "supervise one curated pair only\nand only on lagged range weeks"),
                    ("goal", "stop diluting the signal with\nother pairs and other states"),
                    ("criteria", "improve reduced-universe pair signal\nwithout portfolio metrics"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.pair_state_conditioned_curated_pair.versions.v18_l1.model",
            "model_builder_attr": "build_pair_state_conditioned_curated_pair_model_v18_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.pair_state_conditioned_curated_pair.versions.v18_l1.guide",
            "guide_builder_attr": "build_pair_state_conditioned_curated_pair_guide_v18_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.pair_state_conditioned_curated_pair.versions.v18_l1.model",
                "V18L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
