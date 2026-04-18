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
        "Render model/guide/predict graphs for curated_pair_index_relative versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v17_l1": {
            "version": "v17_l1",
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
                "name": "predict_v17_l1",
                "nodes": (
                    ("baseline", "shared unified latent backbone"),
                    ("family", "v17_l1 curated_pair_index_relative"),
                    ("raw", "head 1\nraw returns for all assets"),
                    ("pairs", "head 2\ncurated reduced-universe pairs"),
                    ("groups", "curated pair / residual\nauxiliary groups"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "raw"),
                    ("family", "pairs"),
                    ("pairs", "groups"),
                    ("raw", "out"),
                    ("groups", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v17_l1",
                "nodes": (
                    ("source", "curated pair stability study found\na tiny stable pair subset"),
                    ("family", "v17_l1 curated_pair_index_relative"),
                    ("thesis", "supervise only the most stable\nweekly reduced-universe pair objects"),
                    ("goal", "avoid broad block dilution and\navoid all-pairs noise"),
                    ("criteria", "improve reduced-universe index signal\nwithout portfolio metrics"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.curated_pair_index_relative.versions.v17_l1.model",
            "model_builder_attr": "build_curated_pair_index_relative_model_v17_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.curated_pair_index_relative.versions.v17_l1.guide",
            "guide_builder_attr": "build_curated_pair_index_relative_guide_v17_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.curated_pair_index_relative.versions.v17_l1.model",
                "V17L1ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
