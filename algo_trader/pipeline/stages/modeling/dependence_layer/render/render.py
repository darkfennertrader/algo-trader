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
        "Render model/guide/predict graphs for dependence_layer versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v4_l1": {
            "version": "v4_l1",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v4_l1",
                "nodes": (
                    ("baseline", "baseline\ncorrected v3_l1 Gaussian unified model"),
                    ("family", "new v4 dependence-layer family"),
                    ("index", "index block only"),
                    ("mix", "shared Gamma scale mixture\nmean-one t overlay"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "mix"),
                    ("mix", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v4_l1",
                "nodes": (
                    ("family", "v4 dependence-layer family"),
                    ("baseline", "corrected v3_l1 Gaussian backbone"),
                    ("overlay", "index-only t-copula-style overlay\nshared Gamma scale mixture"),
                    ("goal", "fix joint index dependence\nwithout disturbing best marginals"),
                    ("criteria", "must beat v3_l1_bug_fixed\non calibration, baskets, residual dependence"),
                ),
                "edges": (
                    ("family", "baseline"),
                    ("baseline", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model",
            "model_builder_attr": "build_dependence_layer_model_v4_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide",
            "guide_builder_attr": "build_dependence_layer_guide_v4_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model",
                "V4L1ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model",
                    "_sample_index_t_copula_mix",
                    ("index_t_copula",),
                ),
            ),
        }
    }


if __name__ == "__main__":
    main()
