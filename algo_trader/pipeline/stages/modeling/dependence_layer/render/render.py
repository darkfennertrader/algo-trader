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
        ,
        "v4_l2": {
            "version": "v4_l2",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v4_l2",
                "nodes": (
                    ("baseline", "baseline\ncorrected v3_l1 Gaussian unified model"),
                    ("family", "v4 dependence-layer family"),
                    ("index", "index block only"),
                    ("broad", "broad shared scale"),
                    ("regional", "regional shared scales\nUS + Europe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "broad"),
                    ("broad", "regional"),
                    ("regional", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v4_l2",
                "nodes": (
                    ("family", "v4 dependence-layer family"),
                    ("baseline", "corrected v3_l1 Gaussian backbone"),
                    ("overlay", "index-only regional t-copula overlay\nbroad + US + Europe shared scales"),
                    ("goal", "reduce remaining index over-width\nwithout disturbing trusted marginals"),
                    ("criteria", "must improve indices baskets\nwhile preserving v4_l1 calibration gains"),
                ),
                "edges": (
                    ("family", "baseline"),
                    ("baseline", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.model",
            "model_builder_attr": "build_dependence_layer_model_v4_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.guide",
            "guide_builder_attr": "build_dependence_layer_guide_v4_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.model",
                "V4L2ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l2.model",
                    "_sample_index_t_copula_mix",
                    ("index_t_copula",),
                ),
            ),
        },
        "v4_l3": {
            "version": "v4_l3",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v4_l3",
                "nodes": (
                    ("baseline", "baseline\ncorrected v3_l1 Gaussian unified model"),
                    ("family", "v4 dependence-layer family"),
                    ("index", "index block only"),
                    ("broad", "broad shared scale"),
                    ("regional", "regional shared scales\nUS + Europe"),
                    ("spread", "shrunk US-vs-Europe\ndifferential scale"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "broad"),
                    ("broad", "regional"),
                    ("regional", "spread"),
                    ("spread", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v4_l3",
                "nodes": (
                    ("family", "v4 dependence-layer family"),
                    ("baseline", "corrected v3_l1 Gaussian backbone"),
                    ("overlay", "regional t-copula overlay\nbroad + US + Europe +\nshrunk differential spread scale"),
                    ("goal", "capture regional tail asymmetry\nwithout disturbing trusted marginals"),
                    ("criteria", "must reduce US-side spread width\nwithout giving back Europe/equal-weight behavior"),
                ),
                "edges": (
                    ("family", "baseline"),
                    ("baseline", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.model",
            "model_builder_attr": "build_dependence_layer_model_v4_l3_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.guide",
            "guide_builder_attr": "build_dependence_layer_guide_v4_l3_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.model",
                "V4L3ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l3.model",
                    "_sample_index_t_copula_mix",
                    ("index_t_copula",),
                ),
            ),
        },
    }


if __name__ == "__main__":
    main()
