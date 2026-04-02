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
        "Render model/guide/predict graphs for residual_copula versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v5_l1": {
            "version": "v5_l1",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v5_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v5 residual-copula family"),
                    ("index", "index residual dependence only"),
                    ("calm", "calm broad scale"),
                    ("stress", "stress broad + regional scales"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "calm"),
                    ("calm", "stress"),
                    ("stress", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v5_l1",
                "nodes": (
                    ("family", "v5 residual-copula family"),
                    ("study", "offline v4_l1 residual study"),
                    (
                        "overlay",
                        "conditional regional residual copula\ncalm broad + stress broad/US/Europe scales",
                    ),
                    ("goal", "capture stress dependence and\nregional tail asymmetry"),
                    (
                        "criteria",
                        "must beat v4_l1 on calibration,\nbaskets, and residual dependence",
                    ),
                ),
                "edges": (
                    ("study", "family"),
                    ("family", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.model",
            "model_builder_attr": "build_residual_copula_model_v5_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.guide",
            "guide_builder_attr": "build_residual_copula_guide_v5_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.model",
                "V5L1ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l1.model",
                    "_sample_index_t_copula_sites",
                    ("index_t_copula",),
                ),
            ),
        },
        "v5_l2": {
            "version": "v5_l2",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v5_l2",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "v5 residual-copula family"),
                    ("index", "index residual dependence only"),
                    ("broad", "calm broad + stress broad scales"),
                    ("asym", "US / Europe\nupper vs lower tail scales"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "broad"),
                    ("broad", "asym"),
                    ("asym", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v5_l2",
                "nodes": (
                    ("family", "v5 residual-copula family"),
                    ("study", "offline v4_l1 residual study"),
                    (
                        "overlay",
                        "asymmetric regional residual copula\nbroad + US/Europe upper/lower tail scales",
                    ),
                    ("goal", "capture regional tail asymmetry\nwithout changing trusted marginals"),
                    (
                        "criteria",
                        "must beat v4_l1 on calibration,\nbaskets, and residual dependence",
                    ),
                ),
                "edges": (
                    ("study", "family"),
                    ("family", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.model",
            "model_builder_attr": "build_residual_copula_model_v5_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.guide",
            "guide_builder_attr": "build_residual_copula_guide_v5_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.model",
                "V5L2ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.residual_copula.versions.v5_l2.model",
                    "_sample_index_t_copula_sites",
                    ("index_t_copula",),
                ),
            ),
        },
    }


if __name__ == "__main__":
    main()
