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
        "Render model/guide/predict graphs for mixture_copula versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v6_l1": {
            "version": "v6_l1",
            "batch": {
                "asset_names": ("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"),
                "state_size": 4,
            },
            "predict_graph": {
                "name": "predict_v6_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v6 mixture-copula family"),
                    ("index", "index dependence only"),
                    ("gate", "soft stress gate\n|index state| driven"),
                    ("mix", "full cross-region\nmixture t-copula scales"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "index"),
                    ("index", "gate"),
                    ("gate", "mix"),
                    ("mix", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v6_l1",
                "nodes": (
                    ("family", "v6 mixture-copula family"),
                    ("study", "phase-2 fitted copula mismatch study"),
                    (
                        "overlay",
                        "soft state-mixture copula\nfull cross-region broad + US/Europe tilts",
                    ),
                    ("goal", "retain cross-region dependence\nwithout hard regime/block splits"),
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
            "model_module": "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.model",
            "model_builder_attr": "build_mixture_copula_model_v6_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.guide",
            "guide_builder_attr": "build_mixture_copula_guide_v6_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.model",
                "V6L1ModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.mixture_copula.versions.v6_l1.model",
                    "_sample_index_t_copula_sites",
                    ("index_t_copula",),
                ),
            ),
        },
    }


if __name__ == "__main__":
    main()
