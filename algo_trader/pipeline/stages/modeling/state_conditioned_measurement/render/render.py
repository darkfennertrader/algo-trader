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
        "Render model/guide/predict graphs for state_conditioned_measurement versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v10_l1": {
            "version": "v10_l1",
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
                "name": "predict_v10_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "new v10 state-conditioned measurement family"),
                    ("latent", "contrast states\nus_style / euro_periphery /\nuk_ch_vs_euro"),
                    ("measurement", "soft composite rows\nIBUS500 / IBEU50"),
                    ("gate", "observed-state gate\nfrom global + index feature magnitude"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "latent"),
                    ("family", "measurement"),
                    ("family", "gate"),
                    ("latent", "out"),
                    ("measurement", "out"),
                    ("gate", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v10_l1",
                "nodes": (
                    ("source", "derived from Family 7 + Family 9 lessons"),
                    ("family", "v10_l1 state-conditioned measurement"),
                    ("thesis", "contrast-state measurement structure\nshould adapt with observed state"),
                    ("goal", "preserve v4_l1 backbone,\nchange only index measurement reliability"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.model",
            "model_builder_attr": "build_state_conditioned_measurement_model_v10_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.guide",
            "guide_builder_attr": "build_state_conditioned_measurement_guide_v10_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.model",
                "V10L1ModelPriors",
                ("base",),
            ),
        },
        "v10_l2": {
            "version": "v10_l2",
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
                "name": "predict_v10_l2",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 / corrected v3_l1 backbone"),
                    ("family", "narrow v10_l2 gate-repair branch"),
                    ("latent", "contrast states\nsame v10_l1 geometry"),
                    ("measurement", "soft composite rows\nmild residual modulation only"),
                    ("gate", "recentered/weakened gate\ncontrast gating heavily shrunk"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "latent"),
                    ("family", "measurement"),
                    ("family", "gate"),
                    ("latent", "out"),
                    ("measurement", "out"),
                    ("gate", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v10_l2",
                "nodes": (
                    ("source", "derived from v10_l1 post-mortem"),
                    ("family", "v10_l2 narrow state-conditioned measurement repair"),
                    ("thesis", "same family, but weaker/recentered gate\nand minimal contrast modulation"),
                    ("goal", "preserve v10_l1 structure,\nfix the over-tightening lever"),
                    ("criteria", "must recover calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l2.model",
            "model_builder_attr": "build_state_conditioned_measurement_model_v10_l2_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l2.guide",
            "guide_builder_attr": "build_state_conditioned_measurement_guide_v10_l2_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l2.model",
                "V10L2ModelPriors",
                ("base",),
            ),
        },
    }


if __name__ == "__main__":
    main()
