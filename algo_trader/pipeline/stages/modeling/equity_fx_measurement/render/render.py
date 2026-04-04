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
        "Render model/guide/predict graphs for equity_fx_measurement versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v12_l1": {
            "version": "v12_l1",
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
                "name": "predict_v12_l1",
                "nodes": (
                    ("baseline", "baseline\ntrusted v4_l1 raw-space backbone"),
                    ("family", "new v12 equity-fx measurement family"),
                    ("local", "local equity states\nus_style / euro_local /\nperiphery / uk_ch_local"),
                    ("fx", "FX translation states\neur / gbp_chf"),
                    ("measurement", "observed indices\ntight composite rows preserved"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                "edges": (
                    ("baseline", "family"),
                    ("family", "local"),
                    ("family", "fx"),
                    ("family", "measurement"),
                    ("local", "out"),
                    ("fx", "out"),
                    ("measurement", "out"),
                ),
            },
            "concept_graph": {
                "name": "concept_v12_l1",
                "nodes": (
                    ("source", "derived from v4-v11 index failures"),
                    ("family", "v12_l1 equity-fx measurement"),
                    ("thesis", "non-US indices are local equity\nplus FX translation measurements"),
                    ("goal", "preserve v4_l1 backbone,\nuse FX knowledge inside the index block"),
                    ("criteria", "must beat v4_l1 on calibration,\nbaskets, and residual dependence"),
                ),
                "edges": (
                    ("source", "family"),
                    ("family", "thesis"),
                    ("thesis", "goal"),
                    ("goal", "criteria"),
                ),
            },
            "model_module": "algo_trader.pipeline.stages.modeling.equity_fx_measurement.versions.v12_l1.model",
            "model_builder_attr": "build_equity_fx_measurement_model_v12_l1_online_filtering",
            "guide_module": "algo_trader.pipeline.stages.modeling.equity_fx_measurement.versions.v12_l1.guide",
            "guide_builder_attr": "build_equity_fx_measurement_guide_v12_l1_online_filtering",
            "split": build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.equity_fx_measurement.versions.v12_l1.model",
                "V12L1ModelPriors",
                ("base",),
            ),
        }
    }


if __name__ == "__main__":
    main()
