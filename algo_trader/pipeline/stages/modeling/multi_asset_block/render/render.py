from __future__ import annotations

from pathlib import Path

from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.render_support import (
    BatchSpec,
    GraphSpec,
    RenderSpec,
    SplitSpec,
    build_v3_l1_based_split,
    run_family,
)


def main() -> None:
    run_family(
        "Render model/guide/predict graphs for multi_asset_block versions.",
        _build_registry(),
        Path(__file__).resolve().parent / "versions",
        build_v3_l1_unified_runtime_batch,
    )


def _batch_spec(asset_names: tuple[str, ...], state_size: int) -> BatchSpec:
    return {"asset_names": asset_names, "state_size": state_size}


def _graph_spec(
    *,
    name: str,
    nodes: tuple[tuple[str, str], ...],
    edges: tuple[tuple[str, str], ...],
    rankdir: str = "LR",
) -> GraphSpec:
    return {"name": name, "nodes": nodes, "edges": edges, "rankdir": rankdir}


def _build_registry() -> dict[str, RenderSpec]:
    return {
        "v3_l1_unified": _simple_spec(
            version="v3_l1_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "XAUUSD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l1_unified",
                nodes=(
                    ("state", "state\nfour-state filtering_state\nFX broad/cross\nindex\ncommodity"),
                    ("batch", "prediction batch\nmixed universe\nX_asset[t]\nX_global[t]"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("ar1", "state rollout\nAR(1) per block"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("cov", "covariance blocks\nstatic global + dynamic\nFX/index/commodity"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("state", "ar1"),
                    ("batch", "mean"),
                    ("classify", "cov"),
                    ("ar1", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
        ),
        "v3_l2_unified": _simple_spec(
            version="v3_l2_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAUUSD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l2_unified",
                nodes=(
                    ("state", "state\nfour-state filtering_state"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("ar1", "state rollout\nAR(1) per block"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("cov", "covariance blocks\n2 global factors\n2 index factors\nFX/index/commodity dynamics"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("state", "ar1"),
                    ("batch", "mean"),
                    ("classify", "cov"),
                    ("ar1", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
        ),
        "v3_l3_unified": _simple_spec(
            version="v3_l3_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "XAUUSD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l3_unified",
                nodes=(
                    ("state", "state\nfour-state filtering_state"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("ar1", "state rollout\nAR(1) per dynamic block"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("static", "static index block\nM_index * B_index_static"),
                    ("cov", "covariance blocks\nglobal + FX + static index +\ndynamic index + commodity"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("state", "ar1"),
                    ("batch", "mean"),
                    ("classify", "static"),
                    ("classify", "cov"),
                    ("ar1", "cov"),
                    ("static", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
        ),
        "v3_l4_unified": _simple_spec(
            version="v3_l4_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "XAUUSD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l4_unified",
                nodes=(
                    ("state", "state\nfour-state filtering_state"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("groups", "infer index groups\nUS / DE / FR / EU / ..."),
                    ("ar1", "state rollout\nAR(1) per dynamic block"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("static", "static index group block\nM_group * diag(lambda_group)"),
                    ("cov", "covariance blocks\nglobal + FX + group index +\ndynamic index + commodity"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("classify", "groups"),
                    ("state", "ar1"),
                    ("batch", "mean"),
                    ("groups", "static"),
                    ("classify", "cov"),
                    ("ar1", "cov"),
                    ("static", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
        ),
        "v3_l5_unified": _simple_spec(
            version="v3_l5_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "XAUUSD"), 5),
            predict_graph=_graph_spec(
                name="predict_v3_l5_unified",
                nodes=(
                    ("state", "state\nbroad + group-aware filtering_state"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("groups", "infer index groups\ndeterministic symbol groups"),
                    ("ar1", "state rollout\nbroad states + group AR(1) vector"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("group", "dynamic group block\nM_group * diag(lambda_group * exp(0.5 g_t))"),
                    ("cov", "covariance blocks\nglobal + FX + dynamic group index +\ndynamic broad index + commodity"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("classify", "groups"),
                    ("state", "ar1"),
                    ("batch", "mean"),
                    ("groups", "group"),
                    ("classify", "cov"),
                    ("ar1", "group"),
                    ("ar1", "cov"),
                    ("group", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
        ),
        "v3_l6_unified": _spec_with_concept(
            version="v3_l6_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAUUSD"), 5),
            predict_graph=_graph_spec(
                name="predict_v3_l6_unified",
                nodes=(
                    ("state", "state\nbroad + spread filtering_state"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("spread", "fixed spread carrier\nUS minus Europe exposure"),
                    ("ar1", "state rollout\nbroad AR(1) states"),
                    ("tail", "spread shock\nheavy-tailed innovation"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("cov", "covariance blocks\nglobal + FX + broad index +\nUS-Europe spread + commodity"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("classify", "spread"),
                    ("state", "ar1"),
                    ("state", "tail"),
                    ("batch", "mean"),
                    ("classify", "cov"),
                    ("spread", "cov"),
                    ("ar1", "cov"),
                    ("tail", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
            concept_graph=_graph_spec(
                name="concept_v3_l6_unified",
                rankdir="TB",
                nodes=(
                    ("baseline", "v3_l1 baseline\nFX + broad index + commodity"),
                    ("carrier", "fixed US-Europe carrier\ncentered / normalized / orthogonalized"),
                    ("state", "dynamic spread state\ns_us_eu[t]"),
                    ("shock", "heavy-tailed shock\nStudentT innovation"),
                    ("index", "index covariance\nbroad mode + spread mode"),
                    ("diagnostics", "target diagnostics\nindices / full / baskets / residual dependence"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("carrier", "index"),
                    ("state", "index"),
                    ("shock", "state"),
                    ("index", "diagnostics"),
                ),
            ),
        ),
        "v3_l7_unified": _spec_with_concept(
            version="v3_l7_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 6),
            predict_graph=_graph_spec(
                name="predict_v3_l7_unified",
                nodes=(
                    ("state", "state\nbroad + two named spread states"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("carrier1", "fixed spread carrier\nUS minus Europe"),
                    ("carrier2", "fixed spread carrier\ncontinental Europe vs UK/CH"),
                    ("ar1", "state rollout\nbroad AR(1) states"),
                    ("tail", "spread shocks\ntwo heavy-tailed innovations"),
                    ("cov", "covariance blocks\nglobal + FX + broad index +\nUS-Europe spread + Europe-vs-UK/CH spread + commodity"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("classify", "carrier1"),
                    ("classify", "carrier2"),
                    ("state", "ar1"),
                    ("state", "tail"),
                    ("batch", "mean"),
                    ("classify", "cov"),
                    ("carrier1", "cov"),
                    ("carrier2", "cov"),
                    ("ar1", "cov"),
                    ("tail", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
            concept_graph=_graph_spec(
                name="concept_v3_l7_unified",
                rankdir="TB",
                nodes=(
                    ("baseline", "v3_l1 baseline\nFX + broad index + commodity"),
                    ("carrier1", "fixed carrier\nUS minus Europe"),
                    ("carrier2", "fixed carrier\ncontinental Europe vs UK/CH"),
                    ("state1", "dynamic spread state\ns_us_eu[t]"),
                    ("state2", "dynamic spread state\ns_eu_vs_uk_ch[t]"),
                    ("shock", "heavy-tailed shocks\nStudentT innovations"),
                    ("index", "index covariance\nbroad mode + US-Europe mode +\nEurope-vs-UK/CH mode"),
                    ("diagnostics", "target diagnostics\nindices / full / baskets / residual dependence"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("carrier1", "index"),
                    ("carrier2", "index"),
                    ("state1", "index"),
                    ("state2", "index"),
                    ("shock", "state1"),
                    ("shock", "state2"),
                    ("index", "diagnostics"),
                ),
            ),
        ),
        "v3_l8_unified": _spec_with_split_and_concept(
            version="v3_l8_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 5),
            predict_graph=_graph_spec(
                name="predict_v3_l8_unified",
                nodes=(
                    ("state", "state\nbroad + regional state"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("carrier", "fixed regional carrier\nUS minus Europe"),
                    ("residual", "small static residual factor\nindex-only learned nugget"),
                    ("ar1", "state rollout\nbroad AR(1) states"),
                    ("tail", "regional shocks\nheavy-tailed innovation"),
                    ("cov", "covariance blocks\nglobal + FX + broad index +\nregional spread + small residual index factor + commodity"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("classify", "carrier"),
                    ("classify", "residual"),
                    ("state", "ar1"),
                    ("state", "tail"),
                    ("batch", "mean"),
                    ("classify", "cov"),
                    ("carrier", "cov"),
                    ("residual", "cov"),
                    ("ar1", "cov"),
                    ("tail", "cov"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
            split={
                "context_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l8_unified",
                "context_attr": "_build_context",
                "structural_attr": "_sample_structural_sites",
                "regime_scales_attr": "_sample_regime_scales",
                "regime_path_attr": "_sample_regime_path",
                "priors_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l8_unified",
                "priors_attr": "V3L8UnifiedModelPriors",
            },
            concept_graph=_graph_spec(
                name="concept_v3_l8_unified",
                nodes=(
                    ("baseline", "v3_l1 baseline\njoint mixed-universe posterior"),
                    ("broad", "broad index factor\nshared market mode"),
                    ("region", "regional factor\nUS minus Europe"),
                    ("nugget", "small static residual\nindex-only covariance term"),
                    ("goal", "goal\nfix index geometry without\ngeneric extra freedom"),
                ),
                edges=(
                    ("baseline", "broad"),
                    ("baseline", "region"),
                    ("baseline", "nugget"),
                    ("broad", "goal"),
                    ("region", "goal"),
                    ("nugget", "goal"),
                ),
            ),
        ),
        "v3_l9_unified": _spec_with_split_and_concept(
            version="v3_l9_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 6),
            predict_graph=_graph_spec(
                name="predict_v3_l9_unified",
                nodes=(
                    ("state", "state\nstructured index submodel"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("classify", "classify assets\nFX / index / commodity"),
                    ("broad", "broad index factor\nshared equity mode"),
                    ("us_eu", "regional factor 1\nUS vs Europe"),
                    ("eu_core", "regional factor 2\ncore Europe vs UK/CH"),
                    ("residual", "small residual factor\nindex-only shrunk nugget"),
                    ("cov", "covariance blocks\nglobal + FX + structured index submodel + commodity"),
                    ("mean", "mean path\nalpha + X_asset w + X_global beta"),
                    ("obs", "LowRankMVN over full universe"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("batch", "classify"),
                    ("state", "broad"),
                    ("state", "us_eu"),
                    ("state", "eu_core"),
                    ("classify", "residual"),
                    ("classify", "cov"),
                    ("broad", "cov"),
                    ("us_eu", "cov"),
                    ("eu_core", "cov"),
                    ("residual", "cov"),
                    ("batch", "mean"),
                    ("mean", "obs"),
                    ("cov", "obs"),
                    ("obs", "out"),
                ),
            ),
            split={
                "context_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l9_unified",
                "context_attr": "_build_context",
                "structural_attr": "_sample_structural_sites",
                "regime_scales_attr": "_sample_regime_scales",
                "regime_path_attr": "_sample_regime_path",
                "priors_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l9_unified",
                "priors_attr": "V3L9UnifiedModelPriors",
            },
            concept_graph=_graph_spec(
                name="concept_v3_l9_unified",
                nodes=(
                    ("baseline", "v3_l1 bugfixed backbone\nFX + commodity + global remain fixed"),
                    ("index", "dedicated index submodel"),
                    ("broad", "broad market factor"),
                    ("reg1", "regional factor\nUS vs Europe"),
                    ("reg2", "regional factor\ncore Europe vs UK/CH"),
                    ("resid", "small residual\nindex-only static factor"),
                    ("diag", "diagnostics target\nindices / baskets / residual dependence"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("index", "broad"),
                    ("index", "reg1"),
                    ("index", "reg2"),
                    ("index", "resid"),
                    ("broad", "diag"),
                    ("reg1", "diag"),
                    ("reg2", "diag"),
                    ("resid", "diag"),
                ),
            ),
        ),
        "v3_l10_unified": _spec_with_split_and_concept(
            version="v3_l10_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 5),
            predict_graph=_graph_spec(
                name="predict_v3_l10_unified",
                nodes=(
                    ("state", "state\ncorrected v3_l6 Gaussian base"),
                    ("batch", "prediction batch\nmixed universe"),
                    ("region", "regional state\nUS vs Europe"),
                    ("cov", "base covariance\nglobal + FX + index + region + commodity"),
                    ("flow", "index-only affine coupling\nidentity-initialized shallow flow"),
                    ("obs", "TransformedDistribution\nflow only warps index coordinates"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("state", "region"),
                    ("region", "cov"),
                    ("batch", "cov"),
                    ("cov", "flow"),
                    ("flow", "obs"),
                    ("obs", "out"),
                ),
            ),
            split={
                "context_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l6_unified",
                "context_attr": "_build_context",
                "structural_attr": "_sample_structural_sites",
                "regime_scales_attr": "_sample_regime_scales",
                "regime_path_attr": "_sample_regime_path",
                "priors_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10_unified",
                "priors_attr": "V3L10UnifiedModelPriors",
                "priors_chain": ("base",),
            },
            concept_graph=_graph_spec(
                name="concept_v3_l10_unified",
                nodes=(
                    ("baseline", "corrected v3_l6 Gaussian base"),
                    ("index", "index block only"),
                    ("flow", "one shallow affine-coupling flow\nidentity initialized"),
                    ("goal", "fix residual index dependence geometry"),
                    ("criteria", "must beat v3_l1_bug_fixed\non calibration, baskets, residual dependence"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("index", "flow"),
                    ("flow", "goal"),
                    ("goal", "criteria"),
                ),
            ),
        ),
        "v3_l10a_clean_unified": _spec_with_split_and_concept(
            version="v3_l10a_clean_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l10a_clean_unified",
                nodes=(
                    ("baseline", "baseline\ncorrected v3_l1 Gaussian unified model"),
                    ("index", "index block only"),
                    ("mix", "shared Gamma scale mixture\nmean-one t overlay"),
                    ("obs", "observation draw\nrow-scaled index covariance"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("index", "mix"),
                    ("mix", "obs"),
                    ("obs", "out"),
                ),
            ),
            split=build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10a_clean_unified",
                "V3L10ACleanUnifiedModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10a_clean_unified",
                    "_sample_index_t_copula_mix",
                    ("index_t_copula",),
                ),
            ),
            concept_graph=_graph_spec(
                name="concept_v3_l10a_clean_unified",
                nodes=(
                    ("baseline", "corrected v3_l1 Gaussian base"),
                    ("overlay", "index-only t-copula-style overlay\nshared Gamma scale mixture"),
                    ("goal", "fix joint index dependence\nwithout disturbing best marginals"),
                    ("criteria", "must beat v3_l1_bug_fixed\non calibration, baskets, residual dependence"),
                ),
                edges=(
                    ("baseline", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            ),
        ),
        "v3_l10b_clean_unified": _spec_with_split_and_concept(
            version="v3_l10b_clean_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l10b_clean_unified",
                nodes=(
                    ("baseline", "baseline\ncorrected v3_l1 Gaussian unified model"),
                    ("index", "index block only"),
                    ("broad", "broad index shared scale\nGamma mean-one overlay"),
                    ("usdiff", "US differential scale\nstrongly shrunk"),
                    ("obs", "observation draw\nfactor-only regional overlay"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("index", "broad"),
                    ("index", "usdiff"),
                    ("broad", "obs"),
                    ("usdiff", "obs"),
                    ("obs", "out"),
                ),
            ),
            split=build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10b_clean_unified",
                "V3L10BCleanUnifiedModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10b_clean_unified",
                    "_sample_index_t_copula_mix",
                    ("index_t_copula",),
                ),
            ),
            concept_graph=_graph_spec(
                name="concept_v3_l10b_clean_unified",
                nodes=(
                    ("baseline", "corrected v3_l1 Gaussian base"),
                    ("overlay", "index-only regional overlay\nbroad scale + shrunk US differential scale"),
                    ("goal", "reduce US-side over-width\nwithout giving back Europe gains"),
                    ("criteria", "must beat v3_l1_bug_fixed\non calibration, baskets, residual dependence"),
                ),
                edges=(
                    ("baseline", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            ),
        ),
        "v3_l10c_clean_unified": _spec_with_split_and_concept(
            version="v3_l10c_clean_unified",
            batch=_batch_spec(("EUR.USD", "IBUS500", "IBDE40", "XAU.USD"), 4),
            predict_graph=_graph_spec(
                name="predict_v3_l10c_clean_unified",
                nodes=(
                    ("baseline", "baseline\ncorrected v3_l1 Gaussian unified model"),
                    ("index", "index block only"),
                    ("broad", "broad index shared scale\nGamma mean-one overlay"),
                    ("usdiff", "US differential scale\nsoftly shrunk"),
                    ("obs", "observation draw\nfactor-only interpolated overlay"),
                    ("out", "outputs\nsamples / mean / covariance"),
                ),
                edges=(
                    ("baseline", "index"),
                    ("index", "broad"),
                    ("index", "usdiff"),
                    ("broad", "obs"),
                    ("usdiff", "obs"),
                    ("obs", "out"),
                ),
            ),
            split=build_v3_l1_based_split(
                "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10c_clean_unified",
                "V3L10CCleanUnifiedModelPriors",
                ("base",),
                (
                    "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l10c_clean_unified",
                    "_sample_index_t_copula_mix",
                    ("index_t_copula",),
                ),
            ),
            concept_graph=_graph_spec(
                name="concept_v3_l10c_clean_unified",
                nodes=(
                    ("baseline", "corrected v3_l1 Gaussian base"),
                    ("overlay", "index-only regional overlay\nbroad scale + softer US differential scale"),
                    ("goal", "keep v3_l10b calibration gains\nwhile recovering more v3_l10a basket balance"),
                    ("criteria", "must beat v3_l1_bug_fixed\non calibration, baskets, residual dependence"),
                ),
                edges=(
                    ("baseline", "overlay"),
                    ("overlay", "goal"),
                    ("goal", "criteria"),
                ),
            ),
        ),
    }


def _simple_spec(version: str, batch: BatchSpec, predict_graph: GraphSpec) -> RenderSpec:
    return {
        "version": version,
        "batch": batch,
        "predict_graph": predict_graph,
        "model_module": f"algo_trader.pipeline.stages.modeling.multi_asset_block.model_{version}",
        "model_builder_attr": f"build_multi_asset_block_model_{version}_online_filtering",
        "guide_module": f"algo_trader.pipeline.stages.modeling.multi_asset_block.guide_{version}",
        "guide_builder_attr": f"build_multi_asset_block_guide_{version}_online_filtering",
    }


def _spec_with_concept(
    *,
    version: str,
    batch: BatchSpec,
    predict_graph: GraphSpec,
    concept_graph: GraphSpec,
) -> RenderSpec:
    spec = _simple_spec(version=version, batch=batch, predict_graph=predict_graph)
    return {**spec, "concept_graph": concept_graph}


def _spec_with_split_and_concept(
    *,
    version: str,
    batch: BatchSpec,
    predict_graph: GraphSpec,
    split: SplitSpec,
    concept_graph: GraphSpec,
) -> RenderSpec:
    spec = _simple_spec(version=version, batch=batch, predict_graph=predict_graph)
    return {**spec, "split": split, "concept_graph": concept_graph}


if __name__ == "__main__":
    main()
