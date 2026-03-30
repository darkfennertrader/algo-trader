from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import graphviz  # pyright: ignore[reportMissingTypeStubs]
import pyro
import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    FilteringState,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch


class BatchSpec(TypedDict):
    asset_names: tuple[str, ...]
    state_size: int


class GraphSpec(TypedDict):
    name: str
    nodes: tuple[tuple[str, str], ...]
    edges: tuple[tuple[str, str], ...]
    rankdir: NotRequired[str]


class SplitSpec(TypedDict):
    context_module: str
    context_attr: str
    structural_attr: str
    regime_scales_attr: str
    regime_path_attr: str
    priors_module: str
    priors_attr: str
    priors_chain: NotRequired[tuple[str, ...]]
    regime_extra_module: NotRequired[str]
    regime_extra_attr: NotRequired[str]
    regime_extra_chain: NotRequired[tuple[str, ...]]


class RenderSpec(TypedDict):
    version: str
    batch: BatchSpec
    predict_graph: GraphSpec
    model_module: str
    model_builder_attr: str
    guide_module: str
    guide_builder_attr: str
    split: NotRequired[SplitSpec]
    concept_graph: NotRequired[GraphSpec]


def build_v3_l1_based_split(
    priors_module: str,
    priors_attr: str,
    priors_chain: tuple[str, ...] = (),
    regime_extra: tuple[str, str, tuple[str, ...]] | None = None,
) -> SplitSpec:
    split: SplitSpec = {
        "context_module": "algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified",
        "context_attr": "_build_context",
        "structural_attr": "_sample_structural_sites",
        "regime_scales_attr": "_sample_regime_scales",
        "regime_path_attr": "_sample_regime_path",
        "priors_module": priors_module,
        "priors_attr": priors_attr,
        "priors_chain": priors_chain,
    }
    if regime_extra is not None:
        extra_module, extra_attr, extra_chain = regime_extra
        split["regime_extra_module"] = extra_module
        split["regime_extra_attr"] = extra_attr
        split["regime_extra_chain"] = extra_chain
    return split


def run_family(
    description: str,
    registry: dict[str, RenderSpec],
    render_root: Path,
    runtime_batch_builder: Any,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "version",
        nargs="?",
        default="all",
        choices=("all", *registry.keys()),
        help="Version to render, or 'all'.",
    )
    args = parser.parse_args()
    render_root.mkdir(parents=True, exist_ok=True)
    versions = registry.keys() if args.version == "all" else (args.version,)
    for version in versions:
        render_version(registry[version], render_root / version, runtime_batch_builder)


def render_version(
    spec: RenderSpec,
    output_dir: Path,
    runtime_batch_builder: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_model(spec, output_dir)
    split = spec.get("split")
    if split is not None:
        _render_structural(spec["batch"], split, output_dir, runtime_batch_builder)
        _render_regime(spec["batch"], split, output_dir, runtime_batch_builder)
    _render_guide(spec, output_dir)
    _render_graph(spec["predict_graph"], output_dir / "predict.png")
    concept_graph = spec.get("concept_graph")
    if concept_graph is not None:
        _render_graph(concept_graph, output_dir / "concept.png")
    print(f"Rendered {spec['version']} into {output_dir}")


def _render_model(spec: RenderSpec, output_dir: Path) -> None:
    model_builder = _load_attr(spec["model_module"], spec["model_builder_attr"])
    pyro.render_model(
        model_builder({}),
        model_args=(_render_batch(spec["batch"]),),
        filename=str(output_dir / "model.png"),
        render_params=True,
        render_distributions=True,
    )


def _render_guide(spec: RenderSpec, output_dir: Path) -> None:
    guide_builder = _load_attr(spec["guide_module"], spec["guide_builder_attr"])
    pyro.clear_param_store()
    pyro.render_model(
        guide_builder({}),
        model_args=(_render_batch(spec["batch"]),),
        filename=str(output_dir / "guide.png"),
        render_params=True,
        render_distributions=True,
    )


def _render_structural(
    batch_spec: BatchSpec,
    split: SplitSpec,
    output_dir: Path,
    runtime_batch_builder: Any,
) -> None:
    def _structural_model(batch: ModelBatch) -> None:
        runtime_batch = runtime_batch_builder(batch)
        context_builder = _load_attr(split["context_module"], split["context_attr"])
        structural_sampler = _load_attr(split["context_module"], split["structural_attr"])
        priors = _build_priors(split)
        context = context_builder(runtime_batch, priors)
        structural_sampler(context)

    pyro.render_model(
        _structural_model,
        model_args=(_render_batch(batch_spec),),
        filename=str(output_dir / "model_structural.png"),
        render_params=True,
        render_distributions=True,
    )


def _render_regime(
    batch_spec: BatchSpec,
    split: SplitSpec,
    output_dir: Path,
    runtime_batch_builder: Any,
) -> None:
    def _regime_model(batch: ModelBatch) -> None:
        runtime_batch = runtime_batch_builder(batch)
        context_builder = _load_attr(split["context_module"], split["context_attr"])
        regime_scales_sampler = _load_attr(
            split["context_module"], split["regime_scales_attr"]
        )
        regime_path_sampler = _load_attr(
            split["context_module"], split["regime_path_attr"]
        )
        priors_root = _load_attr(split["priors_module"], split["priors_attr"])()
        base_priors = _resolve_chain(priors_root, split.get("priors_chain", ()))
        context = context_builder(runtime_batch, base_priors)
        regime_scales = regime_scales_sampler(context)
        regime_path_sampler(context, regime_scales)
        extra_module = split.get("regime_extra_module")
        extra_attr = split.get("regime_extra_attr")
        if extra_module and extra_attr:
            extra_sampler = _load_attr(extra_module, extra_attr)
            extra_priors = _resolve_chain(priors_root, split.get("regime_extra_chain", ()))
            extra_sampler(context, extra_priors)

    pyro.render_model(
        _regime_model,
        model_args=(_render_batch(batch_spec),),
        filename=str(output_dir / "model_regime.png"),
        render_params=True,
        render_distributions=True,
    )


def _build_priors(split: SplitSpec) -> Any:
    priors_root = _load_attr(split["priors_module"], split["priors_attr"])()
    return _resolve_chain(priors_root, split.get("priors_chain", ()))


def _render_graph(spec: GraphSpec, output_path: Path) -> None:
    graph = graphviz.Digraph(spec["name"], format="png")
    graph.attr(rankdir=spec.get("rankdir", "LR"))
    graph.attr("node", shape="box")
    for node_id, label in spec["nodes"]:
        graph.node(node_id, label)
    for source, target in spec["edges"]:
        graph.edge(source, target)
    graph.render(outfile=str(output_path), cleanup=True)


def _render_batch(spec: BatchSpec) -> ModelBatch:
    asset_count = len(spec["asset_names"])
    return ModelBatch(
        X_asset=torch.zeros((2, asset_count, 3), dtype=torch.float32),
        X_global=torch.zeros((2, 2), dtype=torch.float32),
        y=torch.zeros((2, asset_count), dtype=torch.float32),
        asset_names=spec["asset_names"],
        filtering_state=FilteringState(
            h_loc=torch.zeros(spec["state_size"], dtype=torch.float32),
            h_scale=torch.full((spec["state_size"],), 0.15, dtype=torch.float32),
            steps_seen=26,
        ),
    )


def _load_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _resolve_chain(value: Any, chain: tuple[str, ...]) -> Any:
    current = value
    for attr_name in chain:
        current = getattr(current, attr_name)
    return current
