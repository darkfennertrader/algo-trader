from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = graphviz.Digraph("concept_v3_l10c_clean_unified", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")
    graph.node("baseline", "corrected v3_l1 Gaussian base")
    graph.node(
        "overlay",
        "index-only regional overlay\nbroad scale + softer US differential scale",
    )
    graph.node(
        "goal",
        "keep v3_l10b calibration gains\nwhile recovering more v3_l10a basket balance",
    )
    graph.node("criteria", "must beat v3_l1_bug_fixed\non calibration, baskets, residual dependence")
    graph.edge("baseline", "overlay")
    graph.edge("overlay", "goal")
    graph.edge("goal", "criteria")
    output_path = output_dir / "concept_v3_l10c_clean_unified.png"
    graph.render(outfile=str(output_path), cleanup=True)
    print(f"Saved conceptual graph to {output_path}")


if __name__ == "__main__":
    main()
