from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]


def main() -> None:
    graph = graphviz.Digraph("concept_v3_l9_unified", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")
    graph.node("baseline", "v3_l1 bugfixed backbone\nFX + commodity + global remain fixed")
    graph.node("index", "dedicated index submodel")
    graph.node("broad", "broad market factor")
    graph.node("reg1", "regional factor\nUS vs Europe")
    graph.node("reg2", "regional factor\ncore Europe vs UK/CH")
    graph.node("resid", "small residual\nindex-only static factor")
    graph.node("diag", "diagnostics target\nindices / baskets / residual dependence")
    graph.edge("baseline", "index")
    graph.edge("index", "broad")
    graph.edge("index", "reg1")
    graph.edge("index", "reg2")
    graph.edge("index", "resid")
    graph.edge("broad", "diag")
    graph.edge("reg1", "diag")
    graph.edge("reg2", "diag")
    graph.edge("resid", "diag")
    output_path = Path(__file__).resolve().parent / "concept_v3_l9_unified.png"
    graph.render(outfile=str(output_path), cleanup=True)
    print(f"Saved concept graph to {output_path}")


if __name__ == "__main__":
    main()
