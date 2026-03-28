from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "concept_v3_l6_unified.png"
    graph = _build_graph()
    graph.render(outfile=str(output_path), cleanup=True)
    print(f"Saved conceptual graph to {output_path}")


def _build_graph() -> graphviz.Digraph:
    graph = graphviz.Digraph("concept_v3_l6_unified", format="png")
    graph.attr(rankdir="TB")
    graph.attr("node", shape="box")
    graph.node("baseline", "v3_l1 baseline\nFX + broad index + commodity")
    graph.node("carrier", "fixed US-Europe carrier\ncentered / normalized / orthogonalized")
    graph.node("state", "dynamic spread state\ns_us_eu[t]")
    graph.node("shock", "heavy-tailed shock\nStudentT innovation")
    graph.node("index", "index covariance\nbroad mode + spread mode")
    graph.node("diagnostics", "target diagnostics\nindices / full / baskets / residual dependence")
    graph.edge("baseline", "index")
    graph.edge("carrier", "index")
    graph.edge("state", "index")
    graph.edge("shock", "state")
    graph.edge("index", "diagnostics")
    return graph


if __name__ == "__main__":
    main()
