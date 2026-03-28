from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "concept_v3_l7_unified.png"
    graph = _build_graph()
    graph.render(outfile=str(output_path), cleanup=True)
    print(f"Saved conceptual graph to {output_path}")


def _build_graph() -> graphviz.Digraph:
    graph = graphviz.Digraph("concept_v3_l7_unified", format="png")
    graph.attr(rankdir="TB")
    graph.attr("node", shape="box")
    graph.node("baseline", "v3_l1 baseline\nFX + broad index + commodity")
    graph.node("carrier1", "fixed carrier\nUS minus Europe")
    graph.node("carrier2", "fixed carrier\ncontinental Europe vs UK/CH")
    graph.node("state1", "dynamic spread state\ns_us_eu[t]")
    graph.node("state2", "dynamic spread state\ns_eu_vs_uk_ch[t]")
    graph.node("shock", "heavy-tailed shocks\nStudentT innovations")
    graph.node(
        "index",
        "index covariance\nbroad mode + US-Europe mode +\nEurope-vs-UK/CH mode",
    )
    graph.node(
        "diagnostics",
        "target diagnostics\nindices / full / baskets / residual dependence",
    )
    graph.edge("baseline", "index")
    graph.edge("carrier1", "index")
    graph.edge("carrier2", "index")
    graph.edge("state1", "index")
    graph.edge("state2", "index")
    graph.edge("shock", "state1")
    graph.edge("shock", "state2")
    graph.edge("index", "diagnostics")
    return graph


if __name__ == "__main__":
    main()
