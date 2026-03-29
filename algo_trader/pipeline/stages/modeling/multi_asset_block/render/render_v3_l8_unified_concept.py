from __future__ import annotations

from pathlib import Path

import graphviz  # pyright: ignore[reportMissingTypeStubs]


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "concept_v3_l8_unified.png"
    graph = graphviz.Digraph("concept_v3_l8_unified", format="png")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box")
    graph.node("baseline", "v3_l1 baseline\njoint mixed-universe posterior")
    graph.node("broad", "broad index factor\nshared market mode")
    graph.node("region", "regional factor\nUS minus Europe")
    graph.node("nugget", "small static residual\nindex-only covariance term")
    graph.node("goal", "goal\nfix index geometry without\ngeneric extra freedom")
    graph.edge("baseline", "broad")
    graph.edge("baseline", "region")
    graph.edge("baseline", "nugget")
    graph.edge("broad", "goal")
    graph.edge("region", "goal")
    graph.edge("nugget", "goal")
    graph.render(outfile=str(output_path), cleanup=True)
    print(f"Saved concept graph to {output_path}")


if __name__ == "__main__":
    main()
