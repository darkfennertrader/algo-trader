from __future__ import annotations

from algo_trader.pipeline.stages.modeling.factor import model_v1


# Run from repo root:
# uv run python -m algo_trader.pipeline.stages.modeling.factor.render.render_model_v1


def main() -> None:
    output = model_v1.render_model_graph()
    print(f"Saved model graph to {output}")


if __name__ == "__main__":
    main()
