from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from algo_trader.domain.simulation import CPCVSplit

from .index_ranges import indices_to_ranges
from .plotting_backend import require_pyplot


_ROW_HEIGHT = 0.8
_COLORS = {
    "warmup": "#F2CF5B",
    "train": "#4C78A8",
    "test": "#F58518",
    "purged": "#E45756",
    "embargoed": "#9D9DA3",
}


def write_splits_timeline_plot(
    *,
    output_path: Path,
    splits: Sequence[CPCVSplit],
    warmup_idx: np.ndarray | None = None,
) -> None:
    if not splits:
        return
    plt = _require_matplotlib()
    fig, axis = plt.subplots(
        figsize=(14.0, max(4.0, 1.2 + len(splits) * 0.6)),
        constrained_layout=True,
    )
    _draw_warmup(axis, splits=splits, warmup_idx=warmup_idx)
    _draw_split_rows(axis, splits)
    _finalize_axis(axis, splits)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _require_matplotlib():
    return require_pyplot()


def _draw_split_rows(axis, splits: Sequence[CPCVSplit]) -> None:
    for split_id, split in enumerate(splits):
        y0 = split_id - (_ROW_HEIGHT / 2.0)
        _draw_category(axis, split.train_idx, y0, "train", zorder=1)
        _draw_category(axis, split.embargoed_idx, y0, "embargoed", zorder=2)
        _draw_category(axis, split.purged_idx, y0, "purged", zorder=3)
        _draw_category(axis, split.test_idx, y0, "test", zorder=4)


def _draw_category(
    axis, indices: np.ndarray, y0: float, category: str, *, zorder: int
) -> None:
    ranges = [
        (float(start), float(end - start + 1))
        for start, end in indices_to_ranges(indices)
    ]
    if not ranges:
        return
    axis.broken_barh(
        xranges=ranges,
        yrange=(y0, _ROW_HEIGHT),
        facecolors=_COLORS[category],
        edgecolors="none",
        alpha=0.95,
        label=category,
        zorder=zorder,
    )


def _finalize_axis(axis, splits: Sequence[CPCVSplit]) -> None:
    labels = [_split_label(idx, split) for idx, split in enumerate(splits)]
    axis.set_yticks(np.arange(len(splits), dtype=float), labels)
    axis.set_ylim(-0.7, len(splits) - 0.3)
    min_x, max_x = _x_limits(splits)
    axis.set_xlim(float(min_x) - 0.5, float(max_x) + 1.5)
    axis.set_xlabel("Week Index")
    axis.set_ylabel("Inner Split")
    axis.set_title("CPCV Split Timeline")
    axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    handles, names = axis.get_legend_handles_labels()
    ordered = {}
    for handle, name in zip(handles, names):
        ordered.setdefault(name, handle)
    axis.legend(
        ordered.values(),
        ordered.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Segment",
        frameon=True,
    )


def _x_limits(splits: Sequence[CPCVSplit]) -> tuple[int, int]:
    minima: list[int] = []
    maxima: list[int] = []
    for split in splits:
        for values in (
            split.train_idx,
            split.test_idx,
            split.purged_idx,
            split.embargoed_idx,
        ):
            if values.size == 0:
                continue
            minima.append(int(np.min(values)))
            maxima.append(int(np.max(values)))
    return min(minima), max(maxima)


def _split_label(split_id: int, split: CPCVSplit) -> str:
    groups = ",".join(str(int(group)) for group in split.test_group_ids)
    return f"split_{split_id:03d} (test g:{groups})"


def _draw_warmup(
    axis, *, splits: Sequence[CPCVSplit], warmup_idx: np.ndarray | None
) -> None:
    if warmup_idx is None or warmup_idx.size == 0:
        return
    y0 = -0.7
    height = len(splits) + 0.4
    ranges = [
        (float(start), float(end - start + 1))
        for start, end in indices_to_ranges(warmup_idx)
    ]
    if not ranges:
        return
    axis.broken_barh(
        xranges=ranges,
        yrange=(y0, height),
        facecolors=_COLORS["warmup"],
        edgecolors="none",
        alpha=0.25,
        label="warmup",
        zorder=0,
    )
