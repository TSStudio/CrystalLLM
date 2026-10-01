# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Graph Visualization."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from .graph import (
    CrystalGCommPattern,
    CrystalGRank,
    CrystalGraph,
    CrystalGSpan,
    CrystalGSpanId,
    CrystalGSpanPoint,
    DependencyType,
    SpanPointType,
    SpanType,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

NS_TO_MS = 1_000_000

DEFAULT_FIXED_PROC_NS = 30 * NS_TO_MS
DEFAULT_FIXED_COMM_NS = 2 * NS_TO_MS

# Height of a span bar in row units. Rows sit one unit apart, so what is left of the unit
# is the gap that keeps the labels under a bar readable.
_BAR_HEIGHT = 0.4

# Edge colours, kept as matplotlib colour specs so that nothing has to be converted at
# draw time.
_COLOR_COMM_COMM = "tab:green"
_COLOR_PROC_PROC = "tab:purple"
_COLOR_PROC_COMM = "tab:red"
_COLOR_START_CHAIN = "yellow"
_COLOR_REDUNDANT_START_CHAIN = "lightgray"
_COLOR_SYNC = "tab:blue"
_COLOR_ANCHOR_PRED_START = "magenta"
_COLOR_ANCHOR_PRED_STOP = "black"

_P2P_PATTERNS = frozenset({CrystalGCommPattern.P2P_UP, CrystalGCommPattern.P2P_DN})

_SpanKey = tuple[int, CrystalGSpanId]


@dataclass
class _PlotContext:
    """The drawing target and the layout every drawing helper reads.

    Attributes:
        graph: The analyzed graph being drawn.
        ax: The axes all helpers draw on.
        arrow_patch: matplotlib's FancyArrowPatch class, used for every drawn edge.
        ranks: Ranks to draw, top to bottom in this order.
        plotted_ranks: The same ranks as a set, for the membership tests of the edge filters.
        group_keys: Row and colour grouping key of every span of the graph, see _group_key().
        group_colors: Bar colour per grouping key.
        span_extents: (left, right) position of every span of the graph in nanoseconds.
        span_widths: Plot width of every span of the graph in nanoseconds, i.e. the extent it
            would have without the waiting the longest path adds on top.
        span_rows: Row of every span of the plotted ranks.
        rank_rows: Grouping keys of every plotted rank, in row order.
        rank_start_row: Row the first group of a plotted rank occupies.
        total_rows: Rows the figure spans, the gap after the last rank included.
        max_time_ms: Right edge of the last drawn span in milliseconds.
        gray_deps: START-to-START edges to draw gray, see _compute_gray_deps().
    """

    graph: CrystalGraph
    ax: Axes
    arrow_patch: Any
    ranks: list[int]
    plotted_ranks: set[int]
    group_keys: dict[_SpanKey, str]
    group_colors: dict[str, tuple]
    span_extents: dict[_SpanKey, tuple[int, int]]
    span_widths: dict[_SpanKey, int]
    span_rows: dict[_SpanKey, int]
    rank_rows: dict[int, list[str]]
    rank_start_row: dict[int, int]
    total_rows: int
    max_time_ms: float
    gray_deps: set[tuple[CrystalGSpanPoint, CrystalGSpanPoint]]


def _import_matplotlib() -> tuple[Any, Any, Any, Any, Any]:
    """Import matplotlib dependencies.

    Returns:
        Tuple of (mpl, plt, mpatches, FancyArrowPatch, MultipleLocator).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib as mpl
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.ticker import MultipleLocator
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting") from e
    return mpl, plt, mpatches, FancyArrowPatch, MultipleLocator


def _p2p_channels(graph: CrystalGraph) -> set[str]:
    """Return the set of channels that carry point-to-point traffic."""
    channels: set[str] = set()
    for rank in graph.ranks.values():
        for span in rank.spans.values():
            if span.span_type != SpanType.COMM:
                continue
            if not _P2P_PATTERNS.isdisjoint(span.comm_tasks.tasks):
                channels.add(span.comm_tasks.channel)
    return channels


def _group_key(rank: CrystalGRank, span_id: CrystalGSpanId, span: CrystalGSpan) -> str:
    """Return the row and colour grouping key of a span.

    Communication spans group by channel. Computation spans group by their descriptions
    when they carry any, because a captured graph names every computation span "_proc" and
    the semantics live entirely in the descriptions written by the graph marks. Repeated
    descriptions collapse into one key while the label still lists them verbatim, and a
    span marked by two different marks gets its own key, which makes the boundary visible.
    Spans without a description fall back to the type field.
    """
    if span.span_type != SpanType.PROC:
        return span.comm_tasks.channel
    descriptions = rank.span_desc.get(span_id)
    if not descriptions:
        return span_id.type_field
    return "|".join(dict.fromkeys(descriptions))


def _span_group_keys(graph: CrystalGraph) -> dict[_SpanKey, str]:
    """Return the grouping key of every span of the graph, see _group_key().

    The keys drive the rows, the colours and the row ordering alike, so they are computed
    once here instead of being re-derived by each of those steps.
    """
    return {
        (rank_id, span_id): _group_key(rank, span_id, span)
        for rank_id, rank in graph.ranks.items()
        for span_id, span in rank.spans.items()
    }


def _compute_layout(
    graph: CrystalGraph, ranks: list[int], group_keys: dict[_SpanKey, str]
) -> tuple[dict[_SpanKey, int], dict[int, list[str]], dict[int, int], int]:
    """Assign a row to every span of the plotted ranks.

    Ranks are laid out top to bottom in the order given, separated by one empty row. Inside
    a rank the computation groups come first, followed by one row per channel with P2P
    channels before SYMM channels, both in ascending key order.

    Returns:
        Tuple of (span_rows, rank_rows, rank_start_row, total_rows).
    """
    span_rows: dict[_SpanKey, int] = {}
    rank_rows: dict[int, list[str]] = {}
    rank_start_row: dict[int, int] = {}
    p2p_channels = _p2p_channels(graph)

    current_row = 0
    # Rows grow upward on the y axis, so the last rank is placed first to make the
    # first rank of the given order appear at the top of the figure.
    for rank_id in reversed(ranks):
        spans = graph.ranks[rank_id].spans
        proc_keys: set[str] = set()
        rank_p2p_channels: set[str] = set()
        symm_channels: set[str] = set()
        for span_id, span in spans.items():
            key = group_keys[(rank_id, span_id)]
            if span.span_type == SpanType.PROC:
                proc_keys.add(key)
            elif key in p2p_channels:
                rank_p2p_channels.add(key)
            else:
                symm_channels.add(key)

        ordered_keys = sorted(proc_keys) + sorted(rank_p2p_channels) + sorted(symm_channels)
        rank_start_row[rank_id] = current_row
        rank_rows[rank_id] = ordered_keys
        row_of_key = {key: current_row + offset for offset, key in enumerate(ordered_keys)}
        for span_id in spans:
            span_rows[(rank_id, span_id)] = row_of_key[group_keys[(rank_id, span_id)]]

        # One row past the last group of the rank, which leaves the separating gap.
        current_row += len(ordered_keys) + 1

    return span_rows, rank_rows, rank_start_row, current_row


def _compute_span_extents(graph: CrystalGraph, widths: dict[_SpanKey, int]) -> dict[_SpanKey, tuple[int, int]]:
    """Place every span on the time axis by longest-path propagation over the plot widths.

    The dependency structure already built by analyze() is reused: each span contributes
    a START -> STOP internal edge weighted by its plot width, and every UNIDIRECT
    dependency contributes a zero weight edge. BIDIRECT synchronization is honoured
    implicitly because both endpoints live in the same condensed component.

    Args:
        graph: An analyzed execution graph.
        widths: Plot width in nanoseconds per (rank, span_id).

    Returns:
        Mapping from (rank, span_id) to its (left, right) position in nanoseconds.
    """
    if not graph._analyzed:  # noqa: SLF001
        raise RuntimeError("Graph must be analyzed before plotting. Call analyze() first.")

    node_to_comp = graph._node_to_comp  # noqa: SLF001
    adjacency: dict[int, dict[int, int]] = {}

    for dep in graph.dependencies:
        if dep.dep_type != DependencyType.UNIDIRECT:
            continue
        adjacency.setdefault(node_to_comp[dep.v1], {}).setdefault(node_to_comp[dep.v2], 0)

    # The component pair of a span is remembered so that reading the times back below needs
    # no second walk over the points.
    span_comps: dict[_SpanKey, tuple[int, int]] = {}
    for rank_id, rank in graph.ranks.items():
        for span_id in rank.spans:
            start_comp = node_to_comp[CrystalGSpanPoint(rank_id, span_id, SpanPointType.START)]
            stop_comp = node_to_comp[CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)]
            span_comps[(rank_id, span_id)] = (start_comp, stop_comp)

            width = widths[(rank_id, span_id)]
            neighbors = adjacency.setdefault(start_comp, {})
            neighbors[stop_comp] = max(neighbors.get(stop_comp, 0), width)

    comp_time: dict[int, int] = defaultdict(int)
    for comp in graph._comp_topo_order:  # noqa: SLF001
        current_time = comp_time[comp]
        for neighbor_comp, weight in adjacency.get(comp, {}).items():
            comp_time[neighbor_comp] = max(comp_time[neighbor_comp], current_time + weight)

    return {key: (comp_time[start_comp], comp_time[stop_comp]) for key, (start_comp, stop_comp) in span_comps.items()}


def _group_colors(group_keys: Iterable[str], mpl: Any) -> dict[str, tuple]:
    """Assign a bar colour to every grouping key, in ascending key order.

    Palettes are concatenated until they cover the keys, and are wrapped around once no
    palette is left, so a graph with very many groups repeats colours instead of failing.
    """
    sorted_group_keys = sorted(set(group_keys))

    color_palette: list[tuple] = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = mpl.colormaps[name]
        color_palette.extend(cmap(i) for i in range(cmap.N))
        if len(sorted_group_keys) <= len(color_palette):
            break

    return {key: color_palette[idx % len(color_palette)] for idx, key in enumerate(sorted_group_keys)}


def _compute_gray_deps(graph: CrystalGraph) -> set[tuple[CrystalGSpanPoint, CrystalGSpanPoint]]:
    """Return the UNIDIRECT edges whose constraint another edge of the same span already carries.

    A point that depends on both the START and the STOP of one span is reached by two
    parallel zero weight edges, and the one from the STOP is the binding one. The redundant
    edge leaving the START is therefore drawn light gray, which keeps the chain that really
    orders the two spans readable.
    """
    # (target point, source span) -> the source point types the target depends on.
    source_point_types: dict[tuple[CrystalGSpanPoint, _SpanKey], set[SpanPointType]] = defaultdict(set)
    for dep in graph.dependencies:
        if dep.dep_type == DependencyType.UNIDIRECT:
            source_point_types[(dep.v2, (dep.v1.rank, dep.v1.span_id))].add(dep.v1.point_type)

    both_point_types = {SpanPointType.START, SpanPointType.STOP}
    return {
        (CrystalGSpanPoint(rank_id, span_id, SpanPointType.START), v2)
        for (v2, (rank_id, span_id)), point_types in source_point_types.items()
        if point_types == both_point_types
    }


def _point_x_ms(ctx: _PlotContext, point: CrystalGSpanPoint) -> float:
    """Return the x coordinate of a span point in milliseconds."""
    left, right = ctx.span_extents[(point.rank, point.span_id)]
    return (left if point.point_type == SpanPointType.START else right) / NS_TO_MS


def _point_y(ctx: _PlotContext, point: CrystalGSpanPoint) -> float:
    """Return the y coordinate of a span point: the top edge of its bar for a START, the bottom one for a STOP."""
    row = ctx.span_rows[(point.rank, point.span_id)]
    return row + (_BAR_HEIGHT / 2 if point.point_type == SpanPointType.START else -_BAR_HEIGHT / 2)


def _draw_spans(ctx: _PlotContext) -> None:
    """Draw the spans of the plotted ranks as horizontal bars, annotated with their duration and label."""
    for rank_id in ctx.ranks:
        rank = ctx.graph.ranks[rank_id]
        for span_id, span in rank.spans.items():
            row = ctx.span_rows[(rank_id, span_id)]
            left, right = ctx.span_extents[(rank_id, span_id)]
            start_time_ms = left / NS_TO_MS
            drawn_ms = (right - left) / NS_TO_MS
            center_ms = start_time_ms + drawn_ms / 2

            ctx.ax.barh(
                row,
                drawn_ms,
                left=start_time_ms,
                height=_BAR_HEIGHT,
                color=ctx.group_colors[ctx.group_keys[(rank_id, span_id)]],
                edgecolor="black",
                linewidth=1,
            )

            # The bar reaches as far as the longest path allows, which for a span whose STOP
            # point is a synchronization point is the instant the last peer arrives. Annotate
            # the span's own duration so the number never misreports the recorded data, and
            # show the waiting the layout added on top of it separately.
            width = ctx.span_widths[(rank_id, span_id)]
            duration_text = f"{width / NS_TO_MS:.2f}"
            stretch_ns = (right - left) - width
            if stretch_ns > 0:
                duration_text += f" (+{stretch_ns / NS_TO_MS:.2f})"

            ctx.ax.text(
                center_ms,
                row,
                duration_text,
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                fontweight="bold",
            )

            # COMM labels list the descriptions of the span too, like the PROC ones:
            # the policy builders write the schedule semantics of an exchange (e.g.
            # "URF"/"DSF"/"DPAR") there, while runtime captures carry none and the
            # label falls back to the channel and tasks alone. The count_field after
            # the channel is the span's zero-based occurrence index on its rank, which
            # for a runtime capture is the trace_comm_index it was reserved with.
            descriptions = rank.span_desc.get(span_id, [])
            if span.span_type == SpanType.COMM:
                tasks_str = ", ".join(task.value for task in span.comm_tasks.tasks)
                label_text = "\n".join([
                    f"{span.comm_tasks.channel}#{span_id.count_field}#[{tasks_str}]",
                    *descriptions,
                ])
            else:
                label_text = "\n".join([str(span_id), *descriptions])

            ctx.ax.text(
                center_ms,
                row - _BAR_HEIGHT / 2 - 0.15,
                label_text,
                ha="center",
                va="top",
                fontsize=5,
                color="black",
                fontweight="bold",
            )


def _draw_unidirect_dependencies(ctx: _PlotContext) -> None:
    """Draw the UNIDIRECT dependency arrows of the plotted ranks.

    Every edge is solid, and a UNIDIRECT dependency never leaves its rank, so all of them
    are drawn with the same thin translucent line. START-to-START chains (the
    max_connections == 1 serialization) are yellow, or light gray when redundant next to a
    STOP edge of the same span, see _compute_gray_deps(); the remaining edges are colored by
    the span type pair they connect.
    """
    for dep in ctx.graph.dependencies:
        if dep.dep_type != DependencyType.UNIDIRECT or dep.v1.rank not in ctx.plotted_ranks:
            continue

        connectionstyle = "arc3"
        if dep.v1.point_type == SpanPointType.START and dep.v2.point_type == SpanPointType.START:
            color = _COLOR_REDUNDANT_START_CHAIN if (dep.v1, dep.v2) in ctx.gray_deps else _COLOR_START_CHAIN
        else:
            v1_span_type = ctx.graph.ranks[dep.v1.rank].spans[dep.v1.span_id].span_type
            v2_span_type = ctx.graph.ranks[dep.v2.rank].spans[dep.v2.span_id].span_type
            if v1_span_type == SpanType.COMM and v2_span_type == SpanType.COMM:
                color = _COLOR_COMM_COMM
            elif v1_span_type == SpanType.PROC and v2_span_type == SpanType.PROC:
                color = _COLOR_PROC_PROC
            else:
                color = _COLOR_PROC_COMM
                connectionstyle = "arc3,rad=0.1"

        ctx.ax.add_patch(
            ctx.arrow_patch(
                (_point_x_ms(ctx, dep.v1), _point_y(ctx, dep.v1)),
                (_point_x_ms(ctx, dep.v2), _point_y(ctx, dep.v2)),
                arrowstyle="->",
                color=color,
                linewidth=1.0,
                linestyle="-",
                alpha=0.8,
                connectionstyle=connectionstyle,
                mutation_scale=10,
            )
        )


def _plotted_sync_edges(
    graph: CrystalGraph, dep_type: DependencyType, plotted: set[int]
) -> list[tuple[CrystalGSpanPoint, CrystalGSpanPoint]]:
    """Return the synchronization edges of one dependency type to draw for the plotted ranks.

    The graph's cross-rank edges pair adjacent channel members (a relay chain, cyclic
    for P2P), so every direct edge is lost as soon as one of its ends is not drawn,
    even though the drawn ranks still synchronize through the undrawn ones. The list
    returned here closes that gap: it holds every direct edge with both ends drawn,
    then one bridging edge between each adjacent pair of drawn members, in rank order,
    that no direct edge connects anymore. The grouping follows the direct edges of the
    type, so strong and weak components never mix, and a fully drawn view keeps exactly
    the direct edges, P2P cycles included.
    """
    direct_edges = [(dep.v1, dep.v2) for dep in graph.dependencies if dep.dep_type == dep_type]
    direct_set = set(direct_edges)

    # Union-Find over the endpoints of every direct edge, drawn or not. The components
    # it yields are the synchronization groups of that type: for BIDIRECT they match
    # the ones analyze() built, and WEAK_BIDIRECT — which the timing model ignores —
    # gets the same grouping for drawing.
    parent: dict[CrystalGSpanPoint, CrystalGSpanPoint] = {}

    def find(point: CrystalGSpanPoint) -> CrystalGSpanPoint:
        while parent[point] != point:
            parent[point] = parent[parent[point]]  # Path halving, like the one graph.py uses.
            point = parent[point]
        return point

    for v1, v2 in direct_edges:
        parent.setdefault(v1, v1)
        parent.setdefault(v2, v2)
        root_v1, root_v2 = find(v1), find(v2)
        if root_v1 != root_v2:
            parent[root_v1] = root_v2

    edges = [(v1, v2) for v1, v2 in direct_edges if v1.rank in plotted and v2.rank in plotted]

    drawn_by_root: dict[CrystalGSpanPoint, list[CrystalGSpanPoint]] = defaultdict(list)
    for point in parent:
        if point.rank in plotted:
            drawn_by_root[find(point)].append(point)

    for members in drawn_by_root.values():
        if len(members) < 2:  # noqa: PLR2004
            continue
        members.sort(key=lambda point: (point.rank, point.span_id.count_field))
        edges.extend(
            (first, second)
            for first, second in pairwise(members)
            if (first, second) not in direct_set and (second, first) not in direct_set
        )

    return edges


def _draw_bidirect_dependencies(ctx: _PlotContext) -> None:
    """Draw BIDIRECT (and WEAK_BIDIRECT) dependencies as relay chains between the paired spans.

    The direct edges pair adjacent channel members, so drawing them shows every
    synchronization edge, including the ones that reach sandbox ranks, which the
    sandbox-excluded sync span groups no longer carry. A view that skips the members
    between two drawn ranks adds a bridging edge instead of losing their
    synchronization, see _plotted_sync_edges(). A WEAK_BIDIRECT edge — the draw-only
    pairing of an async communication under max_connections == 1, whose peers the
    timing model does not synchronize — is dashed to tell it apart from the strong
    edges.
    """
    for dep_type, linestyle in (
        (DependencyType.BIDIRECT, "-"),
        (DependencyType.WEAK_BIDIRECT, "--"),
    ):
        for v1, v2 in _plotted_sync_edges(ctx.graph, dep_type, ctx.plotted_ranks):
            ctx.ax.add_patch(
                ctx.arrow_patch(
                    (_point_x_ms(ctx, v1), _point_y(ctx, v1)),
                    (_point_x_ms(ctx, v2), _point_y(ctx, v2)),
                    arrowstyle="<->",
                    color=_COLOR_SYNC,
                    linewidth=1.5,
                    linestyle=linestyle,
                    alpha=1,
                    connectionstyle="arc3,rad=0.1",
                    mutation_scale=10,
                )
            )


def _draw_sandbox_background(ctx: _PlotContext) -> None:
    """Shade the rows of every plotted rank that is not a sandbox rank."""
    for rank_id in ctx.ranks:
        if rank_id in ctx.graph.sandbox_ranks:
            continue
        start_row = ctx.rank_start_row[rank_id]
        ctx.ax.axhspan(
            start_row - 0.5,
            start_row + len(ctx.rank_rows[rank_id]) - 0.5,
            facecolor="lightgray",
            alpha=0.3,
            zorder=0,
        )


def _draw_anchor_points(ctx: _PlotContext) -> None:
    """Draw every anchorable point of the plotted ranks with a red star."""
    for anchor_point in ctx.graph.anchorable_points:
        if anchor_point.rank not in ctx.plotted_ranks:
            continue
        ctx.ax.plot(
            _point_x_ms(ctx, anchor_point),
            ctx.span_rows[(anchor_point.rank, anchor_point.span_id)],
            marker="*",
            markersize=8,
            color="red",
            zorder=10,
        )


def _draw_anchor_predecessor_connections(
    ctx: _PlotContext,
    draw_anchor_preds_points: list[CrystalGSpanPoint],
) -> None:
    """Draw the direct anchor predecessors of the given points with thick solid arrows.

    The colour tells the two queries apart: magenta for a START point, i.e. the anchors that
    decide when the span may launch, and black for a STOP point, i.e. the ones that decide
    when it may complete.
    """
    for show_point in draw_anchor_preds_points:
        if show_point.rank not in ctx.plotted_ranks:
            continue

        line_color = (
            _COLOR_ANCHOR_PRED_START if show_point.point_type == SpanPointType.START else _COLOR_ANCHOR_PRED_STOP
        )

        for pred_point in ctx.graph.get_direct_anchor_predecessors(show_point):
            if pred_point.rank not in ctx.plotted_ranks:
                continue
            # The arrow points from the queried point back to its predecessor.
            ctx.ax.add_patch(
                ctx.arrow_patch(
                    (_point_x_ms(ctx, show_point), _point_y(ctx, show_point)),
                    (_point_x_ms(ctx, pred_point), _point_y(ctx, pred_point)),
                    arrowstyle="->",
                    color=line_color,
                    linewidth=1.5,  # Same as a synchronization edge.
                    linestyle="-",
                    alpha=1,
                    connectionstyle="arc3,rad=0.1",
                    mutation_scale=10,
                    zorder=9,
                )
            )


def _configure_axes(ctx: _PlotContext, mpatches: Any, multiple_locator: Any, draw_anchor: bool) -> None:
    """Label one y tick per plotted rank, scale the time axis and add the edge colour legend."""
    y_ticks, y_labels = [], []
    for rank_id in ctx.ranks:
        start_row = ctx.rank_start_row[rank_id]
        # Centre the label on the rows of the rank.
        y_ticks.append(start_row + (len(ctx.rank_rows[rank_id]) - 1) / 2)
        y_labels.append(f"Rank {rank_id}")

    ctx.ax.set_yticks(y_ticks)
    ctx.ax.set_yticklabels(y_labels)
    ctx.ax.set_xlabel("Time (ms)")
    ctx.ax.set_title(f"Execution Graph Timeline (Total: {ctx.max_time_ms:.2f} ms)")
    ctx.ax.grid(True, axis="x", alpha=0.3)
    ctx.ax.set_ylim(-1, ctx.total_rows)
    # Only a short timeline can carry the fine tick spacing; a long one would drown in it.
    ctx.ax.xaxis.set_major_locator(multiple_locator(10 if ctx.max_time_ms < 100 else 100))  # noqa: PLR2004

    legend_elements = [
        mpatches.Patch(color=_COLOR_COMM_COMM, label="Unidirectional (COMM-COMM)"),
        mpatches.Patch(color=_COLOR_PROC_PROC, label="Unidirectional (PROC-PROC)"),
        mpatches.Patch(color=_COLOR_PROC_COMM, label="Unidirectional (PROC-COMM/COMM-PROC)"),
        mpatches.Patch(color=_COLOR_START_CHAIN, label="Unidirectional (START-START)"),
        mpatches.Patch(color=_COLOR_SYNC, label="Bidirectional (Synchronization Group)"),
    ]
    if draw_anchor:
        legend_elements.extend([
            mpatches.Patch(color=_COLOR_ANCHOR_PRED_START, label="Anchor Predecessor (START)"),
            mpatches.Patch(color=_COLOR_ANCHOR_PRED_STOP, label="Anchor Predecessor (STOP)"),
        ])
    ctx.ax.legend(handles=legend_elements, loc="upper right")


def _render_and_output(plt: Any, output_path: str | None, show: bool) -> None:
    """Lay the figure out, save and show it as requested, then close it."""
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=int(os.getenv("PLOT_DPI", "144")), bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close()


def _plot(
    graph: CrystalGraph,
    ranks: list[int] | None,
    widths: dict[_SpanKey, int],
    draw_anchor: bool,
    draw_anchor_preds_points: list[CrystalGSpanPoint] | None,
    output_path: str | None,
    show: bool,
) -> None:
    """Render the Gantt chart for the given plot widths."""
    plotted_ranks = sorted(graph.ranks) if ranks is None else list(ranks)
    for rank_id in plotted_ranks:
        if rank_id not in graph.ranks:
            raise ValueError(f"Rank {rank_id} is not present in the graph")

    # Asking for predecessors is itself a request to draw the anchors they belong to.
    draw_anchor = draw_anchor or draw_anchor_preds_points is not None
    if draw_anchor and not graph.sandbox_ranks:
        raise ValueError("sandbox_ranks must be set before drawing anchors. Call set_sandbox_ranks() first.")

    mpl, plt, mpatches, arrow_patch, multiple_locator = _import_matplotlib()

    group_keys = _span_group_keys(graph)
    span_extents = _compute_span_extents(graph, widths)
    span_rows, rank_rows, rank_start_row, total_rows = _compute_layout(graph, plotted_ranks, group_keys)

    last_stop_ns = max(
        (span_extents[(rank_id, span_id)][1] for rank_id in plotted_ranks for span_id in graph.ranks[rank_id].spans),
        default=0,
    )
    max_time_ms = last_stop_ns / NS_TO_MS

    # The figure grows with the drawn time span and with the row count, with a floor that
    # keeps an almost empty graph readable.
    base_width, base_height, width_per_ms, height_per_row = 5, 4, 0.02, 0.6
    figsize = (max(base_width, max_time_ms * width_per_ms), max(base_height, total_rows * height_per_row))
    _fig, ax = plt.subplots(figsize=figsize)

    ctx = _PlotContext(
        graph=graph,
        ax=ax,
        arrow_patch=arrow_patch,
        ranks=plotted_ranks,
        plotted_ranks=set(plotted_ranks),
        group_keys=group_keys,
        group_colors=_group_colors(group_keys.values(), mpl),
        span_extents=span_extents,
        span_widths=widths,
        span_rows=span_rows,
        rank_rows=rank_rows,
        rank_start_row=rank_start_row,
        total_rows=total_rows,
        max_time_ms=max_time_ms,
        gray_deps=_compute_gray_deps(graph),
    )

    if draw_anchor:
        _draw_sandbox_background(ctx)
    _draw_spans(ctx)
    _draw_unidirect_dependencies(ctx)
    _draw_bidirect_dependencies(ctx)
    if draw_anchor:
        _draw_anchor_points(ctx)
        if draw_anchor_preds_points:
            _draw_anchor_predecessor_connections(ctx, draw_anchor_preds_points)
    _configure_axes(ctx, mpatches, multiple_locator, draw_anchor)
    _render_and_output(plt, output_path, show)


def plot_with_fixed_duration(
    graph: CrystalGraph,
    ranks: list[int] | None = None,
    *,
    proc_duration_ns: int = DEFAULT_FIXED_PROC_NS,
    comm_duration_ns: int = DEFAULT_FIXED_COMM_NS,
    draw_anchor: bool = False,
    draw_anchor_preds_points: list[CrystalGSpanPoint] | None = None,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot an analyzed single-step merged graph using one fixed width per span type.

    Recorded durations are ignored entirely, so the structure of a graph can be
    inspected even when no runtime profiler data is available.

    Args:
        graph: An analyzed single-step merged graph.
        ranks: Ranks to draw, top to bottom in list order. None draws every loaded
            rank in ascending rank order.
        proc_duration_ns: Plot width of every computation span.
        comm_duration_ns: Plot width of every communication span.
        draw_anchor: Whether to shade non-sandbox ranks and mark every anchorable point
            with a red star. Requires set_sandbox_ranks() to have been called.
        draw_anchor_preds_points: Points whose direct anchor predecessors are drawn as
            thick solid arrows, magenta from a START point and black from a STOP point.
            Providing them implies draw_anchor.
        output_path: Path to save the plot. If None, the plot is not saved to file.
        show: Whether to display the plot window.
    """
    widths = {
        (rank_id, span_id): proc_duration_ns if span.span_type == SpanType.PROC else comm_duration_ns
        for rank_id, rank in graph.ranks.items()
        for span_id, span in rank.spans.items()
    }
    _plot(graph, ranks, widths, draw_anchor, draw_anchor_preds_points, output_path, show)


def plot_with_runtime_profiler(
    graph: CrystalGraph,
    ranks: list[int] | None = None,
    *,
    draw_anchor: bool = False,
    draw_anchor_preds_points: list[CrystalGSpanPoint] | None = None,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot an analyzed single-step merged graph using the recorded span durations.

    Spans without a runtime profiler duration collapse to zero width on purpose, which
    makes missing timing data immediately visible instead of hiding it behind a
    fixed-width fallback.

    Args:
        graph: An analyzed single-step merged graph.
        ranks: Ranks to draw, top to bottom in list order. None draws every loaded
            rank in ascending rank order.
        draw_anchor: Whether to shade non-sandbox ranks and mark every anchorable point
            with a red star. Requires set_sandbox_ranks() to have been called.
        draw_anchor_preds_points: Points whose direct anchor predecessors are drawn as
            thick solid arrows, magenta from a START point and black from a STOP point.
            Providing them implies draw_anchor.
        output_path: Path to save the plot. If None, the plot is not saved to file.
        show: Whether to display the plot window.
    """
    widths = {
        (rank_id, span_id): span.duration_ns
        for rank_id, rank in graph.ranks.items()
        for span_id, span in rank.spans.items()
    }
    _plot(graph, ranks, widths, draw_anchor, draw_anchor_preds_points, output_path, show)
