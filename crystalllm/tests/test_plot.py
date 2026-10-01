# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for the Gantt plot of an analyzed graph: the plot widths, the layout and the drawn edges.

The visual artifacts of the policy graphs live in tests/policy_plot. What is locked here is the
data plot.py computes before matplotlib sees it: the width every span occupies under each of the
two duration bases, where the longest path places it on the time axis, which row and colour its
group gets, and which dependency edges a rank selection draws. Rendering is exercised on the
paths only a full run reaches — the default rank selection, a partial view whose synchronization
has to be bridged, the anchor overlay and a graph with missing durations.
"""

import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import matplotlib

from crystalllm.planning import (
    CrystalGCommPattern,
    CrystalGCommTasks,
    CrystalGDep,
    CrystalGraph,
    CrystalGSpan,
    CrystalGSpanId,
    CrystalGSpanPoint,
    DependencyType,
    SpanPointType,
    SpanType,
    plot_with_fixed_duration,
    plot_with_runtime_profiler,
)
from crystalllm.planning import plot as plot_module
from crystalllm.planning.plot import (
    DEFAULT_FIXED_COMM_NS,
    DEFAULT_FIXED_PROC_NS,
    _compute_gray_deps,
    _compute_layout,
    _compute_span_extents,
    _group_colors,
    _plotted_sync_edges,
    _span_group_keys,
)
from crystalllm.planning.policy.megatron_1f1b_interleaved import (
    BACKWARD as INTERLEAVED_BACKWARD,
)
from crystalllm.planning.policy.megatron_1f1b_interleaved import (
    FORWARD as INTERLEAVED_FORWARD,
)
from crystalllm.planning.policy.megatron_1f1b_interleaved import (
    build_megatron_1f1b_interleaved_graph,
)
from crystalllm.planning.policy.parallel_layout import rank_of

PIPELINE_PARALLEL_SIZE = 4
DATA_PARALLEL_SIZE = 3

# One PP chain (every stage at DP position 0) and the DP group of the first stage: the two
# rank selections a plot is usually asked for.
PP_CHAIN = [rank_of(stage, 0, DATA_PARALLEL_SIZE) for stage in range(PIPELINE_PARALLEL_SIZE)]
DP_GROUP = [rank_of(0, dp_index, DATA_PARALLEL_SIZE) for dp_index in range(DATA_PARALLEL_SIZE)]

# One SYMM channel whose members are neither zero-based nor the whole graph.
SYMM_CHANNEL = "dp_group"
SYMM_MEMBERS = [1, 2, 3, 4]

OUTPUT_DIR = Path(__file__).parent / "test_output_plot"


def ms(duration_ms: int) -> int:
    """Convert milliseconds to nanoseconds for test data."""
    return duration_ms * 1_000_000


def build_interleaved_graph(timed_ranks: list[int] | None = None) -> CrystalGraph:
    """Build a PP + DP interleaved graph, analyzed and ready to plot.

    Durations are loaded before analyze() because they are the weights of the condensed
    graph's internal edges, so loading them afterwards would invalidate the analysis.

    Args:
        timed_ranks: Ranks whose computation spans get a duration. Every other rank keeps 0,
            which is what a rank without runtime profiler data looks like.
    """
    graph = build_megatron_1f1b_interleaved_graph(
        num_microbatches=4,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        num_model_chunks=2,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )

    if timed_ranks:
        duration_dict: dict[int, dict[str, list[int]]] = {}
        for rank_id in timed_ranks:
            rank = graph.ranks[rank_id]
            # The computation spans are named like a runtime capture, so the schedule
            # semantics live in span_desc and the durations are one positional list over
            # the rank's _proc spans.
            proc_spans = sorted(
                (span for span in rank.spans.values() if span.span_type == SpanType.PROC),
                key=lambda span: span.span_id.count_field,
            )
            durations = []
            for span in proc_spans:
                descriptions = rank.span_desc.get(span.span_id, [])
                duration_ns = 0
                if INTERLEAVED_FORWARD in descriptions:
                    duration_ns += ms(30)
                if INTERLEAVED_BACKWARD in descriptions:
                    duration_ns += ms(60)
                durations.append(duration_ns)
            duration_dict[rank_id] = {proc_spans[0].span_id.type_field: durations}
        graph.load_span_duration_dict(duration_dict)

    graph.analyze()
    return graph


def build_symm_channel_graph(*, max_connections: int = 8, async_op: bool = False) -> CrystalGraph:
    """Build one SYMM channel over ranks 1..4, with a single collective span per member.

    Args:
        max_connections: Decides whether an async communication pairs its peers weakly (1)
            or strongly, see graph.py's _add_bidirectional_dependency().
        async_op: Whether every member issues its collective asynchronously.
    """
    graph = CrystalGraph()
    graph.set_max_connections(max_connections)
    graph.set_channel_members(SYMM_CHANNEL, SYMM_MEMBERS)
    for rank_id in SYMM_MEMBERS:
        graph.add_span(
            rank_id,
            CrystalGSpan(
                SpanType.COMM,
                CrystalGSpanId("_comm", 0),
                comm_tasks=CrystalGCommTasks(SYMM_CHANNEL, [CrystalGCommPattern.SYMM], async_op=async_op),
                duration_ns=ms(2),
            ),
        )
    graph.analyze()
    return graph


class TestPlotEntrypoints(unittest.TestCase):
    """Test the two duration bases, the rank selection checks and the rendering paths."""

    def setUp(self):
        """Set up the shared output directory."""
        OUTPUT_DIR.mkdir(exist_ok=True)

    def _captured_widths(self, entrypoint, graph, ranks=None, **kwargs) -> dict[tuple[int, CrystalGSpanId], int]:
        """Return the plot widths an entrypoint hands to _plot(), without rendering anything."""
        with mock.patch.object(plot_module, "_plot") as plot_call:
            entrypoint(graph, ranks, show=False, **kwargs)
        plot_call.assert_called_once()
        return plot_call.call_args.args[2]

    def test_fixed_duration_widths_ignore_the_recorded_durations(self):
        """Test that one width per span type is used, whatever the spans recorded.

        Rank 0 carries measured computation durations, yet every computation span must come
        out at the fixed width, both at the default and at an overridden one. The widths also
        have to cover the spans of the ranks that are not drawn, because the longest path
        that places the drawn ones runs over the whole graph.
        """
        graph = build_interleaved_graph(timed_ranks=[0])
        self.assertTrue(
            any(span.duration_ns for span in graph.ranks[0].spans.values()),
            "rank 0 was expected to carry recorded durations",
        )

        for kwargs, proc_ns, comm_ns in (
            ({}, DEFAULT_FIXED_PROC_NS, DEFAULT_FIXED_COMM_NS),
            ({"proc_duration_ns": ms(7), "comm_duration_ns": ms(3)}, ms(7), ms(3)),
        ):
            with self.subTest(**kwargs):
                widths = self._captured_widths(plot_with_fixed_duration, graph, [0], **kwargs)
                self.assertEqual(
                    widths,
                    {
                        (rank_id, span_id): proc_ns if span.span_type == SpanType.PROC else comm_ns
                        for rank_id, rank in graph.ranks.items()
                        for span_id, span in rank.spans.items()
                    },
                )

    def test_runtime_profiler_widths_are_the_recorded_durations(self):
        """Test that the recorded duration is the width, and that a missing one collapses to zero.

        The untimed rank must not fall back to a fixed width: a zero-width bar is what makes
        missing profiler data visible instead of hiding it.
        """
        untimed_rank = rank_of(0, 1, DATA_PARALLEL_SIZE)
        graph = build_interleaved_graph(timed_ranks=PP_CHAIN)

        widths = self._captured_widths(plot_with_runtime_profiler, graph, [*PP_CHAIN, untimed_rank])
        self.assertEqual(
            widths,
            {
                (rank_id, span_id): span.duration_ns
                for rank_id, rank in graph.ranks.items()
                for span_id, span in rank.spans.items()
            },
        )
        self.assertEqual({widths[(untimed_rank, span_id)] for span_id in graph.ranks[untimed_rank].spans}, {0})
        self.assertTrue(any(widths[(PP_CHAIN[0], span_id)] > 0 for span_id in graph.ranks[PP_CHAIN[0]].spans))

    def test_plot_rejects_a_rank_absent_from_the_graph(self):
        """Test that a rank the graph never loaded is reported instead of failing on a lookup later."""
        graph = build_interleaved_graph()

        with self.assertRaises(ValueError) as context:
            plot_with_fixed_duration(graph, [PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE], show=False)
        self.assertIn("not present in the graph", str(context.exception))

    def test_plot_requires_an_analyzed_graph(self):
        """Test that plotting a graph that was never analyzed is refused.

        The span positions come from the condensed graph analyze() builds, so without it
        there is nothing to lay out.
        """
        graph = build_megatron_1f1b_interleaved_graph(num_microbatches=2, pipeline_parallel_size=2, num_model_chunks=2)

        with self.assertRaises(RuntimeError) as context:
            plot_with_fixed_duration(graph, show=False)
        self.assertIn("analyze()", str(context.exception))

    def test_anchor_drawing_requires_sandbox_ranks(self):
        """Test that anchors cannot be drawn before the sandbox boundary is known."""
        graph = build_interleaved_graph()

        # An empty predecessor list still asks for anchor drawing, so both forms must fail.
        for kwargs in ({"draw_anchor": True}, {"draw_anchor_preds_points": []}):
            with self.subTest(**kwargs), self.assertRaises(ValueError) as context:
                plot_with_fixed_duration(graph, show=False, **kwargs)
            self.assertIn("set_sandbox_ranks", str(context.exception))

    def test_default_and_partial_rank_selections_render(self):
        """Test that both rank selections render: every loaded rank by default, the listed ones otherwise.

        Drawing ranks 1 and 3 of the four-member channel is the case whose synchronization
        survives only as a bridging edge, see _plotted_sync_edges().
        """
        graph = build_symm_channel_graph()

        for name, ranks in (("default", None), ("bridged", [1, 3])):
            with self.subTest(view=name):
                output_path = OUTPUT_DIR / f"test_symm_channel_{name}.png"
                plot_with_fixed_duration(graph, ranks, output_path=str(output_path), show=False)
                self.assertTrue(output_path.exists(), f"Plot was not written to {output_path}")

    def test_fixed_duration_renders_the_anchors_of_a_dp_group(self):
        """Test that the anchor overlay renders for the DP group of one stage.

        A DP group is where anchors matter most: its collective is the synchronization that
        crosses the sandbox boundary between PP chains. Only the START twins of the
        anchorable points are queried, i.e. the launch-time predecessors drawn in magenta;
        STOP queries stay supported by the plot (black arrows) but are not exercised here.
        """
        graph = build_interleaved_graph()
        graph.set_sandbox_ranks([DP_GROUP[0]])
        graph.analyze()  # set_sandbox_ranks() invalidates the previous analysis.
        self.assertTrue(graph.anchorable_points, "No anchorable point for this sandbox split")

        show_anchor_points = [
            CrystalGSpanPoint(point.rank, point.span_id, SpanPointType.START)
            for point in sorted(graph.anchorable_points, key=lambda p: (p.rank, p.span_id.count_field))
        ]
        # Without a predecessor to reach, the figure would silently carry no anchor arrow.
        self.assertTrue(any(graph.get_direct_anchor_predecessors(point) for point in show_anchor_points))

        output_path = OUTPUT_DIR / "test_interleaved_dp_group_anchor.png"
        # draw_anchor_preds_points implies draw_anchor, so draw_anchor is not passed.
        plot_with_fixed_duration(
            graph,
            DP_GROUP,
            draw_anchor_preds_points=show_anchor_points,
            output_path=str(output_path),
            show=False,
        )
        self.assertTrue(output_path.exists(), f"Plot was not written to {output_path}")

    def test_runtime_profiler_renders_a_partially_timed_view(self):
        """Test that a view mixing timed and untimed ranks renders on the profiler basis.

        The untimed rank contributes zero-width bars, which the drawing code has to place and
        annotate like any other span.
        """
        graph = build_interleaved_graph(timed_ranks=PP_CHAIN)
        untimed_rank = rank_of(0, 1, DATA_PARALLEL_SIZE)

        output_path = OUTPUT_DIR / "test_interleaved_partially_timed.png"
        plot_with_runtime_profiler(graph, [*PP_CHAIN, untimed_rank], output_path=str(output_path), show=False)
        self.assertTrue(output_path.exists(), f"Plot was not written to {output_path}")


class TestSpanExtents(unittest.TestCase):
    """Test how the longest path over the plot widths places the spans on the time axis."""

    def test_extents_follow_the_dependencies_and_stretch_to_the_last_peer(self):
        """Test the three placement rules on a two-rank collective preceded by unequal computation.

        A span starts at zero when nothing precedes it, a UNIDIRECT dependency pushes its
        successor to the end of the predecessor, and the STOP points of a collective share a
        component, so the peer that arrives first is stretched until the last one is done.
        That stretch is what _draw_spans() annotates as "(+...)" next to the own duration.
        """
        proc_id, comm_id = CrystalGSpanId("_proc", 0), CrystalGSpanId("_comm", 0)
        proc_ms = {0: 10, 1: 30}

        graph = CrystalGraph()
        graph.set_max_connections(8)
        graph.set_channel_members(SYMM_CHANNEL, [0, 1])
        for rank_id, duration_ms in proc_ms.items():
            graph.add_span(rank_id, CrystalGSpan(SpanType.PROC, proc_id, duration_ns=ms(duration_ms)))
            graph.add_span(
                rank_id,
                CrystalGSpan(
                    SpanType.COMM,
                    comm_id,
                    comm_tasks=CrystalGCommTasks(SYMM_CHANNEL, [CrystalGCommPattern.SYMM]),
                    duration_ns=ms(2),
                ),
            )
            graph.add_dependency(
                CrystalGDep(
                    DependencyType.UNIDIRECT,
                    CrystalGSpanPoint(rank_id, proc_id, SpanPointType.STOP),
                    CrystalGSpanPoint(rank_id, comm_id, SpanPointType.START),
                )
            )
        graph.analyze()

        widths = {
            (0, proc_id): ms(proc_ms[0]),
            (1, proc_id): ms(proc_ms[1]),
            (0, comm_id): ms(2),
            (1, comm_id): ms(2),
        }
        self.assertEqual(
            _compute_span_extents(graph, widths),
            {
                (0, proc_id): (0, ms(10)),
                (1, proc_id): (0, ms(30)),
                # Both collectives complete at 30 + 2, so rank 0 waits 20 ms for rank 1.
                (0, comm_id): (ms(10), ms(32)),
                (1, comm_id): (ms(30), ms(32)),
            },
        )


class TestSpanLayout(unittest.TestCase):
    """Test the grouping keys of the spans and the rows and colours they receive."""

    # (descriptions written on the span, expected grouping key)
    PROC_CASES = (
        (["FWD"], "FWD"),
        (["BWD"], "BWD"),
        # Two marks covering the same span: its own key, so the boundary stays visible.
        (["FWD", "BWD"], "FWD|BWD"),
        # A repeated description only shows up in the label, it must not split the row.
        (["FWD", "FWD"], "FWD"),
        # No description at all falls back to the type field.
        ([], "_proc"),
    )

    def test_computation_spans_group_by_their_descriptions(self):
        """Test that the descriptions form the key, and that one key owns one row and one colour.

        A captured graph gives every computation span the type field "_proc", so grouping by
        the span id would collapse a whole iteration onto a single row.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        for index, (descriptions, _) in enumerate(self.PROC_CASES):
            span_id = CrystalGSpanId("_proc", index)
            graph.add_span(0, CrystalGSpan(SpanType.PROC, span_id, duration_ns=ms(10)))
            for description in descriptions:
                graph.add_span_desc(0, span_id, description)
        graph.analyze()

        group_keys = _span_group_keys(graph)
        self.assertEqual(
            group_keys,
            {(0, CrystalGSpanId("_proc", index)): key for index, (_, key) in enumerate(self.PROC_CASES)},
        )

        distinct_keys = sorted(set(group_keys.values()))
        span_rows, rank_rows, rank_start_row, total_rows = _compute_layout(graph, [0], group_keys)
        self.assertEqual(rank_rows[0], distinct_keys)
        self.assertEqual(rank_start_row[0], 0)
        # The rows of the only rank plus the empty row that would separate the next one.
        self.assertEqual(total_rows, len(distinct_keys) + 1)

        rows_of_key: dict[str, set[int]] = defaultdict(set)
        for span_key, row in span_rows.items():
            rows_of_key[group_keys[span_key]].add(row)
        self.assertEqual(dict(rows_of_key), {key: {row} for row, key in enumerate(rank_rows[0])})

        colors = _group_colors(group_keys.values(), matplotlib)
        self.assertEqual(sorted(colors), distinct_keys)
        self.assertEqual(len(set(colors.values())), len(distinct_keys))

    def test_rank_and_channel_row_order(self):
        """Test that the ranks stack top to bottom in list order and that P2P rows precede SYMM rows.

        Rows grow upward, so the first rank of the list must own the highest block. Inside a
        rank the computation groups come first, then the P2P channel although its name sorts
        last, then the SYMM channels in ascending name order.
        """
        p2p_channel, symm_channels = "ZPP", ("AR", "DP")

        graph = CrystalGraph()
        graph.set_max_connections(8)
        for channel in (p2p_channel, *symm_channels):
            graph.set_channel_members(channel, [0, 1])
        for rank_id in (0, 1):
            graph.add_span(rank_id, CrystalGSpan(SpanType.PROC, CrystalGSpanId("_proc", 0), duration_ns=ms(10)))
            # The two ends of the P2P channel are the sending and the receiving pattern.
            p2p_pattern = CrystalGCommPattern.P2P_DN if rank_id == 0 else CrystalGCommPattern.P2P_UP
            channel_patterns = [(p2p_channel, p2p_pattern)]
            channel_patterns += [(channel, CrystalGCommPattern.SYMM) for channel in symm_channels]
            for count_field, (channel, pattern) in enumerate(channel_patterns):
                graph.add_span(
                    rank_id,
                    CrystalGSpan(
                        SpanType.COMM,
                        CrystalGSpanId("_comm", count_field),
                        comm_tasks=CrystalGCommTasks(channel, [pattern]),
                        duration_ns=ms(1),
                    ),
                )
        graph.analyze()

        group_keys = _span_group_keys(graph)
        span_rows, rank_rows, rank_start_row, total_rows = _compute_layout(graph, [0, 1], group_keys)

        expected_keys = ["_proc", p2p_channel, *symm_channels]
        self.assertEqual(rank_rows, {0: expected_keys, 1: expected_keys})
        # Rank 1 is drawn below rank 0, so it takes the lowest block; one row separates them.
        self.assertEqual(rank_start_row, {1: 0, 0: len(expected_keys) + 1})
        self.assertEqual(total_rows, 2 * (len(expected_keys) + 1))
        self.assertEqual(
            span_rows,
            {
                (rank_id, span_id): rank_start_row[rank_id] + expected_keys.index(group_keys[(rank_id, span_id)])
                for rank_id, span_id in group_keys
            },
        )


class TestDrawnEdges(unittest.TestCase):
    """Test which synchronization edges a rank selection draws and which chain edges are redundant."""

    def test_sync_edges_follow_the_drawn_members(self):
        """Test that a drawn pair of channel members always keeps one visible synchronization edge.

        The graph pairs adjacent members into a relay chain, so a view that skips the members
        between two drawn ranks would lose every edge of their synchronization. The bridging
        edges of _plotted_sync_edges() close exactly those gaps, while a direct edge with both
        ends drawn is never duplicated by one.
        """
        graph = build_symm_channel_graph()

        for plotted, expected in (
            # Every member drawn: the relay chain itself, without bridging duplicates.
            ({1, 2, 3, 4}, {(1, 2), (2, 3), (3, 4)}),
            # No direct edge survives, so one bridge stands in, over one or two skipped members.
            ({1, 3}, {(1, 3)}),
            ({1, 4}, {(1, 4)}),
            # The direct (1, 2) stays and (2, 4) bridges the gap that rank 3 leaves.
            ({1, 2, 4}, {(1, 2), (2, 4)}),
            # A single drawn member has no visible synchronization partner.
            ({1}, set()),
        ):
            with self.subTest(plotted=sorted(plotted)):
                edges = _plotted_sync_edges(graph, DependencyType.BIDIRECT, plotted)
                self.assertEqual({(v1.rank, v2.rank) for v1, v2 in edges}, expected)
                self.assertEqual(len(edges), len(expected), f"duplicated edges in {edges}")

    def test_weak_edges_bridge_without_joining_the_strong_ones(self):
        """Test that WEAK_BIDIRECT edges are grouped and bridged on their own.

        An async collective under max_connections == 1 pairs its peers weakly, a pairing the
        timing model ignores. Those edges must still bridge a partial view, and must not merge
        anything into the strong grouping, which stays empty here.
        """
        graph = build_symm_channel_graph(max_connections=1, async_op=True)
        self.assertEqual(
            {dep.dep_type for dep in graph.dependencies if dep.dep_type != DependencyType.UNIDIRECT},
            {DependencyType.WEAK_BIDIRECT},
        )

        weak_edges = _plotted_sync_edges(graph, DependencyType.WEAK_BIDIRECT, {1, 3})
        self.assertEqual([(v1.rank, v2.rank) for v1, v2 in weak_edges], [(1, 3)])
        self.assertEqual(_plotted_sync_edges(graph, DependencyType.BIDIRECT, {1, 3}), [])

    def test_start_edge_is_redundant_only_next_to_a_stop_edge_of_the_same_span(self):
        """Test that the edge leaving a START is marked gray exactly when the same span's STOP also leads there.

        Both edges order the same pair of spans and the one from the STOP is the binding one,
        so only then does the START edge carry no constraint of its own. The classification
        reads the dependency set alone, hence no analyze() is needed.
        """
        first, second = CrystalGSpanId("_proc", 0), CrystalGSpanId("_proc", 1)
        graph = CrystalGraph()
        for span_id in (first, second):
            graph.add_span(0, CrystalGSpan(SpanType.PROC, span_id, duration_ns=ms(10)))

        start_edge = (
            CrystalGSpanPoint(0, first, SpanPointType.START),
            CrystalGSpanPoint(0, second, SpanPointType.START),
        )
        stop_edge = (
            CrystalGSpanPoint(0, first, SpanPointType.STOP),
            CrystalGSpanPoint(0, second, SpanPointType.START),
        )

        graph.add_dependency(CrystalGDep(DependencyType.UNIDIRECT, *start_edge))
        self.assertEqual(_compute_gray_deps(graph), set())

        graph.add_dependency(CrystalGDep(DependencyType.UNIDIRECT, *stop_edge))
        self.assertEqual(_compute_gray_deps(graph), {start_edge})


if __name__ == "__main__":
    unittest.main()
