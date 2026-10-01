# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for the planning graph: authoring, cross-rank matching, analysis and timing queries."""

import unittest

from crystalllm.planning import (
    CrystalGCommPattern,
    CrystalGCommTasks,
    CrystalGDep,
    CrystalGDurationNotIncreasedError,
    CrystalGraph,
    CrystalGSpan,
    CrystalGSpanId,
    CrystalGSpanPoint,
    CrystalGSyncSpanGroup,
    DependencyType,
    SpanPointType,
    SpanType,
)

# Time scale factor: convert test time units to nanoseconds
TIME_SCALE = 1_000_000


def ms(duration_ms: int) -> int:
    """Convert milliseconds to nanoseconds for test data."""
    return duration_ms * TIME_SCALE


def create_span(
    span_type: SpanType,
    type_field: str,
    count_field: int,
    duration_ms: int,
    comm_tasks: CrystalGCommTasks | None = None,
) -> CrystalGSpan:
    """Helper to create a CrystalGSpan with milliseconds duration."""
    return CrystalGSpan(
        span_type=span_type,
        span_id=CrystalGSpanId(type_field, count_field),
        duration_ns=ms(duration_ms),
        comm_tasks=comm_tasks,
    )


def create_proc(type_field: str, count_field: int, duration_ms: int) -> CrystalGSpan:
    """Helper to create a PROC span."""
    return create_span(SpanType.PROC, type_field, count_field, duration_ms)


def create_comm(
    type_field: str,
    count_field: int,
    duration_ms: int,
    pattern: CrystalGCommPattern,
    channel: str = "default",
    async_op: bool = False,
) -> CrystalGSpan:
    """Helper to create a COMM span."""
    return create_span(
        SpanType.COMM,
        type_field,
        count_field,
        duration_ms,
        comm_tasks=CrystalGCommTasks(channel, [pattern], async_op=async_op),
    )


def add_chain_deps(graph: CrystalGraph, rank: int, spans: list[CrystalGSpan]) -> None:
    """Add sequential UNIDIRECT dependencies between spans on the same rank."""
    for i in range(len(spans) - 1):
        graph.add_dependency(
            CrystalGDep(
                dep_type=DependencyType.UNIDIRECT,
                v1=CrystalGSpanPoint(rank=rank, span_id=spans[i].span_id, point_type=SpanPointType.STOP),
                v2=CrystalGSpanPoint(rank=rank, span_id=spans[i + 1].span_id, point_type=SpanPointType.START),
            )
        )


def point(rank: int, span_id: CrystalGSpanId, point_type: SpanPointType) -> CrystalGSpanPoint:
    """Helper to create a CrystalGSpanPoint."""
    return CrystalGSpanPoint(rank=rank, span_id=span_id, point_type=point_type)


class TestGraphConstruction(unittest.TestCase):
    """Test span and dependency authoring: identifiers, ranks and input validation."""

    def test_span_id_string_roundtrip(self):
        """Test the SpanId "{type}#{count}" string format and its parser."""
        self.assertEqual(str(CrystalGSpanId("compute", 5)), "compute#5")
        self.assertEqual(CrystalGSpanId.from_string("timer#10"), CrystalGSpanId("timer", 10))

        with self.assertRaises(ValueError):
            CrystalGSpanId.from_string("invalid")

    def test_add_span_creates_rank_and_rejects_duplicates(self):
        """Test add_span creates the rank on first use and rejects a duplicate span id."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span = create_proc("compute", 0, 100)
        graph.add_span(0, span)
        self.assertIn(0, graph.ranks)
        self.assertIn(span.span_id, graph.ranks[0].spans)

        with self.assertRaises(ValueError):
            graph.add_span(0, span)

    def test_add_span_desc_appends_and_validates_span(self):
        """Test add_span_desc appends items to an existing span and rejects unknown spans."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span = create_proc("compute", 0, 100)
        graph.add_span(0, span)

        graph.add_span_desc(0, span.span_id, "FORWARD")
        graph.add_span_desc(0, span.span_id, "LAYER-1")
        self.assertEqual(graph.ranks[0].span_desc[span.span_id], ["FORWARD", "LAYER-1"])

        with self.assertRaises(ValueError):
            graph.add_span_desc(0, CrystalGSpanId("missing", 0), "DESC")
        with self.assertRaises(ValueError):
            graph.add_span_desc(99, span.span_id, "DESC")

    def test_add_dependency_deduplicates(self):
        """Test that adding the same UNIDIRECT dependency twice keeps one entry."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span1 = create_proc("compute", 0, 100)
        span2 = create_proc("compute", 1, 50)
        graph.add_span(0, span1)
        graph.add_span(0, span2)

        dep = CrystalGDep(
            dep_type=DependencyType.UNIDIRECT,
            v1=point(0, span1.span_id, SpanPointType.STOP),
            v2=point(0, span2.span_id, SpanPointType.START),
        )
        graph.add_dependency(dep)
        graph.add_dependency(dep)
        self.assertEqual(len(graph.dependencies), 1)

    def test_add_dependency_rejects_invalid_dependencies(self):
        """Test the add_dependency input validation rules."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span0 = create_proc("compute", 0, 100)
        span1 = create_proc("compute", 1, 50)
        graph.add_span(0, span0)
        graph.add_span(1, span1)

        with self.subTest(reason="BIDIRECT edges are derived by the graph, not authored"):
            with self.assertRaises(ValueError) as context:
                graph.add_dependency(
                    CrystalGDep(
                        dep_type=DependencyType.BIDIRECT,
                        v1=point(0, span0.span_id, SpanPointType.STOP),
                        v2=point(1, span1.span_id, SpanPointType.STOP),
                    )
                )
            self.assertIn("cannot be added externally", str(context.exception).lower())

        with self.subTest(reason="UNIDIRECT must stay within one rank"):
            with self.assertRaises(ValueError) as context:
                graph.add_dependency(
                    CrystalGDep(
                        dep_type=DependencyType.UNIDIRECT,
                        v1=point(0, span0.span_id, SpanPointType.STOP),
                        v2=point(1, span1.span_id, SpanPointType.START),
                    )
                )
            self.assertIn("same rank", str(context.exception).lower())

        with self.subTest(reason="referenced span must exist"):
            with self.assertRaises(ValueError) as context:
                graph.add_dependency(
                    CrystalGDep(
                        dep_type=DependencyType.UNIDIRECT,
                        v1=point(0, CrystalGSpanId("missing", 0), SpanPointType.STOP),
                        v2=point(0, span0.span_id, SpanPointType.START),
                    )
                )
            self.assertIn("non-existent span", str(context.exception).lower())


class TestAnalyzeAPI(unittest.TestCase):
    """Test the analyze() preconditions and analysis invalidation."""

    def test_timing_apis_require_analyze(self):
        """Test that every timing API rejects a graph that has not been analyzed."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        graph.add_span(0, create_proc("compute", 0, 100))

        probes = {
            "get_points_interval": graph.get_points_interval,
            "increase_span_duration": lambda: graph.increase_span_duration(0, CrystalGSpanId("compute", 0), ms(200)),
            "sync_span_group_launch_time_ns": lambda: graph.sync_span_group_launch_time_ns(
                CrystalGSyncSpanGroup((), ())
            ),
        }
        for name, probe in probes.items():
            with self.subTest(api=name):
                with self.assertRaises(RuntimeError) as context:
                    probe()
                self.assertIn("analyze", str(context.exception))

    def test_analysis_state_reset_on_graph_change(self):
        """Test that every authoring change invalidates the analysis result."""
        span3 = create_proc("compute", 2, 30)
        mutations = {
            "add_span": lambda graph: graph.add_span(0, span3),
            "add_dependency": lambda graph: graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=point(0, CrystalGSpanId("compute", 0), SpanPointType.STOP),
                    v2=point(0, CrystalGSpanId("compute", 1), SpanPointType.START),
                )
            ),
            "set_channel_members": lambda graph: graph.set_channel_members("DP", [0]),
            "load_span_duration_dict": lambda graph: graph.load_span_duration_dict({0: {"compute": [ms(50), ms(50)]}}),
            "set_sandbox_ranks": lambda graph: graph.set_sandbox_ranks([0]),
        }

        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                graph = CrystalGraph()
                graph.set_max_connections(8)
                graph.add_span(0, create_proc("compute", 0, 100))
                graph.add_span(0, create_proc("compute", 1, 50))
                graph.analyze()
                self.assertTrue(graph._analyzed)

                mutate(graph)
                self.assertFalse(graph._analyzed)


class TestGetInterval(unittest.TestCase):
    """Test get_points_interval functionality for computing time intervals between span points."""

    def test_get_points_interval_same_span(self):
        """Test get_points_interval for START to STOP of the same span."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span = create_proc("compute", 0, 100)
        graph.add_span(0, span)
        graph.analyze()

        start_point = point(0, span.span_id, SpanPointType.START)
        end_point = point(0, span.span_id, SpanPointType.STOP)

        self.assertEqual(graph.get_points_interval(start_point, end_point), ms(100))
        # The same point as both ends collapses into one component, so the interval is 0.
        self.assertEqual(graph.get_points_interval(start_point, start_point), 0)

    def test_get_points_interval_chain_dependencies(self):
        """Test get_points_interval along a chain of sequentially dependent spans."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span1 = create_proc("compute", 0, 100)
        span2 = create_proc("compute", 1, 50)
        span3 = create_proc("compute", 2, 30)
        for span in (span1, span2, span3):
            graph.add_span(0, span)
        add_chain_deps(graph, 0, [span1, span2, span3])
        graph.analyze()

        self.assertEqual(
            graph.get_points_interval(
                point(0, span1.span_id, SpanPointType.START),
                point(0, span2.span_id, SpanPointType.STOP),
            ),
            ms(150),
        )
        self.assertEqual(
            graph.get_points_interval(
                point(0, span1.span_id, SpanPointType.START),
                point(0, span3.span_id, SpanPointType.STOP),
            ),
            ms(180),
        )

    def test_get_points_interval_invalid_points(self):
        """Test get_points_interval rejects points that are not in the graph."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span = create_proc("compute", 0, 100)
        graph.add_span(0, span)
        graph.analyze()

        valid_start = point(0, span.span_id, SpanPointType.START)
        valid_end = point(0, span.span_id, SpanPointType.STOP)
        invalid_point = point(99, span.span_id, SpanPointType.START)

        with self.assertRaises(ValueError):
            graph.get_points_interval(invalid_point, valid_end)

        with self.assertRaises(ValueError):
            graph.get_points_interval(valid_start, invalid_point)

    def test_get_points_interval_independent_spans(self):
        """Test get_points_interval between spans with no path between them.

        Both spans are sources that start at time zero, so the interval from one
        START to the other STOP is just the other span's duration.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span1 = create_proc("compute", 0, 100)
        span2 = create_proc("compute", 1, 50)
        graph.add_span(0, span1)
        graph.add_span(0, span2)
        graph.analyze()

        interval = graph.get_points_interval(
            point(0, span1.span_id, SpanPointType.START),
            point(0, span2.span_id, SpanPointType.STOP),
        )
        self.assertEqual(interval, ms(50))

    def test_get_points_interval_topological_order_error(self):
        """Test get_points_interval when end is before start in topological order."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        s1 = create_proc("compute", 0, 100)
        s2 = create_proc("compute", 1, 50)
        graph.add_span(0, s1)
        graph.add_span(0, s2)
        graph.add_dependency(
            CrystalGDep(
                dep_type=DependencyType.UNIDIRECT,
                v1=point(0, s2.span_id, SpanPointType.STOP),
                v2=point(0, s1.span_id, SpanPointType.START),
            )
        )
        graph.analyze()

        with self.assertRaises(ValueError) as context:
            graph.get_points_interval(
                point(0, s1.span_id, SpanPointType.START),
                point(0, s2.span_id, SpanPointType.STOP),
            )
        self.assertIn("topological earlier", str(context.exception).lower())

    def test_get_points_interval_defaults_to_graph_longest_path(self):
        """Test that get_points_interval() without arguments spans the whole graph.

        The fork-join pattern makes the longest path pick max(100, 80) between the
        setup and the join, so the total is 50 + max(100, 80) + 30 = 180 ms.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        setup = create_proc("setup", 0, 50)
        parallel1 = create_proc("parallel1", 0, 100)
        parallel2 = create_proc("parallel2", 0, 80)
        join = create_proc("join", 0, 30)

        for span in [setup, parallel1, parallel2, join]:
            graph.add_span(0, span)

        # Dependencies: setup -> parallel1, setup -> parallel2, parallel1 -> join, parallel2 -> join
        for parallel in [parallel1, parallel2]:
            graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=point(0, setup.span_id, SpanPointType.STOP),
                    v2=point(0, parallel.span_id, SpanPointType.START),
                )
            )
            graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=point(0, parallel.span_id, SpanPointType.STOP),
                    v2=point(0, join.span_id, SpanPointType.START),
                )
            )

        graph.analyze()
        self.assertEqual(graph.get_points_interval(), ms(180))


class TestUpdateSpanDuration(unittest.TestCase):
    """Test increase_span_duration functionality for modifying span durations."""

    def test_increase_span_duration_functionality(self):
        """Test that increase_span_duration is reflected by every timing view.

        Consecutive increases each rewind the longest-path cache, so every query
        after an update must report the newly grown durations.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span1 = create_proc("compute", 0, 100)
        span2 = create_proc("compute", 1, 50)
        graph.add_span(0, span1)
        graph.add_span(0, span2)
        add_chain_deps(graph, 0, [span1, span2])
        graph.analyze()

        interval_args = (
            point(0, span1.span_id, SpanPointType.START),
            point(0, span2.span_id, SpanPointType.STOP),
        )
        self.assertEqual(graph.get_points_interval(*interval_args), ms(150))

        # Grow span1 past its planning value: the chain interval, the whole-graph
        # longest path and the placement of both spans all follow the update.
        graph.increase_span_duration(0, span1.span_id, ms(200))
        self.assertEqual(graph.get_points_interval(*interval_args), ms(250))
        self.assertEqual(graph.get_points_interval(), ms(250))
        self.assertEqual(graph.get_points_interval(end_point=point(0, span1.span_id, SpanPointType.STOP)), ms(200))
        self.assertEqual(graph.get_points_interval(end_point=point(0, span2.span_id, SpanPointType.START)), ms(200))

        # A second increase lands on the already-relaxed cache and must repair it too.
        graph.increase_span_duration(0, span1.span_id, ms(300))
        self.assertEqual(graph.get_points_interval(*interval_args), ms(350))

    def test_increase_span_duration_repairs_partially_relaxed_cache(self):
        """Test that an increase lands while the cache frontier sits inside the updated span.

        A query that needs only part of the graph leaves the frontier in the middle of the
        topological order. When it comes to rest between a span's START and STOP -- the ordinary
        shape of a collective, whose shared STOP waits for the slowest peer -- the START has
        already pushed the old duration and will never be visited again, so the new duration has
        to reach the STOP component by hand rather than through the sweep.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        # Rank 0 reaches the collective early and rank 1 only after a long chain, which orders
        # the shared STOP of the collective behind every span of rank 1.
        r0_compute = create_proc("compute", 0, 10)
        r0_comm = create_comm("ar", 0, 5, CrystalGCommPattern.SYMM, "DP")
        graph.add_span(0, r0_compute)
        graph.add_span(0, r0_comm)
        add_chain_deps(graph, 0, [r0_compute, r0_comm])

        r1_chain = [create_proc("compute", i, 10) for i in range(3)]
        r1_comm = create_comm("ar", 0, 5, CrystalGCommPattern.SYMM, "DP")
        for span in [*r1_chain, r1_comm]:
            graph.add_span(1, span)
        add_chain_deps(graph, 1, [*r1_chain, r1_comm])
        graph.set_channel_members("DP", [0, 1])
        graph.analyze()

        comm_start = point(0, r0_comm.span_id, SpanPointType.START)
        comm_stop = point(0, r0_comm.span_id, SpanPointType.STOP)

        # Relaxing up to rank 1's second span parks the frontier strictly between the two ends
        # of rank 0's collective, which is the state the repair has to cope with.
        graph.get_points_interval(end_point=point(1, r1_chain[1].span_id, SpanPointType.STOP))
        frontier = graph._comp_time_cached_frontier
        self.assertLess(graph._comp_to_topo_idx[graph._node_to_comp[comm_start]], frontier)
        self.assertLess(frontier, graph._comp_to_topo_idx[graph._node_to_comp[comm_stop]])

        # Rank 0 holds the collective for 100 ms from its 10 ms arrival, which outlasts rank 1
        # arriving at 30 ms and waiting the original 5 ms, so both peers stop at 110 ms.
        graph.increase_span_duration(0, r0_comm.span_id, ms(100))
        self.assertEqual(graph.get_points_interval(end_point=comm_stop), ms(110))
        self.assertEqual(graph.get_points_interval(), ms(110))

    def test_increase_span_duration_error_handling(self):
        """Test increase_span_duration error handling for unknown ranks and spans."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span = create_proc("compute", 0, 100)
        graph.add_span(0, span)
        graph.analyze()

        with self.assertRaises(ValueError):
            graph.increase_span_duration(99, span.span_id, ms(200))
        with self.assertRaises(ValueError):
            graph.increase_span_duration(0, CrystalGSpanId("missing", 0), ms(200))

    def test_increase_span_duration_rejects_non_increasing(self):
        """Test that a duration that does not grow is refused without touching the graph.

        The longest-path cache can only be pushed forward, so a shorter measurement is
        reported as a dedicated error the caller is expected to catch and skip.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        span = create_proc("compute", 0, 100)
        graph.add_span(0, span)
        graph.analyze()

        start = point(0, span.span_id, SpanPointType.START)
        stop = point(0, span.span_id, SpanPointType.STOP)

        for rejected_ms in (100, 60):
            with self.subTest(duration_ms=rejected_ms):
                with self.assertRaises(CrystalGDurationNotIncreasedError) as context:
                    graph.increase_span_duration(0, span.span_id, ms(rejected_ms))
                self.assertIn("does not exceed", str(context.exception))
                self.assertEqual(graph.get_points_interval(start, stop), ms(100))

        # A strictly larger duration still goes through.
        graph.increase_span_duration(0, span.span_id, ms(120))
        self.assertEqual(graph.get_points_interval(start, stop), ms(120))


class TestMergeDict(unittest.TestCase):
    """Test merge_dict functionality for merging dictionaries into graphs."""

    def _create_base_graph(self) -> CrystalGraph:
        """Create a base graph with rank 0 and rank 1.

        The single-member "default" channel keeps the COMM span from producing
        any BIDIRECT dependency, so the dependency count stays predictable.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        r0_compute = create_proc("compute", 0, 100)
        r0_comm = create_comm("comm", 0, 50, CrystalGCommPattern.SYMM)
        graph.add_span(0, r0_compute)
        graph.add_span(0, r0_comm)
        add_chain_deps(graph, 0, [r0_compute, r0_comm])

        r1_compute = create_proc("compute", 0, 80)
        graph.add_span(1, r1_compute)
        graph.set_channel_members("default", [0])
        return graph

    def _create_merge_dict(self) -> dict:
        """Create a dictionary for merging with a new rank."""
        return {
            "max_connections": 8,
            "rank_id": 2,
            "spans": {
                "init#0": {
                    "span_type": "proc",
                    "span_id": {"type_field": "init", "count_field": 0},
                    "comm_tasks": None,
                },
                "work#0": {
                    "span_type": "proc",
                    "span_id": {"type_field": "work", "count_field": 0},
                    "comm_tasks": None,
                },
            },
            "span_desc": {
                "init#0": ["INIT"],
            },
            "dependencies": [
                {
                    "dep_type": "unidirect",
                    "v1": {"rank": 2, "span_id": {"type_field": "init", "count_field": 0}, "point_type": "stop"},
                    "v2": {"rank": 2, "span_id": {"type_field": "work", "count_field": 0}, "point_type": "start"},
                },
            ],
        }

    def test_basic_merge_functionality(self):
        """Test basic merge functionality with new ranks and spans."""
        base_graph = self._create_base_graph()
        merge_data = self._create_merge_dict()

        base_graph.merge_dict(merge_data)

        # Ranks 0 and 1 existed, so the payload adds exactly rank 2 with its two
        # spans and its single dependency.
        self.assertEqual(len(base_graph.ranks), 3)
        self.assertIn(2, base_graph.ranks)
        self.assertIn(CrystalGSpanId("init", 0), base_graph.ranks[2].spans)
        self.assertIn(CrystalGSpanId("work", 0), base_graph.ranks[2].spans)
        self.assertEqual(len(base_graph.dependencies), 2)

    def test_merge_duplicate_handling(self):
        """Test that duplicate span_ids are skipped, keeping the existing content.

        Merging is idempotent per span: an already present span and description
        list are kept, while genuinely new spans land.
        """
        base_graph = self._create_base_graph()
        base_graph.analyze()

        original_duration = base_graph.ranks[0].spans[CrystalGSpanId("compute", 0)].duration_ns
        base_graph.ranks[0].span_desc[CrystalGSpanId("compute", 0)] = ["BASE"]

        # Create dictionary with overlapping data
        overlap_data = {
            "max_connections": 8,
            "rank_id": 0,
            "spans": {
                "compute#0": {
                    "span_type": "proc",
                    "span_id": {"type_field": "compute", "count_field": 0},
                    "comm_tasks": None,
                },
                "new_op#0": {
                    "span_type": "proc",
                    "span_id": {"type_field": "new_op", "count_field": 0},
                    "comm_tasks": None,
                },
            },
            "span_desc": {
                "compute#0": ["OLD", "OLD"],
                "new_op#0": ["NEW", "NEW"],
            },
            "dependencies": [
                {
                    "dep_type": "unidirect",
                    "v1": {"rank": 0, "span_id": {"type_field": "new_op", "count_field": 0}, "point_type": "stop"},
                    "v2": {"rank": 0, "span_id": {"type_field": "compute", "count_field": 0}, "point_type": "start"},
                },
            ],
        }

        base_graph.merge_dict(overlap_data)

        # The existing span and description list were kept, the new span landed
        # with its description list preserved verbatim. Serialized spans carry no
        # duration, so the imported span starts from the zero default.
        self.assertEqual(base_graph.ranks[0].spans[CrystalGSpanId("compute", 0)].duration_ns, original_duration)
        self.assertIn(CrystalGSpanId("new_op", 0), base_graph.ranks[0].spans)
        self.assertEqual(base_graph.ranks[0].spans[CrystalGSpanId("new_op", 0)].duration_ns, 0)
        self.assertEqual(base_graph.ranks[0].span_desc[CrystalGSpanId("compute", 0)], ["BASE"])
        self.assertEqual(base_graph.ranks[0].span_desc[CrystalGSpanId("new_op", 0)], ["NEW", "NEW"])

    def test_merge_graph_integrity(self):
        """Test that a merged graph analyzes and times the merged content.

        Rank 0 runs compute(100ms) -> comm(50ms), rank 1 runs compute(80ms) and
        rank 2 contributes init -> work structure only (serialized spans carry no
        duration), so the longest path is rank 0's 150ms chain -- which only holds
        if the merged dependencies landed and the imported spans did not break
        the analysis.
        """
        base_graph = self._create_base_graph()
        base_graph.merge_dict(self._create_merge_dict())
        base_graph.analyze()

        self.assertEqual(base_graph.get_points_interval(), ms(150))

    def test_merge_error_handling(self):
        """Test error handling for invalid dictionaries during merge."""
        complete = {"max_connections": 8, "rank_id": 0, "spans": {}, "span_desc": {}, "dependencies": []}
        for missing_key in ("max_connections", "rank_id", "spans", "span_desc", "dependencies"):
            with self.subTest(missing_key=missing_key):
                payload = {key: value for key, value in complete.items() if key != missing_key}
                with self.assertRaises(KeyError) as context:
                    CrystalGraph().merge_dict(payload)
                self.assertIn(missing_key, str(context.exception))

        with self.assertRaises(ValueError) as context:
            CrystalGraph().merge_dict({
                "max_connections": 8,
                "rank_id": 0,
                "spans": {},
                "span_desc": {"missing#0": ["x"]},
                "dependencies": [],
            })
        self.assertIn("non-existent span", str(context.exception))

    def test_merge_adopts_first_and_rejects_mismatched_max_connections(self):
        """The first payload fixes max_connections; a different one cannot be merged."""
        graph = CrystalGraph()
        graph.merge_dict(self._create_merge_dict())
        self.assertEqual(graph.max_connections, 8)

        mismatched = {**self._create_merge_dict(), "max_connections": 1}
        with self.assertRaises(ValueError) as context:
            graph.merge_dict(mismatched)
        self.assertIn("max_connections", str(context.exception))


class TestToRankDict(unittest.TestCase):
    """Test to_rank_dict functionality for serializing a rank to dictionary."""

    def _create_graph_with_bidirect_dependency(self) -> CrystalGraph:
        """Create a graph with a rank-local dependency and an auto-generated BIDIRECT dependency."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        r0_compute = create_proc("compute", 0, 100)
        r0_send = create_comm("send", 0, 50, CrystalGCommPattern.P2P_DN)
        r1_recv = create_comm("recv", 0, 30, CrystalGCommPattern.P2P_UP)

        graph.add_span(0, r0_compute)
        graph.add_span(0, r0_send)
        graph.add_span(1, r1_recv)
        add_chain_deps(graph, 0, [r0_compute, r0_send])
        graph.set_channel_members("default", [0, 1])
        graph.analyze()
        return graph

    def test_to_rank_dict_excludes_bidirect_and_roundtrips(self):
        """Test that to_rank_dict emits rank-local data only and survives merge_dict.

        The BIDIRECT edge of the analyzed graph is derived state, so the payload
        must carry only the spans and the UNIDIRECT dependency of the rank, and
        merging the payload into a fresh graph must rebuild exactly that content.
        Serialized spans carry no duration, so the rebuilt spans start from zero
        regardless of the durations they were authored with.
        """
        graph = self._create_graph_with_bidirect_dependency()
        bidirect_deps = [dep for dep in graph.dependencies if dep.dep_type == DependencyType.BIDIRECT]
        self.assertEqual(len(bidirect_deps), 1)

        rank_dict = graph.to_rank_dict(0)

        self.assertEqual(rank_dict["max_connections"], 8)
        self.assertEqual(rank_dict["rank_id"], 0)
        self.assertEqual(set(rank_dict["spans"]), {"compute#0", "send#0"})
        self.assertEqual(len(rank_dict["dependencies"]), 1)
        self.assertEqual(rank_dict["dependencies"][0]["dep_type"], "unidirect")

        receiver = CrystalGraph()
        receiver.merge_dict(rank_dict)
        self.assertEqual(len(receiver.ranks[0].spans), 2)
        self.assertEqual(receiver.ranks[0].spans[CrystalGSpanId("compute", 0)].duration_ns, 0)
        self.assertEqual(receiver.ranks[0].spans[CrystalGSpanId("send", 0)].comm_tasks.channel, "default")
        self.assertEqual(len(receiver.dependencies), 1)

    def test_to_rank_dict_requires_max_connections(self):
        """Export refuses a graph whose max_connections was never set."""
        graph = CrystalGraph()
        graph.add_span(0, create_proc("compute", 0, 100))
        with self.assertRaises(ValueError) as context:
            graph.to_rank_dict(0)
        self.assertIn("set_max_connections", str(context.exception))

        graph.set_max_connections(1)
        payload = graph.to_rank_dict(0)
        self.assertEqual(payload["max_connections"], 1)
        # The payload carries exactly the keys merge_dict() requires of it.
        self.assertEqual(set(payload), {"max_connections", "rank_id", "spans", "span_desc", "dependencies"})

    def test_to_rank_dict_error_handling(self):
        """Test that to_rank_dict raises error for non-existent rank."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        with self.assertRaises(KeyError) as context:
            graph.to_rank_dict(99)
        self.assertIn("99", str(context.exception))


class TestLoadSpanDurationDict(unittest.TestCase):
    """Test load_span_duration_dict functionality for loading span durations from dictionary."""

    def test_load_span_duration_dict(self):
        """Test load_span_duration_dict across multiple ranks, and that it invalidates the analysis.

        The position in a duration list is the count_field of the span it updates,
        so one table rewrites every listed span of every listed rank at once.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        for span in (create_proc("FORWARD", 0, 100), create_proc("FORWARD", 1, 200), create_proc("BACKWARD", 0, 50)):
            graph.add_span(0, span)
        for span in (create_proc("COMPUTE", 0, 50), create_proc("COMPUTE", 1, 30)):
            graph.add_span(1, span)
        graph.analyze()

        graph.load_span_duration_dict({
            0: {"FORWARD": [ms(150), ms(250)], "BACKWARD": [ms(75)]},
            1: {"COMPUTE": [ms(60), ms(40)]},
        })

        self.assertEqual(graph.ranks[0].spans[CrystalGSpanId("FORWARD", 0)].duration_ns, ms(150))
        self.assertEqual(graph.ranks[0].spans[CrystalGSpanId("FORWARD", 1)].duration_ns, ms(250))
        self.assertEqual(graph.ranks[0].spans[CrystalGSpanId("BACKWARD", 0)].duration_ns, ms(75))
        self.assertEqual(graph.ranks[1].spans[CrystalGSpanId("COMPUTE", 0)].duration_ns, ms(60))
        self.assertEqual(graph.ranks[1].spans[CrystalGSpanId("COMPUTE", 1)].duration_ns, ms(40))
        # The condensed graph no longer matches the spans, so it must be rebuilt.
        self.assertFalse(graph._analyzed)

    def test_load_span_duration_dict_error_handling(self):
        """Test error handling for invalid ranks and spans in load_span_duration_dict."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        graph.add_span(0, create_proc("FORWARD", 0, 100))

        with self.assertRaises(ValueError) as context:
            graph.load_span_duration_dict({99: {"FORWARD": [ms(100)]}})
        self.assertIn("Rank 99 not found", str(context.exception))

        # A duration list longer than the issued spans misses every span past its end.
        with self.assertRaises(ValueError) as context:
            graph.load_span_duration_dict({0: {"FORWARD": [ms(100), ms(200)]}})
        self.assertIn("not found", str(context.exception).lower())


class TestCrystalGCommTasks(unittest.TestCase):
    """Test CrystalGCommTasks serialization."""

    def test_comm_tasks_roundtrip(self):
        """Test CrystalGCommTasks to_dict and from_dict preserve channel, tasks and async_op."""
        scenarios = [
            ([CrystalGCommPattern.P2P_DN], True),
            ([CrystalGCommPattern.P2P_DN, CrystalGCommPattern.SYMM], False),
        ]
        for tasks, async_op in scenarios:
            with self.subTest(task_count=len(tasks), async_op=async_op):
                comm_tasks = CrystalGCommTasks("TEST_CH", tasks, async_op=async_op)
                data = comm_tasks.to_dict()
                self.assertEqual(data["channel"], "TEST_CH")
                self.assertEqual(data["tasks"], [task.value for task in tasks])
                self.assertEqual(data["async_op"], async_op)
                self.assertEqual(CrystalGCommTasks.from_dict(data), comm_tasks)

    def test_comm_tasks_missing_tasks_normalize_to_empty(self):
        """Test that a null or missing task list is read as no tasks."""
        for data in ({"channel": "CH", "tasks": None}, {"channel": "CH"}):
            with self.subTest(data=data):
                self.assertEqual(CrystalGCommTasks.from_dict(data), CrystalGCommTasks("CH", []))

        self.assertEqual(CrystalGCommTasks("CH", []).to_dict()["tasks"], [])


class TestCrystalGSpanValidation(unittest.TestCase):
    """Test CrystalGSpan comm_tasks validation and serialization."""

    def test_comm_span_requires_comm_tasks_and_proc_allows_none(self):
        """Test that COMM spans must carry comm_tasks while PROC spans may omit it."""
        with self.assertRaises(ValueError) as context:
            CrystalGSpan(
                span_type=SpanType.COMM,
                span_id=CrystalGSpanId("test", 0),
                duration_ns=100,
                comm_tasks=None,
            )
        self.assertIn("comm_tasks", str(context.exception).lower())

        proc_span = CrystalGSpan(
            span_type=SpanType.PROC,
            span_id=CrystalGSpanId("compute", 0),
            duration_ns=100,
            comm_tasks=None,
        )
        self.assertIsNone(proc_span.comm_tasks)

    def test_span_serialization_roundtrip(self):
        """Test CrystalGSpan to_dict/from_dict roundtrip for PROC and COMM spans.

        The static duration is not part of the payload: a serialized span carries
        no measured time, so the restored span starts from the zero default while
        everything else roundtrips unchanged.
        """
        proc_span = create_proc("compute", 0, 100)
        comm_span = create_comm("send", 0, 50, CrystalGCommPattern.P2P_DN, "PP")
        for span in (proc_span, comm_span):
            with self.subTest(span_type=span.span_type):
                self.assertNotIn("duration_ns", span.to_dict())
                restored = CrystalGSpan.from_dict(span.to_dict())
                span.duration_ns = 0
                self.assertEqual(restored, span)

        # A PROC span serializes comm_tasks as null and reads it back as None.
        self.assertIsNone(proc_span.to_dict()["comm_tasks"])


class TestAutoBIDIRECTDependencies(unittest.TestCase):
    """Test automatic BIDIRECT dependency generation during analyze()."""

    def _get_bidirect_deps(self, graph: CrystalGraph):
        """Helper to get all BIDIRECT dependencies from graph."""
        return [dep for dep in graph.dependencies if dep.dep_type == DependencyType.BIDIRECT]

    def test_build_cross_rank_dependencies_requires_max_connections(self):
        """Matching refuses a graph whose max_connections was never set."""
        graph = CrystalGraph()
        graph.add_span(0, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST"))
        graph.set_channel_members("TEST", [0, 1])

        with self.assertRaises(ValueError) as context:
            graph.build_cross_rank_dependencies()
        self.assertIn("set_max_connections", str(context.exception))

    def test_weak_bidirect_for_async_comm_under_one_connection(self):
        """max_connections=1 pairs an async communication with draw-only WEAK_BIDIRECT.

        The weak edge joins no component, so each peer keeps a single-point
        synchronization group; a synchronous communication, or any communication
        under several connections, still pairs with the strong BIDIRECT and the
        peers share one two-point group.
        """
        scenarios = {
            (1, True): DependencyType.WEAK_BIDIRECT,
            (1, False): DependencyType.BIDIRECT,
            (8, True): DependencyType.BIDIRECT,
        }
        for (max_connections, async_op), expected_type in scenarios.items():
            with self.subTest(max_connections=max_connections, async_op=async_op):
                graph = CrystalGraph()
                graph.set_max_connections(max_connections)
                r0_send = create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST", async_op=async_op)
                r1_recv = create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST", async_op=async_op)
                graph.add_span(0, r0_send)
                graph.add_span(1, r1_recv)
                graph.set_channel_members("TEST", [0, 1])
                graph.analyze()

                pair = [
                    dep
                    for dep in graph.dependencies
                    if dep.dep_type in {DependencyType.BIDIRECT, DependencyType.WEAK_BIDIRECT}
                ]
                self.assertEqual(len(pair), 1)
                self.assertEqual(pair[0].dep_type, expected_type)

                send_group = graph.sync_span_group_of_point(CrystalGSpanPoint(0, r0_send.span_id, SpanPointType.STOP))
                recv_group = graph.sync_span_group_of_point(CrystalGSpanPoint(1, r1_recv.span_id, SpanPointType.STOP))
                if expected_type == DependencyType.BIDIRECT:
                    self.assertEqual([p.rank for p in send_group.stop_points], [0, 1])
                    self.assertIs(send_group, recv_group)
                else:
                    self.assertEqual([p.rank for p in send_group.stop_points], [0])
                    self.assertIsNot(send_group, recv_group)
                    self.assertEqual([p.rank for p in recv_group.stop_points], [1])

    def test_cross_rank_dependencies_generation(self):
        """Test analyze() creates correct BIDIRECT deps for a matching P2P pair."""
        graph = CrystalGraph()
        graph.set_max_connections(8)

        r0_comm = create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST")
        r1_comm = create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST")

        graph.add_span(0, r0_comm)
        graph.add_span(1, r1_comm)
        graph.set_channel_members("TEST", [0, 1])

        graph.analyze()

        bidirect_deps = self._get_bidirect_deps(graph)
        self.assertEqual(len(bidirect_deps), 1)

        dep = bidirect_deps[0]
        self.assertEqual(dep.v1.rank, 0)
        self.assertEqual(dep.v2.rank, 1)
        self.assertEqual(dep.v1.point_type, SpanPointType.STOP)
        self.assertEqual(dep.v2.point_type, SpanPointType.STOP)

    def test_cross_rank_dependencies_opposite_directions_on_channels(self):
        """Test BIDIRECT deps for opposite P2P directions carried by separate channels.

        Matching is confined to one channel, so both channels pair their own
        P2P_DN/P2P_UP sequences even though each rank issues traffic on both.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)

        # Rank 0 has downward traffic on CH1 and upward traffic on CH2.
        graph.add_span(0, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "CH1"))
        graph.add_span(0, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "CH2"))

        # Rank 1 has the matching upward traffic on CH1 and downward traffic on CH2.
        graph.add_span(1, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "CH1"))
        graph.add_span(1, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "CH2"))

        graph.set_channel_members("CH1", [0, 1])
        graph.set_channel_members("CH2", [0, 1])

        graph.analyze()

        # Each channel has one matching P2P_DN/P2P_UP pair, so two BIDIRECT deps are created.
        self.assertEqual(len(self._get_bidirect_deps(graph)), 2)

    def test_bidirect_cleared_on_reanalyze(self):
        """Test that BIDIRECT deps are cleared when analyze() is called again."""
        graph = CrystalGraph()
        graph.set_max_connections(8)

        # Rank 0 and 1: both have send and recv for circular connection
        for rank in [0, 1]:
            graph.add_span(rank, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST"))
            graph.add_span(rank, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST"))

        graph.set_channel_members("TEST", [0, 1])
        graph.analyze()
        # With circular connection between 2 ranks: 2 BIDIRECT deps
        self.assertEqual(len(self._get_bidirect_deps(graph)), 2)

        # Add new spans on rank 2 and re-analyze
        graph.add_span(2, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST"))
        graph.add_span(2, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST"))
        graph.set_channel_members("TEST", [0, 1, 2])

        graph.analyze()

        # The stale 2-edge build was dropped and the 3-rank circular connection rebuilt 3 edges.
        self.assertEqual(len(self._get_bidirect_deps(graph)), 3)

    def test_p2p_matching_with_non_continuous_rank_ids(self):
        """Test that P2P matching follows the member table, not rank id continuity."""
        with self.subTest(scenario="single pair"):
            graph = CrystalGraph()
            graph.set_max_connections(8)
            graph.add_span(0, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST"))
            graph.add_span(5, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST"))
            graph.set_channel_members("TEST", [0, 5])
            graph.analyze()

            bidirect_deps = self._get_bidirect_deps(graph)
            self.assertEqual(len(bidirect_deps), 1)
            self.assertEqual({bidirect_deps[0].v1.rank, bidirect_deps[0].v2.rank}, {0, 5})

        with self.subTest(scenario="circular chain"):
            graph = CrystalGraph()
            graph.set_max_connections(8)
            for rank_id in (0, 2, 5):
                graph.add_span(rank_id, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST"))
                graph.add_span(rank_id, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST"))
            graph.set_channel_members("TEST", [0, 2, 5])
            graph.analyze()

            # Circular connection: 0-2, 2-5, and the wrap-around 5-0.
            bidirect_deps = self._get_bidirect_deps(graph)
            self.assertEqual(len(bidirect_deps), 3)
            rank_pairs = {(min(dep.v1.rank, dep.v2.rank), max(dep.v1.rank, dep.v2.rank)) for dep in bidirect_deps}
            self.assertEqual(rank_pairs, {(0, 2), (0, 5), (2, 5)})

    def test_p2p_pair_completes_together(self):
        """Test that a matched P2P pair shares one completion instant."""
        graph = CrystalGraph()
        graph.set_max_connections(8)

        # Rank 0: compute (100ms) -> send (P2P_DN)
        r0_compute = create_proc("compute", 0, 100)
        r0_send = create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "PP")

        # Rank 1: recv (P2P_UP) -> compute (80ms)
        r1_recv = create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "PP")
        r1_compute = create_proc("compute", 0, 80)

        graph.add_span(0, r0_compute)
        graph.add_span(0, r0_send)
        graph.add_span(1, r1_recv)
        graph.add_span(1, r1_compute)

        add_chain_deps(graph, 0, [r0_compute, r0_send])
        add_chain_deps(graph, 1, [r1_recv, r1_compute])
        graph.set_channel_members("PP", [0, 1])

        graph.analyze()

        # The pair synchronizes once rank 0's compute has finished, so both STOP
        # points complete at the same instant, 100ms after the origin.
        send_stop = point(0, r0_send.span_id, SpanPointType.STOP)
        recv_stop = point(1, r1_recv.span_id, SpanPointType.STOP)
        self.assertEqual(graph.get_points_interval(end_point=send_stop), ms(100))
        self.assertEqual(graph.get_points_interval(end_point=recv_stop), ms(100))
        self.assertEqual(graph.get_points_interval(send_stop, recv_stop), 0)

    def test_mismatched_p2p_task_counts_raise(self):
        """Test that unequal P2P_DN/P2P_UP sequences are rejected with the stride hint."""
        with self.subTest(scenario="unequal counts"):
            graph = CrystalGraph()
            graph.set_max_connections(8)
            graph.add_span(0, create_comm("send", 0, 0, CrystalGCommPattern.P2P_DN, "TEST"))
            graph.add_span(0, create_comm("send", 1, 0, CrystalGCommPattern.P2P_DN, "TEST"))
            graph.add_span(1, create_comm("recv", 0, 0, CrystalGCommPattern.P2P_UP, "TEST"))
            graph.set_channel_members("TEST", [0, 1])

            with self.assertRaises(ValueError) as context:
                graph.analyze()
            message = str(context.exception).lower()
            self.assertIn("mismatched", message)
            self.assertIn("p2p", message)

        with self.subTest(scenario="peer rank not loaded"):
            graph = CrystalGraph()
            graph.set_max_connections(8)
            graph.add_span(0, create_comm("send", 0, 10, CrystalGCommPattern.P2P_DN, "PP_0:1"))
            graph.set_channel_members("PP_0:1", [0, 1])

            with self.assertRaises(ValueError) as context:
                graph.analyze()
            self.assertIn("pp chain boundary", str(context.exception).lower())


class TestChannelsAndSyncSpanGroups(unittest.TestCase):
    """Test channel member tables, synchronization groups and anchorable determination."""

    @staticmethod
    def _symm_graph(ranks: list[int], channel: str = "DP", occurrences: int = 1) -> CrystalGraph:
        """Build a graph where every rank performs `occurrences` collectives on one channel."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        for rank in ranks:
            for occurrence in range(occurrences):
                graph.add_span(rank, create_comm("ar", occurrence, 10 * (rank + 1), CrystalGCommPattern.SYMM, channel))
        graph.set_channel_members(channel, ranks)
        return graph

    def test_analyze_requires_channel_members(self):
        """Test that a channel without a member table is rejected."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        graph.add_span(0, create_comm("ar", 0, 10, CrystalGCommPattern.SYMM, "DP"))

        with self.assertRaises(ValueError) as context:
            graph.analyze()
        self.assertIn("member table", str(context.exception))

    def test_analyze_rejects_rank_outside_channel_members(self):
        """Test that a rank with spans on a channel it does not belong to is rejected."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        graph.add_span(0, create_comm("ar", 0, 10, CrystalGCommPattern.SYMM, "DP"))
        graph.add_span(7, create_comm("ar", 0, 10, CrystalGCommPattern.SYMM, "DP"))
        graph.set_channel_members("DP", [0, 1])

        with self.assertRaises(ValueError) as context:
            graph.analyze()
        self.assertIn("not one of its members", str(context.exception))

    def test_symm_matching_uses_relay_bidirect_chain(self):
        """Test that a four party collective forms one group with three BIDIRECT edges."""
        graph = self._symm_graph([0, 1, 2, 3])
        graph.analyze()

        bidirect_deps = [dep for dep in graph.dependencies if dep.dep_type == DependencyType.BIDIRECT]
        self.assertEqual(len(bidirect_deps), 3)

        self.assertEqual(len(graph.sync_span_groups), 1)
        group = graph.sync_span_groups[0]
        self.assertEqual([p.rank for p in group.stop_points], [0, 1, 2, 3])
        self.assertEqual([p.rank for p in group.start_points], [0, 1, 2, 3])
        self.assertTrue(all(p.point_type == SpanPointType.STOP for p in group.stop_points))
        self.assertTrue(all(p.point_type == SpanPointType.START for p in group.start_points))

    def test_symm_matching_groups_each_occurrence_separately(self):
        """Test that the k-th collective of every member forms its own group."""
        graph = self._symm_graph([0, 1, 2], occurrences=3)
        graph.analyze()

        self.assertEqual(len(graph.sync_span_groups), 3)
        for group in graph.sync_span_groups:
            occurrences = {p.span_id.count_field for p in group.stop_points}
            self.assertEqual(len(occurrences), 1)
            self.assertEqual([p.rank for p in group.stop_points], [0, 1, 2])

    def test_symm_mismatched_counts_raise(self):
        """Test that loaded members with differing collective counts are rejected."""
        graph = CrystalGraph()
        graph.set_max_connections(8)
        graph.add_span(0, create_comm("ar", 0, 10, CrystalGCommPattern.SYMM, "DP"))
        graph.add_span(0, create_comm("ar", 1, 10, CrystalGCommPattern.SYMM, "DP"))
        graph.add_span(1, create_comm("ar", 0, 10, CrystalGCommPattern.SYMM, "DP"))
        graph.set_channel_members("DP", [0, 1])

        with self.assertRaises(ValueError) as context:
            graph.analyze()
        self.assertIn("mismatched symm", str(context.exception).lower())

    def test_sync_span_group_of_point(self):
        """Test sync_span_group_of_point resolves COMM points and rejects all others."""
        graph = self._symm_graph([0, 1, 2])
        graph.add_span(0, create_proc("compute", 0, 100))
        graph.analyze()

        span_id = CrystalGSpanId("ar", 0)
        from_start = graph.sync_span_group_of_point(point(1, span_id, SpanPointType.START))
        from_stop = graph.sync_span_group_of_point(point(1, span_id, SpanPointType.STOP))
        self.assertIs(from_start, from_stop)

        # Computation points and points outside the graph belong to no group.
        self.assertIsNone(graph.sync_span_group_of_point(point(0, CrystalGSpanId("compute", 0), SpanPointType.START)))
        self.assertIsNone(graph.sync_span_group_of_point(point(99, span_id, SpanPointType.STOP)))

    def test_unmatched_comm_span_forms_single_point_group(self):
        """Test that a collective without any loaded peer still forms a group of its own."""
        graph = self._symm_graph([0])
        graph.analyze()

        self.assertEqual(len(graph.sync_span_groups), 1)
        self.assertEqual(len(graph.sync_span_groups[0].stop_points), 1)

    @staticmethod
    def _skewed_collective_graph(arrivals: dict[int, int]) -> CrystalGraph:
        """Build a graph where each rank reaches one collective after its own proc span.

        The differing proc durations make the members arrive at the collective at
        different times, so the launch time is a non-trivial max over the START points.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        for rank, arrival_ms in arrivals.items():
            proc = create_proc("compute", 0, arrival_ms)
            comm = create_comm("ar", 0, 5, CrystalGCommPattern.SYMM, "DP")
            graph.add_span(rank, proc)
            graph.add_span(rank, comm)
            add_chain_deps(graph, rank, [proc, comm])
        graph.set_channel_members("DP", sorted(arrivals))
        return graph

    def test_launch_time_matches_max_over_start_points_for_multi_party_group(self):
        """Test that a >=3 member group's join component gives the max START time."""
        graph = self._skewed_collective_graph({0: 10, 1: 40, 2: 25})
        graph.analyze()

        group = graph.sync_span_group_of_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.START))
        # A three member group owns a join component, and it must equal both the latest
        # arrival (40 ms) and the explicit max over the members' START points.
        self.assertEqual(graph.sync_span_group_launch_time_ns(group), ms(40))
        self.assertEqual(
            graph.sync_span_group_launch_time_ns(group),
            max(graph.get_points_interval(end_point=p) for p in group.start_points),
        )

    def test_launch_time_tracks_online_duration_increase(self):
        """Test that the join component reflects a later arrival after increase_span_duration."""
        graph = self._skewed_collective_graph({0: 10, 1: 40, 2: 25})
        graph.analyze()
        group = graph.sync_span_group_of_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.START))
        self.assertEqual(graph.sync_span_group_launch_time_ns(group), ms(40))

        # Rank 2 now arrives last, so the launch time must follow it up.
        graph.increase_span_duration(2, CrystalGSpanId("compute", 0), ms(70))
        self.assertEqual(graph.sync_span_group_launch_time_ns(group), ms(70))

    def test_launch_time_of_two_party_group_uses_start_points(self):
        """Test that a two party group also owns a join component and returns the max START."""
        graph = self._skewed_collective_graph({0: 10, 1: 40})
        graph.analyze()

        group = graph.sync_span_group_of_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.START))
        self.assertIn(graph._node_to_comp[group.stop_points[0]], graph._sync_group_join_comp)
        self.assertEqual(graph.sync_span_group_launch_time_ns(group), ms(40))

    def test_launch_time_of_single_point_group_is_its_own_start(self):
        """Test that a peerless collective is launched at its own START time."""
        graph = self._skewed_collective_graph({0: 15})
        graph.analyze()

        group = graph.sync_span_group_of_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.START))
        self.assertEqual(graph.sync_span_group_launch_time_ns(group), ms(15))

    def test_sync_span_groups_exclude_sandbox_points(self):
        """Test that the sandbox side of a group is dropped from the scheduling view."""
        graph = self._symm_graph([0, 1, 2])
        graph.set_sandbox_ranks([0])
        graph.analyze()

        # The group keeps only the emulated members, and a sandbox point no longer
        # resolves to any group.
        group = graph.sync_span_group_of_point(point(1, CrystalGSpanId("ar", 0), SpanPointType.START))
        self.assertEqual([p.rank for p in group.stop_points], [1, 2])
        self.assertIsNone(graph.sync_span_group_of_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.STOP)))

    def test_group_of_only_sandbox_ranks_disappears(self):
        """Test that a group whose every member is a sandbox rank is not built at all."""
        graph = self._symm_graph([0, 1])
        graph.set_sandbox_ranks([0, 1])
        graph.analyze()

        self.assertEqual(graph.sync_span_groups, [])

    def test_cross_rank_matching_survives_duration_reload(self):
        """Test that reloading durations re-analyzes without rebuilding BIDIRECT edges."""
        graph = self._symm_graph([0, 1])
        graph.analyze()
        self.assertTrue(graph._cross_rank_built)
        bidirect = [dep for dep in graph.dependencies if dep.dep_type == DependencyType.BIDIRECT]

        # A duration change does not touch the spans or the member tables, so the
        # matching is reused and the rebuilt graph carries the same BIDIRECT edges.
        graph.load_span_duration_dict({0: {"ar": [ms(10)]}, 1: {"ar": [ms(10)]}})
        self.assertTrue(graph._cross_rank_built)
        graph.analyze()
        self.assertEqual([dep for dep in graph.dependencies if dep.dep_type == DependencyType.BIDIRECT], bidirect)

        # A structural change invalidates the matching and rebuilds it.
        graph.add_span(0, create_comm("ar", 1, 5, CrystalGCommPattern.SYMM, "DP"))
        graph.add_span(1, create_comm("ar", 1, 5, CrystalGCommPattern.SYMM, "DP"))
        self.assertFalse(graph._cross_rank_built)
        graph.analyze()
        self.assertEqual(len([dep for dep in graph.dependencies if dep.dep_type == DependencyType.BIDIRECT]), 2)

    def test_anchorable_covers_all_non_sandbox_members_of_a_group(self):
        """Test that a multi-party group crossing the sandbox boundary anchors every outside member."""
        graph = self._symm_graph([0, 1, 2, 3])
        graph.set_sandbox_ranks([0])
        graph.analyze()

        self.assertEqual({p.rank for p in graph.anchorable_points}, {1, 2, 3})
        for rank in (1, 2, 3):
            self.assertTrue(graph.is_anchorable_point(point(rank, CrystalGSpanId("ar", 0), SpanPointType.STOP)))
        self.assertFalse(graph.is_anchorable_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.STOP)))

    def test_group_entirely_outside_sandbox_is_not_anchorable(self):
        """Test that a group between purely emulated ranks yields no anchorable point."""
        graph = self._symm_graph([2, 3])
        graph.set_sandbox_ranks([0, 1])
        graph.analyze()

        self.assertEqual(graph.anchorable_points, set())

    def test_anchor_queries_require_analyzed_graph_and_sandbox(self):
        """Test that anchor queries reject an empty sandbox and an unanalyzed graph."""
        graph = self._symm_graph([0, 1, 2])
        query = point(1, CrystalGSpanId("ar", 0), SpanPointType.STOP)

        # Analyzed but no sandbox registered: anchorability is undefined, not all-False.
        graph.analyze()
        for probe in (graph.is_anchorable_point, graph.get_direct_anchor_predecessors):
            with self.assertRaises(RuntimeError) as context:
                probe(query)
            self.assertIn("set_sandbox_ranks()", str(context.exception))

        # Sandbox registered but the analysis is stale: the derivation lags behind.
        graph.set_sandbox_ranks([0])
        for probe in (graph.is_anchorable_point, graph.get_direct_anchor_predecessors):
            with self.assertRaises(RuntimeError) as context:
                probe(query)
            self.assertIn("analyze()", str(context.exception))

        graph.analyze()
        self.assertTrue(graph.is_anchorable_point(query))
        self.assertFalse(graph.is_anchorable_point(point(0, CrystalGSpanId("ar", 0), SpanPointType.STOP)))

    def test_direct_anchor_predecessors_follow_the_graph_order(self):
        """Test that the anchor frontier of a collective is the latest earlier collective.

        Waiting for the later collective implies the earlier one, so an anchor
        that a later anchor already covers is dropped from the minimal set: the
        third collective waits for the second only, never for the first.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        for rank in (0, 1):
            occurrences = [create_comm("ar", i, 10, CrystalGCommPattern.SYMM, "DP") for i in range(3)]
            for comm_span in occurrences:
                graph.add_span(rank, comm_span)
            add_chain_deps(graph, rank, occurrences)
        graph.set_channel_members("DP", [0, 1])
        graph.set_sandbox_ranks([0])
        graph.analyze()

        def anchor_stop(occurrence: int) -> CrystalGSpanPoint:
            return point(1, CrystalGSpanId("ar", occurrence), SpanPointType.STOP)

        # The first collective starts at the origin and has no anchor before it.
        self.assertEqual(
            graph.get_direct_anchor_predecessors(point(1, CrystalGSpanId("ar", 0), SpanPointType.START)),
            [],
        )
        self.assertEqual(
            graph.get_direct_anchor_predecessors(point(1, CrystalGSpanId("ar", 1), SpanPointType.START)),
            [anchor_stop(0)],
        )
        # The first anchor is shadowed by the second, so only the second remains.
        self.assertEqual(
            graph.get_direct_anchor_predecessors(point(1, CrystalGSpanId("ar", 2), SpanPointType.START)),
            [anchor_stop(1)],
        )


if __name__ == "__main__":
    unittest.main()
