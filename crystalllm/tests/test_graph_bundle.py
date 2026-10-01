# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for channel member resolution and the replay graph bundle."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

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
)
from crystalllm.replay.graph_bundle import CrystalGraphBundle, pp_chain_of_rank
from crystalllm.replay.parser import resolve_channel_members
from crystalllm.replay.util import (
    build_graph_cache_dict,
    cached_op_timings,
    generate_graph_cache,
    load_graph_cache,
    validate_graph_cache,
)

# 8 ranks laid out as 4 PP chains of 2 stages, i.e. a PP rank stride of 4.
WORLD_SIZE = 8
PP_RANK_STRIDE = 4
STEP = 3

P2P_COMM_INDEX = 0
DP_COMM_INDEX = 1

NEW_GROUP_TABLE = {
    "ppg0": [0, 4],
    "ppg1": [1, 5],
    "ppg2": [2, 6],
    "ppg3": [3, 7],
    "dpg0": [0, 1, 2, 3],
    "dpg1": [4, 5, 6, 7],
}


def rank_dict(rank: int) -> dict:
    """Build the captured graph of one rank: compute, PP send/recv, compute, DP all-reduce, compute.

    The five spans are chained by intra-rank unidirect dependencies in that order, so
    the serialized payload matches what merge_dict() consumes. Stage role: the first
    rank of every chain is below PP_RANK_STRIDE, sends "p2p_dn" towards the next
    stage and sits in "dpg0", while later stages send "p2p_up" and sit in "dpg1".
    The P2P channel "ppg{chain}_0:1" names group ranks 0 and 1 of that chain's PP
    group, i.e. its two stages. The graphs carry no measured time at all: durations
    live in the profiler files, and only the P2P exchange is asynchronous.
    """
    chain = pp_chain_of_rank(rank, PP_RANK_STRIDE)
    is_first_stage = rank < PP_RANK_STRIDE
    p2p_pattern = "p2p_dn" if is_first_stage else "p2p_up"
    dp_channel = "dpg0" if is_first_stage else "dpg1"

    def proc(index: int) -> dict:
        return {
            "span_type": "proc",
            "span_id": {"type_field": "_proc", "count_field": index},
            "comm_tasks": None,
        }

    def comm(index: int, channel: str, pattern: str, async_op: bool) -> dict:
        return {
            "span_type": "comm",
            "span_id": {"type_field": "_comm", "count_field": index},
            "comm_tasks": {"channel": channel, "tasks": [pattern], "async_op": async_op},
        }

    spans = {
        "_proc#0": proc(0),
        f"_comm#{P2P_COMM_INDEX}": comm(P2P_COMM_INDEX, f"ppg{chain}_0:1", p2p_pattern, True),
        "_proc#1": proc(1),
        f"_comm#{DP_COMM_INDEX}": comm(DP_COMM_INDEX, dp_channel, "symm", False),
        "_proc#2": proc(2),
    }

    order = ["_proc#0", f"_comm#{P2P_COMM_INDEX}", "_proc#1", f"_comm#{DP_COMM_INDEX}", "_proc#2"]
    dependencies = [
        {
            "dep_type": "unidirect",
            "v1": {"rank": rank, "span_id": spans[order[i]]["span_id"], "point_type": "stop"},
            "v2": {"rank": rank, "span_id": spans[order[i + 1]]["span_id"], "point_type": "start"},
        }
        for i in range(len(order) - 1)
    ]

    return {
        "max_connections": 8,
        "rank_id": rank,
        "spans": spans,
        "span_desc": {},
        "dependencies": dependencies,
    }


def write_gzip_json(path: Path, payload: dict) -> None:
    """Write a gzipped JSON file."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


class TestResolveChannelMembers(unittest.TestCase):
    """Test resolving a graph channel name into its member global ranks."""

    def test_symm_channel_is_the_group_itself(self):
        """Test that a collective channel resolves to the whole process group."""
        self.assertEqual(resolve_channel_members("dpg0", NEW_GROUP_TABLE), [0, 1, 2, 3])

    def test_p2p_channel_maps_group_ranks_to_global_ranks(self):
        """Test that a P2P channel resolves its two group ranks to global ranks."""
        self.assertEqual(resolve_channel_members("ppg2_0:1", NEW_GROUP_TABLE), [2, 6])
        self.assertEqual(resolve_channel_members("dpg0_1:3", NEW_GROUP_TABLE), [1, 3])

    def test_unknown_channel_raises(self):
        """Test that an unresolvable channel is rejected instead of silently guessed.

        The three channels cover every failure mode of the resolver: an unknown
        SYMM group name, a P2P channel on an unknown group, and a P2P channel whose
        group rank is out of range.
        """
        for channel in ("nosuchgroup", "nosuchgroup_0:1", "dpg0_0:99"):
            with self.subTest(channel=channel), self.assertRaises(ValueError) as context:
                resolve_channel_members(channel, NEW_GROUP_TABLE)
            self.assertIn("Cannot resolve members", str(context.exception))


class TestCrystalGraphBundle(unittest.TestCase):
    """Test PP chain discovery, pruning, merging and the replay side queries."""

    def setUp(self):
        """Write a full set of graph and trace files, plus profiler data for one chain only."""
        self._tempdir = tempfile.TemporaryDirectory()
        root = Path(self._tempdir.name)
        self.graph_dir = root / "graph"
        self.trace_dir = root / "trace"
        self.profiler_dir = root / "profiler"
        for directory in (self.graph_dir, self.trace_dir, self.profiler_dir):
            directory.mkdir()

        for rank in range(WORLD_SIZE):
            write_gzip_json(self.graph_dir / f"graph_{rank}.json.gz", {str(STEP): rank_dict(rank)})
            # Empty traces suffice: _load() passes new_group_table explicitly, so the
            # traces are never parsed and only need to exist for _load_chain().
            with gzip.open(self.trace_dir / f"trace_{rank}.log.gz", "wt", encoding="utf-8") as f:
                f.write("")

        # Only chain 2 has runtime profiler data, so it is loaded even though it holds
        # neither a sandbox rank nor a launched emulated rank. The comm durations are
        # keyed by channel: the P2P channel "ppg2_0:1" names the two group ranks of
        # chain 2, and the DP channel differs between the two stages.
        for rank in (2, 6):
            dp_channel = "dpg0" if rank < PP_RANK_STRIDE else "dpg1"
            write_gzip_json(
                self.profiler_dir / f"runtime_profiler_{rank}.json.gz",
                {
                    str(STEP): {
                        "rank_id": rank,
                        "proc": {"0": 5_000_000},
                        "comm": {"ppg2_0:1": [1_000_000], dp_channel: [2_000_000]},
                    }
                },
            )

    def tearDown(self):
        """Remove the temporary dataset."""
        self._tempdir.cleanup()

    def _load(self, **overrides) -> CrystalGraphBundle:
        """Load a bundle for the shared scenario, applying keyword overrides."""
        kwargs = {
            "graph_dir": str(self.graph_dir),
            "trace_dir": str(self.trace_dir),
            "profiler_dir": str(self.profiler_dir),
            "pp_rank_stride": PP_RANK_STRIDE,
            "sandbox_ranks": [0, 4],
            "world_size": WORLD_SIZE,
            "own_rank": 1,
            "launched_ranks": [1, 5],
            "new_group_table": NEW_GROUP_TABLE,
        }
        kwargs.update(overrides)
        return CrystalGraphBundle.load(**kwargs)

    def test_chain_selection(self):
        """Test which PP chains load under the three selection scenarios.

        Default: chain 0 is mandatory through the sandbox ranks 0 and 4, chain 1
        through the launched ranks 1 and 5 (and the own rank 1), chain 2 survives on
        profiler data alone, and chain 3 is pruned. Every emulated rank started
        (launched_ranks=None): all four chains become mandatory and pruning is a
        no-op. Without profiler data: chain 2 loses its only claim to usefulness and
        is pruned alongside chain 3.
        """
        scenarios = {
            "mandatory and profiled": (None, [0, 1, 2, 4, 5, 6]),
            "every emulated rank started": ({"launched_ranks": None}, list(range(WORLD_SIZE))),
            "no profiler data": ({"profiler_dir": None}, [0, 1, 4, 5]),
        }
        for name, (overrides, expected_ranks) in scenarios.items():
            with self.subTest(name):
                bundle = self._load(**(overrides or {}))
                self.assertEqual(sorted(bundle.step_graphs[STEP].ranks), expected_ranks)

    def test_runtime_profiler_durations_are_applied(self):
        """Test that profiler durations land on the matching spans and others stay at 0."""
        bundle = self._load()
        # Every graph file was authored with max_connections=8, so the bundle
        # exposes that single shared value.
        self.assertEqual(bundle.max_connections, 8)
        graph = bundle.step_graphs[STEP]

        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_proc", 0)].duration_ns, 5_000_000)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_comm", P2P_COMM_INDEX)].duration_ns, 1_000_000)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_comm", DP_COMM_INDEX)].duration_ns, 2_000_000)
        self.assertEqual(graph.ranks[1].spans[CrystalGSpanId("_proc", 0)].duration_ns, 0)

    def test_step_of_and_sync_span_group_lookup(self):
        """Test the (rank, trace_comm_index) index and the group it resolves to."""
        bundle = self._load()

        self.assertEqual(bundle.step_of_trace_comm_index(1, DP_COMM_INDEX), STEP)
        self.assertIsNone(bundle.step_of_trace_comm_index(1, 99))
        self.assertIsNone(bundle.sync_span_group_of_point(1, 99))

        dp_group = bundle.sync_span_group_of_point(1, DP_COMM_INDEX)
        # The DP collective of stage 0 spans every loaded emulated rank of that stage
        # (rank 3 belongs to the pruned chain 3); the sandbox member of the group is
        # excluded from the scheduling view.
        self.assertEqual([p.rank for p in dp_group.stop_points], [1, 2])

        p2p_group = bundle.sync_span_group_of_point(1, P2P_COMM_INDEX)
        self.assertEqual([p.rank for p in p2p_group.stop_points], [1, 5])

    def test_launch_time_is_the_latest_group_member(self):
        """Test that the launch time of a group is the latest START over all its members."""
        bundle = self._load()
        graph = bundle.step_graphs[STEP]

        dp_group = bundle.sync_span_group_of_point(1, DP_COMM_INDEX)
        expected = max(graph.get_points_interval(end_point=p) for p in dp_group.start_points)
        self.assertEqual(bundle.sync_span_group_launch_time_ns(STEP, dp_group), expected)
        # Rank 2 is the only profiled member, and its P2P span extends the compute span
        # it depends on, so its chain decides the launch.
        self.assertEqual(expected, 6_000_000)

    def test_anchorable_marks_the_non_sandbox_side(self):
        """Test that the sandbox boundary makes the emulated members of a group anchorable."""
        bundle = self._load()

        self.assertTrue(
            bundle.is_anchorable_point(
                STEP, CrystalGSpanPoint(1, CrystalGSpanId("_comm", DP_COMM_INDEX), SpanPointType.STOP)
            )
        )
        self.assertFalse(
            bundle.is_anchorable_point(
                STEP, CrystalGSpanPoint(0, CrystalGSpanId("_comm", DP_COMM_INDEX), SpanPointType.STOP)
            )
        )

    def test_chain_with_mismatched_steps_is_skipped(self):
        """Test that a chain from a different analysis window is skipped, not merged.

        Rank 6 now only offers step 99 instead of the baseline step, so chain 2 is
        rejected even though it carries profiler data: an incomplete chain is worth
        less than the durations it would contribute.
        """
        write_gzip_json(self.graph_dir / "graph_6.json.gz", {"99": rank_dict(6)})

        bundle = self._load()
        self.assertEqual(sorted(bundle.step_graphs[STEP].ranks), [0, 1, 4, 5])

    def test_broken_own_chain_raises(self):
        """Test that the assistant refuses to start when its own PP chain is unusable.

        Rank 5 is the second stage of the own rank's chain 1, so deleting its graph
        file breaks exactly that chain.
        """
        (self.graph_dir / "graph_5.json.gz").unlink()

        with self.assertRaises(ValueError) as context:
            self._load()
        self.assertIn("own rank", str(context.exception))

    def test_profiler_step_outside_the_graph_raises(self):
        """Test that profiler data of a step the graphs do not contain is reported.

        A step outside the graphs' steps cannot be reconciled by defaulting the
        durations: none of the step's spans exists in any merged graph, so the
        profiler and the graph must come from different analysis windows.
        """
        write_gzip_json(
            self.profiler_dir / "runtime_profiler_2.json.gz",
            {"99": {"rank_id": 2, "proc": {"0": 1}, "comm": {}}},
        )

        with self.assertRaises(ValueError) as context:
            self._load()
        self.assertIn("outside the", str(context.exception))

    def test_profiler_missing_durations_keep_default_zero(self):
        """Test that spans the profiler has no duration for keep the default 0.

        Three coverage gaps at once: the proc dict misses two of the three PROC span
        indices, the P2P channel's duration array is empty, and the DP channel is
        absent from the comm dict altogether. The covered spans load their measured
        durations while every uncovered span keeps its default 0.
        """
        write_gzip_json(
            self.profiler_dir / "runtime_profiler_2.json.gz",
            {
                str(STEP): {
                    "rank_id": 2,
                    "proc": {"0": 5_000_000},
                    "comm": {"ppg2_0:1": []},
                }
            },
        )

        graph = self._load().step_graphs[STEP]
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_proc", 0)].duration_ns, 5_000_000)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_proc", 1)].duration_ns, 0)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_proc", 2)].duration_ns, 0)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_comm", P2P_COMM_INDEX)].duration_ns, 0)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_comm", DP_COMM_INDEX)].duration_ns, 0)

    def test_bundle_requires_one_max_connections(self):
        """Bundling refuses step graphs captured under different (or unset) max_connections."""
        graph_8 = CrystalGraph()
        graph_8.set_max_connections(8)
        graph_1 = CrystalGraph()
        graph_1.set_max_connections(1)
        unset = CrystalGraph()

        bundle = CrystalGraphBundle({STEP: graph_8}, {})
        self.assertEqual(bundle.max_connections, 8)

        with self.assertRaises(ValueError) as context:
            CrystalGraphBundle({STEP: graph_8, STEP + 1: graph_1}, {})
        self.assertIn("max_connections", str(context.exception))

        with self.assertRaises(ValueError) as context:
            CrystalGraphBundle({STEP: unset}, {})
        self.assertIn("max_connections", str(context.exception))

    def test_profiler_extra_data_is_ignored(self):
        """Test that profiler data beyond the graph's spans is tolerated.

        The profiler records every real communication of a group, including ones the
        graph has no span for, so an unknown channel or a longer duration array must
        not fail the load; only the spans are filled. An unknown PROC span index is
        likewise ignored.
        """
        write_gzip_json(
            self.profiler_dir / "runtime_profiler_2.json.gz",
            {
                str(STEP): {
                    "rank_id": 2,
                    # "77" is not a span index of the graph.
                    "proc": {"0": 5_000_000, "77": 9_000_000},
                    # "unmapped" never appears in the graph, and "dpg0" has one
                    # graph span but two measured durations.
                    "comm": {"ppg2_0:1": [1_000_000], "dpg0": [2_000_000, 9_000_000], "unmapped": [7_000_000]},
                }
            },
        )

        graph = self._load().step_graphs[STEP]
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_proc", 0)].duration_ns, 5_000_000)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_comm", P2P_COMM_INDEX)].duration_ns, 1_000_000)
        self.assertEqual(graph.ranks[2].spans[CrystalGSpanId("_comm", DP_COMM_INDEX)].duration_ns, 2_000_000)


class TestPredecessorEventKeys(unittest.TestCase):
    """Test that only ranks which really broadcast can be waited on.

    A collective group marks every non-sandbox member anchorable, but mock bootstrap starts
    only the NCCL topology neighbours of the sandbox; the rest are proxy ranks that never run
    and therefore never announce an arrival.
    """

    SANDBOX_RANK = 0
    STARTED_RANK = 1
    PROXY_RANK = 2

    def setUp(self):
        """Build a bundle with two chained collectives over a sandbox, a started and a proxy rank.

        Both collectives run on two channels with the same three members, and the only
        ordering between them is the intra-rank chain comm0.STOP -> comm1.START on
        each rank; there are no other compute spans or dependencies. Both groups
        therefore see the same members minus the sandbox rank, which makes every
        non-sandbox STOP point anchorable, and the predecessors of a
        second-collective anchor are exactly the first-collective STOP points of the
        same rank.
        """
        graph = CrystalGraph()
        graph.set_max_connections(8)
        for rank in (self.SANDBOX_RANK, self.STARTED_RANK, self.PROXY_RANK):
            for index, channel in ((0, "dpg_first"), (1, "dpg_second")):
                graph.add_span(
                    rank,
                    CrystalGSpan(
                        SpanType.COMM,
                        CrystalGSpanId("_comm", index),
                        CrystalGCommTasks(channel, [CrystalGCommPattern.SYMM]),
                    ),
                )
            graph.add_dependency(
                CrystalGDep(
                    DependencyType.UNIDIRECT,
                    CrystalGSpanPoint(rank, CrystalGSpanId("_comm", 0), SpanPointType.STOP),
                    CrystalGSpanPoint(rank, CrystalGSpanId("_comm", 1), SpanPointType.START),
                )
            )
        members = [self.SANDBOX_RANK, self.STARTED_RANK, self.PROXY_RANK]
        graph.set_channel_members("dpg_first", members)
        graph.set_channel_members("dpg_second", members)
        graph.set_sandbox_ranks([self.SANDBOX_RANK])
        graph.analyze()

        self.graph = graph
        trace_comm_index_steps = {(rank, index): STEP for rank in members for index in (0, 1)}
        self.bundle = CrystalGraphBundle({STEP: graph}, trace_comm_index_steps)

    def test_keys_follow_the_broadcast_set_not_the_anchor_state(self):
        """Test that who is waited on is decided by the broadcast set, not by anchorability.

        The first assertion pins the data-side set: the proxy rank is as anchorable
        as the started rank, because either duration could decide a launch time. The
        remaining assertions apply execution-side sets on top: a member yields a key
        exactly when it actually broadcasts, so the filter is by membership in the
        broadcast set rather than by rank identity.
        """
        anchorable_ranks = {p.rank for p in self.graph.anchorable_points}
        self.assertEqual(anchorable_ranks, {self.STARTED_RANK, self.PROXY_RANK})

        self.assertEqual(
            self.bundle.predecessor_event_keys({self.STARTED_RANK, self.PROXY_RANK}),
            {(self.STARTED_RANK, 0), (self.PROXY_RANK, 0)},
        )
        # Here the proxy rank anchors but does not run; without the filter it would
        # be waited on and block the replay forever.
        self.assertEqual(
            self.bundle.predecessor_event_keys({self.STARTED_RANK}),
            {(self.STARTED_RANK, 0)},
        )


class TestGraphCache(unittest.TestCase):
    """Test the prebuilt launch-timing cache against the bundle queries it replaces."""

    SANDBOX_RANKS = [0, 4]

    def setUp(self):
        """Reuse the bundle fixture of TestCrystalGraphBundle, plus a trace group table.

        The cache builds its bundle without a new_group_table, so the fixture
        writes the NEW_GROUP lines the bundle's fallback parses into one; the
        launched rank the cache builds from is rank 1, so its trace carries the
        table. Profiler durations are added per test, because the interesting
        properties are how the cache tracks them and how it stays equivalent
        across launched-rank sets.
        """
        self._tempdir = tempfile.TemporaryDirectory()
        root = Path(self._tempdir.name)
        self.graph_dir = root / "graph"
        self.trace_dir = root / "trace"
        self.profiler_dir = root / "profiler"
        for directory in (self.graph_dir, self.trace_dir, self.profiler_dir):
            directory.mkdir()

        for rank in range(WORLD_SIZE):
            write_gzip_json(self.graph_dir / f"graph_{rank}.json.gz", {str(STEP): rank_dict(rank)})
            with gzip.open(self.trace_dir / f"trace_{rank}.log.gz", "wt", encoding="utf-8") as f:
                for group_name, ranks in NEW_GROUP_TABLE.items():
                    f.write(
                        f"NEW_GROUP#{{'ranks': {ranks}, 'timeout': datetime.timedelta(seconds=1800), "
                        f"'backend': None, 'group_name': '{group_name}', "
                        f"'group_desc': '{group_name}-nccl'}}\n"
                    )

        self._write_profiler_durations()

    def tearDown(self):
        """Remove the temporary dataset."""
        self._tempdir.cleanup()

    def _write_profiler_durations(self) -> None:
        """Give every rank measured durations: 5 ms per PROC, per-comm-channel values."""
        for rank in range(WORLD_SIZE):
            chain = pp_chain_of_rank(rank, PP_RANK_STRIDE)
            dp_channel = "dpg0" if rank < PP_RANK_STRIDE else "dpg1"
            write_gzip_json(
                self.profiler_dir / f"runtime_profiler_{rank}.json.gz",
                {
                    str(STEP): {
                        "rank_id": rank,
                        "proc": {"0": 5_000_000, "1": 5_000_000, "2": 5_000_000},
                        "comm": {f"ppg{chain}_0:1": [3_000_000], dp_channel: [2_000_000]},
                    }
                },
            )

    def _build_cache(self, **overrides) -> dict:
        """Build a cache dict for the shared scenario, applying keyword overrides."""
        kwargs = {
            "graph_dir": str(self.graph_dir),
            "trace_dir": str(self.trace_dir),
            "profiler_dir": str(self.profiler_dir),
            "pp_rank_stride": PP_RANK_STRIDE,
            "sandbox_ranks": list(self.SANDBOX_RANKS),
            "world_size": WORLD_SIZE,
        }
        kwargs.update(overrides)
        return build_graph_cache_dict(**kwargs)

    def _reference_bundle(self, launched_ranks: list[int]) -> CrystalGraphBundle:
        """Load the bundle an assistant with the given launched ranks would hold."""
        return CrystalGraphBundle.load(
            graph_dir=str(self.graph_dir),
            trace_dir=str(self.trace_dir),
            profiler_dir=str(self.profiler_dir),
            pp_rank_stride=PP_RANK_STRIDE,
            sandbox_ranks=list(self.SANDBOX_RANKS),
            world_size=WORLD_SIZE,
            own_rank=1,
            launched_ranks=launched_ranks,
        )

    def _generate(self, output_dir: Path, **overrides) -> Path:
        """Generate a cache for the shared scenario into output_dir."""
        kwargs = {
            "graph_dir": str(self.graph_dir),
            "trace_dir": str(self.trace_dir),
            "profiler_dir": str(self.profiler_dir),
            "pp_rank_stride": PP_RANK_STRIDE,
            "sandbox_ranks": list(self.SANDBOX_RANKS),
            "world_size": WORLD_SIZE,
            "output_dir": output_dir,
        }
        kwargs.update(overrides)
        return generate_graph_cache(**kwargs)

    def test_cache_matches_the_bundle_queries(self):
        """Test that every cached record equals what the bundle query would answer.

        For each emulated rank and each of its COMM spans, the cache records the
        operation exactly when the bundle calls it anchorable, and the recorded
        launch time then equals the bundle's answer; a non-anchorable operation
        carries no record at all, matching the fact that _replay_planned_op() is
        never reached for it.
        """
        cache = self._build_cache()
        for rank in range(WORLD_SIZE):
            if rank in self.SANDBOX_RANKS:
                # Sandbox ranks never replay, so the cache holds no records for them.
                self.assertNotIn(rank, {op["rank"] for op in cache["ops"]})
                continue
            timings = cached_op_timings(cache, rank)
            bundle = self._reference_bundle(launched_ranks=[rank])
            for index in (P2P_COMM_INDEX, DP_COMM_INDEX):
                anchorable = bundle.is_anchorable_point(
                    STEP, CrystalGSpanPoint(rank, CrystalGSpanId("_comm", index), SpanPointType.STOP)
                )
                self.assertEqual(index in timings, anchorable)
                if anchorable:
                    group = bundle.sync_span_group_of_point(rank, index)
                    self.assertEqual(timings[index], bundle.sync_span_group_launch_time_ns(STEP, group))

    def test_cache_with_explicit_launched_ranks(self):
        """Test that a cache built for a launched set matches that set's bundle.

        _run_emulate passes the mock bootstrap member ranks as the launched set, so
        the cache's bundle and a started assistant's bundle are loaded alike: every
        rank of that set must read the same anchorable set and launch times from the
        cache as its own bundle reports.
        """
        launched = [1, 2, 5]
        cache = self._build_cache(launched_ranks=launched)
        bundle = self._reference_bundle(launched_ranks=launched)
        for rank in launched:
            timings = cached_op_timings(cache, rank)
            for index in (P2P_COMM_INDEX, DP_COMM_INDEX):
                anchorable = bundle.is_anchorable_point(
                    STEP, CrystalGSpanPoint(rank, CrystalGSpanId("_comm", index), SpanPointType.STOP)
                )
                self.assertEqual(index in timings, anchorable)
                if anchorable:
                    group = bundle.sync_span_group_of_point(rank, index)
                    self.assertEqual(timings[index], bundle.sync_span_group_launch_time_ns(STEP, group))

    def test_cache_is_independent_of_launched_ranks(self):
        """Test that the default cache still answers a partial launch's bundle.

        The default cache is built with every emulated rank launched. A rank started
        under a partial launch must still read the same anchorable set and launch
        times from it as its own bundle would report: the chains such a full load
        adds hold neither sandbox ranks, launched ranks nor profiler durations, so
        they cannot shift a launch time (a max over the members' START times) or an
        anchorable flag.
        """
        cache = self._build_cache()
        for rank in range(WORLD_SIZE):
            if rank in self.SANDBOX_RANKS:
                continue
            timings = cached_op_timings(cache, rank)
            # A partial launch: this rank only (plus whatever its own chain forces).
            bundle = self._reference_bundle(launched_ranks=[rank])
            for index in (P2P_COMM_INDEX, DP_COMM_INDEX):
                anchorable = bundle.is_anchorable_point(
                    STEP, CrystalGSpanPoint(rank, CrystalGSpanId("_comm", index), SpanPointType.STOP)
                )
                self.assertEqual(index in timings, anchorable)
                if anchorable:
                    group = bundle.sync_span_group_of_point(rank, index)
                    self.assertEqual(timings[index], bundle.sync_span_group_launch_time_ns(STEP, group))

    def test_generate_load_and_validate_roundtrip(self):
        """Test that a generated cache file loads back and validates only for its scenario."""
        cache_dir = Path(self._tempdir.name) / "graph_cache"
        path = self._generate(cache_dir, timestamp="test")
        self.assertEqual(path.name, "graph_cache_test.json.gz")

        cache = load_graph_cache(path)
        validate_graph_cache(cache, world_size=WORLD_SIZE, sandbox_ranks=self.SANDBOX_RANKS)
        self.assertEqual(cached_op_timings(cache, 1), cached_op_timings(self._build_cache(), 1))

        with self.assertRaises(ValueError):
            validate_graph_cache(cache, world_size=WORLD_SIZE + 1, sandbox_ranks=self.SANDBOX_RANKS)
        with self.assertRaises(ValueError):
            validate_graph_cache(cache, world_size=WORLD_SIZE, sandbox_ranks=[0, 8])

    def test_cache_is_reused_when_inputs_are_unchanged(self):
        """Test that an unchanged invocation reuses the newest cache instead of rebuilding.

        The first call writes a cache; a second call with the same arguments returns
        the very same path. Rewriting a profiler file afterwards invalidates the
        reuse, so the third call builds a fresh cache, as does a call with a
        different sandbox split even though the inputs are untouched again.
        """
        cache_dir = Path(self._tempdir.name) / "graph_cache"
        first = self._generate(cache_dir, timestamp="first")
        self.assertEqual(self._generate(cache_dir, timestamp="second"), first)

        self._write_profiler_durations()
        third = self._generate(cache_dir, timestamp="third")
        self.assertNotEqual(third, first)
        self.assertEqual(third.name, "graph_cache_third.json.gz")

        other_split = self._generate(cache_dir, timestamp="fourth", sandbox_ranks=[1, 5])
        self.assertNotEqual(other_split, third)
        self.assertEqual(other_split.name, "graph_cache_fourth.json.gz")

    def test_launch_time_tracks_profiler_durations(self):
        """Test that the cached launch times follow the runtime profiler durations.

        The DP collective of stage 0 sits on ranks 0-3. Rank 1's spans are
        PROC#0(5ms) -> P2P(3ms) -> PROC#1(5ms) -> DP(2ms), so the collective's
        launch time is the last arrival among its emulated members: rank 1 and 2
        both reach it at 13 ms, while rank 3's shorter path (PROC#0 5ms -> P2P
        3ms -> PROC#1 5ms) arrives at 13 ms as well — raising rank 1's P2P to
        5 ms pushes its arrival to 15 ms and the launch to 15 ms, proving the
        cached value follows the profiler durations.
        """
        write_gzip_json(
            self.profiler_dir / "runtime_profiler_1.json.gz",
            {
                str(STEP): {
                    "rank_id": 1,
                    "proc": {"0": 5_000_000, "1": 5_000_000, "2": 5_000_000},
                    "comm": {"ppg1_0:1": [5_000_000], "dpg0": [2_000_000]},
                }
            },
        )
        cache = self._build_cache()
        launch = cached_op_timings(cache, 2)[DP_COMM_INDEX]
        self.assertEqual(launch, 15_000_000)


if __name__ == "__main__":
    unittest.main()
