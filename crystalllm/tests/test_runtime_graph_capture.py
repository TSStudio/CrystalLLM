# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for Crystal runtime graph capture lifecycle.

This file focuses on the local state machine in ``CrystalRuntimeContext`` after
Graph capture was changed from marker-created spans to communication-boundary
spans.  It intentionally does not launch real distributed communication.  The
covered behavior is:

- graph capture is active only in the right-open analysis window ``[S, E)``;
- ``step_start()``/``step_stop()`` create per-step ``_proc`` spans;
- synchronous communication cuts the current proc span and makes the next proc
  span depend on the communication span;
- the next proc span stays pending across the real communication launch and is
  only started afterwards (``cond_start_proc_span_after_comm``), so a
  blocking synchronous launch is never timed inside it; capture mode, whose fake
  work launches nothing, starts the pending span right after the cut;
- same-channel communication spans keep an explicit channel-ordering dependency
  whenever max_connections is not 1; under 1 the issue-wait durations already
  contain the channel serialization, so the chain is left out and a synchronous
  communication keeps its order through the proc-span chain;
- profiler-only span record (graph capture disabled) still allocates logical span
  IDs for wait bookkeeping but persists nothing into a graph;
- coalescing mode reserves one shared communication trace index when the manager is
  entered, cuts the computation span there, collects tasks in order, and materializes
  one communication span at manager end;
- asynchronous communication allows following compute to continue until
  ``wait()`` adds the communication span as a dependency;
- ``cond_crystal_graph_mark`` records descriptions through the pending
  ``span_desc`` path when the touched proc span has not been materialized yet;
- under ``CUDA_DEVICE_MAX_CONNECTIONS=1`` the profiler times no ProcessGroup at
  all, so every communication gets an issue-wait measurement: synchronous
  communications settle at the proc-span start after the launch, asynchronous
  ones at the proc-span start after the wait, and a coalescing manager measures
  from its entry stop event; an asynchronous communication that is never waited
  inside its step is recorded as never waited at the step boundary, keeping its
  issue position with duration 0.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from crystalllm.planning import CrystalGCommPattern, CrystalGCommTasks, CrystalGSpanId, SpanPointType
from crystalllm.runtime import cond_crystal_graph_mark
from crystalllm.runtime.context import CrystalRuntimeContext
from crystalllm.torch.collective_handler import (
    crystal_c10d_coalescing_manager_handler,
    crystal_collectives_handler,
    crystal_runtime_profiler_coalescing_manager_handler,
    crystal_runtime_profiler_collectives_handler,
)
from crystalllm.torch.fake import CrystalFakeWork, CrystalProfiledWork


def reset_context_singleton() -> None:
    """Reset the RuntimeContext singleton so each test owns a fresh state."""
    CrystalRuntimeContext._instance = None
    CrystalRuntimeContext._initialized = False


def step_graph(ctx: CrystalRuntimeContext, step: int):
    """Return the captured graph for a step from the RuntimeContext test state."""
    return ctx._captured_graph_per_steps[step]


def dependency_points(graph) -> set[tuple[CrystalGSpanId, SpanPointType, CrystalGSpanId, SpanPointType]]:
    """Return dependency endpoints in a compact comparable form."""
    return {(dep.v1.span_id, dep.v1.point_type, dep.v2.span_id, dep.v2.point_type) for dep in graph.dependencies}


class TestRuntimeGraphCapture(unittest.TestCase):
    """Validate graph-capture boundaries without starting distributed jobs."""

    def setUp(self) -> None:
        """Create an isolated runtime environment for one test case."""
        reset_context_singleton()
        self._tmpdir = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(
            os.environ,
            {
                "RANK": "0",
                "LOCAL_WORLD_SIZE": "8",
                "CRYSTAL_LOG_DIR": self._tmpdir.name,
                "CRYSTAL_GRAPH_CAPTURING": "true",
                "CRYSTAL_ANALYSIS_STEP_START": "2",
                "CRYSTAL_ANALYSIS_STEP_END": "3",
            },
            clear=False,
        )
        self._env.start()

    def tearDown(self) -> None:
        """Release environment overrides and singleton state."""
        self._env.stop()
        self._tmpdir.cleanup()
        reset_context_singleton()

    def test_graph_capture_only_inside_right_open_analysis_window(self) -> None:
        """Capture only step 2 when the configured analysis window is [2, 3)."""
        ctx = CrystalRuntimeContext()

        ctx.step_start()
        self.assertFalse(ctx.is_graph_capture_active)
        ctx.step_stop()
        self.assertNotIn(1, ctx._captured_graph_per_steps)

        ctx.step_start()
        self.assertTrue(ctx.is_graph_capture_active)
        ctx.step_stop()
        self.assertIn(2, ctx._captured_graph_per_steps)

        ctx.step_start()
        self.assertFalse(ctx.is_graph_capture_active)
        ctx.step_stop()
        self.assertNotIn(3, ctx._captured_graph_per_steps)

    def test_sync_comm_slices_proc_span_and_sets_dependencies(self) -> None:
        """Synchronous comm should become exactly proc -> comm -> next-proc."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()  # step 1, outside window
        ctx.step_stop()
        ctx.step_start()  # step 2, active

        comm_span_id = ctx.cond_record_comm_span(
            10,
            CrystalGCommTasks(channel="pg", tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
        )
        # The real launch of the communication happens between the cut and the
        # pending proc span start.
        ctx.cond_start_proc_span_after_comm()
        ctx.step_stop()

        graph = step_graph(ctx, 2)
        self.assertFalse(graph.ranks[0].spans[comm_span_id].comm_tasks.async_op)
        self.assertEqual(
            set(graph.ranks[0].spans),
            {CrystalGSpanId("_proc", 0), CrystalGSpanId("_comm", 10), CrystalGSpanId("_proc", 1)},
        )
        self.assertEqual(
            dependency_points(graph),
            {
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_comm", 10), SpanPointType.START),
                (CrystalGSpanId("_comm", 10), SpanPointType.STOP, CrystalGSpanId("_proc", 1), SpanPointType.START),
            },
        )

    def test_async_wait_slices_proc_span_and_rejects_duplicate_wait(self) -> None:
        """Async comm should only become a dependency after wait()."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()

        comm_span_id = ctx.cond_record_comm_span(
            11,
            CrystalGCommTasks(channel="pg", tasks=[CrystalGCommPattern.SYMM]),
            async_op=True,
        )
        ctx.cond_start_proc_span_after_comm()
        ctx.cond_comm_span_wait(comm_span_id)
        with self.assertRaises(RuntimeError):
            ctx.cond_comm_span_wait(comm_span_id)
        ctx.step_stop()

        graph = step_graph(ctx, 2)
        self.assertEqual(
            set(graph.ranks[0].spans),
            {
                CrystalGSpanId("_proc", 0),
                CrystalGSpanId("_comm", 11),
                CrystalGSpanId("_proc", 1),
                CrystalGSpanId("_proc", 2),
            },
        )
        # The launch shape of the communication is graph data: the async span
        # records its async_op on the comm tasks it carries.
        self.assertTrue(graph.ranks[0].spans[CrystalGSpanId("_comm", 11)].comm_tasks.async_op)
        # Compute between launch and wait keeps running as _proc 1 chained on
        # _proc 0 rather than on the comm span: this is what distinguishes async
        # comm from sync comm. Only wait() makes the following proc span (_proc 2)
        # depend on the comm span.
        self.assertEqual(
            dependency_points(graph),
            {
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_comm", 11), SpanPointType.START),
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_proc", 1), SpanPointType.START),
                (CrystalGSpanId("_proc", 1), SpanPointType.STOP, CrystalGSpanId("_proc", 2), SpanPointType.START),
                (CrystalGSpanId("_comm", 11), SpanPointType.STOP, CrystalGSpanId("_proc", 2), SpanPointType.START),
            },
        )

    def test_graph_mark_adds_pending_description_to_proc_span(self) -> None:
        """Graph mark should attach desc to a proc span materialized later."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        with cond_crystal_graph_mark("FWD"):
            pass
        ctx.step_stop()

        graph = step_graph(ctx, 2)
        self.assertEqual(graph.ranks[0].span_desc[CrystalGSpanId("_proc", 0)], ["FWD"])

    def test_graph_mark_covers_proc_span_started_after_mark_enter(self) -> None:
        """Graph mark should cover proc spans that start after mark entry."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_record_comm_span(
            20,
            CrystalGCommTasks(channel="pg", tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
        )
        ctx.cond_start_proc_span_after_comm()
        with cond_crystal_graph_mark("LATE"):
            ctx.cond_record_comm_span(
                21,
                CrystalGCommTasks(channel="pg", tasks=[CrystalGCommPattern.SYMM]),
                async_op=False,
            )
            ctx.cond_start_proc_span_after_comm()
        ctx.step_stop()

        graph = step_graph(ctx, 2)
        self.assertNotIn(CrystalGSpanId("_proc", 0), graph.ranks[0].span_desc)
        self.assertEqual(graph.ranks[0].span_desc[CrystalGSpanId("_proc", 1)], ["LATE"])
        self.assertEqual(graph.ranks[0].span_desc[CrystalGSpanId("_proc", 2)], ["LATE"])

    def test_same_channel_comm_spans_have_ordering_dependency(self) -> None:
        """Only comm spans on the same channel should be explicitly ordered."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()

        ctx.cond_record_comm_span(
            30,
            CrystalGCommTasks(channel="same_pg", tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
        )
        ctx.cond_start_proc_span_after_comm()
        ctx.cond_record_comm_span(
            31,
            CrystalGCommTasks(channel="same_pg", tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
        )
        ctx.cond_start_proc_span_after_comm()
        ctx.cond_record_comm_span(
            32,
            CrystalGCommTasks(channel="other_pg", tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
        )
        ctx.cond_start_proc_span_after_comm()
        ctx.step_stop()

        graph = step_graph(ctx, 2)
        self.assertEqual(
            set(graph.ranks[0].spans),
            {
                CrystalGSpanId("_proc", 0),
                CrystalGSpanId("_comm", 30),
                CrystalGSpanId("_proc", 1),
                CrystalGSpanId("_comm", 31),
                CrystalGSpanId("_proc", 2),
                CrystalGSpanId("_comm", 32),
                CrystalGSpanId("_proc", 3),
            },
        )
        # The ordering edge is channel-scoped: the two comm spans on "same_pg"
        # get an explicit STOP-to-START edge (comm 30 -> comm 31), while no edge
        # at all runs from either of them to the "other_pg" comm span.
        self.assertEqual(
            dependency_points(graph),
            {
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_comm", 30), SpanPointType.START),
                (CrystalGSpanId("_comm", 30), SpanPointType.STOP, CrystalGSpanId("_proc", 1), SpanPointType.START),
                (CrystalGSpanId("_proc", 1), SpanPointType.STOP, CrystalGSpanId("_comm", 31), SpanPointType.START),
                (CrystalGSpanId("_comm", 30), SpanPointType.STOP, CrystalGSpanId("_comm", 31), SpanPointType.START),
                (CrystalGSpanId("_comm", 31), SpanPointType.STOP, CrystalGSpanId("_proc", 2), SpanPointType.START),
                (CrystalGSpanId("_proc", 2), SpanPointType.STOP, CrystalGSpanId("_comm", 32), SpanPointType.START),
                (CrystalGSpanId("_comm", 32), SpanPointType.STOP, CrystalGSpanId("_proc", 3), SpanPointType.START),
            },
        )

    def test_same_channel_ordering_dependency_skipped_under_one_connection(self) -> None:
        """Under max_connections=1 the channel chain is not drawn between comm spans.

        The issue-wait durations already contain the channel serialization, so
        chaining the channel's spans would double-count the overlap of consecutive
        communications. A synchronous communication still orders what follows it
        through the proc-span chain (its next proc span links on the comm span),
        while an asynchronous one chains on the ended computation span, leaving
        the two communications free to overlap.
        """
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}, clear=False):
            ctx = CrystalRuntimeContext()
            ctx.step_start()  # step 1, outside window
            ctx.step_stop()
            ctx.step_start()  # step 2, active; step_start opened _proc 0

            # One pair of same-channel communications per launch shape. The proc
            # span counter advances by two per pair: the one each communication
            # cut and the one the following after_comm opened.
            proc_index = 0
            for async_op, channel, count_field in ((False, "sync_pg", 40), (True, "async_pg", 42)):
                with self.subTest(async_op=async_op):
                    first = ctx.cond_record_comm_span(
                        count_field,
                        CrystalGCommTasks(channel=channel, tasks=[CrystalGCommPattern.SYMM]),
                        async_op=async_op,
                    )
                    ctx.cond_start_proc_span_after_comm()
                    second = ctx.cond_record_comm_span(
                        count_field + 1,
                        CrystalGCommTasks(channel=channel, tasks=[CrystalGCommPattern.SYMM]),
                        async_op=async_op,
                    )
                    ctx.cond_start_proc_span_after_comm()

                    dep_pairs = dependency_points(step_graph(ctx, 2))
                    # No channel-ordering edge between the two communication spans.
                    self.assertNotIn((first, SpanPointType.STOP, second, SpanPointType.START), dep_pairs)
                    next_proc = CrystalGSpanId("_proc", proc_index + 1)
                    if async_op:
                        # The following proc span chains on the ended computation
                        # span, not on the communication: the pair stays free to
                        # overlap.
                        self.assertNotIn((first, SpanPointType.STOP, next_proc, SpanPointType.START), dep_pairs)
                    else:
                        # The following proc span chains on the communication span,
                        # keeping the channel order through the proc-span chain.
                        self.assertIn((first, SpanPointType.STOP, next_proc, SpanPointType.START), dep_pairs)
                    proc_index += 2
            ctx.step_stop()

    def test_profiler_only_span_record_returns_ids_without_persisting_graph(self) -> None:
        """Profiler-only span record allocates logical IDs but writes no graph."""
        with (
            mock.patch.dict(
                os.environ,
                {"CRYSTAL_GRAPH_CAPTURING": "false", "ENABLE_CRYSTAL_RUNTIME_PROFILER": "true"},
            ),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.Event"),
        ):
            ctx = CrystalRuntimeContext()
            self.assertFalse(ctx.is_graph_capture_configured)
            ctx.step_start()  # step 1, outside window
            ctx.step_stop()
            ctx.step_start()  # step 2, profiler window active
            self.assertTrue(ctx.is_span_record_active)

            comm_span_id = ctx.cond_record_comm_span(
                50,
                CrystalGCommTasks(channel="pg", tasks=[CrystalGCommPattern.SYMM]),
                async_op=True,
            )
            ctx.cond_start_proc_span_after_comm()

            # The logical ID still drives async wait bookkeeping ...
            ctx.cond_comm_span_wait(comm_span_id)
            with self.assertRaises(RuntimeError):
                ctx.cond_comm_span_wait(comm_span_id)
            ctx.step_stop()

    def test_coalescing_collects_tasks_and_materializes_one_comm_span(self) -> None:
        """Coalescing should cut the proc span at start and create one comm span at end."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()
        reserved_trace_index = ctx.last_trace_comm_index

        # The coalesced operations run inside the manager scope, so computation must
        # already be cut before the first task is collected.
        self.assertFalse(ctx._span_proc_open)
        ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.SYMM)
        ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.P2P_UP)
        comm_span_id = ctx.cond_end_comm_span_coalescing("coalesced_pg", async_op=False)
        # The manager's real launch happens between the cut and the pending span start.
        ctx.cond_start_proc_span_after_comm()
        ctx.step_stop()

        graph = step_graph(ctx, 2)
        self.assertEqual(comm_span_id, CrystalGSpanId("_comm", reserved_trace_index))
        self.assertEqual(
            graph.ranks[0].spans[comm_span_id].comm_tasks.tasks,
            [CrystalGCommPattern.SYMM, CrystalGCommPattern.P2P_UP],
        )
        # The whole manager is exactly one comm span: the proc span cut at manager
        # start is its predecessor, and the pending span started after the launch
        # continues on it.
        self.assertEqual(
            set(graph.ranks[0].spans),
            {CrystalGSpanId("_proc", 0), comm_span_id, CrystalGSpanId("_proc", 1)},
        )
        self.assertEqual(
            dependency_points(graph),
            {
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, comm_span_id, SpanPointType.START),
                (comm_span_id, SpanPointType.STOP, CrystalGSpanId("_proc", 1), SpanPointType.START),
            },
        )

    def test_coalescing_reserves_one_trace_index_without_span_record(self) -> None:
        """The manager's trace index must be reserved even outside the analysis window."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()  # step 1, outside [2, 3)
        self.assertFalse(ctx.is_span_record_active)

        ctx.cond_start_comm_span_coalescing()
        self.assertTrue(ctx.is_comm_trace_counter_frozen)
        reserved_trace_index = ctx.last_trace_comm_index
        # The c10d trace numbers every operation of the manager with that one index.
        self.assertEqual(ctx.reserve_trace_comm_index(), reserved_trace_index)
        self.assertIsNone(ctx.cond_end_comm_span_coalescing("pg", async_op=False))
        self.assertFalse(ctx.is_comm_trace_counter_frozen)
        self.assertEqual(ctx.reserve_trace_comm_index(), reserved_trace_index + 1)
        ctx.step_stop()

    def test_coalescing_end_outside_manager_scope_raises(self) -> None:
        """Closing a manager that was never opened must fail loudly."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()

        with self.assertRaises(RuntimeError):
            ctx.cond_end_comm_span_coalescing("pg", async_op=False)

    def test_capture_coalescing_manager_starts_proc_span_without_real_launch(self) -> None:
        """The capture-mode manager handler must start the pending proc span right after its cut."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()
        reserved_trace_index = ctx.last_trace_comm_index
        ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.SYMM)

        group = mock.Mock()
        group.group_name = "coalesced_pg"
        group.group_desc = "coalesced_pg"
        work = crystal_c10d_coalescing_manager_handler(group, async_ops=True)

        self.assertIsInstance(work, CrystalFakeWork)
        # Fake work launches nothing, so the pending proc span has already started.
        self.assertTrue(ctx._span_proc_open)
        self.assertIsNone(ctx._span_pending_proc_prev_ids)
        self.assertEqual(work._linked_comm_span_id, CrystalGSpanId("_comm", reserved_trace_index))

    def test_runtime_profiler_coalescing_without_work_records_slot(self) -> None:
        """A synchronous coalescing manager returns no Work, so its comm span alone must justify the slot."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()
        ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.SYMM)

        group = mock.Mock()
        group.group_name = "coalesced_pg"
        group.group_desc = "coalesced_pg"
        with mock.patch.object(ctx, "cond_record_profiler_comm_slot") as record_slot:
            crystal_runtime_profiler_coalescing_manager_handler(group, async_ops=False, launch_func=lambda: None)

        # The manager's real collectives are synchronous, so launch_func returns no
        # Work at all; the materialized comm span alone must still record one slot to
        # keep the slot/completion sequences aligned.
        record_slot.assert_called_once_with(group, "coalesced_pg")

    def test_runtime_profiler_coalescing_without_tasks_records_slot(self) -> None:
        """A coalescing manager without tasks still needs a slot for its work."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()

        group = mock.Mock()
        group.group_name = "local_pg"
        group.group_desc = "local_pg"
        work = object()
        with mock.patch.object(ctx, "cond_record_profiler_comm_slot") as record_slot:
            returned = crystal_runtime_profiler_coalescing_manager_handler(
                group, async_ops=True, launch_func=lambda: work
            )
        ctx.step_stop()

        # The manager produced work but collected no task, so no comm span exists and
        # the work is returned unwrapped; the work still reports a completion, which
        # needs its slot under the group's channel.
        record_slot.assert_called_once_with(group, "local_pg")
        self.assertIs(returned, work)
        graph = step_graph(ctx, 2)
        self.assertEqual(set(graph.ranks[0].spans), {CrystalGSpanId("_proc", 0), CrystalGSpanId("_proc", 1)})

    def test_runtime_profiler_rendezvous_coalescing_creates_no_comm_span(self) -> None:
        """A rendezvous coalescing manager must drop its tasks and only resume computation."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()
        ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.SYMM)

        group = mock.Mock()
        group.group_name = "rendezvous_pg"
        group.group_desc = "_crystal_replay_rendezvous_pg"
        work = object()
        with mock.patch.object(ctx, "cond_record_profiler_comm_slot") as record_slot:
            returned = crystal_runtime_profiler_coalescing_manager_handler(
                group, async_ops=True, launch_func=lambda: work
            )
        ctx.step_stop()

        record_slot.assert_not_called()
        self.assertIs(returned, work)
        graph = step_graph(ctx, 2)
        # The collected tasks are dropped by the rendezvous channel=None path, so
        # the graph keeps only the computation spans of the step.
        self.assertEqual(set(graph.ranks[0].spans), {CrystalGSpanId("_proc", 0), CrystalGSpanId("_proc", 1)})

    def test_runtime_profiler_coalescing_launches_work_before_proc_span_start(self) -> None:
        """The real launch must run after the cut and before the pending proc span starts."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()
        reserved_trace_index = ctx.last_trace_comm_index
        ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.SYMM)

        group = mock.Mock()
        group.group_name = "coalesced_pg"
        group.group_desc = "coalesced_pg"
        work = object()
        seen = {}

        def launch_func():
            # The real coalesced kernels are launched here: the manager scope must be
            # closed and the span chain already cut, but the next proc span must still
            # be pending so this possibly blocking launch cannot leak into it.
            seen["coalescing"] = ctx._span_coalescing
            seen["proc_span_open"] = ctx._span_proc_open
            seen["pending_proc_span"] = ctx._span_pending_proc_prev_ids
            return work

        with mock.patch.object(ctx, "cond_record_profiler_comm_slot") as record_slot:
            returned = crystal_runtime_profiler_coalescing_manager_handler(
                group, async_ops=True, launch_func=launch_func
            )

        self.assertFalse(seen["coalescing"])
        self.assertFalse(seen["proc_span_open"])
        # An async-typed manager lets compute overlap until wait(), so the pending
        # proc span chains on the proc span ended at manager start, not the comm span.
        self.assertEqual(seen["pending_proc_span"], [CrystalGSpanId("_proc", 0)])
        # After the launch returns, the pending proc span has been started.
        self.assertTrue(ctx._span_proc_open)
        self.assertIsNone(ctx._span_pending_proc_prev_ids)
        record_slot.assert_called_once_with(group, "coalesced_pg")
        self.assertIsInstance(returned, CrystalProfiledWork)
        self.assertIs(returned._work, work)
        self.assertEqual(returned._linked_comm_span_id, CrystalGSpanId("_comm", reserved_trace_index))
        ctx.step_stop()

    def test_runtime_profiler_coalescing_fallback_proc_span_starts_before_launch(self) -> None:
        """Without collected tasks the next proc span starts right away, before the launch."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        ctx.cond_start_comm_span_coalescing()

        group = mock.Mock()
        group.group_name = "local_pg"
        group.group_desc = "local_pg"
        seen = {}

        def launch_func():
            # The manager collected no task, so no comm span exists and there is no
            # communication whose launch time must stay out of the next proc span:
            # it is already open while this runs; this manager produced no work at
            # all.
            seen["proc_span_open"] = ctx._span_proc_open
            seen["pending_proc_span"] = ctx._span_pending_proc_prev_ids

        with mock.patch.object(ctx, "cond_record_profiler_comm_slot") as record_slot:
            crystal_runtime_profiler_coalescing_manager_handler(group, async_ops=False, launch_func=launch_func)

        # The fallback span continues the proc span ended at manager start and is
        # already running during the launch.
        self.assertTrue(seen["proc_span_open"])
        self.assertIsNone(seen["pending_proc_span"])
        ctx.step_stop()

        # No task collected and no work produced: no slot at all.
        record_slot.assert_not_called()
        graph = step_graph(ctx, 2)
        self.assertIn(
            (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_proc", 1), SpanPointType.START),
            dependency_points(graph),
        )

    def test_start_proc_span_after_comm_without_pending_span_is_noop(self) -> None:
        """Starting without a pending span must not touch the open proc span."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()

        self.assertTrue(ctx._span_proc_open)
        ctx.cond_start_proc_span_after_comm()
        # Nothing was cut, so the start neither opened a new span nor ended one; the
        # per-step count still points at the span opened by step_start().
        self.assertTrue(ctx._span_proc_open)
        self.assertEqual(ctx._span_proc_count, 0)
        self.assertIsNone(ctx._span_pending_proc_prev_ids)
        ctx.step_stop()

    def test_capture_collective_starts_proc_span_without_real_launch(self) -> None:
        """Capture-mode fake work launches nothing, so the pending span starts right after the cut."""
        ctx = CrystalRuntimeContext()
        ctx.step_start()
        ctx.step_stop()
        ctx.step_start()
        # The trace writer wrapper reserves the trace index before the handler runs.
        ctx.reserve_trace_comm_index()

        group = mock.Mock()
        group.group_name = "pg"
        group.group_desc = "pg"

        def all_reduce(tensor, op=None, group=None, async_op=False):
            raise AssertionError("capture mode must not launch the real collective")

        with mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 8]):
            work = crystal_collectives_handler(all_reduce, [object(), None, group, True], {})

        self.assertIsInstance(work, CrystalFakeWork)
        # The cut is complete right away: fake work has no launch window, so the next
        # proc span is already open for the compute that follows.
        self.assertTrue(ctx._span_proc_open)
        self.assertIsNone(ctx._span_pending_proc_prev_ids)
        # The async fake wait still ends that span and chains the one after it on
        # the comm span.
        work.wait()
        self.assertTrue(ctx._span_proc_open)
        ctx.step_stop()

        dep_pairs = dependency_points(step_graph(ctx, 2))
        # Async launch shape: compute between cut and wait keeps running as _proc 1
        # chained on _proc 0, not on the comm span; only the fake wait makes the
        # span after it (_proc 2) depend on the comm span.
        self.assertEqual(
            dep_pairs,
            {
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_comm", 0), SpanPointType.START),
                (CrystalGSpanId("_proc", 0), SpanPointType.STOP, CrystalGSpanId("_proc", 1), SpanPointType.START),
                (CrystalGSpanId("_proc", 1), SpanPointType.STOP, CrystalGSpanId("_proc", 2), SpanPointType.START),
                (CrystalGSpanId("_comm", 0), SpanPointType.STOP, CrystalGSpanId("_proc", 2), SpanPointType.START),
            },
        )

    def test_profiler_collective_starts_proc_span_after_real_launch(self) -> None:
        """A blocking synchronous collective must not be timed inside the next proc span."""
        with (
            mock.patch.dict(
                os.environ,
                {"CRYSTAL_GRAPH_CAPTURING": "false", "ENABLE_CRYSTAL_RUNTIME_PROFILER": "true"},
            ),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.Event"),
        ):
            ctx = CrystalRuntimeContext()
            ctx.step_start()  # step 1, outside window
            ctx.step_stop()
            ctx.step_start()  # step 2, profiler window active

            group = mock.Mock()
            group.group_name = "pg"
            group.group_desc = "pg"
            seen = {}
            start_calls = []

            def all_reduce(tensor, op=None, group=None, async_op=False):
                # This is the real launch point: the span chain must already be cut
                # with the next proc span still pending, so the blocking
                # communication cannot leak into it.
                seen["proc_span_open"] = ctx._span_proc_open
                seen["pending_proc_span"] = ctx._span_pending_proc_prev_ids
                seen["start_calls_during_launch"] = list(start_calls)

            with (
                mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 8]),
                mock.patch.object(
                    ctx._runtime_profiler,
                    "start_proc_span",
                    side_effect=start_calls.append,
                ),
            ):
                returned = crystal_runtime_profiler_collectives_handler(all_reduce, [object(), None, group, False], {})

            self.assertIsNone(returned)
            self.assertFalse(seen["proc_span_open"])
            # The synchronous comm span chains the pending proc span on itself.
            self.assertEqual(seen["pending_proc_span"], [CrystalGSpanId("_comm", 0)])
            # start_proc_span was not called for the next span until after the launch.
            self.assertEqual(seen["start_calls_during_launch"], [])
            # After the launch returns, the pending proc span is started.
            self.assertTrue(ctx._span_proc_open)
            self.assertEqual(start_calls, [1])
            self.assertIsNone(ctx._span_pending_proc_prev_ids)
            ctx.step_stop()

    def _issue_wait_context(self, events: list) -> CrystalRuntimeContext:
        """Build a profiler-mode context under max_connections=1 with tracked events.

        ``events`` collects one mock per created ``torch.cuda.Event`` in creation
        order, so the tests can assert exactly which events bound an issue-wait
        measurement.
        """
        env = mock.patch.dict(
            os.environ,
            {
                "CRYSTAL_GRAPH_CAPTURING": "false",
                "ENABLE_CRYSTAL_RUNTIME_PROFILER": "true",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)

        def fake_event(**kwargs):
            event = mock.Mock()
            events.append(event)
            return event

        event_patch = mock.patch("torch.cuda.Event", side_effect=fake_event)
        event_patch.start()
        self.addCleanup(event_patch.stop)

        cuda_patch = mock.patch("torch.cuda.is_available", return_value=True)
        cuda_patch.start()
        self.addCleanup(cuda_patch.stop)

        return CrystalRuntimeContext()

    def _issue_wait_group(self) -> mock.Mock:
        """Build a mocked cross-node ProcessGroup named ``async_pg``."""
        group = mock.Mock()
        group.group_name = "async_pg"
        group.group_desc = "async_pg"
        return group

    def test_profiler_sync_comm_records_issue_wait_span(self) -> None:
        """A synchronous communication reports an issue-wait span ending after its launch."""
        events: list = []
        ctx = self._issue_wait_context(events)
        group = self._issue_wait_group()

        def all_reduce(tensor, op=None, group=None, async_op=False):
            return None

        with mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 8]):
            ctx.step_start()  # step 1, outside the window
            ctx.step_stop()
            # Step 2 enters the window: under max_connections=1 the profiler times no
            # ProcessGroup at all, so every communication reports an issue-wait duration.
            ctx.step_start()

            # Synchronous communication: the issue-wait span runs from the proc-span
            # stop at the cut to the proc-span start after the launch.
            crystal_runtime_profiler_collectives_handler(all_reduce, [object(), None, group, False], {})
        ctx.step_stop()

        profiler = ctx._runtime_profiler
        self.assertEqual(len(profiler._issue_wait_spans), 1)
        span = profiler._issue_wait_spans[0]
        self.assertEqual(span.channel, "async_pg")
        self.assertEqual(span.step, 2)
        # events[0]: step-2 initial proc span start; events[1]: stop at the comm cut
        # (issue-wait start); events[2]: start after the launch (issue-wait end).
        self.assertIs(span.start_event, events[1])
        self.assertIs(span.end_event, events[2])

    def test_profiler_async_comm_records_issue_wait_span_after_wait(self) -> None:
        """An async communication reports its issue-wait span at wait()."""
        events: list = []
        ctx = self._issue_wait_context(events)
        group = self._issue_wait_group()
        real_work = mock.Mock()

        def all_reduce(tensor, op=None, group=None, async_op=False):
            return real_work

        with mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 8]):
            ctx.step_start()
            ctx.step_stop()
            ctx.step_start()  # window active: profiler times no group under max_connections=1

            work = crystal_runtime_profiler_collectives_handler(all_reduce, [object(), None, group, True], {})
            self.assertIsInstance(work, CrystalProfiledWork)
            work.wait()
        ctx.step_stop()

        profiler = ctx._runtime_profiler
        self.assertEqual(len(profiler._issue_wait_spans), 1)
        span = profiler._issue_wait_spans[0]
        # events[0]: step-2 initial start; events[1]: stop at the cut (issue-wait
        # start); events[2]: start after the launch; events[3]: stop at the wait;
        # events[4]: start after the wait (issue-wait end).
        self.assertIs(span.start_event, events[1])
        self.assertIs(span.end_event, events[4])

    def test_profiler_unwaited_async_comm_settles_at_step_boundary(self) -> None:
        """An async communication nobody waits is recorded as never waited at step_stop().

        A pipeline schedule may leave the last send handle unwaited, so no
        wait-side bracket ever closes its measurement: step_stop() must hand it
        to the profiler as a never-waited communication, which keeps its issue
        position in the channel's duration array but is reported with duration 0.
        """
        events: list = []
        ctx = self._issue_wait_context(events)
        group = self._issue_wait_group()
        real_work = mock.Mock()

        def all_reduce(tensor, op=None, group=None, async_op=False):
            return real_work

        with mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 8]):
            ctx.step_start()
            ctx.step_stop()
            ctx.step_start()  # window active: profiler times no group under max_connections=1

            work = crystal_runtime_profiler_collectives_handler(all_reduce, [object(), None, group, True], {})
            self.assertIsInstance(work, CrystalProfiledWork)
            # Nobody calls work.wait(): the measurement stays in flight.
            self.assertEqual(len(ctx._runtime_profiler._issue_wait_spans), 0)
        ctx.step_stop()

        profiler = ctx._runtime_profiler
        self.assertEqual(len(profiler._issue_wait_spans), 1)
        span = profiler._issue_wait_spans[0]
        self.assertEqual(span.step, 2)
        self.assertEqual(span.channel, "async_pg")
        # The entry is keyed by the trace index the launch reserved, so it keeps
        # its issue position; the issue-side start event is kept while the missing
        # end event marks the never-waited measurement: duration 0.
        self.assertEqual(span.trace_comm_index, 0)
        self.assertIs(span.start_event, events[1])
        self.assertIsNone(span.end_event)
        # The settlement drained the in-flight bookkeeping.
        self.assertEqual(ctx._issue_wait_in_flight, {})
        self.assertIsNone(ctx._issue_wait_pending)

    def test_profiler_async_coalescing_records_issue_wait_span(self) -> None:
        """An async coalescing manager measures from its entry stop event to the wait."""
        events: list = []
        ctx = self._issue_wait_context(events)
        group = self._issue_wait_group()
        real_work = mock.Mock()

        with mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 8]):
            ctx.step_start()
            ctx.step_stop()
            ctx.step_start()  # window active: profiler times no group under max_connections=1

            ctx.cond_start_comm_span_coalescing()
            ctx.cond_add_coalescing_comm_task(CrystalGCommPattern.SYMM)
            work = crystal_runtime_profiler_coalescing_manager_handler(
                group, async_ops=True, launch_func=lambda: real_work
            )
            work.wait()
        ctx.step_stop()

        profiler = ctx._runtime_profiler
        self.assertEqual(len(profiler._issue_wait_spans), 1)
        span = profiler._issue_wait_spans[0]
        self.assertEqual(span.channel, "async_pg")
        # events[0]: step-2 initial start; events[1]: stop at the manager entry
        # (issue-wait start); events[2]: start after the manager launch; events[3]:
        # stop at the wait; events[4]: start after the wait (issue-wait end).
        self.assertIs(span.start_event, events[1])
        self.assertIs(span.end_event, events[4])

    def test_step_lifecycle_is_noop_without_graph_capture_or_profiler(self) -> None:
        """A run with neither graph capture nor runtime profiler steps cleanly.

        Emulate-mode sandbox training runs the full step lifecycle without either
        span-recording feature: the span state is never initialized, so step_stop()
        must not reach the issue-wait settlement (or touch any span attribute).
        """
        with mock.patch.dict(
            os.environ,
            {"CRYSTAL_GRAPH_CAPTURING": "false", "ENABLE_CRYSTAL_RUNTIME_PROFILER": "false"},
        ):
            ctx = CrystalRuntimeContext()
            self.assertFalse(ctx.is_span_record_configured)
            ctx.step_start()
            self.assertFalse(ctx.is_span_record_active)
            ctx.step_stop()
            ctx.destroy()


if __name__ == "__main__":
    unittest.main()
