# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for the Crystal runtime profiler communication bookkeeping.

These tests drive ``CrystalRuntimeProfiler`` without CUDA and without real
distributed communication: ProcessGroups are mocked and completion records are
produced by calling the installed hook directly. The covered behavior is:

- slots and completion-hook records are matched one to one per ProcessGroup;
- every duration is reported under its channel, whatever the graph span status
  of the communication, in issue order;
- leaving the right-open analysis window stops accepting new completions, so
  communication issued after the window cannot break the matching;
- group discovery enables timing only for cross-node, non-rendezvous groups;
- under ``max_connections == 1`` no group gets collectives timing: every
  communication reports an issue-wait duration instead, the channel's duration
  array follows the communication trace index (issue order) rather than the
  recording order, and a communication that was never waited keeps its issue
  position with duration 0;
- under ``ignore_async_issue_wait`` the issue-wait duration of an asynchronous
  communication is reported as 0 while keeping its issue position, so the
  array stays aligned with the graph's spans and synchronous durations are
  untouched.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest import mock

from crystalllm.runtime.profiler import CrystalRuntimeProfiler, discover_cross_node_groups


def work_info(*, seq: int, duration_s: float, started: str, finished: str) -> types.SimpleNamespace:
    """Build a minimal stand-in for the WorkInfo passed to a completion hook."""
    return types.SimpleNamespace(
        op_type="ALLREDUCE",
        seq=seq,
        active_duration=duration_s,
        time_started=started,
        time_finished=finished,
    )


def mock_group(group_name: str, group_desc: str = "pg") -> mock.Mock:
    """Build a mocked ProcessGroup that only exposes the attributes we need."""
    group = mock.Mock()
    group.group_name = group_name
    group.group_desc = group_desc
    return group


class TestRuntimeProfilerCommMatching(unittest.TestCase):
    """Validate slot/completion matching without CUDA or real collectives."""

    def setUp(self) -> None:
        """Create a profiler whose analysis window is [2, 3)."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"}, clear=False)
        self._env.start()
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.profiler = CrystalRuntimeProfiler(
                rank=0,
                profiler_dir=self._tmpdir.name,
                step_start=2,
                step_end=3,
            )
        self.group = mock_group("timed_pg")

    def tearDown(self) -> None:
        """Release the temporary output directory and environment overrides."""
        self._env.stop()
        self._tmpdir.cleanup()

    def enter_window(self) -> None:
        """Enter the analysis window and pretend the mocked group is recording."""
        self.profiler.step_start(2)
        self.profiler._registered_groups["timed_pg"] = self.group

    def snapshot(self) -> dict:
        """Take a snapshot, skipping the CUDA synchronization it normally performs."""
        with mock.patch("torch.cuda.synchronize"):
            return self.profiler.snapshot()

    def test_leaving_window_drops_later_completions(self) -> None:
        """Completions of comms issued after the window must not break the matching."""
        self.enter_window()
        self.profiler.record_comm_slot(self.group, step=2, channel="timed_pg")
        hook = self.profiler._make_hook("timed_pg")
        hook(work_info(seq=1, duration_s=0.001, started="t1", finished="t2"))

        # Step 3 is outside [2, 3): the window closes and hooks stop being accepted.
        self.profiler.step_start(3)
        self.assertFalse(self.profiler.is_active)
        self.group._wait_for_pending_works.assert_called_once_with()

        self.profiler.record_comm_slot(self.group, step=3, channel="timed_pg")
        hook(work_info(seq=2, duration_s=0.004, started="t3", finished="t4"))

        result = self.snapshot()
        self.assertEqual(result["2"]["comm"], {"timed_pg": [1_000_000]})
        self.assertNotIn("3", result)

    def test_slot_completion_count_mismatch_raises(self) -> None:
        """An unmatched completion must fail loudly instead of shifting durations."""
        self.enter_window()
        self.profiler.record_comm_slot(self.group, step=2, channel="timed_pg")
        hook = self.profiler._make_hook("timed_pg")
        hook(work_info(seq=1, duration_s=0.001, started="t1", finished="t2"))
        hook(work_info(seq=2, duration_s=0.002, started="t3", finished="t4"))

        with self.assertRaises(RuntimeError):
            self.snapshot()

    def test_durations_group_by_channel_in_issue_order(self) -> None:
        """Slots and completions match one to one; durations split per channel in issue order."""
        self.enter_window()
        self.profiler.record_comm_slot(self.group, step=2, channel="timed_pg")
        self.profiler.record_comm_slot(self.group, step=2, channel="timed_pg_0:1")
        self.profiler.record_comm_slot(self.group, step=2, channel="timed_pg")

        # The hooks fire in completion order, which an overtaking async operation
        # makes differ from the issue order (seq 3 retires first). Matching sorts
        # the records back into issue order by launch time, so each duration
        # still lands on the channel of its own launch.
        hook = self.profiler._make_hook("timed_pg")
        hook(work_info(seq=3, duration_s=0.003, started="t5", finished="t6"))
        hook(work_info(seq=1, duration_s=0.002, started="t1", finished="t2"))
        hook(work_info(seq=2, duration_s=0.001, started="t3", finished="t4"))

        result = self.snapshot()
        self.assertEqual(
            result["2"]["comm"],
            {"timed_pg": [2_000_000, 3_000_000], "timed_pg_0:1": [1_000_000]},
        )

    def test_stop_recording_is_idempotent(self) -> None:
        """Closing the window twice must not drain the groups twice."""
        self.enter_window()
        self.profiler.stop_recording()
        self.profiler.stop_recording()
        self.group._wait_for_pending_works.assert_called_once_with()


class TestRuntimeProfilerIssueWaitTiming(unittest.TestCase):
    """Validate the max_connections-aware timing skip and issue-wait recording."""

    def setUp(self) -> None:
        """Create the environment shared by the issue-wait tests."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"}, clear=False)
        self._env.start()

    def tearDown(self) -> None:
        """Release the temporary output directory and environment overrides."""
        self._env.stop()
        self._tmpdir.cleanup()

    def make_profiler(self, max_connections: int, ignore_async_issue_wait: bool = False) -> CrystalRuntimeProfiler:
        """Build a profiler with the given timing knobs, bypassing the CUDA check."""
        with mock.patch("torch.cuda.is_available", return_value=True):
            return CrystalRuntimeProfiler(
                rank=0,
                profiler_dir=self._tmpdir.name,
                step_start=2,
                step_end=3,
                max_connections=max_connections,
                ignore_async_issue_wait=ignore_async_issue_wait,
            )

    def test_prepare_skips_all_groups_with_one_connection(self) -> None:
        """Under max_connections == 1 no group is timed at all."""
        profiler = self.make_profiler(1)
        group = mock_group("async_pg")

        prepared = profiler.prepare_for_timing([group])

        self.assertEqual(prepared, [])
        group._enable_collectives_timing.assert_not_called()
        self.assertTrue(profiler.uses_issue_wait_timing)

    def test_issue_wait_spans_fill_comm_durations_in_snapshot(self) -> None:
        """Issue-wait event pairs are turned into per-channel durations at snapshot time."""
        profiler = self.make_profiler(1)
        start_event = mock.Mock()
        start_event.elapsed_time.return_value = 1.5
        end_event = mock.Mock()
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=4, start_event=start_event, end_event=end_event
        )

        with mock.patch("torch.cuda.synchronize"):
            result = profiler.snapshot()

        # An untimed group has neither slots nor completions, so its issue-wait
        # spans are the only source of the channel's durations.
        self.assertEqual(result["2"]["comm"], {"async_pg": [1_500_000]})

    def test_issue_wait_spans_are_ordered_by_trace_index(self) -> None:
        """The channel's array follows issue order, not the recording order.

        An asynchronous communication is recorded when its wait returns, which can
        overtake the wait of a later-issued communication, so the array must be
        sorted by the communication trace index to keep the graph's positional
        matching aligned.
        """
        profiler = self.make_profiler(1)
        late_start, late_end = mock.Mock(), mock.Mock()
        late_start.elapsed_time.return_value = 3.0
        early_start, early_end = mock.Mock(), mock.Mock()
        early_start.elapsed_time.return_value = 2.0
        # Recorded in wait-return order: the communication issued later (index 5)
        # is waited first.
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=5, start_event=late_start, end_event=late_end
        )
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=3, start_event=early_start, end_event=early_end
        )

        with mock.patch("torch.cuda.synchronize"):
            result = profiler.snapshot()

        self.assertEqual(result["2"]["comm"], {"async_pg": [2_000_000, 3_000_000]})

    def test_unwaited_comm_keeps_issue_position_with_zero_duration(self) -> None:
        """A never-waited communication keeps its issue position with duration 0."""
        profiler = self.make_profiler(1)
        waited_start, waited_end = mock.Mock(), mock.Mock()
        waited_start.elapsed_time.return_value = 1.0
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=3, start_event=waited_start, end_event=waited_end
        )
        profiler.record_unwaited_comm_span(step=2, channel="async_pg", trace_comm_index=4, start_event=mock.Mock())
        unwaited_start, unwaited_end = mock.Mock(), mock.Mock()
        unwaited_start.elapsed_time.return_value = 5.0
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=5, start_event=unwaited_start, end_event=unwaited_end
        )

        with mock.patch("torch.cuda.synchronize"):
            result = profiler.snapshot()

        # The unwaited entry stays at its issue position with duration 0 instead of
        # shifting the durations of the communications issued after it.
        self.assertEqual(result["2"]["comm"], {"async_pg": [1_000_000, 0, 5_000_000]})

    def test_ignore_async_issue_wait_zeroes_async_duration_keeping_issue_position(self) -> None:
        """Under ignore_async_issue_wait an async entry reports 0 at its issue position.

        Dropping the entry outright would shift every later duration onto the
        wrong graph span, so the zero must stay at the async communication's
        position in the channel's array.
        """
        profiler = self.make_profiler(1, ignore_async_issue_wait=True)
        sync_start, sync_end = mock.Mock(), mock.Mock()
        sync_start.elapsed_time.return_value = 1.0
        async_start, async_end = mock.Mock(), mock.Mock()
        async_start.elapsed_time.return_value = 7.0
        late_sync_start, late_sync_end = mock.Mock(), mock.Mock()
        late_sync_start.elapsed_time.return_value = 2.0
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=3, start_event=sync_start, end_event=sync_end
        )
        profiler.record_issue_wait_comm_span(
            step=2,
            channel="async_pg",
            trace_comm_index=4,
            start_event=async_start,
            end_event=async_end,
            async_op=True,
        )
        profiler.record_issue_wait_comm_span(
            step=2, channel="async_pg", trace_comm_index=5, start_event=late_sync_start, end_event=late_sync_end
        )

        with mock.patch("torch.cuda.synchronize"):
            result = profiler.snapshot()

        self.assertEqual(result["2"]["comm"], {"async_pg": [1_000_000, 0, 2_000_000]})

    def test_ignore_async_issue_wait_false_keeps_async_duration(self) -> None:
        """Without ignore_async_issue_wait the async issue-wait duration is exported as measured."""
        profiler = self.make_profiler(1)
        async_start, async_end = mock.Mock(), mock.Mock()
        async_start.elapsed_time.return_value = 7.0
        profiler.record_issue_wait_comm_span(
            step=2,
            channel="async_pg",
            trace_comm_index=4,
            start_event=async_start,
            end_event=async_end,
            async_op=True,
        )

        with mock.patch("torch.cuda.synchronize"):
            result = profiler.snapshot()

        self.assertEqual(result["2"]["comm"], {"async_pg": [7_000_000]})


class TestRuntimeProfilerGroupDiscovery(unittest.TestCase):
    """Validate which ProcessGroups get timing enabled."""

    def test_discovery_excludes_rendezvous_and_same_node_groups(self) -> None:
        """Only cross-node, non-rendezvous groups may get timing enabled."""
        rendezvous = mock_group("rendezvous", group_desc="_crystal_replay_rendezvous_pg")
        local = mock_group("local")
        cross_node = mock_group("cross_node")
        ranks_by_group = {rendezvous: [0, 8], local: [0, 1], cross_node: [0, 8]}

        with (
            mock.patch("crystalllm.runtime.profiler._world") as world,
            mock.patch("torch.distributed.get_process_group_ranks", side_effect=lambda group: ranks_by_group[group]),
        ):
            world.pg_map = dict.fromkeys(ranks_by_group)
            world.pg_names = {rendezvous: "rendezvous", local: "local", cross_node: "cross_node"}
            groups = discover_cross_node_groups(local_world_size=8)

        # The rendezvous group never gets a slot, and a same-node group never gets
        # a graph comm span; timing either would leave unmatched completions.
        self.assertEqual(groups, [cross_node])


if __name__ == "__main__":
    unittest.main()
