# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Runtime Profiler."""

from __future__ import annotations

import gzip
import json
import os
import sys
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _world  # noqa: PLC2701


@dataclass(frozen=True)
class CrystalRuntimeProfilerCommSlot:
    """Launch-side metadata for one communication operation.

    Every communication on a recording group gets a slot, because the per-PG
    slot/completion sequences must stay aligned one to one. ``channel`` is the
    graph channel the duration is reported under: communications without a graph
    span still get their group's channel, since the profiler records durations per
    channel and extra data is tolerated when loading into the graph.
    """

    step: int
    channel: str
    slot_index: int


@dataclass(frozen=True)
class CrystalRuntimeProfilerIssueWaitSpan:
    """One communication duration measured between reused proc-span CUDA events.

    Under ``max_connections == 1`` no ProcessGroup gets collectives timing at all,
    because the timing events would serialize the overlapped communications. Every
    duration is approximated by the issue-wait span instead: from the stop event of
    the proc span ended at the communication issue to the start event of the proc
    span opened after the launch (synchronous communication) or after the wait
    (asynchronous communication). The span therefore includes overlapped computation
    and may exceed the actual communication time, which does not affect the
    correctness of the built graph.

    ``trace_comm_index`` is the communication trace index of the measured
    communication. It never leaves the profiler: the reported durations are arrays
    in issue order, and the index is only the key the channel's entries are sorted
    by, because the recording order (synchronous launch return, asynchronous wait
    return, or the step boundary of a never-waited communication) is not the issue
    order. ``start_event`` — the stop event of the proc span ended at the issue —
    always exists, because the measurement is only created while the profiler
    window records it. ``end_event`` is None exactly when the step ended without a
    wait — a pipeline schedule may leave the last send handle unwaited — and such
    an entry is deliberately reported with duration 0.

    ``async_op`` marks the measurement of an asynchronous communication. When the
    profiler runs with ``ignore_async_issue_wait`` (max_connections == 1 graphs whose
    async_op flags were cleared during adjustment), such an entry keeps its issue
    position in the channel's array but is reported with duration 0: the interval
    from issue to wait also contains the overlapped computation, which the
    adjusted graph no longer accounts for, and dropping the entry outright would
    shift every later duration onto the wrong span.
    """

    step: int
    channel: str
    trace_comm_index: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event | None
    async_op: bool = False


def process_group_name(group: dist.ProcessGroup) -> str:
    """Return a stable c10d process-group identity.

    The PG's ``group_name`` attribute is preferred; otherwise the c10d ``_world``
    name mapping is consulted. No name at all means the group never passed through
    c10d bookkeeping, which is treated as a caller error.
    """
    group_name = getattr(group, "group_name", "")
    if group_name:
        return str(group_name)
    mapped_name = _world.pg_names.get(group)
    if mapped_name:
        return str(mapped_name)
    raise RuntimeError("ProcessGroup has no registered c10d group_name")


def members_cross_nodes(members: Sequence[int], local_world_size: int) -> bool:
    """Return whether the given global ranks span more than one node.

    A rank's node is identified as ``global_rank // local_world_size``, so ranks on
    the same node share the same quotient.
    """
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    return len({rank // local_world_size for rank in members}) > 1


def discover_cross_node_groups(local_world_size: int) -> list[dist.ProcessGroup]:
    """Discover currently registered cross-node c10d ProcessGroups.

    Only groups with members on several nodes need timing. Groups without a ``rank``
    attribute cannot be inspected and are skipped. Rendezvous barriers never get a
    slot, so timing them would leave completions that cannot be matched. Iterating
    by group name keeps the discovery order deterministic across ranks.
    """
    groups: list[dist.ProcessGroup] = []
    for group in sorted(_world.pg_map.keys(), key=lambda item: _world.pg_names.get(item, "")):
        if not hasattr(group, "rank"):
            continue
        if is_replay_rendezvous_group(group):
            continue
        members = list(dist.get_process_group_ranks(group))
        if members and members_cross_nodes(members, local_world_size):
            groups.append(group)
    return groups


def is_replay_rendezvous_group(group: object) -> bool:
    """Return whether the group is the internal replay rendezvous ProcessGroup.

    The rendezvous PG carries no training communication, so it is excluded from
    timing and slot recording.
    """
    return str(group.group_desc) == "_crystal_replay_rendezvous_pg"


def _duration_to_ns(value: object) -> int:
    """Convert a timedelta or a plain second count to an integer nanosecond count."""
    if hasattr(value, "total_seconds"):
        return round(value.total_seconds() * 1_000_000_000)
    return round(float(value) * 1_000_000_000)


def _time_to_str(value: object) -> str:
    """Render a datetime as ISO format, anything else via str()."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _op_type_name(value: object) -> str:
    """Render a WorkInfo op_type (an enum) as its bare name, e.g. ``ALLREDUCE``."""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.rsplit(".", maxsplit=1)[-1]


def _seq_sort_value(value: object) -> int:
    """Return the completion sequence number, or maxsize when it failed to parse."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return sys.maxsize


def _completion_sort_key(completion: dict[str, object]) -> tuple[str, int, str]:
    """Order completions back into launch (issue) order.

    Completions arrive in completion order, which differs from issue order whenever
    asynchronous operations overtake each other. ``time_started`` is the primary key
    because it reflects the launch instant; ``seq`` (and ``time_finished`` last)
    break ties so the order stays stable and matches the slot sequence one to one.
    """
    return (
        str(completion.get("time_started", "")),
        _seq_sort_value(completion.get("seq")),
        str(completion.get("time_finished", "")),
    )


class CrystalRuntimeProfiler:
    """Runtime-internal profiler for proc/comm span durations.

    Proc spans are timed with pairs of CUDA events; comm spans are timed by c10d
    completion hooks on a patched ProcessGroup. The launch side records one slot per
    communication, and ``snapshot`` matches slots with completion records per group
    in issue order to attribute each measured duration to its graph channel.

    Under ``max_connections == 1`` no ProcessGroup gets collectives timing at all —
    the timing events would serialize the overlapped communications — so every
    duration is approximated by issue-wait spans built from the reused proc-span
    boundary events instead.
    """

    def __init__(
        self,
        *,
        rank: int,
        profiler_dir: str | Path,
        step_start: int,
        step_end: int,
        max_connections: int = 8,
        ignore_async_issue_wait: bool = False,
    ) -> None:
        """Initialize profiler state and output configuration.

        Args:
            rank: Global rank writing the result file.
            profiler_dir: Directory receiving ``runtime_profiler_{rank}.json.gz``.
            step_start: First step (inclusive) of the profiling window.
            step_end: Last step (exclusive) of the profiling window.
            max_connections: CUDA_DEVICE_MAX_CONNECTIONS of the run; with 1, no
                ProcessGroup gets collectives timing and issue-wait durations are
                reported instead.
            ignore_async_issue_wait: Skip the issue-wait durations of asynchronous
                communications (used with max_connections == 1 graphs whose
                async_op flags were cleared during adjustment): an async
                issue-wait interval also contains overlapped computation, which
                the adjusted graph no longer accounts for. Their channel arrays
                keep the entry positions (the graph loader tolerates shorter
                arrays) but the async entries are never appended.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CrystalRuntimeProfiler requires CUDA")

        self.rank = rank
        self.profiler_dir = Path(profiler_dir)
        self.step_start_bound = step_start
        self.step_end_bound = step_end
        self._max_connections = max_connections
        self._ignore_async_issue_wait = ignore_async_issue_wait
        self._step = 0
        self._active = False
        self._timing_started = False
        self._recording_stopped = False
        self._lock = threading.Lock()
        self._prepared_groups: dict[str, dist.ProcessGroup] = {}
        self._registered_groups: dict[str, dist.ProcessGroup] = {}
        self._issue_wait_spans: list[CrystalRuntimeProfilerIssueWaitSpan] = []
        self._slots: dict[str, list[CrystalRuntimeProfilerCommSlot]] = defaultdict(list)
        self._completions: list[dict[str, object]] = []
        self._errors: list[str] = []
        self._proc_start_events: dict[int, dict[int, torch.cuda.Event]] = defaultdict(dict)
        self._proc_stop_events: dict[int, dict[int, torch.cuda.Event]] = defaultdict(dict)

        if self._max_connections == 1:
            print(
                "Crystal runtime profiler initialized with max_connections=1: no ProcessGroup gets collectives "
                "timing, every communication reports an issue-wait duration instead (from the communication "
                "issue to the wait, or to the launch for synchronous calls). An issue-wait duration includes "
                "overlapped computation, so it may be longer than the actual communication, but the overlap does "
                "not affect the correctness of the built graph."
            )
        if self._ignore_async_issue_wait:
            print(
                "Crystal runtime profiler initialized with ignore_async_issue_wait: the issue-wait durations of "
                "asynchronous communications are reported as 0, keeping their positions in the channel "
                "arrays"
            )

    @property
    def is_active(self) -> bool:
        """Whether the current step is inside the right-open profiling window ``[step_start, step_end)``."""
        return self._active

    def step_start(self, step: int) -> None:
        """Advance to a new step, lazily starting and later closing the profiling window.

        On the first active step the timing infrastructure is installed once:
        discover the cross-node groups, enable their collectives timing, drain the
        timed warmup works (so warmup completions never reach the hooks), and only
        then register the completion hooks. On the first step past the window the
        recording is closed via ``stop_recording``.
        """
        was_active = self._active
        self._step = step
        self._active = self.step_start_bound <= step < self.step_end_bound
        if self._active and not self._timing_started:
            groups = discover_cross_node_groups(_local_world_size())
            self.prepare_for_timing(groups)
            self.drain_prepared()
            self.start_recording()
            self._timing_started = True
        elif was_active and not self._active:
            self.stop_recording()

    def step_stop(self) -> None:
        """End-of-step no-op kept so callers can bracket every step symmetrically."""

    def start_proc_span(self, span_index: int) -> torch.cuda.Event | None:
        """Record a timing CUDA event at the start of the computation span.

        The event is returned so the runtime context can reuse it as the end point
        of an issue-wait duration; None when the profiler window is closed.
        """
        if not self._active:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._proc_start_events[self._step][span_index] = event
        return event

    def stop_proc_span(self, span_index: int) -> torch.cuda.Event | None:
        """Record a timing CUDA event at the end of the computation span.

        Both events are enqueued on the current CUDA stream, so the pair measures
        GPU time and ignores host-side launch gaps. The event is returned so the
        runtime context can reuse it as the start point of an issue-wait duration;
        None when the profiler window is closed.
        """
        if not self._active:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._proc_stop_events[self._step][span_index] = event
        return event

    def prepare_for_timing(self, groups: Iterable[dist.ProcessGroup]) -> list[str]:
        """Enable collectives timing on the given groups; return the newly prepared names.

        Idempotent: already prepared groups are skipped, so repeated calls with the
        same discovery result do nothing. Under ``max_connections == 1`` no group is
        timed at all: the timing events would serialize every overlapped
        communication, so all durations come from issue-wait measurements instead
        (see ``record_issue_wait_comm_span``).
        """
        if self._max_connections == 1:
            return []
        prepared: list[str] = []
        for group in groups:
            group_name = process_group_name(group)
            if group_name in self._prepared_groups:
                continue
            group._enable_collectives_timing()  # noqa: SLF001
            self._prepared_groups[group_name] = group
            prepared.append(group_name)
        return prepared

    @property
    def uses_issue_wait_timing(self) -> bool:
        """Whether communications are measured through issue-wait durations.

        True under ``max_connections == 1``, where no ProcessGroup gets collectives
        timing at all because its timing events would serialize the overlapped
        communications.
        """
        return self._max_connections == 1

    def record_issue_wait_comm_span(
        self,
        *,
        step: int,
        channel: str,
        trace_comm_index: int,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        async_op: bool = False,
    ) -> None:
        """Record one issue-wait duration between two already recorded CUDA events.

        The events are the proc-span boundaries around one communication; the
        profiler only stores the pair and turns it into a duration at snapshot time.
        ``trace_comm_index`` orders the channel's entries by issue at snapshot time,
        because the recording order — the wait return for an asynchronous
        communication — need not be the issue order. ``async_op`` marks the
        measurement of an asynchronous communication, reported with duration 0
        when the profiler runs with ``ignore_async_issue_wait`` (see
        ``CrystalRuntimeProfilerIssueWaitSpan.async_op``).
        """
        self._issue_wait_spans.append(
            CrystalRuntimeProfilerIssueWaitSpan(
                step=step,
                channel=channel,
                trace_comm_index=trace_comm_index,
                start_event=start_event,
                end_event=end_event,
                async_op=async_op,
            )
        )

    def record_unwaited_comm_span(
        self, *, step: int, channel: str, trace_comm_index: int, start_event: torch.cuda.Event
    ) -> None:
        """Record one asynchronous communication that its step ended without a wait.

        A pipeline schedule may leave the last send handle unwaited, so no wait-side
        boundary ever closes its measurement. The entry keeps the issue-side start
        event and its issue position in the channel's duration array but carries no
        end event: it is deliberately reported with duration 0 — nothing in the
        graph chains on the STOP of an unwaited communication — and ``snapshot``
        prints a warning naming the communication.
        """
        self._issue_wait_spans.append(
            CrystalRuntimeProfilerIssueWaitSpan(
                step=step,
                channel=channel,
                trace_comm_index=trace_comm_index,
                start_event=start_event,
                end_event=None,
            )
        )

    def drain_prepared(self) -> None:
        """Drain timed warmup works before hooks are installed.

        Enabling timing may itself enqueue bookkeeping work; draining here keeps its
        completion records out of the analysis window.
        """
        for group in self._prepared_groups.values():
            group._wait_for_pending_works()  # noqa: SLF001

    def start_recording(self) -> list[str]:
        """Install completion hooks on prepared groups; return the newly hooked names."""
        return [
            group_name
            for group_name in sorted(self._prepared_groups)
            if self._register_group(self._prepared_groups[group_name])
        ]

    def group_is_recording(self, group: dist.ProcessGroup) -> bool:
        """Return whether a ProcessGroup has a registered completion hook."""
        return process_group_name(group) in self._registered_groups

    def record_comm_slot(self, group: dist.ProcessGroup, *, step: int, channel: str) -> None:
        """Record launch-side metadata for the op being launched on group.

        Every operation on a recording group must be recorded, otherwise slots and
        completion-hook records cannot be matched one to one. ``channel`` names the
        graph channel of the operation and groups the reported durations.
        """
        if not self._active or not self.group_is_recording(group):
            return
        group_name = process_group_name(group)
        with self._lock:
            group_slots = self._slots[group_name]
            group_slots.append(
                CrystalRuntimeProfilerCommSlot(
                    step=step,
                    channel=channel,
                    slot_index=len(group_slots),
                )
            )

    def stop_recording(self) -> None:
        """Close the recording window: drain pending works, then ignore later completions.

        c10d has no public API to remove a completion hook, so hooks installed for
        the analysis window keep firing afterwards. Draining first guarantees that
        every communication launched inside the window is accounted for (the drain
        also waits for the hook callbacks, which c10d runs on a separate thread);
        the flag then drops the completions of communications launched after the
        window, which have no slot.
        """
        with self._lock:
            if self._recording_stopped:
                return
        self.drain_recorded()
        with self._lock:
            self._recording_stopped = True

    def drain_recorded(self) -> None:
        """Wait until every hooked group's pending works (and hook callbacks) finished."""
        for group in self._registered_groups.values():
            group._wait_for_pending_works()  # noqa: SLF001

    def snapshot(self) -> dict[str, Any]:
        """Return matched duration records keyed by step, raising on incomplete data.

        The returned layout is ``{step: {"rank_id": rank, "proc": {span: ns},
        "comm": {channel: [ns, ...]}}}``. Raises when any hook failed or when the
        slot/completion sequences of a group do not match one to one.
        """
        torch.cuda.synchronize()
        with self._lock:
            slots = {key: list(value) for key, value in self._slots.items()}
            completions = list(self._completions)
            errors = list(self._errors)
            issue_wait_spans = list(self._issue_wait_spans)
        if errors:
            raise RuntimeError("Runtime profiler hook errors: " + "; ".join(errors))
        result: dict[str, Any] = {}
        self._fill_proc_durations(result)
        self._fill_comm_durations(result, slots, completions)
        self._fill_issue_wait_durations(result, issue_wait_spans)
        return result

    def save(self) -> Path:
        """Write profiler result to runtime_profiler_{rank}.json.gz."""
        result = self.snapshot()
        self.profiler_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.profiler_dir / f"runtime_profiler_{self.rank}.json.gz"
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Crystal runtime profiler file stored at {file_path}")
        return file_path

    def _register_group(self, group: dist.ProcessGroup) -> bool:
        """Install this profiler's completion hook on the group; False when already hooked by it."""
        group_name = process_group_name(group)
        if group_name in self._registered_groups:
            return False
        if group._has_hooks():  # noqa: SLF001
            raise RuntimeError(f"ProcessGroup {group_name} already has completion hooks")
        group._register_on_completion_hook(self._make_hook(group_name))  # noqa: SLF001
        self._registered_groups[group_name] = group
        return True

    def _make_hook(self, group_name: str):
        """Build the completion hook for one group.

        c10d invokes the hook from a separate thread once a work retires, passing a
        WorkInfo. The hook copies the fields needed for matching — op type,
        per-group sequence number, active duration (time on the wire, excluding
        enqueue-to-start wait), and wall-clock start/finish stamps — into a
        completion record. Records and hook failures are dropped silently once
        recording has stopped.
        """

        def hook(work_info: object) -> None:
            try:
                completion = {
                    "rank": self.rank,
                    "group_name": group_name,
                    "op_type": _op_type_name(getattr(work_info, "op_type", "?")),
                    "seq": getattr(work_info, "seq", "?"),
                    "active_duration_ns": _duration_to_ns(getattr(work_info, "active_duration", 0)),
                    "time_started": _time_to_str(getattr(work_info, "time_started", "?")),
                    "time_finished": _time_to_str(getattr(work_info, "time_finished", "?")),
                }
                with self._lock:
                    if self._recording_stopped:
                        return
                    self._completions.append(completion)
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                with self._lock:
                    if self._recording_stopped:
                        return
                    self._errors.append(f"hook({group_name}) failed: {type(exc).__name__}: {exc}")

        return hook

    def _ensure_step_entry(self, result: dict[str, Any], step: int) -> dict[str, Any]:
        """Return the result entry of the step, creating the proc/comm skeleton on first use."""
        step_key = str(step)
        if step_key not in result:
            result[step_key] = {"rank_id": self.rank, "proc": {}, "comm": {}}
        return result[step_key]

    def _fill_proc_durations(self, result: dict[str, Any]) -> None:
        """Fill proc span durations from the recorded CUDA event pairs.

        ``elapsed_time`` returns milliseconds; the multiplication converts to
        nanoseconds, the unit of every duration in the output. A span without its
        stop event is treated as a bug and raises.
        """
        for step, start_events in self._proc_start_events.items():
            stop_events = self._proc_stop_events.get(step, {})
            step_entry = self._ensure_step_entry(result, step)
            for span_index, start_event in start_events.items():
                if span_index not in stop_events:
                    raise RuntimeError(f"Missing proc stop event for step={step}, span={span_index}")
                duration_ns = int(start_event.elapsed_time(stop_events[span_index]) * 1_000_000)
                step_entry["proc"][str(span_index)] = duration_ns

    def _fill_comm_durations(
        self,
        result: dict[str, Any],
        slots_by_group_name: dict[str, list[CrystalRuntimeProfilerCommSlot]],
        completions: list[dict[str, object]],
    ) -> None:
        """Match each group's slots with its completions in issue order.

        Slots are appended per group at launch time, so their order is the issue
        order; completions are re-sorted into issue order via ``_completion_sort_key``.
        Pairing the two ordered sequences with a strict zip attributes every measured
        duration to the channel of the corresponding launch. A count mismatch means
        an operation was lost on either side, which is reported instead of guessed.
        """
        completions_by_group_name: dict[str, list[dict[str, object]]] = defaultdict(list)
        for completion in completions:
            completions_by_group_name[str(completion.get("group_name", "?"))].append(completion)

        errors: list[str] = []
        for group_name in sorted(set(slots_by_group_name) | set(completions_by_group_name)):
            slots = sorted(slots_by_group_name.get(group_name, []), key=lambda slot: slot.slot_index)
            group_completions = sorted(completions_by_group_name.get(group_name, []), key=_completion_sort_key)
            if len(slots) != len(group_completions):
                errors.append(f"group_name={group_name}: slots={len(slots)} completions={len(group_completions)}")
                continue
            for slot, completion in zip(slots, group_completions, strict=True):
                step_entry = self._ensure_step_entry(result, slot.step)
                # Each channel's array ends up in issue order: slots are ordered by
                # slot_index and completions by their sort key, so the n-th duration
                # belongs to the channel's n-th launched communication, matching the
                # count_field order of the channel's spans in the graph.
                step_entry["comm"].setdefault(slot.channel, []).append(int(completion["active_duration_ns"]))
        if errors:
            raise RuntimeError("Runtime profiler unmatched slot/completion: " + "; ".join(errors))

    def _fill_issue_wait_durations(
        self,
        result: dict[str, Any],
        issue_wait_spans: list[CrystalRuntimeProfilerIssueWaitSpan],
    ) -> None:
        """Fill the comm durations of the groups measured through issue-wait spans.

        Each span reuses the proc-span boundary events of one communication, so the
        ``elapsed_time`` between them approximates its duration (including overlapped
        computation). The entries are sorted by communication trace index inside each
        step and channel, so each channel's array ends up in issue order — matching
        the count_field order of the channel's spans in the graph — whatever order the
        measurements were recorded in. A span without an end event is a never-waited
        communication: it contributes 0 and draws a warning at snapshot time. Under
        ``ignore_async_issue_wait`` an asynchronous span keeps its position in the array but
        contributes 0 as well (see ``CrystalRuntimeProfilerIssueWaitSpan.async_op``).
        A group never mixes both paths: it is either timed (slots/completions) or
        issue-wait measured.
        """
        unwaited: list[CrystalRuntimeProfilerIssueWaitSpan] = []
        async_ignored = 0
        for span in sorted(
            issue_wait_spans,
            key=lambda item: (item.step, item.channel, item.trace_comm_index),
        ):
            step_entry = self._ensure_step_entry(result, span.step)
            if span.end_event is None:
                unwaited.append(span)
                duration_ns = 0
            else:
                duration_ns = int(span.start_event.elapsed_time(span.end_event) * 1_000_000)
                if self._ignore_async_issue_wait and span.async_op:
                    async_ignored += 1
                    duration_ns = 0
            step_entry["comm"].setdefault(span.channel, []).append(duration_ns)
        for span in unwaited:
            print(
                f"Crystal runtime profiler: communication with trace index {span.trace_comm_index} "
                f"on channel '{span.channel}' at step {span.step} of rank {self.rank} was never "
                "waited; its duration is recorded as 0"
            )
        if async_ignored:
            print(
                f"Crystal runtime profiler: {async_ignored} asynchronous issue-wait durations of "
                f"rank {self.rank} are reported as 0 (ignore_async_issue_wait)"
            )


def _local_world_size() -> int:
    """Read the number of ranks per node from LOCAL_WORLD_SIZE (fallback: GPUS_PER_NODE)."""
    value = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("GPUS_PER_NODE")
    if value is None:
        raise RuntimeError("LOCAL_WORLD_SIZE or GPUS_PER_NODE must be set")
    return int(value)
