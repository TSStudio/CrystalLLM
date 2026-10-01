# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Runtime Context.

One singleton context drives four independently-configured subsystems, layered as:

- span record — the base layer. A per-step state machine that cuts the
  computation span at every communication launch/wait boundary and allocates
  the logical ``_proc``/``_comm`` span IDs. It is configured when either
  consumer below is, because both consume its IDs and boundaries.
- graph capture — persists every span, dependency, and description into a
  per-step ``CrystalGraph`` and serializes the graphs at ``destroy()``.
  Depends on span record for IDs and chain boundaries.
- runtime profiler — measures PROC/COMM span durations (CUDA events, comm
  slots, issue-wait intervals) keyed by the span record's IDs. Depends on
  span record for IDs and chain boundaries.
- c10d trace — logs every c10d collective into a trace file. Independent of
  the three layers above.

Every combination of graph capture and profiler is valid: neither (e.g.
emulate-mode sandbox training, where the step lifecycle degenerates to a
no-op), graph capture only, profiler only, or both.

Guard naming: ``is_*_configured`` answers whether a subsystem was enabled by
the environment (static for the whole process); ``is_*_active`` additionally
requires the current step to be inside that subsystem's analysis window (and,
for the span record, inside the step lifecycle). Every ``cond_*`` method is a
safe no-op while its subsystem is inactive.

State naming: ``_span_*`` belongs to the span record, ``_graph_*`` to graph
capture, ``_issue_wait_*`` to the profiler's issue-wait bookkeeping. Each
group is created and reset only while its own subsystem is configured, so a
guard can never touch state of a subsystem that was never initialized.

One shared communication counter numbers every communication for the c10d
trace, the graph comm span IDs, and the profiler, so all three artifacts refer
to the same communication by the same index.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from typing_extensions import ParamSpec, Self

from ..planning import (
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
from ..util import is_truthy_value
from .profiler import CrystalRuntimeProfiler
from .rendezvous import replay_rendezvous

_T = TypeVar("_T")
_P = ParamSpec("_P")
_PROC_TYPE_FIELD = "_proc"
_COMM_TYPE_FIELD = "_comm"


@dataclass
class _IssueWaitMeasurement:
    """One in-flight issue-wait duration measurement on the context side.

    Created when a communication cuts the proc span while the profiler measures
    communications through issue-wait durations (``max_connections == 1``):
    ``start_event`` is the stop event of that proc span. Synchronous
    communications settle at ``cond_start_proc_span_after_comm``, asynchronous
    ones at ``cond_end_comm_span_wait`` after their wait.

    ``trace_comm_index`` is the shared communication trace index, which is also the
    count field of the communication's span ID: the counter never resets and the
    type field is constant, so the index alone identifies the communication —
    and it is the very number the profiler keys its durations by.
    """

    trace_comm_index: int
    channel: str
    step: int
    async_op: bool
    start_event: torch.cuda.Event


class CrystalRuntimeContext:
    """Singleton class for managing CrystalLLM runtime context.

    Layer ownership of the instance state:

    - step lifecycle: ``_rank``, ``_step``, ``_in_step``, ``_max_connections``,
      ``_analysis_step_start``/``_analysis_step_end`` (the window shared by
      graph capture and the profiler).
    - shared numbering: ``_comm_trace_counter``, ``_comm_trace_counter_last``,
      ``_comm_trace_counter_frozen``.
    - span record (``_span_*``): the per-step span state machine, initialized
      while span record is configured (graph capture or profiler enabled).
    - graph capture (``_graph_*`` plus ``_captured_graph_per_steps``):
      initialized while graph capture is configured.
    - profiler (``_runtime_profiler`` plus ``_issue_wait_*``): initialized
      while the profiler is configured.
    - c10d trace (``_c10d_trace``): initialized while tracing is configured.
    """

    _instance: CrystalRuntimeContext | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> Self:
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the runtime context."""
        if not CrystalRuntimeContext._initialized:
            with CrystalRuntimeContext._lock:
                if not CrystalRuntimeContext._initialized:
                    self._initialize()
                    CrystalRuntimeContext._initialized = True

    def _initialize(self) -> None:
        """Read environment configuration and set up every configured subsystem."""
        self._rank = int(os.getenv("RANK"))
        self._step = 0
        # CUDA_DEVICE_MAX_CONNECTIONS bounds how many channels the GPU can overlap;
        # the graph builder turns it into a serialization dependency when it is 1.
        self._max_connections: int = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "8"))
        self._analysis_step_start: int | None = None
        self._analysis_step_end: int | None = None

        # Shared communication numbering (see the section below).
        self._comm_trace_counter = 0
        self._comm_trace_counter_last = -1
        self._comm_trace_counter_frozen = False

        # Step lifecycle flag: spans may only be recorded between step_start()
        # and step_stop().
        self._in_step = False

        graph_capture_configured = is_truthy_value(os.getenv("CRYSTAL_GRAPH_CAPTURING"))
        profiler_configured = is_truthy_value(os.getenv("ENABLE_CRYSTAL_RUNTIME_PROFILER"))

        # Both graph capture and the profiler consume the span record and share
        # its analysis window, so the window envs are required as soon as either
        # is configured. Each layer's state is initialized under its own flag.
        if graph_capture_configured or profiler_configured:
            self._analysis_step_start = int(os.getenv("CRYSTAL_ANALYSIS_STEP_START"))
            self._analysis_step_end = int(os.getenv("CRYSTAL_ANALYSIS_STEP_END"))
            self._reset_step_span_state()
            if graph_capture_configured:
                self._reset_step_graph_state()
            if profiler_configured:
                self._reset_step_issue_wait_state()

        # c10d trace: independent of the span-record layers.
        if is_truthy_value(os.getenv("ENABLE_CRYSTAL_C10D_TRACE")):
            from ..torch.c10d_trace import CrystalC10DTracer

            self._c10d_trace = CrystalC10DTracer()
            self._c10d_trace.init_trace_file()
        else:
            self._c10d_trace = None

        # Graph capture.
        if graph_capture_configured:
            self._captured_graph_per_steps: dict[int, CrystalGraph] | None = {}
            print("Crystal graph capture initialized!")
        else:
            self._captured_graph_per_steps = None

        # Runtime profiler.
        if profiler_configured:
            profiler_dir = os.getenv("CRYSTAL_RUNTIME_PROFILER_DIR")
            if profiler_dir is None:
                profiler_dir = str(Path(os.getenv("CRYSTAL_LOG_DIR")) / "profiler" / "crystal")
            assert self._analysis_step_start is not None
            assert self._analysis_step_end is not None
            # Planning sets this alongside a max_connections == 1 graph whose async_op
            # flags were cleared during adjustment: the issue-wait interval of an
            # asynchronous communication also contains overlapped computation, which
            # the adjusted graph no longer accounts for, so its duration is exported
            # as 0 instead (see CrystalRuntimeProfiler's ignore_async_issue_wait).
            ignore_async_issue_wait = is_truthy_value(os.getenv("CRYSTAL_RUNTIME_PROFILER_IGNORE_ASYNC_ISSUE_WAIT"))
            self._runtime_profiler: CrystalRuntimeProfiler | None = CrystalRuntimeProfiler(
                rank=self._rank,
                profiler_dir=profiler_dir,
                step_start=self._analysis_step_start,
                step_end=self._analysis_step_end,
                max_connections=self._max_connections,
                ignore_async_issue_wait=ignore_async_issue_wait,
            )
            print("Crystal runtime profiler initialized!")
        else:
            self._runtime_profiler = None

    # ---------------------------------------------------------------------------
    # Subsystem configuration and step lifecycle.
    # ---------------------------------------------------------------------------

    @property
    def is_span_record_configured(self) -> bool:
        """Whether any span-record consumer (graph capture or profiler) is configured."""
        return self.is_graph_capture_configured or self.is_profiler_configured

    @property
    def is_graph_capture_configured(self) -> bool:
        """Whether graph capture is configured for the whole run."""
        return self._captured_graph_per_steps is not None

    @property
    def is_profiler_configured(self) -> bool:
        """Whether the runtime profiler is configured for the whole run."""
        return self._runtime_profiler is not None

    @property
    def is_c10d_trace_configured(self) -> bool:
        """Whether the c10d trace is configured for the whole run."""
        return self._c10d_trace is not None

    @property
    def is_graph_capture_active(self) -> bool:
        """Whether graph capture is configured and the current step is in its window."""
        if not self.is_graph_capture_configured:
            return False
        assert self._analysis_step_start is not None
        assert self._analysis_step_end is not None
        return self._analysis_step_start <= self._step < self._analysis_step_end

    @property
    def is_profiler_active(self) -> bool:
        """Whether the profiler is configured and its recording window is open."""
        return self.is_profiler_configured and self._runtime_profiler.is_active

    @property
    def is_span_record_active(self) -> bool:
        """Whether the span state machine should run right now.

        ``span record`` and ``graph capture`` are intentionally different:

        - span record is the base layer that cuts computation spans at
          communication launch/wait boundaries, allocates logical ``_proc`` and
          ``_comm`` span IDs, tracks coalescing tasks, and records pending
          ``span_desc``. The profiler consumes this state even when no graph
          file is being written, so profiler-only mode still aligns CUDA events
          and c10d completion-hook durations with the same logical span IDs.
        - graph capture is the persistence layer guarded by
          ``is_graph_capture_active``. Only graph capture writes ``CrystalGSpan``
          objects and dependencies into ``CrystalGraph`` for later serialization.

        Active requires both a consumer inside its analysis window (graph capture
        or the profiler) and the run being between ``step_start()`` and
        ``step_stop()``: spans may not be recorded outside the step lifecycle.
        """
        return self._in_step and (self.is_graph_capture_active or self.is_profiler_active)

    @property
    def is_comm_span_coalescing(self) -> bool:
        """Whether a coalescing manager is currently open with the span record active."""
        # The span state only exists once _reset_step_span_state() has run, which is
        # guaranteed whenever is_span_record_active is true, so the short-circuit
        # above makes the direct attribute access safe.
        return self.is_span_record_active and self._span_coalescing

    def step_start(self) -> None:
        """Advance to the next training step and reset per-step subsystem state.

        In order: close a proc span left open by a missing ``step_stop``, advance
        the step counter, mark the iteration in the c10d trace, reset each
        layer's per-step state, create the step's empty graph (graph capture),
        drive the runtime profiler window, and run the emulation rendezvous. A
        fresh initial proc span opens immediately when the span record is active.
        """
        if self.is_span_record_configured and self._span_proc_open:
            self.step_stop()

        self._step += 1

        if self.is_c10d_trace_configured:
            # Mark the iteration boundary in the trace so the replayer can align
            # captured operations with the training loop.
            self.cond_custom_write_c10d_trace("SCHEDULE", f"iteration {self._step} begin")

        if self.is_span_record_configured:
            self._reset_step_span_state()
        if self.is_graph_capture_configured:
            self._reset_step_graph_state()
        if self.is_profiler_configured:
            self._reset_step_issue_wait_state()

        if self.is_graph_capture_active:
            assert self._captured_graph_per_steps is not None
            step_graph = CrystalGraph()
            # The export refuses a graph without the value, and merging refuses to
            # combine graphs captured under different values.
            step_graph.set_max_connections(self._max_connections)
            self._captured_graph_per_steps[self._step] = step_graph

        if self.is_profiler_configured:
            self._runtime_profiler.step_start(self._step)

        if os.getenv("CRYSTAL_MODE") == "EMULATE":
            replay_rendezvous(self._step)

        self._in_step = True

        if self.is_span_record_active:
            self._open_proc_span([])

    def step_stop(self) -> None:
        """End the current training step, closing any open proc span.

        Every issue-wait measurement still outstanding is settled here: an
        asynchronous communication may never be waited inside its step (a
        pipeline schedule can leave the last send handle unwaited), and without
        a fallback the channel's duration array would silently lose its entry
        and shift every later span's duration. Such a communication is recorded
        as never waited, which reports duration 0 and draws a warning at
        profiler snapshot time.
        """
        if self.is_span_record_active and self._span_proc_open:
            self._close_proc_span()

        self._in_step = False

        if self.is_profiler_configured:
            # Settle before the step bracket closes: every outstanding measurement
            # was created while the profiler was configured.
            self._settle_outstanding_issue_wait()
            self._runtime_profiler.step_stop()

    def destroy(self) -> None:
        """Flush and persist every subsystem's artifacts at the end of the run.

        Order matters: the profiler's recording window is closed first but its file
        is written last, so a failed slot/completion match cannot take the already
        captured graph and trace down with it; in between, the c10d trace file is
        closed and the captured graphs are serialized per rank.
        """
        if self.is_span_record_configured and self._span_proc_open:
            self.step_stop()

        if self.is_profiler_configured:
            self._runtime_profiler.stop_recording()

        if self.is_c10d_trace_configured:
            self._c10d_trace.close_trace_file()

        if self.is_graph_capture_configured:
            assert not self.is_comm_span_coalescing
            assert self._captured_graph_per_steps is not None

            graph_per_steps_dict: dict[int, dict] = {}
            for step, step_graph in self._captured_graph_per_steps.items():
                if self._rank in step_graph.ranks:
                    graph_per_steps_dict[step] = step_graph.to_rank_dict(self._rank)
                else:
                    print(f"Crystal graph capture nothing for rank {self._rank} at step {step}")

            graph_dir = Path(os.getenv("CRYSTAL_LOG_DIR")) / "graph"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_file_name = "graph_" + str(self._rank) + ".json.gz"
            graph_file_path = graph_dir / graph_file_name

            with gzip.open(graph_file_path, "wt", encoding="utf-8") as f:
                json.dump(graph_per_steps_dict, f, indent=2)

            print(f"Crystal graph file stored at {graph_file_path}")

        if self.is_profiler_configured:
            self._runtime_profiler.save()

    # ---------------------------------------------------------------------------
    # Shared communication numbering (c10d trace / graph comm spans / profiler).
    # ---------------------------------------------------------------------------

    @property
    def last_trace_comm_index(self) -> int:
        """Last reserved communication trace index."""
        return self._comm_trace_counter_last

    def reserve_trace_comm_index(self) -> int:
        """Reserve a communication trace index shared by trace/graph/profiler.

        While a coalescing manager holds the counter frozen, every operation inside that
        manager reserves the manager's own index instead of a new one, so the whole
        manager stays one logical communication.
        """
        trace_index = self._comm_trace_counter
        self._comm_trace_counter_last = trace_index
        if not self._comm_trace_counter_frozen:
            self._comm_trace_counter += 1
        return trace_index

    @property
    def is_comm_trace_counter_frozen(self) -> bool:
        """Whether a coalescing manager currently holds the shared trace counter frozen."""
        return self._comm_trace_counter_frozen

    @property
    def _coalescing_trace_comm_index(self) -> int:
        """Communication trace index reserved by the coalescing manager in scope."""
        if not self._comm_trace_counter_frozen:
            raise RuntimeError("coalescing bookkeeping is used outside a coalescing manager")
        return self._comm_trace_counter_last

    # ---------------------------------------------------------------------------
    # c10d trace.
    # ---------------------------------------------------------------------------

    def cond_custom_write_c10d_trace(self, log_type: str, log_detail: str) -> None:
        """Conditionally write a custom trace entry if tracing is configured."""
        if self.is_c10d_trace_configured:
            self._c10d_trace._write_trace(log_type, log_detail)  # noqa: SLF001

    def cond_c10d_trace_writer(self, func: Callable[_P, _T], args: list[Any], kwargs: dict[str, Any]) -> None:
        """Conditionally trace collective communication operations under a reserved index.

        Every intercepted call reserves a fresh index here (the same one the capture-side
        graph span then reuses via ``last_trace_comm_index``); inside a coalescing
        manager the frozen counter keeps all operations on the manager's index.
        """
        if self.is_c10d_trace_configured:
            self._c10d_trace.c10d_trace_writer(self.reserve_trace_comm_index(), func, args, kwargs)

    def cond_c10d_trace_start_coalescing(
        self, group_name: str, group_desc: str, device_set: bool, async_ops: bool
    ) -> None:
        """Conditionally trace the start of a coalescing operation.

        Must be called inside the manager scope opened by ``cond_start_comm_span_coalescing``,
        whose reserved index is the one being traced.
        """
        if self.is_c10d_trace_configured:
            self._c10d_trace.trace_start_coalescing(
                self._coalescing_trace_comm_index, group_name, group_desc, device_set, async_ops
            )

    def cond_c10d_trace_end_coalescing(
        self, group_name: str, group_desc: str, device_set: bool, async_ops: bool
    ) -> None:
        """Conditionally trace the end of a coalescing operation.

        Must be called before ``cond_end_comm_span_coalescing`` releases the manager scope.
        """
        if self.is_c10d_trace_configured:
            self._c10d_trace.trace_end_coalescing(
                self._coalescing_trace_comm_index, group_name, group_desc, device_set, async_ops
            )

    # ---------------------------------------------------------------------------
    # Span record: the per-step span state machine.
    # ---------------------------------------------------------------------------

    def _reset_step_span_state(self) -> None:
        """Reset the span state machine for a new step.

        Exists only while span record is configured; graph capture and the
        profiler carry their own per-step resets.
        """
        # The open proc span and its chain bookkeeping.
        self._span_proc_prev_ids: list[CrystalGSpanId] = []
        self._span_proc_count = 0
        self._span_proc_open = False
        # The next proc span deferred by a communication cut, waiting for the
        # communication launch (see cond_start_proc_span_after_comm).
        self._span_pending_proc_prev_ids: list[CrystalGSpanId] | None = None
        # Async communications that already got their wait.
        self._span_waited_comm_ids: set[CrystalGSpanId] = set()
        # Coalescing manager scope.
        self._span_coalescing = False
        self._span_coalescing_tasks: list[CrystalGCommPattern] = []
        self._span_coalescing_entry_proc_span_id: CrystalGSpanId | None = None
        # Stop event of the proc span closed at manager entry (None without an
        # active profiler); reused as the issue-wait start event when the
        # manager's comm span is created at exit.
        self._span_coalescing_entry_stop_event: torch.cuda.Event | None = None

    def _open_proc_span(self, prev_span_ids: list[CrystalGSpanId]) -> torch.cuda.Event | None:
        """Open a new computation span chained on the given predecessor span IDs.

        The span is indexed by ``_span_proc_count`` (not advanced until the span
        ends) and the profiler's start CUDA event is recorded when the profiler
        window is open. The event is returned so callers can reuse it as the end
        point of an issue-wait duration. None entries in ``prev_span_ids`` are
        dropped.
        """
        self._span_proc_prev_ids = [span_id for span_id in prev_span_ids if span_id is not None]
        self._span_proc_open = True
        start_event = None
        if self.is_profiler_active:
            start_event = self._runtime_profiler.start_proc_span(self._span_proc_count)
        return start_event

    def _close_proc_span(self) -> tuple[CrystalGSpanId | None, torch.cuda.Event | None]:
        """Close the open computation span and return ``(span_id, stop_event)``.

        The ID is allocated even in profiler-only mode (where no graph is written),
        so async wait bookkeeping keeps working; only graph capture materializes the
        span, applies descriptions registered before the span existed, and wires the
        predecessor dependencies. The stop event is the one the profiler just
        recorded (None when the profiler window is closed) so callers can reuse it
        as the start point of an issue-wait duration. Returns ``(None, None)`` when
        no span is open.
        """
        if not self._span_proc_open:
            return None, None
        span_index = self._span_proc_count
        stop_event = None
        if self.is_profiler_active:
            stop_event = self._runtime_profiler.stop_proc_span(span_index)
        span_id = CrystalGSpanId(_PROC_TYPE_FIELD, span_index)
        if self.is_graph_capture_active:
            span_id = self._add_span_to_graph(
                SpanType.PROC,
                _PROC_TYPE_FIELD,
                span_index,
                None,
                self._span_proc_prev_ids,
            )
            self._apply_pending_span_descs(span_id)
        self._span_proc_count += 1
        self._span_proc_open = False
        return span_id, stop_event

    def _record_comm_span(
        self,
        trace_comm_index: int,
        comm_tasks: CrystalGCommTasks,
        proc_span_id: CrystalGSpanId | None,
        *,
        async_op: bool,
        issue_wait_start_event: torch.cuda.Event | None = None,
    ) -> CrystalGSpanId:
        """Allocate one comm span ID and defer the next proc span after it.

        ``proc_span_id`` is the proc span the caller has already closed, so the
        moment computation stops may be earlier than the moment the comm span
        becomes known: a coalescing manager launches its communication inside the
        manager scope, but the coalesced tasks are only complete when that scope
        is left.

        The comm span ID is allocated here (span record); graph capture
        materializes the span and its dependencies when it is active. The next
        proc span stays pending and is only started by
        ``cond_start_proc_span_after_comm`` once the caller has launched the real
        communication: starting it earlier would time a blocking synchronous
        launch inside the following computation span.

        Under ``max_connections == 1`` no channel-ordering dependency is drawn
        between the communication spans of one channel: their issue-wait
        durations are overlapping soft intervals that already contain the channel
        serialization, so the chain would double-count the overlap, and the order
        of a synchronous communication survives through the proc-span chain
        anyway.

        ``issue_wait_start_event`` drives the issue-wait duration measurement (see
        the profiler section): the stop event of the proc span just closed is the
        start point of the measurement, which ends at the next proc-span start
        event after the launch (synchronous) or after the wait (asynchronous). The
        argument changes neither the graph nor the span chain.
        """
        # The async-ness of the communication is graph data: record it on the tasks
        # the caller hands over, coalesced or single, so the graph keeps the launch
        # shape of every communication.
        comm_tasks.async_op = async_op
        comm_span_id = CrystalGSpanId(_COMM_TYPE_FIELD, trace_comm_index)
        if self.is_graph_capture_active:
            channel = comm_tasks.channel
            prev_span_ids = [] if proc_span_id is None else [proc_span_id]
            # Under max_connections == 1 every duration is an issue-wait interval that
            # already contains the channel serialization, and consecutive
            # communications' intervals overlap in reality; chaining the channel's
            # spans STOP-to-START would serialize those overlapping intervals and
            # inflate the total time. The synchronous order survives through the
            # proc-span chain, which links on a synchronous comm span, so the channel
            # chain is only drawn when connections overlap.
            if self._max_connections != 1 and channel in self._graph_channel_last_span_id:
                prev_span_ids.append(self._graph_channel_last_span_id[channel])
            comm_span_id = self._add_span_to_graph(
                SpanType.COMM, _COMM_TYPE_FIELD, trace_comm_index, comm_tasks, prev_span_ids
            )
            self._graph_channel_last_span_id[channel] = comm_span_id

        # Async comm lets computation overlap the communication until wait(), so the
        # pending proc span chains on the ended proc span; sync comm blocks inside
        # the launch, so it chains on the comm span itself.
        pending_prev_span_ids = ([] if proc_span_id is None else [proc_span_id]) if async_op else [comm_span_id]
        self._span_pending_proc_prev_ids = pending_prev_span_ids

        self.cond_start_issue_wait_measurement(comm_span_id, comm_tasks.channel, async_op, issue_wait_start_event)
        return comm_span_id

    def cond_record_comm_span(
        self,
        trace_comm_index: int,
        comm_tasks: CrystalGCommTasks,
        *,
        async_op: bool,
    ) -> CrystalGSpanId | None:
        """Record one communication span when span record is active; otherwise no-op.

        This only cuts the span chain at the communication launch: the next proc
        span stays pending until the caller launches the real communication and
        starts it with ``cond_start_proc_span_after_comm``.
        """
        if not self.is_span_record_active:
            return None
        proc_span_id, stop_event = self._close_proc_span()
        return self._record_comm_span(
            trace_comm_index,
            comm_tasks,
            proc_span_id,
            async_op=async_op,
            issue_wait_start_event=stop_event,
        )

    def cond_start_proc_span_after_comm(self) -> None:
        """Start the pending next proc span once the real communication launch returned.

        Pairs with ``cond_record_comm_span`` and ``cond_end_comm_span_coalescing``:
        both leave the next proc span pending before the real kernels are launched
        whenever a comm span exists, and this method starts it only once they are
        on the wire, so the duration of a blocking synchronous launch never lands
        inside the following computation span. Capture mode calls it right after
        its cut because fake work launches nothing at all, and an empty coalescing
        manager (no comm span) has already started its next proc span, so for both
        this is a no-op. This is the launch-side counterpart of the wait-side
        bracket in ``cond_start/end_comm_span_wait``. No-op when span record is
        inactive or nothing is pending.
        """
        if not self.is_span_record_active:
            return
        pending_prev_span_ids = self._span_pending_proc_prev_ids
        if pending_prev_span_ids is None:
            return
        self._span_pending_proc_prev_ids = None
        start_event = self._open_proc_span(pending_prev_span_ids)
        self.cond_settle_pending_issue_wait(start_event)

    def cond_start_comm_span_coalescing(self) -> None:
        """Enter one coalescing manager: reserve its trace index and close the current proc span.

        The whole manager is one logical communication, so its communication trace index
        is reserved here and the counter is frozen until
        ``cond_end_comm_span_coalescing`` releases it. Reserving is unconditional because
        the c10d trace numbers operations outside the analysis window as well; cutting
        the computation span only happens while span record is active. The coalesced
        operations are launched inside the manager scope, so computation has to stop when
        the manager is entered even though the comm span can only be materialized when it
        is left.
        """
        self._comm_trace_counter_frozen = True
        self.reserve_trace_comm_index()
        if not self.is_span_record_active or self._span_coalescing:
            return
        self._span_coalescing = True
        self._span_coalescing_tasks = []
        (
            self._span_coalescing_entry_proc_span_id,
            self._span_coalescing_entry_stop_event,
        ) = self._close_proc_span()

    def cond_add_coalescing_comm_task(self, task: CrystalGCommPattern) -> None:
        """Collect one coalesced comm task when coalescing is active; otherwise no-op."""
        if self.is_span_record_active and self._span_coalescing:
            self._span_coalescing_tasks.append(task)

    def cond_end_comm_span_coalescing(self, channel: str | None, *, async_op: bool) -> CrystalGSpanId | None:
        """Leave one coalescing manager: record the coalesced comm span and release the trace index.

        The comm span reuses the trace index reserved when the manager was entered, and
        the counter advances past it here so the next operation gets a fresh index. None
        is returned when the manager becomes no comm span at all: span record is
        inactive, no task was collected, or ``channel`` is None because the caller must
        not turn the collected tasks into a communication. With no comm span there is
        no communication whose launch time must stay out of the next proc span, so it
        is started right away and the caller's later
        ``cond_start_proc_span_after_comm`` finds nothing pending; only a
        materialized comm span leaves the next proc span pending until the manager's
        real work is on the wire.
        """
        trace_comm_index = self._coalescing_trace_comm_index
        comm_span_id: CrystalGSpanId | None = None
        if self.is_span_record_active and self._span_coalescing:
            self._span_coalescing = False
            comm_tasks = self._span_coalescing_tasks
            self._span_coalescing_tasks = []
            proc_span_id = self._span_coalescing_entry_proc_span_id
            self._span_coalescing_entry_proc_span_id = None
            if channel is None or len(comm_tasks) == 0:
                # No comm span exists, so there is no communication launch whose blocking
                # time must be kept out of the next proc span: start it right away
                # instead of leaving it pending, which would drop the manager's launch
                # time into the gap between spans. The caller's later
                # cond_start_proc_span_after_comm() becomes a no-op because
                # nothing is pending.
                self._open_proc_span([] if proc_span_id is None else [proc_span_id])
            else:
                comm_span_id = self._record_comm_span(
                    trace_comm_index,
                    CrystalGCommTasks(channel=channel, tasks=comm_tasks),
                    proc_span_id,
                    async_op=async_op,
                    issue_wait_start_event=self._span_coalescing_entry_stop_event,
                )
        self._span_coalescing_entry_stop_event = None
        self._comm_trace_counter_frozen = False
        self.reserve_trace_comm_index()
        return comm_span_id

    def cond_start_comm_span_wait(self, comm_span_id: CrystalGSpanId | None) -> CrystalGSpanId | None:
        """Close the current proc span before an async wait; returns the closed span ID.

        Computation stops at the wait call, so the span must be cut here; waiting
        twice on the same comm span would corrupt the chain and raises. No-op (and
        None) when the span record is inactive or there is no comm span to wait on.
        """
        if comm_span_id is None or not self.is_span_record_active:
            return None
        if comm_span_id in self._span_waited_comm_ids:
            raise RuntimeError(f"Communication span {comm_span_id} has already been waited")
        proc_span_id, _ = self._close_proc_span()
        return proc_span_id

    def cond_end_comm_span_wait(
        self,
        comm_span_id: CrystalGSpanId | None,
        proc_span_id: CrystalGSpanId | None,
    ) -> None:
        """Open the next proc span after an async wait when span record is active; otherwise no-op.

        The new span chains on both the closed proc span and the waited comm span:
        the comm span guarantees the communication is complete before computation
        resumes. Its start event closes the in-flight issue-wait measurement of
        that communication (see the profiler section).
        """
        if comm_span_id is None or not self.is_span_record_active:
            return
        self._span_waited_comm_ids.add(comm_span_id)
        prev_span_ids = [] if proc_span_id is None else [proc_span_id]
        prev_span_ids.append(comm_span_id)
        start_event = self._open_proc_span(prev_span_ids)
        self.cond_settle_in_flight_issue_wait(comm_span_id, start_event)

    def cond_comm_span_wait(self, comm_span_id: CrystalGSpanId | None) -> None:
        """Process wait() for a fake async communication span: the start/end pair in one call."""
        proc_span_id = self.cond_start_comm_span_wait(comm_span_id)
        self.cond_end_comm_span_wait(comm_span_id, proc_span_id)

    # ---------------------------------------------------------------------------
    # Graph capture: span and dependency persistence into per-step CrystalGraph.
    # ---------------------------------------------------------------------------

    def _reset_step_graph_state(self) -> None:
        """Reset graph-capture chain state for a new step.

        Exists only while graph capture is configured; the step's ``CrystalGraph``
        itself is created separately, and only for steps inside the analysis
        window.
        """
        # Last span overall: feeds the START-to-START chain under max_connections == 1.
        self._graph_last_span_id: CrystalGSpanId | None = None
        # Previous comm span of each channel: feeds the channel STOP-to-START
        # chain whenever max_connections is not 1.
        self._graph_channel_last_span_id: dict[str, CrystalGSpanId] = {}
        # Descriptions registered before their proc span materialized in the graph.
        self._graph_pending_span_descs: dict[CrystalGSpanId, list[str]] = {}

    @property
    def _this_step_captured_graph(self) -> CrystalGraph:
        """Graph of the current step.

        Only valid while ``is_graph_capture_active`` is true: every active step got
        its empty ``CrystalGraph`` in ``step_start``, while inactive steps never
        entered the dict and would raise a KeyError.
        """
        assert self._captured_graph_per_steps is not None
        return self._captured_graph_per_steps[self._step]

    def _add_span_to_graph(
        self,
        span_type: SpanType,
        type_field: str,
        count_field: int,
        comm_tasks: CrystalGCommTasks | None,
        prev_span_ids: list[CrystalGSpanId],
    ) -> CrystalGSpanId:
        """Materialize one span in the captured graph with its happens-before chain.

        Dependencies: every predecessor in ``prev_span_ids`` contributes a
        STOP(prev) -> START(new) edge, the basic happens-before chain between spans.
        """
        span_id = CrystalGSpanId(type_field=type_field, count_field=count_field)

        new_span = CrystalGSpan(span_type=span_type, span_id=span_id, comm_tasks=comm_tasks, duration_ns=0)
        self._this_step_captured_graph.add_span(self._rank, new_span)
        for prev_id in prev_span_ids:
            self._this_step_captured_graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=CrystalGSpanPoint(self._rank, prev_id, SpanPointType.STOP),
                    v2=CrystalGSpanPoint(self._rank, new_span.span_id, SpanPointType.START),
                )
            )
        # With a single CUDA connection the device cannot overlap communication with
        # computation, so the replay scheduler must not overlap any spans: chain the
        # new span's START to the previous span's START, forcing one global order
        # that the scheduler has to respect.
        if self._max_connections == 1 and self._graph_last_span_id is not None:
            self._this_step_captured_graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=CrystalGSpanPoint(self._rank, self._graph_last_span_id, SpanPointType.START),
                    v2=CrystalGSpanPoint(self._rank, new_span.span_id, SpanPointType.START),
                )
            )
        self._graph_last_span_id = new_span.span_id
        return new_span.span_id

    def _apply_pending_span_descs(self, span_id: CrystalGSpanId) -> None:
        """Attach descriptions registered before the span existed, now that it does."""
        descriptions = self._graph_pending_span_descs.pop(span_id, [])
        for description in descriptions:
            self._add_span_desc_to_graph(span_id, description)

    def cond_graph_mark_start(self) -> int | None:
        """Return the first capturable proc span index for graph mark, or None when inactive."""
        if self.is_graph_capture_active:
            return self._span_proc_count
        return None

    def cond_graph_mark_stop(self, start_proc_index: int | None, description: str) -> None:
        """Attach the description to every proc span opened since ``cond_graph_mark_start``.

        With a span still open, ``_span_proc_count`` is the index of that
        not-yet-ended span; with all spans closed, ``count - 1`` is the last ended
        one. Either way the covered range is ``[start_proc_index, end_proc_index]``.
        No-op when graph capture is inactive.
        """
        if start_proc_index is None or not self.is_graph_capture_active:
            return
        end_proc_index = self._span_proc_count if self._span_proc_open else self._span_proc_count - 1
        for proc_index in range(start_proc_index, end_proc_index + 1):
            self._register_proc_span_desc(proc_index, description)

    def _register_proc_span_desc(self, proc_index: int, description: str) -> None:
        """Register a description for one proc span, now or once the span exists.

        A span already materialized in the graph is annotated directly; a span that
        is still open (or not yet created) is not in the graph, so the description
        waits in ``_graph_pending_span_descs`` until ``_close_proc_span`` applies it
        through ``_apply_pending_span_descs``.
        """
        span_id = CrystalGSpanId(_PROC_TYPE_FIELD, proc_index)
        if (
            self._rank in self._this_step_captured_graph.ranks
            and span_id in self._this_step_captured_graph.ranks[self._rank].spans
        ):
            self._add_span_desc_to_graph(span_id, description)
        else:
            self._graph_pending_span_descs.setdefault(span_id, []).append(description)

    def _add_span_desc_to_graph(self, span_id: CrystalGSpanId, description: str) -> None:
        """Write the description onto an existing span in the captured graph."""
        self._this_step_captured_graph.add_span_desc(self._rank, span_id, description)

    # ---------------------------------------------------------------------------
    # Runtime profiler: PROC/COMM span durations keyed by span-record IDs.
    # ---------------------------------------------------------------------------

    def _reset_step_issue_wait_state(self) -> None:
        """Reset the issue-wait bookkeeping for a new step.

        Exists only while the profiler is configured: no measurement can be
        created without the profiler, and none may outlive its step.
        """
        # One measurement right after a communication cut, before the real launch
        # returned; keyed by trace_comm_index once the launch turned out asynchronous,
        # until its wait.
        self._issue_wait_pending: _IssueWaitMeasurement | None = None
        self._issue_wait_in_flight: dict[int, _IssueWaitMeasurement] = {}

    def cond_start_issue_wait_measurement(
        self,
        comm_span_id: CrystalGSpanId,
        channel: str,
        async_op: bool,
        start_event: torch.cuda.Event | None,
    ) -> None:
        """Create the pending issue-wait measurement of one communication, if qualified.

        Qualification, narrowing one guarantee at a time: the profiler exists, its
        window is open (``start_event`` is the stop event of the just-closed proc
        span, recorded only inside the window), and it measures communications
        through issue-wait durations (``max_connections == 1``, where timing
        events would serialize them). No-op while the profiler is not configured.
        """
        if not (
            self.is_profiler_configured and start_event is not None and self._runtime_profiler.uses_issue_wait_timing
        ):
            return
        self._issue_wait_pending = _IssueWaitMeasurement(
            trace_comm_index=comm_span_id.count_field,
            channel=channel,
            step=self._step,
            async_op=async_op,
            start_event=start_event,
        )

    def cond_settle_pending_issue_wait(self, start_event: torch.cuda.Event | None) -> None:
        """Settle the pending measurement once the real launch returned.

        A synchronous communication ends here: the just recorded start event is its
        end point, so the measurement is handed to the profiler. An asynchronous one
        keeps running until its wait, so the measurement moves to
        ``_issue_wait_in_flight`` keyed by the comm index; its end point is the
        start event that ``cond_end_comm_span_wait`` records after the wait.
        No-op while the profiler is not configured.
        """
        if not self.is_profiler_configured:
            return
        measurement = self._issue_wait_pending
        if measurement is None:
            return
        self._issue_wait_pending = None
        if measurement.async_op:
            self._issue_wait_in_flight[measurement.trace_comm_index] = measurement
            return
        # A pending measurement only exists while the profiler is configured with
        # its window open, so the launch-side start event must have been recorded.
        assert self._runtime_profiler is not None
        assert start_event is not None
        self._runtime_profiler.record_issue_wait_comm_span(
            step=measurement.step,
            channel=measurement.channel,
            trace_comm_index=measurement.trace_comm_index,
            start_event=measurement.start_event,
            end_event=start_event,
            async_op=measurement.async_op,
        )

    def cond_settle_in_flight_issue_wait(
        self, comm_span_id: CrystalGSpanId, start_event: torch.cuda.Event | None
    ) -> None:
        """Settle the in-flight measurement of one communication after its wait.

        No-op while the profiler is not configured.
        """
        if not self.is_profiler_configured:
            return
        measurement = self._issue_wait_in_flight.pop(comm_span_id.count_field, None)
        if measurement is None:
            return
        # An in-flight measurement only exists while the profiler is configured
        # with its window open, so the wait-side start event must have been recorded.
        assert self._runtime_profiler is not None
        assert start_event is not None
        self._runtime_profiler.record_issue_wait_comm_span(
            step=measurement.step,
            channel=measurement.channel,
            trace_comm_index=measurement.trace_comm_index,
            start_event=measurement.start_event,
            end_event=start_event,
            async_op=measurement.async_op,
        )

    def _settle_outstanding_issue_wait(self) -> None:
        """Settle every measurement still outstanding at the step boundary.

        An outstanding measurement exists for each communication issued in this
        step whose wait has not returned yet (or never happens at all). Each is
        recorded as a never-waited communication with its issue-side start event:
        it keeps its position in the channel's issue-ordered duration array but is
        deliberately reported with duration 0, since nothing in the graph chains
        on the STOP of an unwaited communication.

        Only called from the profiler bracket of ``step_stop``, so the profiler
        is configured and the issue-wait bookkeeping exists.
        """
        outstanding: list[_IssueWaitMeasurement] = []
        if self._issue_wait_pending is not None:
            outstanding.append(self._issue_wait_pending)
            self._issue_wait_pending = None
        outstanding.extend(self._issue_wait_in_flight.values())
        self._issue_wait_in_flight.clear()
        if not outstanding:
            return
        # Every outstanding measurement was created while the profiler was
        # configured with its window open, so the profiler and each start event
        # must be there.
        assert self._runtime_profiler is not None
        for measurement in outstanding:
            self._runtime_profiler.record_unwaited_comm_span(
                step=measurement.step,
                channel=measurement.channel,
                trace_comm_index=measurement.trace_comm_index,
                start_event=measurement.start_event,
            )

    def cond_record_profiler_comm_slot(self, group: Any, channel: str) -> None:
        """Record a profiler comm slot when the profiler is active; otherwise no-op.

        Every communication on a recording group is recorded, whatever its graph
        span status, so the profiler can match it with the completion-hook record.
        """
        if self.is_profiler_active:
            self._runtime_profiler.record_comm_slot(
                group,
                step=self._step,
                channel=channel,
            )


def initialize_crystal_runtime() -> None:
    """Initialize the runtime context."""
    _ = CrystalRuntimeContext()


def crystal_runtime_step_start() -> None:
    """Start the current Crystal runtime step."""
    CrystalRuntimeContext().step_start()


def crystal_runtime_step_stop() -> None:
    """Stop the current Crystal runtime step."""
    CrystalRuntimeContext().step_stop()


def destroy_crystal_runtime() -> None:
    """Destroy the runtime context."""
    CrystalRuntimeContext().destroy()


def get_context() -> CrystalRuntimeContext:
    """Get the CrystalRuntimeContext singleton instance."""
    return CrystalRuntimeContext()
