# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Per-rank span construction state machine shared by the pipeline policy builders.

The builder replays the span-cutting state machine that CrystalRuntimeContext
(runtime/context.py) runs under collective_handler.py, so a policy graph and the
runtime capture of the same schedule produce the same spans and dependencies:
a rank always has one open ``_proc`` computation span, which every communication
launch or wait boundary cuts (empty spans included), communication spans are
numbered ``_comm`` on one shared per-rank counter and connect themselves to the
just-closed computation span and to the previous span issued on their channel,
and the schedule's compute steps only attach their semantics ("FWD"/"BWD") as
``span_desc`` graph marks, never as span identities.

A schedule therefore never threads ``prev_span_ids`` through its steps; it marks
compute steps with graph_mark(), launches communication with add_comm_span(),
and draws wait boundaries with comm_span_wait(), exactly where schedules.py
executes or waits the corresponding operation.
"""

from __future__ import annotations

from ..graph import (
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

PROC_TYPE_FIELD = "_proc"
COMM_TYPE_FIELD = "_comm"


class RankSpanBuilder:
    """Build the spans of one rank the way runtime graph capture cuts them.

    Method mapping to CrystalRuntimeContext:
    - ``start_proc_span``/``end_proc_span``: ``_open_proc_span`` and
      ``_close_proc_span``, materializing every computation span including
      the empty ones a boundary opens, so the span sequence matches a capture.
    - ``add_comm_span``: ``cond_record_comm_span``, numbering comm spans on
      the rank's shared comm counter like the runtime's comm trace counter.
    - ``comm_span_wait``: the ``cond_start/end_comm_span_wait`` pair that
      ``Work.wait()`` triggers under the runtime collective handlers.
    - ``graph_mark_start``/``graph_mark_stop``: ``cond_graph_mark_start`` and
      ``cond_graph_mark_stop``, attaching descriptions to the computation spans
      a marked region covers, with the same pending mechanism for spans that
      materialize later.
    - ``step_start``/``step_stop``: the span-record part of ``step_start`` and
      ``step_stop``.
    - ``_last_span``: ``_graph_last_span_id``, the START-to-START chain of
      consecutive spans when max_connections is 1.
    - ``_channel_last_span``: ``_graph_channel_last_span_id``, the previous comm
      span of each channel, which sequences a channel's operations in issue order
      whenever max_connections is not 1 (under 1 the issue-wait durations already
      contain the channel serialization, so the chain would double-count the
      overlap of consecutive communications).
    - ``_waited_comm_span_ids``: ``_span_waited_comm_ids``, guarding that
      a schedule waits every async communication span at most once.
    """

    def __init__(self, graph: CrystalGraph, rank: int, max_connections: int) -> None:
        """Create the builder of one rank's schedule inside a graph.

        Args:
            graph: The graph the spans and dependencies are added to.
            rank: The global rank whose schedule is being built.
            max_connections: If 1, chain consecutive spans with a START-to-START
                dependency, like the runtime does under one CUDA connection.
        """
        self._graph = graph
        self._rank = rank
        self._max_connections = max_connections
        self._last_span: CrystalGSpanId | None = None
        self._channel_last_span: dict[str, CrystalGSpanId] = {}
        self._proc_span_open = False
        self._current_proc_prev_span_ids: list[CrystalGSpanId] = []
        self._proc_span_count = 0
        self._comm_span_count = 0
        self._waited_comm_span_ids: set[CrystalGSpanId] = set()
        self._pending_span_desc: dict[CrystalGSpanId, list[str]] = {}

    def _add_span(
        self,
        span_type: SpanType,
        type_field: str,
        count_field: int,
        comm_tasks: CrystalGCommTasks | None,
        prev_span_ids: list[CrystalGSpanId],
    ) -> CrystalGSpanId:
        """Add one span with STOP-to-START dependencies on its predecessors."""
        new_span = CrystalGSpan(
            span_type=span_type,
            span_id=CrystalGSpanId(type_field=type_field, count_field=count_field),
            comm_tasks=comm_tasks,
            duration_ns=0,
        )
        self._graph.add_span(self._rank, new_span)
        for prev_id in prev_span_ids:
            self._graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=CrystalGSpanPoint(self._rank, prev_id, SpanPointType.STOP),
                    v2=CrystalGSpanPoint(self._rank, new_span.span_id, SpanPointType.START),
                )
            )
        # Add START-to-START dependency from previous span if max_connections=1
        if self._max_connections == 1 and self._last_span is not None:
            self._graph.add_dependency(
                CrystalGDep(
                    dep_type=DependencyType.UNIDIRECT,
                    v1=CrystalGSpanPoint(self._rank, self._last_span, SpanPointType.START),
                    v2=CrystalGSpanPoint(self._rank, new_span.span_id, SpanPointType.START),
                )
            )
        self._last_span = new_span.span_id
        return new_span.span_id

    def step_start(self) -> None:
        """Open the step's leading computation span (span-record part of ``step_start``)."""
        self.start_proc_span()

    def step_stop(self) -> None:
        """Materialize the step's trailing computation span (span-record part of ``step_stop``)."""
        self.end_proc_span()

    def start_proc_span(self, prev_span_ids: list[CrystalGSpanId] | None = None) -> None:
        """Open the next computation span (``_open_proc_span``)."""
        self._current_proc_prev_span_ids = [span_id for span_id in prev_span_ids or [] if span_id is not None]
        self._proc_span_open = True

    def end_proc_span(self) -> CrystalGSpanId | None:
        """Close and materialize the open computation span (``_close_proc_span``).

        The span is materialized even when no graph mark named it: a boundary
        that opens an empty computation span cuts the rank's timeline the same
        way at capture time, so the empty span stays in the graph.
        """
        if not self._proc_span_open:
            return None
        span_id = self._add_span(
            SpanType.PROC,
            PROC_TYPE_FIELD,
            self._proc_span_count,
            None,
            self._current_proc_prev_span_ids,
        )
        self._apply_pending_proc_span_desc(self._proc_span_count)
        self._proc_span_count += 1
        self._proc_span_open = False
        return span_id

    def add_comm_span(
        self,
        comm_tasks: CrystalGCommTasks,
        *,
        async_op: bool,
        description: str | None = None,
    ) -> CrystalGSpanId:
        """Add one communication span (``cond_record_comm_span``).

        Ends the open computation span, sequences the new span after it, then opens
        the next computation span: after the comm span itself when the operation is
        synchronous, after the closed computation span when it is async and a later
        ``comm_span_wait`` draws the boundary. The previous span of the same channel
        also sequences the new one whenever max_connections is not 1; under 1 the
        issue-wait durations already contain the channel serialization, so that
        chain is left out (see the class docstring).

        Args:
            comm_tasks: Channel and communication patterns of the operation.
            async_op: Whether the schedule waits the operation right away. Recorded
                on the comm tasks like the runtime context does, so the graph keeps
                the launch shape of every communication.
            description: Schedule-level annotation (e.g. the p2p operation's
                name) recorded as span description; pure annotation that no
                timing query reads, like every span_desc entry.

        Returns:
            The new communication span's ID.
        """
        comm_tasks.async_op = async_op
        proc_span_id = self.end_proc_span()
        prev_span_ids = [] if proc_span_id is None else [proc_span_id]
        channel = comm_tasks.channel
        # Mirror the runtime: under max_connections == 1 the issue-wait durations
        # already contain the channel serialization, so chaining the channel's
        # spans would double-count the overlap of consecutive communications; a
        # synchronous exchange keeps its order through the proc-span chain, which
        # links on its comm span.
        if self._max_connections != 1 and channel in self._channel_last_span:
            prev_span_ids.append(self._channel_last_span[channel])
        comm_span_id = self._add_span(SpanType.COMM, COMM_TYPE_FIELD, self._comm_span_count, comm_tasks, prev_span_ids)
        self._comm_span_count += 1
        self._channel_last_span[channel] = comm_span_id
        if description is not None:
            self._graph.add_span_desc(self._rank, comm_span_id, description)
        if async_op:
            self.start_proc_span([] if proc_span_id is None else [proc_span_id])
        else:
            self.start_proc_span([comm_span_id])
        return comm_span_id

    def comm_span_wait(self, comm_span_id: CrystalGSpanId | None) -> None:
        """Wait an async communication span (the runtime wait boundary pair).

        Cuts the open computation span at the wait and opens the next one after
        both the closed span and the waited communication, like ``Work.wait()``
        does under the runtime collective handlers.
        """
        if comm_span_id is None:
            return
        if comm_span_id in self._waited_comm_span_ids:
            raise RuntimeError(f"Communication span {comm_span_id} has already been waited")
        self._waited_comm_span_ids.add(comm_span_id)
        proc_span_id = self.end_proc_span()
        prev_span_ids = [] if proc_span_id is None else [proc_span_id]
        prev_span_ids.append(comm_span_id)
        self.start_proc_span(prev_span_ids)

    def graph_mark_start(self) -> int:
        """Return the computation span index the marked region starts at (``cond_graph_mark_start``)."""
        return self._proc_span_count

    def graph_mark_stop(self, start_index: int, description: str) -> None:
        """Attach a description to the spans a marked region covers (``cond_graph_mark_stop``).

        The region covers every computation span materialized since start_index,
        including the currently open one, so a step that a communication boundary
        cuts in the middle ends up described on all of its pieces.
        """
        end_index = self._proc_span_count if self._proc_span_open else self._proc_span_count - 1
        for proc_index in range(start_index, end_index + 1):
            self._record_proc_span_desc(proc_index, description)

    def graph_mark(self, description: str) -> None:
        """Mark an instantaneous compute step, like the runtime's graph mark decorator.

        A Megatron compute step of the modeled schedules runs without a
        communication boundary inside it, so its mark covers exactly the
        computation span that is open at this point. When no boundary separates
        two consecutive steps, their marks land on the same span, which is also
        what a capture of that schedule records.
        """
        self.graph_mark_stop(self.graph_mark_start(), description)

    def _record_proc_span_desc(self, proc_index: int, description: str) -> None:
        span_id = CrystalGSpanId(PROC_TYPE_FIELD, proc_index)
        if self._rank in self._graph.ranks and span_id in self._graph.ranks[self._rank].spans:
            self._graph.add_span_desc(self._rank, span_id, description)
        else:
            self._pending_span_desc.setdefault(span_id, []).append(description)

    def _apply_pending_proc_span_desc(self, proc_index: int) -> None:
        span_id = CrystalGSpanId(PROC_TYPE_FIELD, proc_index)
        for description in self._pending_span_desc.pop(span_id, []):
            self._graph.add_span_desc(self._rank, span_id, description)
