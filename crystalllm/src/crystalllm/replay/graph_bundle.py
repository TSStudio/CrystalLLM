# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Replay Graph Bundle.

Loads the merged CrystalGraph of every useful PP chain and exposes the queries the
replay side needs: the step an operation belongs to, its synchronization group, the
launch time of that group, and the arrivals a rank may have to wait for. A chain is
useful when it holds a sandbox rank, a launched emulated rank, or runtime profiler
data. Depends only on the planning graph and the trace parser, so plotting tools can
load graphs without going through the assistant.
"""

from __future__ import annotations

import gzip
import json
import logging
import pathlib
import time
from collections import defaultdict

from ..planning import (
    CrystalGraph,
    CrystalGSpanId,
    CrystalGSpanPoint,
    CrystalGSyncSpanGroup,
    SpanPointType,
)
from ..planning.graph import UNSET_MAX_CONNECTIONS
from .parser import CrystalTraceParser, resolve_channel_members

logger = logging.getLogger(__name__)

# Span type_field values shared by the captured graph files and the runtime profiler
# files: "_proc" marks a compute span, "_comm" a communication span. Centralizing them
# keeps graph span IDs and profiler duration sections keyed identically.
PROC_TYPE_FIELD = "_proc"
COMM_TYPE_FIELD = "_comm"

# Process group name of the default (world) process group in the c10d trace.
DEFAULT_GROUP_NAME = "0"


def pp_chain_of_rank(rank: int, pp_rank_stride: int) -> int:
    """Return the PP chain a rank belongs to.

    Ranks are laid out so that one PP chain is every stride-th rank: chain, chain +
    pp_rank_stride, chain + 2 * pp_rank_stride, ... The chain index is therefore the
    rank modulo the stride, and the ranks of chain c are range(c, world_size,
    pp_rank_stride).
    """
    return rank % pp_rank_stride


def _graph_file(graph_dir: str, rank: int) -> pathlib.Path:
    """Return the captured graph file of a rank."""
    return pathlib.Path(graph_dir) / f"graph_{rank}.json.gz"


def _trace_file(trace_dir: str, rank: int) -> pathlib.Path:
    """Return the c10d trace file of a rank."""
    return pathlib.Path(trace_dir) / f"trace_{rank}.log.gz"


def _profiler_file(profiler_dir: str, rank: int) -> pathlib.Path:
    """Return the runtime profiler file of a rank."""
    return pathlib.Path(profiler_dir) / f"runtime_profiler_{rank}.json.gz"


def _load_gzip_json(path: pathlib.Path) -> dict:
    """Read a gzipped JSON file into a dict."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class CrystalGraphBundle:
    """Merged per-step graphs of all useful PP chains plus the replay-side queries."""

    def __init__(
        self,
        step_graphs: dict[int, CrystalGraph],
        trace_comm_index_steps: dict[tuple[int, int], int],
    ) -> None:
        """Initialize the bundle from already merged and analyzed per-step graphs.

        Every graph must have had set_sandbox_ranks() and analyze() called already:
        all query methods assume the synchronization groups, anchorable points and
        longest-path caches exist. load() is the normal construction path; building a
        bundle directly is meant for graphs prepared by other means.

        Every step graph must carry its max_connections (set_max_connections()) and
        all of them must agree: the graphs model one device overlap capability, and
        mixing steps captured under different values would time one replay against
        several incompatible serializations.

        Args:
            step_graphs: Analyzed merged graph per step, keyed by step number.
            trace_comm_index_steps: Mapping from (rank, trace_comm_index) of every COMM span
                in the graphs to its step. A key that is absent means the operation is
                outside the analysis window.

        Raises:
            ValueError: If the step graphs disagree on max_connections, or any of
                them never had it set.
        """
        max_connections_values = {graph.max_connections for graph in step_graphs.values()}
        if len(max_connections_values) != 1:
            raise ValueError(
                "A bundle requires one max_connections across its step graphs, got "
                f"{sorted(max_connections_values)} for steps {sorted(step_graphs)}"
            )
        (max_connections,) = max_connections_values
        if max_connections == UNSET_MAX_CONNECTIONS:
            raise ValueError("Every step graph must have max_connections set before bundling")

        self.step_graphs = step_graphs
        self._trace_comm_index_steps = trace_comm_index_steps
        self._max_connections = max_connections

    @property
    def max_connections(self) -> int:
        """The CUDA_DEVICE_MAX_CONNECTIONS the bundled step graphs were captured under.

        Construction verified that every step graph carries the same value, so the
        property is safe to branch replay behavior on.
        """
        return self._max_connections

    @classmethod
    def load(
        cls,
        *,
        graph_dir: str,
        trace_dir: str,
        profiler_dir: str | None,
        pp_rank_stride: int,
        sandbox_ranks: list[int],
        world_size: int,
        own_rank: int,
        launched_ranks: list[int] | None = None,
        new_group_table: dict[str, list[int]] | None = None,
    ) -> CrystalGraphBundle:
        """Discover, prune, validate and merge the graphs of all useful PP chains.

        Pipeline:
        1. Check that pp_rank_stride divides world_size, so every chain has the same
           number of stages and the per-step merge below is well defined.
        2. Collect the mandatory chains: those holding a sandbox rank, a launched
           emulated rank, or the own rank.
        3. Prune the remaining chains that have no runtime profiler data either (see
           _select_chains), and read the baseline step list from the own rank's graph.
        4. Load every candidate chain all-or-nothing (see _load_chain); a rejected
           chain is skipped, unless it is the own rank's chain, which is fatal.
        5. Merge, for each baseline step, the graph payload of every loaded rank into
           one CrystalGraph and inject the channel member tables.
        6. Overwrite span durations from the runtime profiler files (see
           _load_profiler_durations). This runs before analyze() so the longest-path
           weights are built from the measured durations.
        7. Index every COMM span as (rank, trace_comm_index) -> step, then run
           set_sandbox_ranks() + analyze() on each merged graph, which every bundle
           query requires.

        Args:
            graph_dir: Directory holding graph_{rank}.json.gz.
            trace_dir: Directory holding trace_{rank}.log.gz.
            profiler_dir: Directory holding runtime_profiler_{rank}.json.gz, or None to
                keep every span duration at 0.
            pp_rank_stride: Rank distance between adjacent PP stages, i.e. the number of
                ranks in one PP stage, which is also the number of PP chains.
            sandbox_ranks: Ranks running the real training program.
            world_size: Total number of ranks in the target scenario.
            own_rank: The rank owning this bundle; its chain must load successfully.
            launched_ranks: Emulated ranks actually started for this run. Their chains
                are mandatory. None means every emulated rank is started.
            new_group_table: Process group name to member global ranks. Parsed from the
                own rank trace when not provided.

        Returns:
            A bundle whose per-step graphs are analyzed and have sandbox ranks applied.

        Raises:
            ValueError: If pp_rank_stride does not divide world_size, if the own rank's
                PP chain is unusable, if the runtime profiler data references a step
                outside the graphs, or if the merged step graphs disagree on
                max_connections (checked when the bundle is assembled).
        """
        if world_size % pp_rank_stride != 0:
            raise ValueError(
                f"pp_rank_stride {pp_rank_stride} must divide world size {world_size} so that every "
                f"PP chain has the same number of stages"
            )

        load_start = time.perf_counter()
        if launched_ranks is None:
            launched_ranks = [rank for rank in range(world_size) if rank not in set(sandbox_ranks)]

        mandatory_chains = {
            pp_chain_of_rank(rank, pp_rank_stride) for rank in [*sandbox_ranks, *launched_ranks, own_rank]
        }
        candidate_chains, pruned_chains = cls._select_chains(
            pp_rank_stride=pp_rank_stride,
            world_size=world_size,
            mandatory_chains=mandatory_chains,
            profiler_dir=profiler_dir,
        )

        baseline_steps = sorted(int(step) for step in _load_gzip_json(_graph_file(graph_dir, own_rank)))
        loaded_ranks: list[int] = []
        loaded_chains: list[int] = []
        rank_graphs: dict[int, dict[int, dict]] = {}

        for chain in candidate_chains:
            chain_ranks = list(range(chain, world_size, pp_rank_stride))
            chain_graphs, rejection = cls._load_chain(graph_dir, trace_dir, chain_ranks, baseline_steps)
            if rejection is not None:
                if chain == pp_chain_of_rank(own_rank, pp_rank_stride):
                    raise ValueError(
                        f"PP chain {chain} of the own rank {own_rank} is unusable: {rejection}. "
                        f"The assistant cannot schedule its own operations without it."
                    )
                logger.warning("Skipping PP chain %s: %s", chain, rejection)
                continue
            rank_graphs.update(chain_graphs)
            loaded_ranks.extend(chain_ranks)
            loaded_chains.append(chain)

        loaded_ranks.sort()
        logger.info(
            "Loaded graph files of %d/%d candidate PP chains in %.1fs",
            len(loaded_chains),
            len(candidate_chains),
            time.perf_counter() - load_start,
        )

        if new_group_table is None:
            group_parse_start = time.perf_counter()
            new_group_table = cls._parse_new_group_table(trace_dir, own_rank)
            logger.info(
                "Parsed the process group table from the own rank trace in %.1fs",
                time.perf_counter() - group_parse_start,
            )
        new_group_table = {DEFAULT_GROUP_NAME: list(range(world_size)), **new_group_table}

        merge_start = time.perf_counter()
        step_graphs: dict[int, CrystalGraph] = {}
        for step in baseline_steps:
            graph = CrystalGraph()
            for rank in loaded_ranks:
                graph.merge_dict(rank_graphs[rank][step])
            cls._inject_channel_members(graph, new_group_table)
            step_graphs[step] = graph
        logger.info(
            "Merged %d step graphs over %d ranks in %.1fs",
            len(step_graphs),
            len(loaded_ranks),
            time.perf_counter() - merge_start,
        )

        profiled_ranks = cls._load_profiler_durations(step_graphs, profiler_dir, loaded_ranks)

        trace_comm_index_steps: dict[tuple[int, int], int] = {}
        analyze_start = time.perf_counter()
        for step, graph in step_graphs.items():
            for rank_id, rank in graph.ranks.items():
                for span_id in rank.spans:
                    if span_id.type_field == COMM_TYPE_FIELD:
                        trace_comm_index_steps[(rank_id, span_id.count_field)] = step
            graph.set_sandbox_ranks(sandbox_ranks)
            graph.analyze()

        logger.info(
            "Analyzed %d step graphs in %.1fs",
            len(step_graphs),
            time.perf_counter() - analyze_start,
        )

        logger.info(
            "Loaded %d/%d PP chains (%d pruned as useless), %d ranks, %d with runtime profiler durations, "
            "steps %s, total %.1fs",
            len(loaded_chains),
            pp_rank_stride,
            len(pruned_chains),
            len(loaded_ranks),
            len(profiled_ranks),
            baseline_steps,
            time.perf_counter() - load_start,
        )

        return cls(step_graphs, trace_comm_index_steps)

    @staticmethod
    def _select_chains(
        *,
        pp_rank_stride: int,
        world_size: int,
        mandatory_chains: set[int],
        profiler_dir: str | None,
    ) -> tuple[list[int], list[int]]:
        """Split all PP chains into the ones worth loading and the useless ones.

        A chain that holds neither a sandbox rank nor a launched emulated rank and has
        no runtime profiler data at all would only contribute spans of duration 0 that
        no launched rank ever updates online during replay. Under the max() semantics
        of a launch time such a member can never decide a group's launch, so it is
        dropped before any graph or trace file is read. Pruning whole chains rather
        than single ranks keeps the all-or-nothing validation of _load_chain()
        intact. Profiler presence is detected by file existence only, and mandatory
        chains are candidates unconditionally.

        Returns:
            Tuple of (candidate chain indices, pruned chain indices).
        """
        candidates: list[int] = []
        pruned: list[int] = []
        for chain in range(pp_rank_stride):
            if chain in mandatory_chains:
                candidates.append(chain)
                continue
            has_profiler = profiler_dir is not None and any(
                _profiler_file(profiler_dir, rank).exists() for rank in range(chain, world_size, pp_rank_stride)
            )
            if has_profiler:
                candidates.append(chain)
            else:
                pruned.append(chain)
        return candidates, pruned

    @staticmethod
    def _load_chain(
        graph_dir: str, trace_dir: str, chain_ranks: list[int], baseline_steps: list[int]
    ) -> tuple[dict[int, dict[int, dict]], str | None]:
        """Validate and read one PP chain.

        A chain is loaded as a whole or not at all: a partially loaded chain would
        silently skew the merged graph, because every stage's spans feed both the
        cross-rank dependency reconstruction and the timing propagation to the stages
        that follow it. Three conditions reject a chain: a missing graph file (the
        rank's spans are unavailable), a missing trace file (its channel members
        cannot be resolved), and a differing step set (the ranks come from different
        analysis windows, which would mix spans of different iterations into the
        same step).

        Returns:
            Tuple of (per-rank graph dicts, rejection reason). The dicts map rank to
            {step: the rank's serialized graph payload of that step}, ready for
            merge_dict(); the reason is None on success.
        """
        chain_graphs: dict[int, dict[int, dict]] = {}
        for rank in chain_ranks:
            if not _graph_file(graph_dir, rank).exists():
                return {}, f"missing graph file for rank {rank}, which would break the PP chain"
            if not _trace_file(trace_dir, rank).exists():
                return {}, f"missing trace file for rank {rank}, whose channel members cannot be resolved"
            rank_graph = {int(step): data for step, data in _load_gzip_json(_graph_file(graph_dir, rank)).items()}
            if sorted(rank_graph) != baseline_steps:
                return {}, (
                    f"rank {rank} has steps {sorted(rank_graph)} instead of the baseline {baseline_steps}, "
                    f"so it comes from a different analysis window or run"
                )
            chain_graphs[rank] = rank_graph
        return chain_graphs, None

    @staticmethod
    def _parse_new_group_table(trace_dir: str, rank: int) -> dict[str, list[int]]:
        """Read the NEW_GROUP table of a rank's trace.

        Process group creation is a world-wide operation, so a single rank's trace
        already records every group. Parsing runs with include_replay_op=False, which
        skips the SCHEDULE/COALESCING/communication lines and reads NEW_GROUP lines
        only, keeping the cost proportional to the number of groups rather than the
        trace length.

        Returns:
            Mapping from process group name to member global ranks.
        """
        parser = CrystalTraceParser(str(_trace_file(trace_dir, rank)), include_replay_op=False)
        return {group.group_name: list(group.ranks) for group in parser.comm_groups_ordered_list}

    @staticmethod
    def _inject_channel_members(graph: CrystalGraph, new_group_table: dict[str, list[int]]) -> None:
        """Resolve and attach the member table of every channel appearing in the graph.

        The merged graph carries spans only; analyze() needs the channel member tables
        to rebuild the cross-rank dependencies by peer matching. The tables come from
        the process groups of the trace rather than the spans themselves, so a group
        member whose spans were pruned or never captured is still covered. The
        membership check deduplicates resolution: many spans usually share one
        channel, and resolve_channel_members() is only paid once per channel.
        """
        for rank in graph.ranks.values():
            for span in rank.spans.values():
                if span.comm_tasks is None:
                    continue
                channel = span.comm_tasks.channel
                if channel not in graph.channels:
                    graph.set_channel_members(channel, resolve_channel_members(channel, new_group_table))

    @staticmethod
    def _load_profiler_durations(
        step_graphs: dict[int, CrystalGraph], profiler_dir: str | None, loaded_ranks: list[int]
    ) -> list[int]:
        """Write runtime profiler durations into the merged graphs.

        Must run before analyze(): the longest-path edge weights are captured from the
        static span durations there, so a duration written afterwards would be ignored.

        PROC durations are keyed by span index: every graph PROC span looks its index
        up in the profiler dict, and one whose index has no entry keeps its default
        duration 0 after a warning. COMM durations are grouped per graph channel in
        issue order, and each array entry matches the channel's spans in count_field
        order, which is also issue order. A channel absent from the profiler, or an
        array shorter than the graph's spans on it, leaves the uncovered spans at
        their default duration 0 after a warning. Profiler data without a graph span
        — an unknown channel, an unknown span index, or a longer array than the graph
        has spans — is tolerated and ignored.

        Returns:
            The loaded ranks that had a profiler file, for coverage logging.

        Raises:
            ValueError: If the profiler data references a step the graphs do not
                contain.
        """
        if profiler_dir is None:
            logger.warning("No runtime profiler directory given, every span duration stays 0")
            return []

        profiler_start = time.perf_counter()
        profiled_ranks: list[int] = []
        for rank in loaded_ranks:
            profiler_path = _profiler_file(profiler_dir, rank)
            if not profiler_path.exists():
                continue
            profiled_ranks.append(rank)
            for step_str, step_data in _load_gzip_json(profiler_path).items():
                step = int(step_str)
                graph = step_graphs.get(step)
                if graph is None:
                    raise ValueError(
                        f"Runtime profiler of rank {rank} contains step {step}, which is outside the "
                        f"steps present in the graph {sorted(step_graphs)}"
                    )
                spans = graph.ranks[rank].spans
                proc_durations = step_data["proc"]
                for span_id, span in spans.items():
                    if span_id.type_field != PROC_TYPE_FIELD:
                        continue
                    duration_ns = proc_durations.get(str(span_id.count_field))
                    if duration_ns is None:
                        logger.debug(
                            "Runtime profiler of rank %d has no duration for PROC span %s at step %d, "
                            "keeping the default duration 0",
                            rank,
                            span_id,
                            step,
                        )
                        continue
                    span.duration_ns = int(duration_ns)
                comm_durations = step_data["comm"]
                spans_by_channel: dict[str, list[CrystalGSpanId]] = defaultdict(list)
                for span_id, span in spans.items():
                    if span_id.type_field == COMM_TYPE_FIELD and span.comm_tasks is not None:
                        spans_by_channel[span.comm_tasks.channel].append(span_id)
                for channel, ordered_ids in spans_by_channel.items():
                    ordered_ids.sort(key=lambda sid: sid.count_field)
                    if channel not in comm_durations:
                        logger.debug(
                            "Runtime profiler of rank %d at step %d has no durations on channel %r, "
                            "keeping its %d spans at the default duration 0",
                            rank,
                            step,
                            channel,
                            len(ordered_ids),
                        )
                        continue
                    durations = comm_durations[channel]
                    if len(durations) < len(ordered_ids):
                        logger.debug(
                            "Runtime profiler of rank %d at step %d has %d durations on channel %r, "
                            "but the graph has %d spans on it; the uncovered spans keep the default "
                            "duration 0",
                            rank,
                            step,
                            len(durations),
                            channel,
                            len(ordered_ids),
                        )
                    for offset, span_id in enumerate(ordered_ids):
                        if offset < len(durations):
                            spans[span_id].duration_ns = int(durations[offset])

        total_ranks = len(loaded_ranks)
        logger.info(
            "Runtime profiler coverage: %d/%d loaded ranks have durations in %.1fs",
            len(profiled_ranks),
            total_ranks,
            time.perf_counter() - profiler_start,
        )
        return profiled_ranks

    def step_of_trace_comm_index(self, rank: int, trace_comm_index: int) -> int | None:
        """Return the step of a communication operation, or None when outside the window."""
        return self._trace_comm_index_steps.get((rank, trace_comm_index))

    def sync_span_group_of_point(self, rank: int, trace_comm_index: int) -> CrystalGSyncSpanGroup | None:
        """Return the synchronization group of a communication operation.

        The lookup goes through the START point of the operation's span; the graph
        indexes the START and STOP points of every member alike, so either resolves to
        the same group, i.e. the participants of the underlying communication with the
        sandbox members excluded.

        Args:
            rank: The rank issuing the operation.
            trace_comm_index: The operation's trace communication index.

        Returns:
            The synchronization group, or None when the operation is outside the analysis
            window and therefore absent from the graph.
        """
        step = self.step_of_trace_comm_index(rank, trace_comm_index)
        if step is None:
            return None
        point = CrystalGSpanPoint(rank, CrystalGSpanId(COMM_TYPE_FIELD, trace_comm_index), SpanPointType.START)
        return self.step_graphs[step].sync_span_group_of_point(point)

    def sync_span_group_launch_time_ns(self, step: int, group: CrystalGSyncSpanGroup) -> int:
        """Return the launch time of a synchronization group relative to the step start.

        A multi-party communication cannot start before its last participant has
        arrived, so the launch time is the latest START time over all members. The
        graph derives that max from the group's join component, so the query costs the
        same no matter how many members the group has. A single-point group has no
        join component and degenerates to the START time of its only member.

        Args:
            step: The step whose graph owns the group.
            group: A synchronization group of that step, e.g. from
                sync_span_group_of_point().

        Returns:
            The launch time in nanoseconds.
        """
        return self.step_graphs[step].sync_span_group_launch_time_ns(group)

    def is_anchorable_point(self, step: int, point: CrystalGSpanPoint) -> bool:
        """Return whether a point is anchorable in the given step's graph.

        Anchorable points are the STOP points of the synchronization groups that cross
        the sandbox boundary; see the anchor state built by the graph's analyze().
        """
        return self.step_graphs[step].is_anchorable_point(point)

    def predecessor_event_keys(self, broadcasting_ranks: set[int]) -> set[tuple[int, int]]:
        """Return the (rank, trace_comm_index) arrivals the replay may wait for.

        Only the ranks passed in can ever announce an arrival, so predecessors on any
        other rank are dropped. That filter is required, not an optimization: anchorable
        is the data-side set, i.e. every non-sandbox member of a boundary-crossing
        group, because any of their durations may decide the launch time. The ranks that
        actually run are the execution-side set, and mock bootstrap only starts the NCCL
        topology neighbours of the sandbox while leaving the rest of a group as proxy
        ranks. The two sets coincide for P2P, where every peer of a sandbox rank is
        started, but differ by an order of magnitude for a large collective, and
        waiting on a proxy rank would block forever.

        Dropping them costs no accuracy: a proxy rank only ever contributes its static
        runtime profiler duration, and a synchronization group completes at a single
        instant because its members' STOP points share one component, so the update
        from any started member already pushes that component to the real completion
        time.

        The union is taken over every step and every anchorable span. Each anchor is
        queried through the START point of its own span: the direct anchor predecessors
        of that point are exactly what must have completed before the anchor can be
        launched, whereas querying the STOP point would pull in the anchor's own group
        as its own predecessor. The key keeps the span's count_field, which is the
        communication trace index the runtime side announces arrivals under.

        Args:
            broadcasting_ranks: Ranks that execute operations and therefore broadcast.

        Returns:
            The set of predecessor keys worth creating an event for.
        """
        keys: set[tuple[int, int]] = set()
        for graph in self.step_graphs.values():
            for anchor_point in graph.anchorable_points:
                start_point = CrystalGSpanPoint(anchor_point.rank, anchor_point.span_id, SpanPointType.START)
                for pred_point in graph.get_direct_anchor_predecessors(start_point):
                    if pred_point.rank in broadcasting_ranks:
                        keys.add((pred_point.rank, pred_point.span_id.count_field))
        return keys
