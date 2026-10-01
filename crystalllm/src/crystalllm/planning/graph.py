# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Graph."""

from __future__ import annotations

import enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any


class CrystalGDurationNotIncreasedError(ValueError):
    """Raised when a span duration update does not strictly increase the current value.

    increase_span_duration() only rewinds the longest-path cache to the updated span, which
    is sound solely because component times can grow, never shrink. A measured duration
    below the one already in the graph is an expected outcome rather than a defect, e.g.
    when the runtime profiler value came from a different planning batch, so callers are
    meant to catch this and skip the update.
    """


class SpanType(enum.Enum):
    """Enumeration of span operation types."""

    PROC = "proc"  # Computation operation, has no cross-rank peer
    COMM = "comm"  # Communication operation, matched with its peers by channel


class CrystalGCommPattern(enum.Enum):
    """Enumeration of communication patterns, i.e. how a task is matched with its peers.

    The pattern decides how build_cross_rank_dependencies() pairs the task with the tasks
    of the other members of its channel. "Upstream" and "downstream" refer to the member
    order of the channel: the downstream of group rank g is group rank g + 1 (mod size).
    """

    SYMM = "symm"  # Symmetric: every channel member joins, e.g. all-reduce between DP ranks
    P2P_UP = "p2p_up"  # Point-to-point paired with the upstream member's P2P_DN
    P2P_DN = "p2p_dn"  # Point-to-point paired with the downstream member's P2P_UP


@dataclass
class CrystalGCommTasks:
    """Represents a collection of communication tasks on the same channel.

    Attributes:
        channel: Channel name identifying the process group the tasks run on. Peer
            matching is confined to one channel, and the tasks a rank issues on a channel
            are matched with those of its peers in issue order.
        tasks: Communication patterns of this span, in issue order. A single span carries
            several tasks when the operations were coalesced, e.g. a batched P2P exchange.
        async_op: Whether the communication was issued asynchronously, i.e. whether
            computation may overlap it until its wait. The runtime context records it
            when the span is authored, so the graph keeps the launch shape of every
            communication without timing it.
    """

    channel: str
    tasks: list[CrystalGCommPattern]
    async_op: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "channel": self.channel,
            "tasks": [task.value for task in (self.tasks or [])],
            "async_op": self.async_op,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrystalGCommTasks:
        """Deserialize from dictionary.

        A missing or null task list is read as no tasks, so that a span serialized without
        its patterns still yields a usable object instead of raising; a missing async_op
        reads as False for the same reason.
        """
        return cls(
            channel=data["channel"],
            tasks=[CrystalGCommPattern(task) for task in (data.get("tasks") or [])],
            async_op=bool(data.get("async_op", False)),
        )


class DependencyType(enum.Enum):
    """Enumeration of dependency types."""

    UNIDIRECT = "unidirect"  # v2 must be after v1, always within one rank
    BIDIRECT = "bidirect"  # v1 and v2 must complete simultaneously, always across ranks
    WEAK_BIDIRECT = "weak_bidirect"  # Draw-only pairing of an async communication under max_connections == 1


class SpanPointType(enum.Enum):
    """Enumeration of span points."""

    START = "start"  # Start point of a span
    STOP = "stop"  # Stop point of a span


@dataclass(frozen=True)
class CrystalGSpanId:
    """Span identifier with format {type_field}#{count_field}.

    Frozen so it can be used as a dictionary key.

    Attributes:
        type_field: Operation kind, e.g. "FORWARD". Must not contain "#", otherwise
            from_string() cannot parse back what __str__() produced.
        count_field: Zero-based occurrence index of this kind on its rank. Cross-rank
            matching relies on it to replay the issue order of a rank, so the counters of
            a rank must be assigned in issue order.
    """

    type_field: str
    count_field: int

    def __str__(self) -> str:
        """Serialize to string format {type_field}#{count_field}."""
        return f"{self.type_field}#{self.count_field}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SpanId({self.type_field!r}, {self.count_field})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type_field": self.type_field, "count_field": self.count_field}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrystalGSpanId:
        """Deserialize from dictionary."""
        return cls(type_field=data["type_field"], count_field=data["count_field"])

    @classmethod
    def from_string(cls, s: str) -> CrystalGSpanId:
        """Parse SpanId from string.

        Args:
            s: String in format "{type_field}#{count_field}"

        Returns:
            Parsed SpanId instance.

        Raises:
            ValueError: If string format is invalid.
        """
        parts = s.split("#")
        if len(parts) != 2:  # noqa: PLR2004
            raise ValueError(f"Invalid SpanId format: {s}")
        return cls(type_field=parts[0], count_field=int(parts[1]))


@dataclass(frozen=True)
class CrystalGSpanPoint:
    """Represents a point on a Span (START or STOP).

    A point is a node of the execution graph: a span becomes the START -> STOP edge
    between its two points. Frozen so it can be used as a dictionary key.

    Attributes:
        rank: Global rank the span runs on.
        span_id: Identifier of the span this point belongs to.
        point_type: Whether the point is the start or the stop of the span.
    """

    rank: int
    span_id: CrystalGSpanId
    point_type: SpanPointType

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Point(rank={self.rank}, span_id={self.span_id}, point_type={self.point_type})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rank": self.rank,
            "span_id": self.span_id.to_dict(),
            "point_type": self.point_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrystalGSpanPoint:
        """Deserialize from dictionary."""
        return cls(
            rank=data["rank"],
            span_id=CrystalGSpanId.from_dict(data["span_id"]),
            point_type=SpanPointType(data["point_type"]),
        )


@dataclass
class CrystalGSpan:
    """Represents an execution unit (computation or communication).

    Attributes:
        span_type: Type of operation (PROC or COMM).
        span_id: Identifier of this span, unique within its rank.
        comm_tasks: Communication tasks for COMM type spans, None for PROC spans.
        duration_ns: Static duration in nanoseconds. analyze() turns it into the weight of
            the span's START -> STOP edge in the condensed graph. Online updates go through
            increase_span_duration(), which edits that edge and leaves this field untouched.
            The value is never serialized: an authored graph carries no measured time, and
            a loaded graph receives its durations from the runtime profiler through
            load_span_duration_dict().
    """

    span_type: SpanType
    span_id: CrystalGSpanId
    comm_tasks: CrystalGCommTasks | None = None
    duration_ns: int = 0

    def __post_init__(self):
        """Validate that COMM spans have non-None comm_tasks."""
        if self.span_type == SpanType.COMM and self.comm_tasks is None:
            raise ValueError(f"COMM type span {self.span_id} must have non-None comm_tasks")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        The static duration is deliberately absent, see the class docstring.
        """
        return {
            "span_type": self.span_type.value,
            "span_id": self.span_id.to_dict(),
            "comm_tasks": self.comm_tasks.to_dict() if self.comm_tasks is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrystalGSpan:
        """Deserialize from dictionary, starting from the zero default duration."""
        comm_tasks_data = data.get("comm_tasks")
        comm_tasks = CrystalGCommTasks.from_dict(comm_tasks_data) if comm_tasks_data is not None else None
        return cls(
            span_type=SpanType(data["span_type"]),
            span_id=CrystalGSpanId.from_dict(data["span_id"]),
            comm_tasks=comm_tasks,
        )


@dataclass(frozen=True)
class CrystalGDep:
    """Represents a dependency between two span points.

    Frozen so equal dependencies collapse in the dependency set of the graph.

    Attributes:
        dep_type: Type of dependency (UNIDIRECT, BIDIRECT or WEAK_BIDIRECT).
        v1: First span point. For UNIDIRECT the earlier point; for BIDIRECT the point on
            the smaller rank, which is what makes the two possible orderings of a peer
            pair collapse into one entry.
        v2: Second span point.
    """

    dep_type: DependencyType
    v1: CrystalGSpanPoint
    v2: CrystalGSpanPoint

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dep_type": self.dep_type.value,
            "v1": self.v1.to_dict(),
            "v2": self.v2.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrystalGDep:
        """Deserialize from dictionary."""
        return cls(
            dep_type=DependencyType(data["dep_type"]),
            v1=CrystalGSpanPoint.from_dict(data["v1"]),
            v2=CrystalGSpanPoint.from_dict(data["v2"]),
        )


@dataclass(frozen=True)
class CrystalGSyncSpanGroup:
    """One synchronization group: the COMM spans of a single communication across ranks.

    This is a different concept from a process group (called a channel at the graph
    layer and a comm_group at the replay layer): a channel carries one sync group per
    communication it performs.

    Attributes:
        stop_points: STOP points of the group members. They all belong to the same BIDIRECT
            component of the condensed graph, which is what identifies the group.
        start_points: START points of the same spans, positionally aligned with
            stop_points. Unlike the stop points they stay in separate components, see
            _build_group_join_nodes() for why.
    """

    stop_points: tuple[CrystalGSpanPoint, ...]
    start_points: tuple[CrystalGSpanPoint, ...]


@dataclass
class CrystalGRank:
    """Represents all spans executing on a single rank.

    Attributes:
        rank_id: The rank identifier.
        spans: Mapping from span IDs to span objects on this rank.
        span_desc: Mapping from span IDs to their description lists. Only computation spans
            are described in practice: a captured graph gives every computation span the
            same type field, so the semantics live in these descriptions.
    """

    rank_id: int
    spans: dict[CrystalGSpanId, CrystalGSpan] = field(default_factory=dict)
    span_desc: dict[CrystalGSpanId, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rank_id": self.rank_id,
            "spans": {str(k): v.to_dict() for k, v in self.spans.items()},
            "span_desc": {str(k): v for k, v in self.span_desc.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrystalGRank:
        """Deserialize from dictionary."""
        rank = cls(rank_id=data["rank_id"])
        for span_id_str, span_data in data["spans"].items():
            span_id = CrystalGSpanId.from_string(span_id_str)
            rank.spans[span_id] = CrystalGSpan.from_dict(span_data)
        for span_id_str, descriptions in data.get("span_desc", {}).items():
            rank.span_desc[CrystalGSpanId.from_string(span_id_str)] = list(descriptions)
        return rank


# Sentinel for a CrystalGraph whose max_connections was never set. A negative
# connection count is impossible, so it unambiguously marks the unset state.
UNSET_MAX_CONNECTIONS = -1


@dataclass
class CrystalGraph:
    """Represents the complete distributed execution graph.

    The graph is authored per rank: callers add spans and the UNIDIRECT dependencies that
    order them inside a rank, plus the member table of every channel. analyze() then joins
    the ranks by matching the communication tasks of each channel and condenses the result
    into a weighted DAG on which all timing queries run.

    Attributes:
        max_connections: CUDA_DEVICE_MAX_CONNECTIONS the graph was authored under,
            recorded through set_max_connections() before any export. It feeds no
            timing query; it travels with every exported payload and is checked on
            merge, so graphs authored under different device overlap capabilities
            can never be combined.
        ranks: Mapping from rank IDs to CrystalGRank objects.
        dependencies: Set of dependencies between span points. Holds the UNIDIRECT
            dependencies added by callers plus the BIDIRECT ones derived by
            build_cross_rank_dependencies().
        channels: Mapping from channel name to member global ranks, ordered by group rank.
        sync_span_groups: Synchronization groups produced by analyze().
        sandbox_ranks: Ranks registered by set_sandbox_ranks(), an input to analyze().
        anchorable_points: Points computed by analyze() whose time the replay side may use
            as an anchor, see _build_anchor_state().
    """

    max_connections: int = UNSET_MAX_CONNECTIONS
    ranks: dict[int, CrystalGRank] = field(default_factory=dict)
    dependencies: set[CrystalGDep] = field(default_factory=set)
    channels: dict[str, list[int]] = field(default_factory=dict)

    # Validity flags of the derived state below. Cross-rank matching reads only spans and
    # channel member tables, so it outlives the changes that merely invalidate _analyzed.
    _cross_rank_built: bool = field(default=False, init=False, repr=False)
    _analyzed: bool = field(default=False, init=False, repr=False)

    # Condensed DAG built by analyze(): points connected by BIDIRECT dependencies collapse
    # into one component, and an edge weight is the duration a component must wait before
    # its successor may start. Components are identified by opaque integer ids.
    _node_to_comp: dict[CrystalGSpanPoint, int] = field(default_factory=dict, init=False, repr=False)
    _condensed_graph_adj: dict[int, dict[int, int]] = field(default_factory=dict, init=False, repr=False)
    _comp_topo_order: list[int] = field(default_factory=list, init=False, repr=False)
    _comp_to_topo_idx: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    # Longest-path time per component, filled in incrementally along the topological order.
    # Components before the frontier have already propagated their outgoing edges, so every
    # component up to and including the frontier holds its final time.
    _comp_time_cache: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _comp_time_cached_frontier: int = field(default=0, init=False, repr=False)

    # Synchronization groups and their join components, see _build_sync_span_groups() and
    # _build_group_join_nodes(). Keyed by the shared BIDIRECT component of a group.
    sync_span_groups: list[CrystalGSyncSpanGroup] = field(default_factory=list, init=False, repr=False)
    _point_to_sync_group: dict[CrystalGSpanPoint, CrystalGSyncSpanGroup] = field(
        default_factory=dict, init=False, repr=False
    )
    _sync_group_join_comp: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    # Sandbox split and the anchor state derived from it, see _build_anchor_state().
    sandbox_ranks: set[int] = field(default_factory=set, init=False, repr=False)
    anchorable_points: set[CrystalGSpanPoint] = field(default_factory=set, init=False, repr=False)
    _comp_anchor_predecessors: dict[int, list[CrystalGSpanPoint]] = field(default_factory=dict, init=False, repr=False)

    def set_max_connections(self, max_connections: int) -> None:
        """Record the CUDA_DEVICE_MAX_CONNECTIONS the graph was authored under.

        The value is part of every exported payload and is checked on merge, so graphs
        authored under different device overlap capabilities can never be combined. It
        feeds no timing query, so the derived state stays valid.

        Args:
            max_connections: The CUDA_DEVICE_MAX_CONNECTIONS value to record.
        """
        self.max_connections = max_connections

    def add_span(self, rank_id: int, span: CrystalGSpan) -> None:
        """Add a span to the specified rank, creating the rank on first use."""
        if rank_id not in self.ranks:
            self.ranks[rank_id] = CrystalGRank(rank_id=rank_id)
        if span.span_id in self.ranks[rank_id].spans:
            raise ValueError(f"Span {span.span_id} already exists in rank {rank_id}")
        self.ranks[rank_id].spans[span.span_id] = span
        self._cross_rank_built = False
        self._analyzed = False

    def add_span_desc(self, rank_id: int, span_id: CrystalGSpanId, description: str) -> None:
        """Append a description item to a span, see CrystalGRank.span_desc.

        Descriptions are pure annotations that no timing query reads, so the derived state
        stays valid.
        """
        if rank_id not in self.ranks or span_id not in self.ranks[rank_id].spans:
            raise ValueError(f"Span {span_id} not found in rank {rank_id}")
        self.ranks[rank_id].span_desc.setdefault(span_id, []).append(description)

    def _add_bidirectional_dependency(
        self, rank1: int, span_id1: CrystalGSpanId, rank2: int, span_id2: CrystalGSpanId
    ) -> None:
        """Add a BIDIRECT (or WEAK_BIDIRECT) dependency between the STOP points of two communication spans.

        Only the STOP points are related: peers of one communication complete together,
        while their start times are free to differ by the arrival skew.

        Under ``max_connections == 1`` an asynchronously issued communication pairs
        its peers with WEAK_BIDIRECT instead: the timing model keeps no cross-rank
        synchronization for it (each side's duration already contains the wait for
        its peer), so the dependency exists for drawing only and analyze() neither
        merges its points nor orders them. Either side being async is enough.
        """
        # Ensure smaller rank is always first, so that matching the same pair from either
        # side yields one identical dependency in the set.
        if rank1 > rank2:
            rank1, rank2 = rank2, rank1
            span_id1, span_id2 = span_id2, span_id1

        dep_type = DependencyType.BIDIRECT
        if self.max_connections == 1 and (
            self.ranks[rank1].spans[span_id1].comm_tasks.async_op
            or self.ranks[rank2].spans[span_id2].comm_tasks.async_op
        ):
            dep_type = DependencyType.WEAK_BIDIRECT

        self.dependencies.add(
            CrystalGDep(
                dep_type=dep_type,
                v1=CrystalGSpanPoint(rank1, span_id1, SpanPointType.STOP),
                v2=CrystalGSpanPoint(rank2, span_id2, SpanPointType.STOP),
            )
        )

    def add_dependency(self, dep: CrystalGDep) -> None:
        """Add a UNIDIRECT dependency to the graph.

        Only intra-rank ordering can be authored: everything that crosses a rank boundary
        is a BIDIRECT dependency derived by build_cross_rank_dependencies() from the
        communication tasks, so it is rejected here.

        Args:
            dep: The dependency to add.

        Raises:
            ValueError: If a referenced span does not exist, if the dependency is BIDIRECT,
                or if its two points are not on the same rank.
        """
        if dep.v1.rank not in self.ranks or dep.v1.span_id not in self.ranks[dep.v1.rank].spans:
            raise ValueError(f"Point v1 references non-existent span: {dep.v1}")
        if dep.v2.rank not in self.ranks or dep.v2.span_id not in self.ranks[dep.v2.rank].spans:
            raise ValueError(f"Point v2 references non-existent span: {dep.v2}")

        if dep.dep_type != DependencyType.UNIDIRECT:
            raise ValueError(
                "BIDIRECT and WEAK_BIDIRECT dependencies cannot be added externally. "
                "They are automatically managed by the graph."
            )

        if dep.v1.rank != dep.v2.rank:
            raise ValueError(
                f"UNIDIRECT dependency requires spans to be on the same rank, "
                f"but v1 is on rank {dep.v1.rank} and v2 is on rank {dep.v2.rank}"
            )

        self.dependencies.add(dep)
        self._analyzed = False

    def set_channel_members(self, channel: str, members: list[int]) -> None:
        """Register the member global ranks of a channel, ordered by group rank.

        The member table drives peer matching, so it must cover every channel the spans
        refer to, including members whose spans were never loaded.
        """
        self.channels[channel] = list(members)
        self._cross_rank_built = False
        self._analyzed = False

    def to_rank_dict(self, rank: int) -> dict[str, Any]:
        """Serialize the specified rank to a dictionary consumable by merge_dict().

        Only the dependencies internal to the rank are emitted. The cross-rank ones are
        derived from the communication tasks, so every receiver rebuilds them itself.

        Raises:
            KeyError: If the rank is not part of the graph.
            ValueError: If set_max_connections() was never called.
        """
        if rank not in self.ranks:
            raise KeyError(f"Rank {rank} not found in the graph")
        if self.max_connections == UNSET_MAX_CONNECTIONS:
            raise ValueError("set_max_connections() must be called before exporting the graph")

        internal_deps = [
            dep.to_dict()
            for dep in self.dependencies
            if dep.dep_type == DependencyType.UNIDIRECT and dep.v1.rank == rank and dep.v2.rank == rank
        ]

        return {"max_connections": self.max_connections, **self.ranks[rank].to_dict(), "dependencies": internal_deps}

    def merge_dict(self, data: dict[str, Any]) -> None:
        """Merge the payload of one rank, as produced by to_rank_dict(), into this graph.

        Merging is idempotent per span: content already present is kept and the incoming
        copy is skipped with a notice, which lets the payloads of the same rank be replayed
        without wiping the state built from an earlier one. The first merged payload fixes
        max_connections when it is still unset; a payload authored under a different value
        is rejected, because its spans would model a different device overlap capability
        than the ones already merged.

        Raises:
            KeyError: If a required key is missing from the payload.
            ValueError: If a description or dependency references an unknown span, or if
                the payload's max_connections differs from the graph's.
        """
        for key in ("max_connections", "rank_id", "spans", "span_desc", "dependencies"):
            if key not in data:
                raise KeyError(f"Missing required key: '{key}'")

        if self.max_connections == UNSET_MAX_CONNECTIONS:
            self.max_connections = data["max_connections"]
        elif self.max_connections != data["max_connections"]:
            raise ValueError(
                f"Cannot merge rank {data['rank_id']} payload authored with max_connections="
                f"{data['max_connections']} into a graph authored with max_connections="
                f"{self.max_connections}"
            )

        rank_id = data["rank_id"]

        for span_id_str, span_data in data["spans"].items():
            span_id = CrystalGSpanId.from_string(span_id_str)
            if rank_id in self.ranks and span_id in self.ranks[rank_id].spans:
                print(f"Skipping duplicate span {span_id} in rank {rank_id}")
                continue
            self.add_span(rank_id, CrystalGSpan.from_dict(span_data))

        # Descriptions are merged whole: a span either keeps the list it already has or
        # receives the incoming one, so the two are never interleaved.
        for span_id_str, descriptions in data["span_desc"].items():
            span_id = CrystalGSpanId.from_string(span_id_str)
            if rank_id not in self.ranks or span_id not in self.ranks[rank_id].spans:
                raise ValueError(f"span_desc references non-existent span {span_id} in rank {rank_id}")
            if span_id in self.ranks[rank_id].span_desc:
                print(f"Skipping duplicate span_desc for span {span_id} in rank {rank_id}")
                continue
            self.ranks[rank_id].span_desc[span_id] = list(descriptions)

        # add_dependency() validates the payload and deduplicates through the set.
        for dep_data in data["dependencies"]:
            self.add_dependency(CrystalGDep.from_dict(dep_data))

    def load_span_duration_dict(self, duration_dict: dict[int, dict[str, list[int]]]) -> None:
        """Overwrite the static span durations from a measured duration table.

        The table is keyed {rank: {type_field: [duration_ns, ...]}}, where the position in a
        duration list is the count_field of the span it belongs to. That is what lets a
        profiler report a bare per-kind sequence of measurements: the graph reconstructs the
        span ids as it walks the list, and every list entry must hit an existing span.

        Args:
            duration_dict: Dictionary mapping rank IDs to type_field -> duration list mappings.

        Raises:
            ValueError: If a specified rank or span is not found in the graph.
        """
        for rank_id, type_field_dict in duration_dict.items():
            if rank_id not in self.ranks:
                raise ValueError(f"Rank {rank_id} not found in the graph")

            rank = self.ranks[rank_id]

            for type_field, duration_list in type_field_dict.items():
                for count_field, duration_ns in enumerate(duration_list):
                    span_id = CrystalGSpanId(type_field=type_field, count_field=count_field)
                    if span_id not in rank.spans:
                        raise ValueError(f"Span {span_id} not found in rank {rank_id}")
                    rank.spans[span_id].duration_ns = duration_ns

        # The condensed graph no longer matches the spans. The cross-rank matching
        # itself stays valid: it reads only spans and channel member tables.
        self._analyzed = False

    def build_cross_rank_dependencies(self) -> None:
        """Build cross-rank BIDIRECT (or WEAK_BIDIRECT) dependencies by matching communication tasks.

        Matching is driven by the channel member table (the process group members), not by
        the set of loaded ranks, and the tasks of a rank are matched in its issue order:
        - P2P: for every ordered adjacent member pair (g, g+1 mod size), the first member's
          P2P_DN sequence is paired one-to-one with the second member's P2P_UP sequence. A
          2-member P2P channel yields both directions. Both ends must be loaded, hence a
          P2P channel is always loaded as a whole chain, otherwise the two sequences differ
          in length and the mismatch is reported.
        - SYMM: the sequences of all participating members are aligned and the k-th
          occurrence forms one synchronization group. Unlike P2P this tolerates a partial
          load, since a collective still synchronizes the members that are present.

        A matched pair becomes WEAK_BIDIRECT instead of BIDIRECT when the graph models
        max_connections == 1 and either peer was issued asynchronously, see
        _add_bidirectional_dependency(); such edges join no timing model.

        The result depends only on the spans and the channel member tables, so it is cached
        behind _cross_rank_built: duration updates and sandbox changes keep it, while adding
        spans or changing member tables rebuild it.

        Raises:
            ValueError: If set_max_connections() was never called, if a channel has no
                member table, if a rank has spans on a channel it does not belong to, or
                if the matched sequences differ in length.
        """
        if self.max_connections == UNSET_MAX_CONNECTIONS:
            raise ValueError(
                "set_max_connections() must be called before build_cross_rank_dependencies(): the value "
                "decides whether an async communication pairs its peers with BIDIRECT or WEAK_BIDIRECT"
            )

        # Drop the cross-rank edges of any previous build (BIDIRECT and WEAK_BIDIRECT);
        # they are fully rebuilt below.
        self.dependencies = {dep for dep in self.dependencies if dep.dep_type == DependencyType.UNIDIRECT}

        # channel -> rank -> [(span_id, task_idx, pattern), ...]
        channel_tasks: dict[str, dict[int, list[tuple[CrystalGSpanId, int, CrystalGCommPattern]]]] = {}

        for rank_id, rank in self.ranks.items():
            for span_id, span in rank.spans.items():
                if span.span_type != SpanType.COMM or span.comm_tasks is None:
                    continue
                per_rank = channel_tasks.setdefault(span.comm_tasks.channel, {})
                entries = per_rank.setdefault(rank_id, [])
                for task_idx, pattern in enumerate(span.comm_tasks.tasks):
                    entries.append((span_id, task_idx, pattern))

        for channel, per_rank in channel_tasks.items():
            if channel not in self.channels:
                raise ValueError(
                    f"Channel '{channel}' has no member table. "
                    f"Call set_channel_members() for every channel before analyze()."
                )
            members = self.channels[channel]
            member_set = set(members)

            # (rank, pattern) -> span ids in the issue order of that rank, which is the
            # order peers are matched in: spans by their per-rank counter, and the tasks
            # coalesced into one span by their position inside it.
            issued: dict[tuple[int, CrystalGCommPattern], list[CrystalGSpanId]] = {}
            for rank_id, entries in per_rank.items():
                if rank_id not in member_set:
                    raise ValueError(
                        f"Rank {rank_id} has spans on channel '{channel}' but is not one of its members {members}"
                    )
                entries.sort(key=lambda entry: (entry[0].count_field, entry[1]))
                for span_id, _task_idx, pattern in entries:
                    issued.setdefault((rank_id, pattern), []).append(span_id)

            for group_rank, upstream in enumerate(members):
                downstream = members[(group_rank + 1) % len(members)]
                # A single-member channel is its own downstream and has nobody to pair with.
                if upstream == downstream:
                    continue
                sent = issued.get((upstream, CrystalGCommPattern.P2P_DN), [])
                received = issued.get((downstream, CrystalGCommPattern.P2P_UP), [])
                if len(sent) != len(received):
                    raise ValueError(
                        f"Mismatched P2P task counts between rank {upstream} and rank {downstream} "
                        f"on channel '{channel}': rank {upstream} has {len(sent)} P2P_DN tasks, "
                        f"rank {downstream} has {len(received)} P2P_UP tasks. A P2P channel is loaded "
                        f"as a whole PP chain, so this means the P2P peer crosses a PP chain boundary "
                        f"and the configured PP rank stride does not match the real parallel layout."
                    )
                for span_id_dn, span_id_up in zip(sent, received, strict=True):
                    self._add_bidirectional_dependency(upstream, span_id_dn, downstream, span_id_up)

            # Only members that actually issued SYMM traffic take part. An unloaded member,
            # or one whose spans on this channel are all P2P, must not widen the group.
            symm_sequences: dict[int, list[CrystalGSpanId]] = {}
            for member in members:
                member_symm = issued.get((member, CrystalGCommPattern.SYMM))
                if member_symm:
                    symm_sequences[member] = member_symm
            if not symm_sequences:
                continue

            occurrence_counts = {len(seq) for seq in symm_sequences.values()}
            if len(occurrence_counts) != 1:
                raise ValueError(
                    f"Mismatched SYMM task counts on channel '{channel}' between loaded members: "
                    f"{ {member: len(seq) for member, seq in symm_sequences.items()} }"
                )
            (occurrence_count,) = occurrence_counts

            # One occurrence is one collective, so its members must end up in a single
            # BIDIRECT component. A relay chain over the participants in ascending rank
            # order achieves that with N-1 edges instead of the O(N^2) of a full mesh:
            # Union-Find in _build_condensed_graph() merges a chain and a mesh alike.
            symm_members = sorted(symm_sequences)
            for occurrence in range(occurrence_count):
                for first, second in pairwise(symm_members):
                    self._add_bidirectional_dependency(
                        first,
                        symm_sequences[first][occurrence],
                        second,
                        symm_sequences[second][occurrence],
                    )

        self._cross_rank_built = True

    def analyze(self) -> None:
        """Join the ranks and condense the graph into the DAG that every timing query runs on.

        The sandbox ranks must be registered first (set_sandbox_ranks()): they decide which
        points the synchronization groups carry and which points are anchorable. The steps:
        1. Rebuild the cross-rank BIDIRECT dependencies unless they are still valid, see
           build_cross_rank_dependencies(). WEAK_BIDIRECT edges (the draw-only pairing of
           an async communication under max_connections == 1) join nothing below.
        2. Merge the BIDIRECT-connected points into components with Union-Find and build the
           condensed DAG whose nodes are those components.
        3. Build the synchronization groups from the non-sandbox COMM points.
        4. Add one join component to every multi-member synchronization group, so that the
           launch time of a group becomes a longest-path value instead of a query-time max.
        5. Order the components topologically, which is also where cycles are rejected.
        6. Derive the anchorable points and the anchor frontiers from the sandbox split.

        All edge weights come from the static span durations, so any duration pushed in by
        increase_span_duration() is dropped and the longest-path cache restarts empty.

        Raises:
            ValueError: If circular dependency is detected.
        """

        def _build_condensed_graph() -> None:
            """Condense the BIDIRECT components of the point graph into a weighted DAG.

            A BIDIRECT dependency states that two points happen at the same instant, so the
            points it connects are interchangeable for every timing query and may be merged
            into a single node. What remains is a DAG whose edges are the UNIDIRECT
            dependencies plus the START -> STOP edge of every span:
            1. Number the points, then split the dependencies into edges and merge pairs.
            2. Merge the pairs with Union-Find, which identifies a component by the index of
               its root point. Only STOP points are ever merged, so every component id is a
               valid point index, which is the range _build_group_join_nodes() extends.
            3. Project the edges onto the components. Parallel edges collapse into their
               maximum weight, because the slowest of them is the binding constraint.

            Results are stored in self._node_to_comp and self._condensed_graph_adj.
            """
            node_to_idx: dict[CrystalGSpanPoint, int] = {node: i for i, node in enumerate(node_list)}

            # Edges as (from_idx, to_idx, weight). Only the START -> STOP edge of a span
            # carries a duration; an ordering dependency merely forbids an earlier start.
            weighted_edges: list[tuple[int, int, int]] = []
            bidirect_pairs: list[tuple[int, int]] = []

            for dep in self.dependencies:
                v1_idx = node_to_idx[dep.v1]
                v2_idx = node_to_idx[dep.v2]

                if dep.dep_type == DependencyType.UNIDIRECT:
                    weighted_edges.append((v1_idx, v2_idx, 0))
                elif dep.dep_type == DependencyType.BIDIRECT:
                    bidirect_pairs.append((v1_idx, v2_idx))
                # WEAK_BIDIRECT pairs an async communication for drawing only: it merges
                # no component and orders nothing, so the timing model never sees it and
                # each peer keeps its own single-point synchronization group.

            for rank_id, rank in self.ranks.items():
                for span_id, span in rank.spans.items():
                    start_idx = node_to_idx[CrystalGSpanPoint(rank_id, span_id, SpanPointType.START)]
                    stop_idx = node_to_idx[CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)]
                    weighted_edges.append((start_idx, stop_idx, span.duration_ns))

            # Union-Find with path halving inside find(). Union by rank would buy little
            # here: the merges only ever chain the members of one communication together.
            parent = list(range(len(node_list)))

            def find(x: int) -> int:
                root = x
                while parent[root] != root:
                    root = parent[root]
                while parent[x] != root:
                    parent[x], x = root, parent[x]
                return root

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for idx1, idx2 in bidirect_pairs:
                union(idx1, idx2)

            comp_of_idx = [find(idx) for idx in range(len(node_list))]
            self._node_to_comp = dict(zip(node_list, comp_of_idx, strict=True))

            self._condensed_graph_adj = defaultdict(dict)
            for from_idx, to_idx, weight in weighted_edges:
                neighbors = self._condensed_graph_adj[comp_of_idx[from_idx]]
                to_comp = comp_of_idx[to_idx]
                neighbors[to_comp] = max(neighbors.get(to_comp, weight), weight)

        def _build_sync_span_groups() -> None:
            """Group the non-sandbox COMM STOP points by BIDIRECT component.

            One component of merged STOP points is exactly one communication, so grouping by
            component recovers the participants of each communication from the flat set of
            dependencies. Members are sorted to give the group a deterministic order, which
            the positional alignment of start_points and the use of stop_points[0] as the
            group key both build on.

            Sandbox points are excluded: they are executed for real, their durations can
            never be updated online, and the groups only drive the replay-side scheduling of
            emulated ranks. A group whose every member is a sandbox rank therefore
            disappears, while a communication span without any peer still forms a
            single-point group that can be launched by its own START point. The sandbox side
            still shapes the time model: its STOP points stay in the BIDIRECT components of
            the condensed graph, so a sandbox span's static duration keeps propagating to the
            members that follow it.
            """
            comp_to_stop_points: dict[int, list[CrystalGSpanPoint]] = defaultdict(list)
            for rank_id, rank in self.ranks.items():
                if rank_id in self.sandbox_ranks:
                    continue
                for span_id, span in rank.spans.items():
                    if span.span_type != SpanType.COMM:
                        continue
                    stop_point = CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)
                    comp_to_stop_points[self._node_to_comp[stop_point]].append(stop_point)

            self.sync_span_groups = []
            self._point_to_sync_group = {}
            for stop_points in comp_to_stop_points.values():
                stop_points.sort(key=lambda p: (p.rank, p.span_id.count_field, p.span_id.type_field))
                group = CrystalGSyncSpanGroup(
                    stop_points=tuple(stop_points),
                    start_points=tuple(CrystalGSpanPoint(p.rank, p.span_id, SpanPointType.START) for p in stop_points),
                )
                self.sync_span_groups.append(group)
                for point in group.stop_points:
                    self._point_to_sync_group[point] = group
                for point in group.start_points:
                    self._point_to_sync_group[point] = group

        def _build_group_join_nodes() -> None:
            """Add one join component to every synchronization group of two or more members.

            A group is launched when its last member arrives, i.e. its launch time is the max
            over the START points of all members. A join component with a zero weight edge
            from every member's START component makes the longest path of the condensed graph
            compute that max, so sync_span_group_launch_time_ns() becomes a single cache
            lookup instead of one get_points_interval() call per member. A single-point group
            has nothing to take the max over and keeps using get_points_interval() on its own
            START point.

            A join component has no outgoing edge. It therefore cannot shift the time of any
            other component and cannot close a cycle, which is what separates it from
            condensing the START points into the BIDIRECT component: the latter would replace
            the correct max(start_i + duration_i) of the shared STOP point with
            max(start_i) + max(duration_i), inflating it by the arrival skew, and would turn
            opposite launch orders of two groups on the same ranks into a cycle.
            """
            self._sync_group_join_comp = {}
            # Component ids so far are point indices, so counting on past the last one keeps
            # the join ids collision free.
            next_comp = len(node_list)
            for group in self.sync_span_groups:
                if len(group.start_points) < 2:  # noqa: PLR2004
                    continue
                join_comp = next_comp
                next_comp += 1
                for point in group.start_points:
                    self._condensed_graph_adj[self._node_to_comp[point]][join_comp] = 0
                # Materialize the join component as an explicit sink.
                self._condensed_graph_adj[join_comp] = {}
                # All stop points of a group share one BIDIRECT component, so that component
                # identifies the group and stop_points[0] is a valid representative key for
                # the O(1) lookup in sync_span_group_launch_time_ns().
                self._sync_group_join_comp[self._node_to_comp[group.stop_points[0]]] = join_comp

        def _compute_topological_order() -> None:
            """Order the components with Kahn's algorithm and reject cyclic graphs.

            The order is what makes the longest-path cache work: relaxing the components in
            it reaches every predecessor of a component before the component itself, so a
            single forward sweep is enough. Kahn's algorithm doubles as the cycle check,
            because a component on a cycle never drops to in-degree zero and is therefore
            missing from the result.

            Results are stored in self._comp_topo_order and self._comp_to_topo_idx.

            Raises:
                ValueError: If circular dependency is detected.
            """
            # Join components carry no point, so they only show up in the adjacency and have
            # to be collected from their own table.
            all_comps = set(self._node_to_comp.values()) | set(self._sync_group_join_comp.values())

            in_degree: dict[int, int] = defaultdict(int)
            for neighbors in self._condensed_graph_adj.values():
                for to_comp in neighbors:
                    in_degree[to_comp] += 1

            queue: deque[int] = deque(comp for comp in all_comps if in_degree[comp] == 0)

            self._comp_topo_order = []
            while queue:
                comp = queue.popleft()
                self._comp_topo_order.append(comp)
                for neighbor_comp in self._condensed_graph_adj[comp]:
                    in_degree[neighbor_comp] -= 1
                    if in_degree[neighbor_comp] == 0:
                        queue.append(neighbor_comp)

            if len(self._comp_topo_order) != len(all_comps):
                raise ValueError("Circular dependency detected in graph")

            self._comp_to_topo_idx = {comp: idx for idx, comp in enumerate(self._comp_topo_order)}

        def _build_anchor_state() -> None:
            """Compute the anchorable points and their anchor frontiers from the sandbox split.

            A synchronization group crosses the sandbox boundary when its BIDIRECT component
            holds a sandbox point; every point of such a group is then anchorable. The check
            uses the component rather than the group members because the groups themselves
            already exclude the sandbox points. For a two-party group this matches the
            pairwise definition, and it is the only correct rule for the relay-style
            multi-party groups produced by SYMM matching.

            The frontier of a component c is the minimal set of anchorable components that
            reach c: those with a path to c that no nearer anchor already covers. It is
            computed in a single sweep along the topological order from two sets per
            component:
            - ancestors(c): every anchorable component with a path to c, i.e. the union over
              the predecessors p of ancestors(p), plus p itself whenever p is anchorable.
            - frontier(c) = ancestors(c) minus the union of frontier(a) for a in
              ancestors(c). That subtraction drops exactly the shadowed anchors: if an anchor
              a1 lies before another anchor of ancestors(c), then a1 belongs to the frontier
              of the first anchor following it on that path, and that anchor is itself in
              ancestors(c).
            A component whose only predecessor is not anchorable inherits both sets
            unchanged, which collapses the long linear stretches that the span chain of a
            rank produces. Sets are interned so that the many components sharing a frontier
            hold one object instead of a copy each.
            """
            sandbox_comps = set()
            for rank_id in self.sandbox_ranks:
                for span_id, span in self.ranks[rank_id].spans.items():
                    if span.span_type != SpanType.COMM:
                        continue
                    sandbox_comps.add(self._node_to_comp[CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)])

            # A group occupies exactly one component, so that component is also the key its
            # anchorable points are grouped under. Walking the groups rather than the point
            # set keeps the member order deterministic.
            self.anchorable_points = set()
            comp_anchorable_points: dict[int, list[CrystalGSpanPoint]] = {}
            for group in self.sync_span_groups:
                comp = self._node_to_comp[group.stop_points[0]]
                if comp not in sandbox_comps:
                    continue
                self.anchorable_points.update(group.stop_points)
                comp_anchorable_points[comp] = list(group.stop_points)

            comp_predecessors: dict[int, list[int]] = defaultdict(list)
            for from_comp, neighbors in self._condensed_graph_adj.items():
                for to_comp in neighbors:
                    comp_predecessors[to_comp].append(from_comp)

            empty: frozenset[int] = frozenset()
            interned: dict[frozenset[int], frozenset[int]] = {empty: empty}

            def intern(comps: set[int]) -> frozenset[int]:
                key = frozenset(comps)
                return interned.setdefault(key, key)

            comp_ancestor_anchors: dict[int, frozenset[int]] = {}
            comp_frontier_anchors: dict[int, frozenset[int]] = {}

            for comp in self._comp_topo_order:
                this_comp_predecessors = comp_predecessors.get(comp)
                if not this_comp_predecessors:
                    comp_ancestor_anchors[comp] = empty
                    comp_frontier_anchors[comp] = empty
                    continue
                if len(this_comp_predecessors) == 1 and this_comp_predecessors[0] not in comp_anchorable_points:
                    only_predecessor = this_comp_predecessors[0]
                    comp_ancestor_anchors[comp] = comp_ancestor_anchors[only_predecessor]
                    comp_frontier_anchors[comp] = comp_frontier_anchors[only_predecessor]
                    continue

                this_comp_reachable_anchors: set[int] = set()
                for predecessor in this_comp_predecessors:
                    if predecessor in comp_anchorable_points:
                        this_comp_reachable_anchors.add(predecessor)
                    this_comp_reachable_anchors.update(comp_ancestor_anchors[predecessor])
                this_comp_ancestor_anchors = intern(this_comp_reachable_anchors)
                comp_ancestor_anchors[comp] = this_comp_ancestor_anchors

                covered: set[int] = set()
                for anchor_comp in this_comp_ancestor_anchors:
                    covered.update(comp_frontier_anchors[anchor_comp])
                comp_frontier_anchors[comp] = (
                    this_comp_ancestor_anchors if not covered else intern(this_comp_reachable_anchors - covered)
                )

            # Expand the component level frontier down to points eagerly, so that
            # get_direct_anchor_predecessors() stays a pure lookup during replay. Components
            # that share an interned frontier also share the expanded list, which must
            # therefore be treated as read-only. Anchors are listed in topological order, and
            # every anchor component is a key of comp_anchorable_points by construction.
            points_of_frontier: dict[frozenset[int], list[CrystalGSpanPoint]] = {}
            self._comp_anchor_predecessors = {}
            for comp, anchor_comps in comp_frontier_anchors.items():
                anchor_points = points_of_frontier.get(anchor_comps)
                if anchor_points is None:
                    anchor_points = [
                        anchor_point
                        for anchor_comp in sorted(anchor_comps, key=lambda c: self._comp_to_topo_idx[c])
                        for anchor_point in comp_anchorable_points[anchor_comp]
                    ]
                    points_of_frontier[anchor_comps] = anchor_points
                self._comp_anchor_predecessors[comp] = anchor_points

        # Cross-rank matching only reads spans and channel member tables, so it
        # survives duration updates and sandbox changes; rebuild only when invalidated.
        if not self._cross_rank_built:
            self.build_cross_rank_dependencies()

        node_list: list[CrystalGSpanPoint] = []

        for rank_id, rank in self.ranks.items():
            for span_id in rank.spans:
                start_node = CrystalGSpanPoint(rank_id, span_id, SpanPointType.START)
                stop_node = CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)
                node_list.extend((start_node, stop_node))

        _build_condensed_graph()

        # Synchronization groups only need the component layout, and the join components they
        # contribute must exist before the topological order is computed.
        _build_sync_span_groups()
        _build_group_join_nodes()

        _compute_topological_order()

        # Anchor information depends on the component layout, the topological order and the
        # sandbox split, all of which are available at this point.
        _build_anchor_state()

        # Discard the times cached for the previous edge weights.
        self._comp_time_cached_frontier = 0
        self._comp_time_cache = defaultdict(int)

        self._analyzed = True

    def _ensure_cache_up_to(self, target_topo_idx: int) -> None:
        """Relax the components before target_topo_idx so that its own time becomes final.

        Longest paths on a DAG need no priority queue: one sweep in topological order that
        pushes the time of every component along its outgoing edges suffices, and the sweep
        may stop early. Once all components before target_topo_idx have pushed, the component
        at that index has heard from every predecessor and its cached time can no longer grow,
        while the components behind it stay untouched until a later query needs them. A missing
        cache entry reads as zero, which puts every source of the graph at time zero.

        A target the sweep has already passed is clamped to the frontier, so a caller may ask
        for any index without checking first and an already relaxed prefix is never redone.
        """
        for i in range(self._comp_time_cached_frontier, target_topo_idx):
            comp = self._comp_topo_order[i]
            current_time = self._comp_time_cache[comp]

            for neighbor_comp, edge_weight in self._condensed_graph_adj[comp].items():
                self._comp_time_cache[neighbor_comp] = max(
                    self._comp_time_cache[neighbor_comp], current_time + edge_weight
                )

        self._comp_time_cached_frontier = max(self._comp_time_cached_frontier, target_topo_idx)

    def get_points_interval(
        self,
        start_point: CrystalGSpanPoint | None = None,
        end_point: CrystalGSpanPoint | None = None,
    ) -> int:
        """Get the time interval between two points on the condensed graph.

        Both ends are longest-path values, i.e. the earliest instant a point can be reached
        under the durations the edges currently hold, so the interval is the earliest one that
        satisfies every dependency of every rank. Only the part of the cache the query needs
        is computed, see _ensure_cache_up_to().

        Args:
            start_point: The starting point. If None, the first component of the topological
                order is used; being a source it sits at time zero, the origin of the axis all
                times are measured on.
            end_point: The ending point. If None, the interval reaches the end of the whole
                graph, i.e. the component that finishes last.

        Returns:
            The time interval in nanoseconds (end_time - start_time).

        Raises:
            RuntimeError: If analyze() has not been called.
            ValueError: If the points are invalid or end_point is topologically earlier than start_point.
        """
        if not self._analyzed:
            raise RuntimeError("Graph must be analyzed before calling get_points_interval(). Call analyze() first.")

        if start_point is None:
            if not self._comp_topo_order:
                raise ValueError("Graph has no components")
            start_comp = self._comp_topo_order[0]
        else:
            if start_point not in self._node_to_comp:
                raise ValueError(f"Invalid start_point: {start_point}")
            start_comp = self._node_to_comp[start_point]

        if end_point is None:
            # The longest path of the graph ends at whichever component finishes last, and the
            # topological order does not single that one out: several ranks, or the join
            # components of the synchronization groups, leave the graph with several sinks. A
            # join component can never win this max, because it holds max(start_i) of its group
            # while the STOP component of the same span is at least start_i + duration_i.
            # The scan lives here rather than in the sweep on purpose: the sweep is walked on
            # the replay hot path and on every cache rewind, while a whole-graph query is rare,
            # so maintaining the max per relaxation step would pay for it in the wrong place.
            self._ensure_cache_up_to(len(self._comp_topo_order))
            end_time = max(self._comp_time_cache[comp] for comp in self._comp_topo_order)
        else:
            if end_point not in self._node_to_comp:
                raise ValueError(f"Invalid end_point: {end_point}")
            end_comp = self._node_to_comp[end_point]
            if start_comp == end_comp:
                return 0

            start_topo_idx = self._comp_to_topo_idx[start_comp]
            end_topo_idx = self._comp_to_topo_idx[end_comp]
            if start_topo_idx > end_topo_idx:
                raise ValueError(f"end_point {end_point} topological earlier than start_point {start_point}")

            # The cached time of end_comp is final once every earlier component has propagated,
            # and start_comp is earlier still, so both reads below are answered from the cache.
            self._ensure_cache_up_to(end_topo_idx)
            end_time = self._comp_time_cache[end_comp]

        return end_time - self._comp_time_cache[start_comp]

    def increase_span_duration(self, rank_id: int, span_id: CrystalGSpanId, duration_ns: int) -> None:
        """Raise the duration a span occupies in the condensed graph.

        Only the weight of the span's START -> STOP edge is rewritten. The static
        CrystalGSpan.duration_ns keeps its planning value, so a later analyze() rebuilds the
        edge from it and the increase is lost.

        The cached longest paths are repaired rather than dropped. The sweep consumes this edge
        when it passes the START component, so only an edge already behind the cache frontier
        needs repairing: its new weight is applied to the STOP component by hand and the
        frontier is rewound to that component, which makes the next query re-relax everything
        from there on. This is sound precisely because a duration may only grow: every cached
        time is then at worst too small, and re-relaxing can only push times up.

        Args:
            rank_id: The rank ID where the span is located.
            span_id: The ID of the span to update.
            duration_ns: The new duration in nanoseconds.

        Raises:
            ValueError: If the span does not exist in the graph.
            CrystalGDurationNotIncreasedError: If the new duration does not exceed the
                current one. Callers are meant to catch this and skip the update.
            RuntimeError: If analyze() has not been called.
        """
        if not self._analyzed:
            raise RuntimeError("Graph must be analyzed before calling increase_span_duration(). Call analyze() first.")

        if rank_id not in self.ranks:
            raise ValueError(f"Rank {rank_id} not found in the graph")
        if span_id not in self.ranks[rank_id].spans:
            raise ValueError(f"Span {span_id} not found in rank {rank_id}")

        # Adding a span invalidates _analyzed, so an analyzed graph maps all of its points.
        start_comp = self._node_to_comp[CrystalGSpanPoint(rank_id, span_id, SpanPointType.START)]
        stop_comp = self._node_to_comp[CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)]

        former_duration_ns = self._condensed_graph_adj[start_comp][stop_comp]
        if duration_ns <= former_duration_ns:
            raise CrystalGDurationNotIncreasedError(
                f"New duration {duration_ns} does not exceed the current duration {former_duration_ns} "
                f"of span {span_id} on rank {rank_id}, so the longest path cannot absorb it"
            )

        self._condensed_graph_adj[start_comp][stop_comp] = duration_ns

        # The sweep reads this edge when it passes start_comp, so that is what decides whether
        # the cache needs repairing at all. Had the sweep not reached start_comp yet, it would
        # pick up the new weight on its own.
        if self._comp_to_topo_idx[start_comp] < self._comp_time_cached_frontier:
            # start_comp already pushed the old weight and will never push again, so apply the
            # new one by hand. Its own time is final, the sweep having passed it.
            self._comp_time_cache[stop_comp] = max(
                self._comp_time_cache[stop_comp], self._comp_time_cache[start_comp] + duration_ns
            )
            # Rewind so everything from stop_comp on is relaxed again. min() keeps the frontier
            # from ever moving forward: when stop_comp lies beyond it, the components in
            # between have not been relaxed yet and must not be declared done.
            self._comp_time_cached_frontier = min(self._comp_time_cached_frontier, self._comp_to_topo_idx[stop_comp])

    def sync_span_group_of_point(self, point: CrystalGSpanPoint) -> CrystalGSyncSpanGroup | None:
        """Look up the synchronization group a COMM point belongs to.

        Args:
            point: Either the START or the STOP point of a communication span.

        Returns:
            The synchronization group, or None when the point has none: a computation point, a
            point of a sandbox rank, or a point that is not in the graph at all.
        """
        return self._point_to_sync_group.get(point)

    def sync_span_group_launch_time_ns(self, group: CrystalGSyncSpanGroup) -> int:
        """Get the time at which a whole synchronization group can be launched.

        A multi-party communication cannot start before its last participant has arrived, so
        the launch time is the latest START time over all members. Every group of two or more
        members owns a join component that already holds exactly that max, so advancing the
        same incremental longest-path cache get_points_interval() uses turns the query into a
        single lookup. A single-point group has no join component and degenerates to the START
        time of its only member.

        Args:
            group: A synchronization group of this graph, e.g. from sync_span_group_of_point().

        Returns:
            The launch time in nanoseconds, measured like get_points_interval() from the
            earliest component of the graph.

        Raises:
            RuntimeError: If analyze() has not been called.
        """
        if not self._analyzed:
            raise RuntimeError(
                "Graph must be analyzed before calling sync_span_group_launch_time_ns(). Call analyze() first."
            )

        join_comp = self._sync_group_join_comp.get(self._node_to_comp[group.stop_points[0]])
        if join_comp is not None:
            self._ensure_cache_up_to(self._comp_to_topo_idx[join_comp])

            # The first component of the topological order is a source and therefore sits at
            # zero; subtracting it keeps this path on the same origin as get_points_interval().
            return self._comp_time_cache[join_comp] - self._comp_time_cache[self._comp_topo_order[0]]

        return self.get_points_interval(end_point=group.start_points[0])

    def set_sandbox_ranks(self, all_sandbox_ranks: list[int]) -> None:
        """Register the sandbox ranks as an input to analyze().

        The sandbox split decides which points the synchronization groups carry and
        which points are anchorable, so it must be set before analyze() and every
        change to it requires a fresh analyze() call. Ranks absent from the graph
        are dropped.

        Args:
            all_sandbox_ranks: List of rank IDs to mark as sandbox ranks.
        """
        self.sandbox_ranks = {rank_id for rank_id in all_sandbox_ranks if rank_id in self.ranks}
        self._analyzed = False

    def is_anchorable_point(self, point: CrystalGSpanPoint) -> bool:
        """Return whether a point is anchorable.

        Requires analyze() to have run with a non-empty sandbox registered: without a
        sandbox split no point can ever be anchorable, so returning False everywhere
        would silently mask a missing set_sandbox_ranks() call.

        Raises:
            RuntimeError: If analyze() has not been called, or if no sandbox rank was
                registered by set_sandbox_ranks().
        """
        if not self._analyzed:
            raise RuntimeError("Graph must be analyzed before calling is_anchorable_point(). Call analyze() first.")
        if not self.sandbox_ranks:
            raise RuntimeError("set_sandbox_ranks() must be called with a non-empty set before is_anchorable_point().")

        return point in self.anchorable_points

    def get_direct_anchor_predecessors(self, point: CrystalGSpanPoint) -> list[CrystalGSpanPoint]:
        """Get the direct anchor predecessors for a given point.

        Direct anchor predecessors are the minimal set of anchorable points that precede the
        given point: every one of them reaches the point, and none of them reaches another
        member of the set. Dropping an anchor that another anchor already reaches loses no
        information, because waiting for the later one implies the earlier one, so what remains
        is the smallest set of times the replay side has to wait for. All points of one
        synchronization group are listed together, and the groups follow the topological order.
        See _build_anchor_state() for how the set is derived.

        Args:
            point: The point to query direct anchor predecessors for.

        Returns:
            List of direct anchor predecessor points, empty if none exist. Shared between the
            points that resolve to the same set, hence to be treated as read-only.

        Raises:
            RuntimeError: If analyze() has not been called, or if no sandbox rank was
                registered by set_sandbox_ranks().
            KeyError: If the point is not found in the graph.
        """
        if not self._analyzed:
            raise RuntimeError("analyze() must be called before get_direct_anchor_predecessors().")
        if not self.sandbox_ranks:
            raise RuntimeError(
                "set_sandbox_ranks() must be called with a non-empty set before get_direct_anchor_predecessors()."
            )

        comp = self._node_to_comp[point]
        return self._comp_anchor_predecessors.get(comp, [])
