# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM PyTorch Distributed C10d Collective Handlers."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from typing_extensions import ParamSpec

from ..planning import CrystalGCommPattern, CrystalGCommTasks, CrystalGSpanId
from ..runtime import get_context
from ..util import get_from_args_kwargs
from .fake import CrystalFakeProcessGroup, CrystalFakeWork, CrystalProfiledWork

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _local_world_size() -> int:
    """Read the number of ranks per node from LOCAL_WORLD_SIZE (fallback: GPUS_PER_NODE)."""
    value = os.getenv("LOCAL_WORLD_SIZE") or os.getenv("GPUS_PER_NODE")
    if value is None:
        raise RuntimeError("LOCAL_WORLD_SIZE or GPUS_PER_NODE must be set")
    return int(value)


def _get_group(args: list[Any], kwargs: dict[str, Any], position: int | None = None, keyword: str = "group") -> Any:
    """Extract the ProcessGroup argument from args/kwargs, falling back to the default c10d group.

    ``position`` is the index of the group parameter in the intercepted collective's
    signature: positional arguments win over the keyword form, and an omitted group
    means the default world group.
    """
    group = get_from_args_kwargs(args, kwargs, position=position, keyword=keyword)
    if group is not None:
        return group
    return torch.distributed.distributed_c10d._get_default_group()  # noqa: SLF001


def _get_async_op(args: list[Any], kwargs: dict[str, Any], position: int, keyword: str = "async_op") -> bool:
    """Extract the async_op flag from args/kwargs, defaulting to a synchronous call."""
    async_op = get_from_args_kwargs(args, kwargs, position=position, keyword=keyword)
    return bool(async_op) if async_op is not None else False


def _participants_cross_nodes(global_ranks: list[int]) -> bool:
    """Return whether the given global ranks span more than one node.

    A rank's node is identified as ``global_rank // local_world_size``, so ranks on
    the same node share the same quotient and the set of quotients collapses to one
    element.
    """
    return len({rank // _local_world_size() for rank in global_ranks}) > 1


def _process_group_name(group: Any) -> str:
    """Return a stable c10d process-group identity.

    The PG's ``group_name`` attribute is preferred (real and fake PGs both carry it);
    otherwise the c10d ``_world`` name mapping is consulted so a bare ProcessGroup
    still resolves. No name at all means the group never passed through c10d
    bookkeeping, which is treated as a caller error.
    """
    group_name = getattr(group, "group_name", "")
    if group_name:
        return str(group_name)
    mapped_name = torch.distributed.distributed_c10d._world.pg_names.get(group)  # noqa: SLF001
    if mapped_name:
        return str(mapped_name)
    raise RuntimeError("ProcessGroup has no registered c10d group_name")


def _is_replay_rendezvous_group(group: Any) -> bool:
    """Return whether the group is the internal replay rendezvous ProcessGroup.

    The rendezvous PG carries no training communication, so collectives on it must
    never become comm spans or profiler slots.
    """
    return str(group.group_desc) == "_crystal_replay_rendezvous_pg"


def _extract_group_and_async(func_name: str, args: list[Any], kwargs: dict[str, Any]) -> tuple[Any | None, bool]:
    """Extract ``(group, async_op)`` from a collective call based on its function name.

    The two values sit at different argument positions in different c10d signatures,
    because collectives place ``group``/``async_op`` after their tensor arguments:

    - ``isend``/``irecv`` ``(tensor, dst/src, group, tag, group_dst/group_src)``:
      group at 2; asynchronous by definition, so async_op is always True.
    - ``broadcast``/``all_reduce``/``all_gather_into_tensor``/``_all_gather_base``:
      group at 2, async_op at 3.
    - ``reduce_scatter_tensor``/``_reduce_scatter_base`` ``(output, input, op, ...)``:
      group at 3, async_op at 4.
    - ``all_to_all_single`` ``(output, input, out_splits, in_splits, ...)``:
      group at 4, async_op at 5.
    - ``barrier`` ``(group, async_op)``: group at 0, async_op at 1.
    """
    if func_name in {"isend", "irecv"}:
        return _get_group(args, kwargs, position=2), True
    if func_name == "broadcast":
        return _get_group(args, kwargs, position=2), _get_async_op(args, kwargs, position=3)
    if func_name == "all_reduce":
        return _get_group(args, kwargs, position=2), _get_async_op(args, kwargs, position=3)
    if func_name in {"all_gather_into_tensor", "_all_gather_base"}:
        return _get_group(args, kwargs, position=2), _get_async_op(args, kwargs, position=3)
    if func_name in {"reduce_scatter_tensor", "_reduce_scatter_base"}:
        return _get_group(args, kwargs, position=3), _get_async_op(args, kwargs, position=4)
    if func_name == "all_to_all_single":
        return _get_group(args, kwargs, position=4), _get_async_op(args, kwargs, position=5)
    if func_name == "barrier":
        return _get_group(args, kwargs, position=0), _get_async_op(args, kwargs, position=1)
    raise NotImplementedError(f"Unsupported collective operation: {func_name}")


def _p2p_comm_tasks(func_name: str, group: Any, args: list[Any], kwargs: dict[str, Any]) -> CrystalGCommTasks | None:
    """Build the comm tasks of an isend/irecv call, or None when the peer is on the same node.

    Peer resolution: the patched isend/irecv accept ``group_dst``/``group_src``
    (group-relative) directly; otherwise the global ``dst``/``src`` argument is
    translated through the group's rank mapping.

    Pattern selection assumes cross-node P2P traffic only flows between ring
    neighbours, as in pipeline parallelism: a send to the downstream member
    ``(group_rank + 1) % size`` is ``P2P_DN``, a send to the upstream member is
    ``P2P_UP`` (mirrored for irecv), so the sender's task always pairs with the
    receiver's complementary task. A peer that shares the node is intra-node
    traffic and gets no comm span at all, so the neighbour rule is only enforced
    for cross-node peers: a non-neighbour would be misclassified into a pattern
    that cannot be paired across ranks, so it raises instead.

    The channel name carries the unordered rank pair ``min:max`` so both peers derive
    the same string regardless of who sends and who receives.
    """
    group_rank = int(group.rank())
    downstream = (group_rank + 1) % group.size()
    upstream = (group_rank - 1) % group.size()
    if func_name == "isend":
        group_peer = get_from_args_kwargs(args, kwargs, position=4, keyword="group_dst")
        if group_peer is None:
            dst = get_from_args_kwargs(args, kwargs, position=1, keyword="dst")
            group_peer = int(torch.distributed.get_group_rank(group, int(dst)))
        group_peer = int(group_peer)
        peer_global = int(torch.distributed.get_global_rank(group, group_peer))
        self_global = int(torch.distributed.get_global_rank(group, group_rank))
        comm_task = CrystalGCommPattern.P2P_DN if group_peer == downstream else CrystalGCommPattern.P2P_UP
    else:  # irecv
        group_peer = get_from_args_kwargs(args, kwargs, position=4, keyword="group_src")
        if group_peer is None:
            src = get_from_args_kwargs(args, kwargs, position=1, keyword="src")
            group_peer = int(torch.distributed.get_group_rank(group, int(src)))
        group_peer = int(group_peer)
        peer_global = int(torch.distributed.get_global_rank(group, group_peer))
        self_global = int(torch.distributed.get_global_rank(group, group_rank))
        comm_task = CrystalGCommPattern.P2P_UP if group_peer == upstream else CrystalGCommPattern.P2P_DN

    if not _participants_cross_nodes([self_global, peer_global]):
        return None

    # A cross-node peer outside the neighbour pair would be silently misclassified
    # into the wrong pattern above, breaking the cross-rank pairing of the graph.
    if group_peer not in {downstream, upstream}:
        raise RuntimeError(
            f"{func_name}: cross-node peer group rank {group_peer} is not a ring neighbour of "
            f"group rank {group_rank} (upstream={upstream}, downstream={downstream}, size={group.size()})"
        )

    first_rank = min(group_rank, group_peer)
    second_rank = max(group_rank, group_peer)
    return CrystalGCommTasks(
        channel=_process_group_name(group) + f"_{first_rank}:{second_rank}",
        tasks=[comm_task],
    )


def _symm_comm_tasks(func_name: str, group: Any) -> CrystalGCommTasks | None:
    """Build the comm tasks of a symmetric collective (all members participate), or None.

    ``all_to_all_single`` is skipped on purpose: its exchanges stay inside one
    emulation sandbox and are not modelled as cross-node communication. For every
    other collective the whole group joins, so the pattern is ``SYMM`` on the
    group-name channel; a group whose members all share one node gets no comm span.
    """
    if func_name == "all_to_all_single":
        return None
    members = list(torch.distributed.get_process_group_ranks(group))
    if not _participants_cross_nodes(members):
        return None
    return CrystalGCommTasks(channel=_process_group_name(group), tasks=[CrystalGCommPattern.SYMM])


def _comm_tasks_for_call(
    func_name: str,
    group: Any,
    args: list[Any],
    kwargs: dict[str, Any],
) -> CrystalGCommTasks | None:
    """Dispatch to the P2P or symmetric task builder for one collective call."""
    if group.group_desc.startswith("CONTEXT-"):
        # CP stays inside the Sandbox, in both capture and profiling.
        return None
    if func_name in {"isend", "irecv"}:
        return _p2p_comm_tasks(func_name, group, args, kwargs)
    return _symm_comm_tasks(func_name, group)


def crystal_collectives_handler(
    func: Callable[_P, _T], args: list[Any], kwargs: dict[str, Any]
) -> CrystalFakeWork | None:
    """Capture-mode handler: record the comm span and replace the real launch with fake work.

    Nothing is actually communicated in capture mode, so the intercepted function is
    never called and the caller gets ``CrystalFakeWork`` (async calls) whose wait()
    drives the span bookkeeping, or None (sync calls, whose real return value would
    also be None). The comm span reuses ``last_trace_comm_index`` because the trace
    writer wrapper, which runs one layer outside this handler, has already reserved
    the index for this call under CRYSTAL_C10D_CAPTURING. Collectives without a
    group, on the rendezvous PG, or without cross-node participants stay unrecorded.
    """
    group, async_op = _extract_group_and_async(func.__name__, args, kwargs)
    comm_span_id: CrystalGSpanId | None = None

    if group is not None and not _is_replay_rendezvous_group(group):
        comm_tasks = _comm_tasks_for_call(func.__name__, group, args, kwargs)
        if comm_tasks is not None:
            if get_context().is_comm_span_coalescing:
                # Inside a coalescing manager the single calls contribute tasks to the
                # manager's span; the span itself is closed when the manager exits.
                for task in comm_tasks.tasks:
                    get_context().cond_add_coalescing_comm_task(task)
            else:
                comm_span_id = get_context().cond_record_comm_span(
                    get_context().last_trace_comm_index,
                    comm_tasks,
                    async_op=async_op,
                )
                # Fake work launches nothing, so the pending proc span starts at
                # once; the profiler handler starts the same span only after its
                # real launch.
                get_context().cond_start_proc_span_after_comm()

    return CrystalFakeWork(comm_span_id) if async_op else None


def crystal_runtime_profiler_collectives_handler(
    func: Callable[_P, _T],
    args: list[Any],
    kwargs: dict[str, Any],
) -> _T:
    """Profiler-mode handler: record the comm span/slot around a real communication launch.

    Unlike capture mode the collective actually runs. The trace index is reserved
    here because the trace writer wrapper only reserves under CRYSTAL_C10D_CAPTURING,
    which is off in profiler mode. Inside a coalescing manager the call only
    contributes tasks to the manager's span. Every call on a timed group is followed
    by exactly one completion-hook report, so the slot is recorded even for calls
    that produced no comm span (see below).
    """
    group, async_op = _extract_group_and_async(func.__name__, args, kwargs)
    if group is None or _is_replay_rendezvous_group(group):
        return func(*args, **kwargs)
    if get_context().is_comm_span_coalescing:
        comm_tasks = _comm_tasks_for_call(func.__name__, group, args, kwargs)
        if comm_tasks is not None:
            for task in comm_tasks.tasks:
                get_context().cond_add_coalescing_comm_task(task)
        return func(*args, **kwargs)

    comm_span_id: CrystalGSpanId | None = None
    comm_tasks = _comm_tasks_for_call(func.__name__, group, args, kwargs)
    if comm_tasks is not None:
        trace_comm_index = get_context().reserve_trace_comm_index()
        comm_span_id = get_context().cond_record_comm_span(trace_comm_index, comm_tasks, async_op=async_op)
    # A timed process group reports a completion for every operation, so operations
    # without a comm span still need a slot to keep the sequences aligned; their
    # duration is reported under the group's own channel and tolerated as extra data
    # when the graph bundle loads the profiler output.
    channel = comm_tasks.channel if comm_tasks is not None else _process_group_name(group)
    get_context().cond_record_profiler_comm_slot(group, channel)

    result = func(*args, **kwargs)
    # The pending next proc span starts only now that the real communication is on
    # the wire: starting it before func() would time a blocking synchronous
    # communication inside the following computation span. Without a pending span
    # (local-only operation) this is a no-op.
    get_context().cond_start_proc_span_after_comm()

    if async_op and result is not None and comm_span_id is not None:
        return CrystalProfiledWork(result, comm_span_id)
    return result


def crystal_c10d_coalescing_manager_handler(group: CrystalFakeProcessGroup, async_ops: bool) -> CrystalFakeWork:
    """Handle the exit of a capture-mode coalescing manager.

    Closes the manager's single comm span over all tasks collected inside the scope,
    then hands back fake work exactly like a single captured collective: fake work
    launches nothing, so the pending proc span starts at once, keeping the order
    around the launch aligned with the profiler-mode manager handler.
    """
    comm_span_id = get_context().cond_end_comm_span_coalescing(group.group_name, async_op=async_ops)
    get_context().cond_start_proc_span_after_comm()
    return CrystalFakeWork(comm_span_id if async_ops else None)


def crystal_runtime_profiler_coalescing_manager_handler(
    group: Any,
    async_ops: bool,
    launch_func: Callable[[], object | None],
) -> object | None:
    """Profiler-mode coalescing manager exit: close the span, launch the work, record the slot.

    The ordering mirrors ``crystal_runtime_profiler_collectives_handler``: the span
    chain is cut before the real kernels are launched, and the next proc span stays
    pending until the launch returns, so a blocking synchronous launch is never
    timed inside the following computation span. The slot is recorded after the
    launch because a manager without a comm span still needs a slot when its work
    reports a completion.
    """
    # The whole manager is one comm span on its own PG name — the same collective
    # channel naming as _symm_comm_tasks() and the value the capture side passes via
    # group.group_name; a manager has no per-op comm_tasks to take a channel from. A
    # rendezvous manager must never become a communication, so its collected tasks
    # are dropped by passing channel=None.
    channel = None if _is_replay_rendezvous_group(group) else _process_group_name(group)
    comm_span_id = get_context().cond_end_comm_span_coalescing(channel, async_op=async_ops)
    # The real coalesced kernels are launched only after the span chain is cut above.
    # With a comm span the pending next proc span starts only after the launch returns,
    # keeping the launch duration in the gap between spans; an empty manager (no comm
    # span) has already started its next proc span, since there is no communication
    # whose launch time must stay out of it.
    work = launch_func()
    get_context().cond_start_proc_span_after_comm()
    # Every manager reports a slot under its group's channel, whether it became a
    # comm span or not, to keep the slot/completion sequences aligned; the graph
    # bundle tolerates durations without a graph span.
    if comm_span_id is not None or (work is not None and channel is not None):
        get_context().cond_record_profiler_comm_slot(group, channel)

    if async_ops and work is not None and comm_span_id is not None:
        return CrystalProfiledWork(work, comm_span_id)
    return work
