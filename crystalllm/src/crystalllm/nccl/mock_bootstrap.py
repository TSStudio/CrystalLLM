# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM NCCL Mock Bootstrap functions."""

from __future__ import annotations

import ctypes
import os
from ctypes import c_char_p, c_int

import redis

from ..replay import CrystalTraceParser
from ..replay.data import ReplayerCommOpInfo, ReplayerReplayOpType
from .util import get_mock_lib_path

# Default process group name and description used by PyTorch.
_DEFAULT_GROUP_NAME = "0"
_DEFAULT_GROUP_DESC = "default_pg"

# Base UUID for mock NCCL host ids.
_BASE_HOSTID = "483b9e2d-0723-4e06-b94f-7f5681e3817b"

# The ncclRedisControlSetNeighbors public entry is exported under its C++ mangled name.
_SET_NEIGHBORS_SYMBOL = "_Z28ncclRedisControlSetNeighborsPKc"

# Module-level cache for the loaded mock library so successive calls in the
# same process reuse a single ctypes handle.
_NCCL_MOCK_LIB: ctypes.CDLL | None = None


def _load_mock_lib() -> ctypes.CDLL:
    """Load the mock NCCL .so once and reuse."""
    global _NCCL_MOCK_LIB  # noqa: PLW0603

    if _NCCL_MOCK_LIB is not None:
        return _NCCL_MOCK_LIB

    lib = ctypes.CDLL(str(get_mock_lib_path()))
    _NCCL_MOCK_LIB = lib
    return lib


def get_redis_client(host="localhost", port=6379) -> redis.Redis:
    """Create a Redis client for NCCL mock bootstrap coordination."""
    # Set env vars so ncclRedisControlSetNeighbors can use the same Redis server.
    os.environ["CRYSTAL_REDIS_HOST"] = host
    os.environ["CRYSTAL_REDIS_PORT"] = str(port)
    return redis.Redis(host=host, port=port, db=0)


def get_mock_node_hostid(node_id: int) -> str:
    """Return a deterministic mock NCCL host id for a logical node."""
    prefix, _ = _BASE_HOSTID.rsplit("-", 1)
    return f"{prefix}-{node_id:012x}"


def init_redis_db(rc: redis.Redis, world_size: int, gpus_per_node: int, sandbox_ranks: list[int]) -> None:
    """Initialize Redis DB and write global keys for NCCL mock bootstrap coordination."""
    rc.flushdb()
    rc.set("global:world_size", str(world_size))
    rc.set("global:gpus_per_node", str(gpus_per_node))
    if sandbox_ranks:
        rc.rpush("global:sandbox:ranks", *[str(r) for r in sandbox_ranks])

    node_count = (world_size + gpus_per_node - 1) // gpus_per_node
    for node_id in range(node_count):
        rc.set(f"global:node_hostid:{node_id}", get_mock_node_hostid(node_id))


def _init_group_redis(rc: redis.Redis, group_desc: str, ranks: list[int]) -> None:
    """Write the per-group keys ncclRedisControlSetNeighbors reads."""
    rc.set(f"group:{group_desc}:nranks", str(len(ranks)))
    rc.delete(f"group:{group_desc}:ranks")
    if ranks:
        rc.rpush(f"group:{group_desc}:ranks", *[str(r) for r in ranks])


def _call_set_neighbors(group_desc: str) -> None:
    lib = _load_mock_lib()
    fn = getattr(lib, _SET_NEIGHBORS_SYMBOL)
    fn.argtypes = [c_char_p]
    fn.restype = c_int
    rc = fn(c_char_p(group_desc.encode()))
    if rc != 0:
        raise RuntimeError(f"ncclRedisControlSetNeighbors({group_desc}) returned {rc}")


def _set_group_neighbors(
    rc: redis.Redis,
    group_desc: str,
    ranks: list[int],
    sandbox_ranks: list[int],
    world_size: int,
    gpus_per_node: int,
) -> list[int]:
    """Init Redis and call ncclRedisControlSetNeighbors for one process group then returns the neighbor."""
    print(
        f"[mock_bootstrap/set_group_neighbors()] group_desc={group_desc} ranks={ranks} sandbox_ranks={sandbox_ranks}"
        f" world_size={world_size} gpus_per_node={gpus_per_node}",
        flush=True,
    )

    _init_group_redis(rc, group_desc, ranks)

    rc.delete(f"group:{group_desc}:sandbox:neighbors")
    _call_set_neighbors(group_desc)

    rc.delete(f"group:{group_desc}:sandbox:active_ranks")
    rc.delete(f"group:{group_desc}:sandbox:proxy_ranks")

    vals = rc.lrange(f"group:{group_desc}:sandbox:neighbors", 0, -1)

    print(
        f"[mock_bootstrap/set_group_neighbors()] group_desc={group_desc} "
        f"raw_neighbors={vals} sandbox_ranks={sandbox_ranks}",
        flush=True,
    )

    return sorted(int(v) for v in vals if int(v) not in sandbox_ranks)


def _publish_group_active_proxy_ranks(
    rc: redis.Redis,
    group_desc: str,
    ranks: list[int],
    sandbox_ranks_set: set[int],
    extended_member_ranks_set: set[int],
) -> None:
    """Populate the per-group active/proxy rank lists used by mockBootstrapProxy.

    Active ranks are extended member ranks in the group, excluding sandbox ranks.
    Proxy ranks are non-member, non-sandbox ranks in the group. Both lists are
    stored as commRank (index within ``ranks``). The input rank order must match
    ``group:<groupDesc>:ranks``.
    """
    ranks_set = set(ranks)
    active_ranks = sorted((ranks_set & extended_member_ranks_set) - sandbox_ranks_set)
    proxy_ranks = sorted(ranks_set - sandbox_ranks_set - extended_member_ranks_set)

    global_rank_to_comm_rank = {g: c for c, g in enumerate(ranks)}
    try:
        active_comm_ranks = sorted(global_rank_to_comm_rank[g] for g in active_ranks)
        proxy_comm_ranks = sorted(global_rank_to_comm_rank[g] for g in proxy_ranks)
    except KeyError as e:
        raise ValueError(
            f"[mock_bootstrap/publish_group_active_proxy_ranks()] global rank {e.args[0]} not "
            f"in ranks for group_desc={group_desc}; caller passed "
            f"ranks={ranks} extended_member_ranks_set={extended_member_ranks_set} "
            f"sandbox_set={sandbox_ranks_set}"
        ) from e

    rc.delete(f"group:{group_desc}:sandbox:active_ranks")
    if active_comm_ranks:
        rc.rpush(f"group:{group_desc}:sandbox:active_ranks", *[str(r) for r in active_comm_ranks])

    rc.delete(f"group:{group_desc}:sandbox:proxy_ranks")
    if proxy_comm_ranks:
        rc.rpush(f"group:{group_desc}:sandbox:proxy_ranks", *[str(r) for r in proxy_comm_ranks])

    print(
        f"[mock_bootstrap/publish_group_active_proxy_ranks()] group_desc={group_desc} "
        f"active_ranks=(n={len(active_comm_ranks)}) global→comm {active_ranks}→{active_comm_ranks} "
        f"proxy_ranks=(n={len(proxy_comm_ranks)}) global→comm {proxy_ranks}→{proxy_comm_ranks}",
        flush=True,
    )


def _is_p2p_comm_op(op_info: ReplayerCommOpInfo) -> bool:
    """Return whether the replay op is any isend/irecv, including batch_isend_irecv."""
    return op_info.c10d_func_name in {"isend", "irecv"}


def _is_batch_isend_irecv_op(op_info: ReplayerCommOpInfo) -> bool:
    """Return whether the isend/irecv op comes from torch.distributed.batch_isend_irecv."""
    return _is_p2p_comm_op(op_info) and op_info.signature.endswith("batch_isend_irecv")


def _is_point_to_point_p2p_op(op_info: ReplayerCommOpInfo) -> bool:
    """Return whether the replay op is a standalone point-to-point isend/irecv."""
    return _is_p2p_comm_op(op_info) and not _is_batch_isend_irecv_op(op_info)


def _get_p2p_peer_global_rank(
    group_name: str,
    group_desc: str,
    ranks: list[int],
    op_info: ReplayerCommOpInfo,
) -> int:
    """Extract one isend/irecv peer and normalize it to global rank.

    Regular isend/irecv records ``dst`` or ``src`` as global rank. Calls emitted from
    ``batch_isend_irecv`` record ``group_dst`` or ``group_src`` as commRank, so they
    must be translated through the original group ``ranks`` list used by Redis.
    """
    if op_info.c10d_func_name == "isend":
        peer_field_candidates = ("group_dst",) if _is_batch_isend_irecv_op(op_info) else ("dst", "group_dst")
    elif op_info.c10d_func_name == "irecv":
        peer_field_candidates = ("group_src",) if _is_batch_isend_irecv_op(op_info) else ("src", "group_src")
    else:
        raise ValueError(f"Unexpected non-P2P op: {op_info}")

    peer_field = next((field for field in peer_field_candidates if field in op_info.meta), None)
    if peer_field is None:
        raise ValueError(
            f"[mock_bootstrap/prepare_mock_bootstrap_redis()] missing P2P peer field from "
            f"{peer_field_candidates} for group_name={group_name} group_desc={group_desc} "
            f"op=(trace_comm_index={op_info.trace_comm_index}, c10d_func_name={op_info.c10d_func_name}, "
            f"signature={op_info.signature}, meta={op_info.meta})"
        )

    peer_rank = op_info.meta[peer_field]
    if peer_field in {"group_dst", "group_src"}:
        # group_src/group_dst are commRank values; convert through ranks[commRank] to globalRank.
        if peer_rank < 0 or peer_rank >= len(ranks):
            raise ValueError(
                f"[mock_bootstrap/prepare_mock_bootstrap_redis()] invalid group peer rank {peer_rank} "
                f"for group_name={group_name} group_desc={group_desc} ranks={ranks} "
                f"op=(trace_comm_index={op_info.trace_comm_index}, c10d_func_name={op_info.c10d_func_name}, "
                f"signature={op_info.signature})"
            )
        return ranks[peer_rank]

    return peer_rank


def _inspect_group_replay_ops(
    group_name: str,
    group_desc: str,
    ranks: list[int],
    replay_ops_by_trace_index: dict[int, list],
) -> tuple[bool, list[int]]:
    """Inspect group replay ops and return only the results needed downstream.

    Returns:
        A tuple containing:
            - whether this is a pure standalone point-to-point isend/irecv group;
              such groups skip neighbor setup and publish.
            - all isend/irecv peer global ranks; these peers must become members
              regardless of whether the group is published.
    """
    comm_ops = []
    for trace_comm_index in sorted(replay_ops_by_trace_index):
        comm_ops.extend(
            replay_op.info
            for replay_op in replay_ops_by_trace_index[trace_comm_index]
            if replay_op.type == ReplayerReplayOpType.COMM and isinstance(replay_op.info, ReplayerCommOpInfo)
        )

    p2p_ops = []
    point_to_point_p2p_ops = []
    for op in comm_ops:
        if _is_p2p_comm_op(op):
            p2p_ops.append(op)
        if _is_point_to_point_p2p_op(op):
            point_to_point_p2p_ops.append(op)

    p2p_peer_ranks = sorted({_get_p2p_peer_global_rank(group_name, group_desc, ranks, op) for op in p2p_ops})
    group_is_p2p_only = bool(point_to_point_p2p_ops) and len(point_to_point_p2p_ops) == len(comm_ops)
    return group_is_p2p_only, p2p_peer_ranks


def prepare_mock_bootstrap_redis(
    redis_host: str,
    trace_dir: str,
    sandbox_nodes: list[int],
    nnodes: int,
    gpus_per_node: int,
    master_rank: int = 0,
) -> dict[str, list[int]]:
    r"""Prepare Redis for NCCL mock bootstrap from trace and return scheduling results.

    The mock bootstrap mechanism addresses the memory overhead issue caused by large communication
    groups that require many assistant processes during initialization. The key insight is that
    in simplified ring or tree communication patterns, only a small subset of ranks actually
    participate in data transfer, so only these ranks need to join the initialization process.

    ``default_pg`` is handled as an internal special group because it represents the global NCCL
    process group and is required to seed default group neighbors before other process groups are
    published.

    Terminology:
        - "Member ranks": Non-sandbox ranks that are required by default group
          neighbors, filtered communication groups, or P2P peer endpoints.
        - "Member nodes": Node-level scheduling units that contain member ranks plus
          the master rank node.
        - "Extended member ranks": All ranks on member nodes. Once computed, Redis
          publish uses this node-granular set instead of the raw member ranks.

    Args:
        redis_host: Redis host address for NCCL mock bootstrap coordination.
        trace_dir: Directory containing ``trace_0.log.gz``.
        sandbox_nodes: List of logical node ids running training scripts.
        nnodes: Total number of logical nodes in the target cluster.
        gpus_per_node: Number of GPUs per node.
        master_rank: The rank of the master process (default: 0).

    Returns:
        A dict containing:
            - extended_member_ranks: Sorted list of all ranks on member nodes.
            - member_nodes: Sorted list of member nodes.
    """
    sandbox_ranks = []
    for node_id in sandbox_nodes:
        sandbox_ranks.extend(list(range(node_id * gpus_per_node, (node_id + 1) * gpus_per_node)))

    world_size = nnodes * gpus_per_node
    trace_file = f"{trace_dir}/trace_0.log.gz"
    trace_parser = CrystalTraceParser(trace_file)
    comm_groups_info = trace_parser.comm_groups_ordered_list

    rc = get_redis_client(redis_host)
    init_redis_db(rc, world_size, gpus_per_node, sandbox_ranks)

    # Step 1: Filter out groups that do not contain any sandbox ranks.
    sandbox_ranks_set = set(sandbox_ranks)
    filtered_comm_groups_info = [
        group_info for group_info in comm_groups_info if any(rank in sandbox_ranks_set for rank in group_info.ranks)
    ]

    # Step 2: Collect member_ranks. default_pg is handled first as the global special group.
    p2p_only_group_descs = []
    group_is_p2p_only_by_group_name: dict[str, bool] = {}

    default_group_ranks = list(range(world_size))
    default_group_replay_ops = trace_parser.replay_comm_ops_dict[_DEFAULT_GROUP_NAME]
    default_group_is_p2p_only, default_group_p2p_peer_ranks = _inspect_group_replay_ops(
        _DEFAULT_GROUP_NAME, _DEFAULT_GROUP_DESC, default_group_ranks, default_group_replay_ops
    )

    member_ranks_set = set(default_group_p2p_peer_ranks)

    if default_group_is_p2p_only:
        # For a pure point-to-point default_pg trace, peers are enough; mock neighbor init is not needed.
        p2p_only_group_descs.append(_DEFAULT_GROUP_DESC)
    else:
        default_group_neighbor_ranks = _set_group_neighbors(
            rc, _DEFAULT_GROUP_DESC, default_group_ranks, sandbox_ranks, world_size, gpus_per_node
        )
        member_ranks_set.update(default_group_neighbor_ranks)

    for group_info in filtered_comm_groups_info:
        replay_ops_by_trace_index = trace_parser.replay_comm_ops_dict[group_info.group_name]
        group_is_p2p_only, group_p2p_peer_ranks = _inspect_group_replay_ops(
            group_info.group_name, group_info.group_desc, group_info.ranks, replay_ops_by_trace_index
        )
        group_is_p2p_only_by_group_name[group_info.group_name] = group_is_p2p_only
        # All P2P peers must be scheduled as member ranks, even when the group itself is not published.
        member_ranks_set.update(group_p2p_peer_ranks)

        if group_is_p2p_only:
            # P2P-only groups do not participate in NCCL mock init, so skip neighbor/publish for them.
            p2p_only_group_descs.append(group_info.group_desc)
        elif not group_info.group_desc:
            if len(group_info.ranks) > 1:
                # If a group has no group_desc but has more than one rank, NCCL mock bootstrap could not handle it.
                member_ranks_set.update(group_info.ranks)
        elif group_info.group_desc == _DEFAULT_GROUP_DESC:
            # default_pg is handled once by its PyTorch group_name "0" path above.
            continue
        elif len(group_info.ranks) <= 1:
            # If a group has only one rank, it is a no-op in creating communication groups.
            continue
        else:
            member_ranks_set.update(
                _set_group_neighbors(
                    rc, group_info.group_desc, list(group_info.ranks), sandbox_ranks, world_size, gpus_per_node
                )
            )

    member_ranks_set.difference_update(sandbox_ranks_set)
    member_ranks = sorted(member_ranks_set)

    # Step 3: Calculate member_nodes that can contain all member_ranks.
    member_nodes_set = {rank // gpus_per_node for rank in member_ranks}
    # Only root rank calculates the unique id of the default NCCL group.
    if master_rank not in sandbox_ranks_set and master_rank not in member_ranks_set:
        member_nodes_set.add(master_rank // gpus_per_node)
    member_nodes = sorted(member_nodes_set)

    # Step 4: Extend member ranks to all ranks on member nodes because CrystalLLM schedules by node.
    extended_member_ranks = sorted(
        rank for node_id in member_nodes for rank in range(node_id * gpus_per_node, (node_id + 1) * gpus_per_node)
    )
    extended_member_ranks_set = set(extended_member_ranks)

    # Step 5: Publish active/proxy ranks for default_pg and filtered process groups.
    if not default_group_is_p2p_only:
        _publish_group_active_proxy_ranks(
            rc,
            _DEFAULT_GROUP_DESC,
            default_group_ranks,
            sandbox_ranks_set,
            extended_member_ranks_set,
        )

    for group_info in filtered_comm_groups_info:
        group_is_p2p_only = group_is_p2p_only_by_group_name[group_info.group_name]
        if (
            group_is_p2p_only
            or not group_info.group_desc
            or group_info.group_desc == _DEFAULT_GROUP_DESC
            or len(group_info.ranks) <= 1
        ):
            # Groups skipped during neighbor setup must not publish active/proxy ranks either.
            continue
        _publish_group_active_proxy_ranks(
            rc,
            group_info.group_desc,
            group_info.ranks,
            sandbox_ranks_set,
            extended_member_ranks_set,
        )

    # Step 6: Print information with length first, then elements.
    print("[mock_bootstrap/prepare_mock_bootstrap_redis()] Summary:")
    print(f"  - Number of original groups: {len(comm_groups_info)}")
    print(f"  - Number of filtered groups (containing sandbox_ranks): {len(filtered_comm_groups_info)}")
    print(f"  - P2P-only groups without neighbor/publish (len={len(p2p_only_group_descs)}): {p2p_only_group_descs}")
    print(f"  - Sandbox ranks (len={len(sandbox_ranks)}): {sandbox_ranks}")
    print(f"  - Member ranks (len={len(member_ranks)}): {member_ranks}")
    print(f"  - Member nodes (len={len(member_nodes)}): {member_nodes}")
    print(f"  - Extended member ranks (len={len(extended_member_ranks)}): {extended_member_ranks}")

    return {
        "extended_member_ranks": extended_member_ranks,
        "member_nodes": member_nodes,
    }
