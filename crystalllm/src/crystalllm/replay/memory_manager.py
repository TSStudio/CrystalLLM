# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Unified Memory Manager for Replay Assistant.

See ASSISTANT_MEMORY_MANAGER.md at the repository root for the full design.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib.util
import json
import logging
import math
import os
import pathlib
import pickle
from collections import defaultdict
from dataclasses import dataclass

import torch

from .data import (
    ReplayerCommGroupInfo,
    ReplayerCommOpInfo,
    ReplayerReplayOpInfo,
    ReplayerReplayOpType,
    TensorMemoryLevel,
    TensorProvider,
)

# Configure logging with default StreamHandler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage A: constants, demand/share key encoding, Redis key builders, Lua
# script wrappers.
# ---------------------------------------------------------------------------

PLACEHOLDER = "<placeholder>"

MEMORY_MANAGER_REDIS_DB = 2
MEMORY_MANAGER_REDIS_HOST = "localhost"
MEMORY_MANAGER_REDIS_PORT = 6379

MEMORY_TENSOR_INPUT = "INPUT"
MEMORY_TENSOR_OUTPUT = "OUTPUT"

STATE_READY = "READY"
STATE_EVICTING = "EVICTING"
STATE_EVICTED = "EVICTED"
STATE_DESTROYING = "DESTROYING"
STATE_DESTROYED = "DESTROYED"

ACQUIRE_READY = "READY"
ACQUIRE_NOT_READY = "NOT_READY"
ACQUIRE_VERSION_MISMATCH = "VERSION_MISMATCH"

CLAIM_CLAIMED = "CLAIMED"
CLAIM_SKIP = "SKIP"

OWNER_KIND_SHARED = "SHARED"
OWNER_KIND_PERSISTENT = "PERSISTENT"
OWNER_KIND_INSTANT = "INSTANT"


def encode_demand_id(
    rank: int,
    c10d_func_name: str,
    signature: str,
    direction: str,
    shape: list[int],
    dtype: str,
) -> str:
    """Encode a tensor demand identity into a reversible, Redis-key-safe string."""
    payload = [rank, c10d_func_name, signature, direction, shape, dtype]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def decode_demand_id(demand_id: str) -> tuple[int, str, str, str, list[int], str]:
    """Decode a demand_id produced by :func:`encode_demand_id`."""
    padding = "=" * (-len(demand_id) % 4)
    raw = base64.urlsafe_b64decode((demand_id + padding).encode("ascii")).decode("utf-8")
    rank, c10d_func_name, signature, direction, shape, dtype = json.loads(raw)
    return rank, c10d_func_name, signature, direction, shape, dtype


def encode_share_key(
    c10d_func_name: str,
    signature: str,
    direction: str,
    shape: list[int],
    dtype: str,
) -> str:
    """Encode the fields shared by SHARED-level demands (demand_id without rank)."""
    payload = [c10d_func_name, signature, direction, shape, dtype]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def decode_share_key(share_key: str) -> tuple[str, str, str, list[int], str]:
    """Decode a share_key produced by :func:`encode_share_key`."""
    padding = "=" * (-len(share_key) % 4)
    raw = base64.urlsafe_b64decode((share_key + padding).encode("ascii")).decode("utf-8")
    c10d_func_name, signature, direction, shape, dtype = json.loads(raw)
    return c10d_func_name, signature, direction, shape, dtype


def demand_id_from_share_key(share_key: str, rank: int) -> str:
    """Re-derive a demand_id for a given rank from a share_key (used by tests/debugging)."""
    c10d_func_name, signature, direction, shape, dtype = decode_share_key(share_key)
    return encode_demand_id(rank, c10d_func_name, signature, direction, shape, dtype)


def local_namespace(local_rank: int) -> str:
    """The Redis key namespace prefix owned by the manager of a given local_rank."""
    return f"memory:{local_rank}:"


def redis_field(prefix: str, name: str) -> str:
    """Build a full Redis key by appending a field suffix to a resource key prefix."""
    return f"{prefix}:{name}"


# --- init: ---


def key_init_done(local_rank: int) -> str:
    """Redis key indicating MemoryManager initialization is complete."""
    return f"{local_namespace(local_rank)}init:done"


def key_init_managed_ranks(local_rank: int) -> str:
    """Redis list key containing ranks managed by this local rank."""
    return f"{local_namespace(local_rank)}init:managed_ranks"


def key_init_reserved_memory_ratio(local_rank: int) -> str:
    """Redis key storing the reserved GPU memory ratio for this manager."""
    return f"{local_namespace(local_rank)}init:RESERVED_MEMORY_RATIO"


# --- demand: ---


def key_demand_prefix(local_rank: int, demand_id: str) -> str:
    """Redis key prefix for metadata of one tensor demand."""
    return f"{local_namespace(local_rank)}demand:{demand_id}"


def key_demand_level(local_rank: int, demand_id: str) -> str:
    """Redis key storing the memory level of one tensor demand."""
    return redis_field(key_demand_prefix(local_rank, demand_id), "level")


def key_demand_target_ref(local_rank: int, demand_id: str) -> str:
    """Redis key storing the current target resource reference of one demand."""
    return redis_field(key_demand_prefix(local_rank, demand_id), "target_ref")


# --- pool:anything:<dtype>: ---


def key_pool_anything(local_rank: int, dtype_str: str) -> str:
    """Resource key prefix for the ANYTHING shared pool of a given dtype."""
    return f"{local_namespace(local_rank)}pool:anything:{dtype_str}"


# --- allocation:<owner_kind>:<seq>: ---


def key_allocation(local_rank: int, owner_kind: str, seq: int) -> str:
    """Resource key prefix for a SHARED/PERSISTENT/INSTANT allocation.

    ``owner_kind`` must be the lower-case key segment: ``shared`` / ``persistent`` / ``instant``.
    """
    return f"{local_namespace(local_rank)}allocation:{owner_kind}:{seq:06d}"


# --- instant:<demand_id>: ---


def key_instant_prefix(local_rank: int, demand_id: str) -> str:
    """Redis key prefix for one INSTANT demand's per-use metadata."""
    return f"{local_namespace(local_rank)}instant:{demand_id}"


def key_instant_next_use_index(local_rank: int, demand_id: str) -> str:
    """Redis key storing the next expected use index of an INSTANT demand."""
    return redis_field(key_instant_prefix(local_rank, demand_id), "next_use_index")


def key_instant_total_use_count(local_rank: int, demand_id: str) -> str:
    """Redis key storing the total use count of an INSTANT demand."""
    return redis_field(key_instant_prefix(local_rank, demand_id), "total_use_count")


def key_instant_use_allocation_ref(local_rank: int, demand_id: str, use_index: int) -> str:
    """Redis key storing the allocation reference for one INSTANT use index."""
    return f"{key_instant_prefix(local_rank, demand_id)}:use:{use_index}:allocation_ref"


# --- request_queue / request:<request_id>: ---


def key_request_queue(local_rank: int) -> str:
    """Redis list key used as the manager's request queue."""
    return f"{local_namespace(local_rank)}request_queue"


def key_request_prefix(local_rank: int, request_id: str) -> str:
    """Redis key prefix for one assistant request."""
    return f"{local_namespace(local_rank)}request:{request_id}"


def key_request_demand_id(local_rank: int, request_id: str) -> str:
    """Redis key storing the demand id for one request."""
    return redis_field(key_request_prefix(local_rank, request_id), "demand_id")


def key_request_use_index(local_rank: int, request_id: str) -> str:
    """Redis key storing the INSTANT use index requested by one request."""
    return redis_field(key_request_prefix(local_rank, request_id), "use_index")


def key_request_target_ref(local_rank: int, request_id: str) -> str:
    """Redis key storing the resolved target resource reference of one request."""
    return redis_field(key_request_prefix(local_rank, request_id), "target_ref")


def make_request_id(rank: int, local_request_seq: int) -> str:
    """Build a request_id in the ``r<rank>_<local_request_seq>`` format."""
    return f"r{rank}_{local_request_seq}"


def parse_request_id_rank(request_id: str) -> int:
    """Extract the rank encoded in a request_id produced by :func:`make_request_id`."""
    body = request_id[1:]
    rank_str = body.split("_", 1)[0]
    return int(rank_str)


# --- pub/sub channel ---


def key_rank_events_channel(local_rank: int, rank: int) -> str:
    """Redis pub/sub channel used to wake one assistant rank."""
    return f"{local_namespace(local_rank)}rank:{rank}:events"


# ---------------------------------------------------------------------------
# Lua scripts for atomic acquire / claim-for-release semantics.
# ---------------------------------------------------------------------------

_ACQUIRE_IF_READY_LUA = """
local state_key = KEYS[1] .. ':state'
local version_key = KEYS[1] .. ':version'
local user_count_key = KEYS[1] .. ':user_count'
local expected_version = ARGV[1]

local state = redis.call('GET', state_key)
if state ~= 'READY' then
    return 'NOT_READY'
end

local current_version = redis.call('GET', version_key)
if expected_version ~= '' and expected_version ~= current_version then
    return 'VERSION_MISMATCH'
end

redis.call('INCRBY', user_count_key, 1)
return 'READY'
"""

_CLAIM_FOR_RELEASE_LUA = """
local state_key = KEYS[1] .. ':state'
local user_count_key = KEYS[1] .. ':user_count'
local transition_state = ARGV[1]

local state = redis.call('GET', state_key)
local user_count = tonumber(redis.call('GET', user_count_key))

if state == 'READY' and user_count == 0 then
    redis.call('SET', state_key, transition_state)
    return 'CLAIMED'
end
return 'SKIP'
"""


class MemoryManagerRedisClient:
    """Thin wrapper bundling Redis connections and Lua scripts for a given local_rank.

    Keeps a dedicated blocking connection for consuming the request queue (``BRPOP``)
    separate from the general command connection, so a blocking pop never stalls
    ordinary GET/SET/EVALSHA traffic.
    """

    def __init__(self, local_rank: int):
        """Create Redis clients and register the manager Lua scripts."""
        import redis

        self.local_rank = local_rank

        self.command = redis.Redis(
            host=MEMORY_MANAGER_REDIS_HOST,
            port=MEMORY_MANAGER_REDIS_PORT,
            db=MEMORY_MANAGER_REDIS_DB,
        )
        self.queue = redis.Redis(
            host=MEMORY_MANAGER_REDIS_HOST,
            port=MEMORY_MANAGER_REDIS_PORT,
            db=MEMORY_MANAGER_REDIS_DB,
        )

        self._acquire_if_ready_script = self.command.register_script(_ACQUIRE_IF_READY_LUA)
        self._claim_for_release_script = self.command.register_script(_CLAIM_FOR_RELEASE_LUA)

    def acquire_if_ready(self, target_ref: str, expected_version: int | None = None) -> str:
        """Atomically check READY state (and optionally version) then bump user_count.

        Returns one of ``ACQUIRE_READY`` / ``ACQUIRE_NOT_READY`` / ``ACQUIRE_VERSION_MISMATCH``.
        """
        version_arg = "" if expected_version is None else str(expected_version)
        result = self._acquire_if_ready_script(keys=[target_ref], args=[version_arg])
        return result.decode("ascii") if isinstance(result, bytes) else result

    def claim_for_release(self, target_ref: str, transition_state: str) -> str:
        """Atomically transition a READY, unused (``user_count == 0``) resource.

        Returns ``CLAIM_CLAIMED`` or ``CLAIM_SKIP``.
        """
        result = self._claim_for_release_script(keys=[target_ref], args=[transition_state])
        return result.decode("ascii") if isinstance(result, bytes) else result

    @staticmethod
    def new_pubsub():
        """Create a fresh Redis connection dedicated to pub/sub subscriptions."""
        import redis

        rc = redis.Redis(
            host=MEMORY_MANAGER_REDIS_HOST,
            port=MEMORY_MANAGER_REDIS_PORT,
            db=MEMORY_MANAGER_REDIS_DB,
        )
        return rc.pubsub()


# ---------------------------------------------------------------------------
# Stage B: tensor demand construction and Provider matching.
# ---------------------------------------------------------------------------

DEFAULT_GROUP_NAME = "0"
DEFAULT_GROUP_DESC = "default_pg"


def dtype_to_str(dtype) -> str:
    """Canonical string form of a torch.dtype used in demand_id / share_key encoding."""
    return str(dtype)


def load_tensor_providers(module_path: str | None) -> list[TensorProvider]:
    """Dynamically load the ``TENSOR_PROVIDERS`` list from a user-provided module.

    A provider module must expose a module-level ``TENSOR_PROVIDERS: list[TensorProvider]``
    whose order defines matching priority (first match wins).
    """
    if module_path is None:
        return []

    if not pathlib.Path(module_path).exists():
        logger.warning("Tensor provider module not found: %s", module_path)
        return []

    module_name = pathlib.Path(module_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    providers = getattr(module, "TENSOR_PROVIDERS", None)
    if providers is None:
        logger.warning("Module %s does not define TENSOR_PROVIDERS; no providers loaded", module_path)
        return []

    logger.info("Loaded %d tensor providers from %s", len(providers), module_path)
    return list(providers)


@dataclass
class TensorDemand:
    """A single tensor demand, deduplicated by demand_id within a MemoryManager."""

    demand_id: str
    rank: int
    c10d_func_name: str
    signature: str
    direction: str
    shape: list[int]
    dtype: str
    level: TensorMemoryLevel
    comm_group: ReplayerCommGroupInfo
    share_key: str | None
    # The TensorProvider that matched this demand (None for OUTPUT/ANYTHING demands
    # with no match); reused directly by Stage D/E instead of re-matching.
    provider: TensorProvider | None = None
    # Ordered list of the ReplayerCommOpInfo instances that produced this demand_id,
    # one entry per occurrence in trace order. Length == total use count for INSTANT.
    occurrences: list[ReplayerCommOpInfo] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage C: resource planning (pure Python) and the CUDA-independent Redis
# scaffold (demand / instant / init keys). Pool and allocation resource keys
# themselves (state, cuda_ipc_handle, ...) are only meaningful once a real
# CUDA allocation attempt has been made, so they are written in Stage D.
# ---------------------------------------------------------------------------

RESERVED_MEMORY_RATIO_DEFAULT = 0.1


def shape_numel(shape: list[int]) -> int:
    """Number of elements in a tensor shape (1 for the empty/scalar shape)."""
    return math.prod(shape)


def dtype_itemsize(dtype_str: str) -> int:
    """Bytes per element for a dtype string produced by :func:`dtype_to_str`."""
    return torch.empty((), dtype=dtype_from_str(dtype_str)).element_size()


def dtype_from_str(dtype_str: str) -> torch.dtype:
    """Reverse of :func:`dtype_to_str`: ``"torch.float16"`` -> ``torch.float16``."""
    name = dtype_str.split(".", 1)[1]
    return getattr(torch, name)


@dataclass
class PoolPlan:
    """Plan for a single ANYTHING shared pool (one per dtype)."""

    dtype_str: str
    max_elements: int
    nbytes: int
    demand_ids: list[str]


@dataclass
class AllocationPlan:
    """Plan for a single SHARED or PERSISTENT allocation."""

    owner_kind: str
    seq: int
    key: str
    num_elements: int
    nbytes: int
    dtype_str: str
    demand_ids: list[str]
    representative_demand_id: str


@dataclass
class InstantPlan:
    """Plan for a single INSTANT demand's use-index bookkeeping."""

    demand_id: str
    total_use_count: int


def _default_group_info(world_size: int) -> ReplayerCommGroupInfo:
    """Synthesize the implicit default process group ("0"), covering the full world."""
    return ReplayerCommGroupInfo(
        ranks=list(range(world_size)),
        backend=None,
        group_name=DEFAULT_GROUP_NAME,
        group_desc=DEFAULT_GROUP_DESC,
        timeout=None,
    )


def _resolve_peer_rank(comm_group: ReplayerCommGroupInfo, meta: dict, direct_key: str, group_key: str) -> int:
    """Resolve a peer's global rank from op meta, mirroring torch.distributed.get_global_rank."""
    if group_key in meta and direct_key not in meta:
        return comm_group.ranks[meta[group_key]]
    return meta[direct_key]


def export_cuda_ipc_handle(tensor: torch.Tensor) -> bytes:
    """Export a CUDA tensor's storage as a pickled IPC handle for Redis storage."""
    storage = tensor.untyped_storage()
    handle = storage._share_cuda_()
    return pickle.dumps(handle)


def fill_legal_random_(tensor: torch.Tensor) -> None:
    """Fill a tensor in-place with dtype-legal values (no NaN/Inf for float dtypes)."""
    if tensor.dtype.is_floating_point:
        tensor.normal_()
    else:
        tensor.random_()


class MemoryManager:
    """Unified per-GPU memory manager for the Replay Assistant.

    Mirrors AssistantProcess's construction style: it consumes already-parsed
    ReplayerCommGroupInfo / ReplayerReplayOpInfo lists (one list per managed
    rank) rather than parsing trace files itself. Trace parsing only happens
    in the ``if __name__ == "__main__":`` entry point.
    """

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        sandbox_ranks: list[int],
        rank_comm_groups: dict[int, list[ReplayerCommGroupInfo]],
        rank_replay_ops: dict[int, list[ReplayerReplayOpInfo]],
        tensor_providers: list[TensorProvider] | None = None,
        reserved_memory_ratio: float = RESERVED_MEMORY_RATIO_DEFAULT,
    ):
        """Initialize the MemoryManager.

        Args:
            local_rank: The local GPU index this manager owns.
            world_size: Total number of ranks in the emulated experiment.
            sandbox_ranks: Global ranks running in sandbox (not emulated by any assistant).
            rank_comm_groups: Per managed-rank ordered ReplayerCommGroupInfo list, as parsed
                from that rank's own trace file.
            rank_replay_ops: Per managed-rank ordered ReplayerReplayOpInfo list, as parsed
                from that rank's own trace file.
            tensor_providers: Ordered list of TensorProvider instances; first match wins.
            reserved_memory_ratio: Fraction of total GPU memory never used by the manager.
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.sandbox_ranks = list(sandbox_ranks)
        self.emulated_ranks = [r for r in range(world_size) if r not in self.sandbox_ranks]
        self.managed_ranks = sorted(rank_replay_ops.keys())

        self.rank_comm_groups = rank_comm_groups
        self.rank_replay_ops = rank_replay_ops
        self.tensor_providers = list(tensor_providers) if tensor_providers else []
        self.reserved_memory_ratio = reserved_memory_ratio

        self.redis = MemoryManagerRedisClient(local_rank)

        # Populated by build_demands().
        self.demands: dict[str, TensorDemand] = {}

        # Populated by plan_resources().
        self.pool_plans: dict[str, PoolPlan] = {}
        self.allocation_plans: list[AllocationPlan] = []
        self.instant_plans: dict[str, InstantPlan] = {}
        self.demand_target_ref: dict[str, str] = {}

    def _intra_assistant_only(self, peers: list[int]) -> bool:
        return all(peer in self.emulated_ranks for peer in peers)

    def _member_groups_for_rank(self, rank: int) -> dict[str, ReplayerCommGroupInfo]:
        """Groups this rank would actually join, mirroring AssistantProcess.comm_groups_init()."""
        known = {DEFAULT_GROUP_NAME: _default_group_info(self.world_size)}
        for comm_group in self.rank_comm_groups.get(rank, []):
            if rank in comm_group.ranks:
                known[comm_group.group_name] = comm_group
        return known

    def _match_provider(
        self, rank: int, replay_op: ReplayerCommOpInfo, comm_group: ReplayerCommGroupInfo
    ) -> tuple[TensorProvider | None, TensorMemoryLevel]:
        for provider in self.tensor_providers:
            if provider.can_handle(rank, replay_op, comm_group):
                return provider, provider.tensor_memory_level(rank, replay_op, comm_group)
        return None, TensorMemoryLevel.ANYTHING

    def _record_demand(
        self,
        rank: int,
        c10d_func_name: str,
        signature: str,
        direction: str,
        shape: list[int],
        dtype,
        comm_group: ReplayerCommGroupInfo,
        replay_op: ReplayerCommOpInfo,
        *,
        force_anything: bool,
    ) -> None:
        dtype_str = dtype_to_str(dtype)
        demand_id = encode_demand_id(rank, c10d_func_name, signature, direction, shape, dtype_str)

        existing = self.demands.get(demand_id)
        if existing is not None:
            existing.occurrences.append(replay_op)
            return

        if force_anything:
            provider, level = None, TensorMemoryLevel.ANYTHING
        else:
            provider, level = self._match_provider(rank, replay_op, comm_group)

        share_key = None
        if level == TensorMemoryLevel.SHARED:
            share_key = encode_share_key(c10d_func_name, signature, direction, shape, dtype_str)

        self.demands[demand_id] = TensorDemand(
            demand_id=demand_id,
            rank=rank,
            c10d_func_name=c10d_func_name,
            signature=signature,
            direction=direction,
            shape=shape,
            dtype=dtype_str,
            level=level,
            comm_group=comm_group,
            share_key=share_key,
            provider=provider,
            occurrences=[replay_op],
        )

    def _build_rank_demands(self, rank: int) -> None:
        member_groups = self._member_groups_for_rank(rank)
        pp_group_has_init: set[str] = set()
        skip_until_coalescing_end = False

        for op in self.rank_replay_ops.get(rank, []):
            if op.type == ReplayerReplayOpType.ITERATION_BEGIN:
                continue

            if op.type == ReplayerReplayOpType.COALESCING_END:
                skip_until_coalescing_end = False
                continue

            if op.type == ReplayerReplayOpType.COALESCING_START:
                skip_until_coalescing_end = op.info.group_name not in member_groups
                continue

            # op.type == COMM
            if skip_until_coalescing_end:
                continue

            info = op.info
            if info.group_name not in member_groups:
                continue
            comm_group = member_groups[info.group_name]
            func = info.c10d_func_name

            if func == "isend":
                dst = _resolve_peer_rank(comm_group, info.meta, "dst", "group_dst")
                if dst in self.emulated_ranks:
                    pp_key = f"{info.group_name}-{dst}"
                    if pp_key in pp_group_has_init:
                        continue
                    pp_group_has_init.add(pp_key)
                self._record_demand(
                    rank,
                    func,
                    info.signature,
                    MEMORY_TENSOR_INPUT,
                    info.input_shape,
                    info.input_dtype,
                    comm_group,
                    info,
                    force_anything=False,
                )
            elif func == "irecv":
                src = _resolve_peer_rank(comm_group, info.meta, "src", "group_src")
                if src in self.emulated_ranks:
                    pp_key = f"{info.group_name}-{src}"
                    if pp_key in pp_group_has_init:
                        continue
                    pp_group_has_init.add(pp_key)
                self._record_demand(
                    rank,
                    func,
                    info.signature,
                    MEMORY_TENSOR_OUTPUT,
                    info.output_shape,
                    info.output_dtype,
                    comm_group,
                    info,
                    force_anything=True,
                )
            elif func in {"broadcast", "all_reduce"}:
                if self._intra_assistant_only(comm_group.ranks):
                    continue
                self._record_demand(
                    rank,
                    func,
                    info.signature,
                    MEMORY_TENSOR_INPUT,
                    info.input_shape,
                    info.input_dtype,
                    comm_group,
                    info,
                    force_anything=False,
                )
            elif func in {
                "all_gather_into_tensor",
                "_all_gather_base",
                "reduce_scatter_tensor",
                "_reduce_scatter_base",
            }:
                if self._intra_assistant_only(comm_group.ranks):
                    continue
                self._record_demand(
                    rank,
                    func,
                    info.signature,
                    MEMORY_TENSOR_INPUT,
                    info.input_shape,
                    info.input_dtype,
                    comm_group,
                    info,
                    force_anything=False,
                )
                self._record_demand(
                    rank,
                    func,
                    info.signature,
                    MEMORY_TENSOR_OUTPUT,
                    info.output_shape,
                    info.output_dtype,
                    comm_group,
                    info,
                    force_anything=True,
                )
            elif func == "all_to_all_single":
                if self._intra_assistant_only(comm_group.ranks):
                    continue
                raise ValueError("all-to-all should not cross the boundary of sandbox")
            elif func == "barrier":
                continue
            else:
                raise ValueError(f"Unsupported op: {op}")

    def build_demands(self) -> None:
        """Scan every managed rank's replay ops and populate ``self.demands``.

        Must be called before resource planning (Stage C). Idempotent only if
        called once; calling twice would duplicate occurrence counts.
        """
        self.demands = {}
        for rank in self.managed_ranks:
            self._build_rank_demands(rank)
        logger.info("Built %d tensor demands across %d managed ranks", len(self.demands), len(self.managed_ranks))

    def plan_resources(self) -> None:
        """Plan pools/allocations/instant bookkeeping from ``self.demands`` (pure Python).

        Must be called after :meth:`build_demands`. Populates ``pool_plans``,
        ``allocation_plans`` (SHARED then PERSISTENT, in a deterministic but not
        yet size-ordered sequence: see :meth:`allocation_plans_by_size_asc`),
        ``instant_plans``, and ``demand_target_ref``.
        """
        self.pool_plans = {}
        self.allocation_plans = []
        self.instant_plans = {}
        self.demand_target_ref = {}

        self._plan_anything_pools()
        seq = self._plan_shared_allocations(start_seq=1)
        self._plan_persistent_allocations(start_seq=seq)
        self._plan_instant_demands()

        self._allocation_plan_by_key = {p.key: p for p in self.allocation_plans}

        logger.info(
            "Planned %d pools, %d shared/persistent allocations, %d instant demands",
            len(self.pool_plans),
            len(self.allocation_plans),
            len(self.instant_plans),
        )

    def _plan_anything_pools(self) -> None:
        demands_by_dtype: dict[str, list[TensorDemand]] = defaultdict(list)
        for demand in self.demands.values():
            if demand.level == TensorMemoryLevel.ANYTHING:
                demands_by_dtype[demand.dtype].append(demand)

        for dtype_str in sorted(demands_by_dtype):
            this_dtype_demands = demands_by_dtype[dtype_str]
            max_elements = max(shape_numel(d.shape) for d in this_dtype_demands)
            itemsize = dtype_itemsize(dtype_str)
            pool_key = key_pool_anything(self.local_rank, dtype_str)
            self.pool_plans[dtype_str] = PoolPlan(
                dtype_str=dtype_str,
                max_elements=max_elements,
                nbytes=max_elements * itemsize,
                demand_ids=sorted(d.demand_id for d in this_dtype_demands),
            )
            for demand in this_dtype_demands:
                self.demand_target_ref[demand.demand_id] = pool_key

    def _plan_shared_allocations(self, start_seq: int) -> int:
        demands_by_share_key: dict[str, list[TensorDemand]] = defaultdict(list)
        for demand in self.demands.values():
            if demand.level == TensorMemoryLevel.SHARED:
                demands_by_share_key[demand.share_key].append(demand)

        seq = start_seq
        for share_key in sorted(demands_by_share_key):
            this_share_key_demands = sorted(demands_by_share_key[share_key], key=lambda d: d.rank)
            representative = this_share_key_demands[0]
            num_elements = shape_numel(representative.shape)
            itemsize = dtype_itemsize(representative.dtype)
            alloc_key = key_allocation(self.local_rank, "shared", seq)

            self.allocation_plans.append(
                AllocationPlan(
                    owner_kind=OWNER_KIND_SHARED,
                    seq=seq,
                    key=alloc_key,
                    num_elements=num_elements,
                    nbytes=num_elements * itemsize,
                    dtype_str=representative.dtype,
                    demand_ids=[d.demand_id for d in this_share_key_demands],
                    representative_demand_id=representative.demand_id,
                )
            )
            for demand in this_share_key_demands:
                self.demand_target_ref[demand.demand_id] = alloc_key
            seq += 1
        return seq

    def _plan_persistent_allocations(self, start_seq: int) -> int:
        seq = start_seq
        persistent_demands = sorted(
            (d for d in self.demands.values() if d.level == TensorMemoryLevel.PERSISTENT),
            key=lambda d: d.demand_id,
        )
        for demand in persistent_demands:
            num_elements = shape_numel(demand.shape)
            itemsize = dtype_itemsize(demand.dtype)
            alloc_key = key_allocation(self.local_rank, "persistent", seq)

            self.allocation_plans.append(
                AllocationPlan(
                    owner_kind=OWNER_KIND_PERSISTENT,
                    seq=seq,
                    key=alloc_key,
                    num_elements=num_elements,
                    nbytes=num_elements * itemsize,
                    dtype_str=demand.dtype,
                    demand_ids=[demand.demand_id],
                    representative_demand_id=demand.demand_id,
                )
            )
            self.demand_target_ref[demand.demand_id] = alloc_key
            seq += 1
        return seq

    def _plan_instant_demands(self) -> None:
        for demand in self.demands.values():
            if demand.level == TensorMemoryLevel.INSTANT:
                self.instant_plans[demand.demand_id] = InstantPlan(
                    demand_id=demand.demand_id,
                    total_use_count=len(demand.occurrences),
                )
                self.demand_target_ref[demand.demand_id] = key_instant_prefix(self.local_rank, demand.demand_id)

    # -----------------------------------------------------------------
    # Stage D: real CUDA allocation for ANYTHING pools and SHARED/PERSISTENT
    # allocations, with CPU fallback when the memory budget is exceeded.
    # -----------------------------------------------------------------

    def compute_allocatable_bytes(self, total_bytes: int) -> int:
        """Budget usable for manager-owned pools/allocations, after reserving headroom."""
        return int(total_bytes * (1 - self.reserved_memory_ratio))

    def allocation_plans_by_size_asc(self) -> list[AllocationPlan]:
        """Return SHARED+PERSISTENT allocation plans ordered smallest-to-largest.

        This is the required order for the actual pre-allocation attempt order (Stage D).
        """
        return sorted(self.allocation_plans, key=lambda p: (p.nbytes, p.key))

    def allocate(self, total_bytes: int | None = None) -> None:
        """Perform real CUDA allocation for ANYTHING pools then SHARED/PERSISTENT.

        Must be called after :meth:`plan_resources`. Requires a CUDA device.

        Args:
            total_bytes: Total GPU memory to budget against. Defaults to the real
                device's total memory; tests may inject a smaller value to
                deterministically exercise the CPU-fallback path.
        """
        torch.cuda.set_device(self.local_rank)
        if total_bytes is None:
            _, total_bytes = torch.cuda.mem_get_info(device=self.local_rank)

        self.allocatable_bytes = self.compute_allocatable_bytes(total_bytes)
        self.used_bytes = 0

        self._pool_registry: dict[str, torch.Tensor] = {}
        self._allocation_registry: dict[str, torch.Tensor] = {}
        self._cpu_backup_registry: dict[str, torch.Tensor] = {}
        self._live_instant_allocations: set[str] = set()
        self._next_instant_seq = 1

        self._allocate_anything_pools()
        self._allocate_shared_persistent()

    def _allocate_anything_pools(self) -> None:
        device = f"cuda:{self.local_rank}"
        for dtype_str, plan in self.pool_plans.items():
            if self.used_bytes + plan.nbytes > self.allocatable_bytes:
                raise RuntimeError(
                    f"Not enough memory budget for ANYTHING pool dtype={dtype_str}: "
                    f"need {plan.nbytes} bytes, only {self.allocatable_bytes - self.used_bytes} left"
                )

            dtype = dtype_from_str(dtype_str)
            tensor = torch.empty(plan.max_elements, dtype=dtype, device=device)
            fill_legal_random_(tensor)
            handle_bytes = export_cuda_ipc_handle(tensor)

            self._pool_registry[dtype_str] = tensor
            self.used_bytes += plan.nbytes

            pool_key = key_pool_anything(self.local_rank, dtype_str)
            pipe = self.redis.command.pipeline()
            pipe.set(redis_field(pool_key, "user_count"), 0)
            pipe.set(redis_field(pool_key, "version"), 0)
            pipe.set(redis_field(pool_key, "max_elements"), plan.max_elements)
            pipe.set(redis_field(pool_key, "nbytes"), plan.nbytes)
            pipe.set(redis_field(pool_key, "cuda_ipc_handle"), handle_bytes)
            pipe.set(redis_field(pool_key, "state"), STATE_READY)
            pipe.execute()
            logger.info(
                "Allocated ANYTHING pool dtype=%s elements=%d nbytes=%d", dtype_str, plan.max_elements, plan.nbytes
            )

    @staticmethod
    def _provide_for_demand(demand: TensorDemand, use_index: int) -> torch.Tensor:
        if demand.provider is None:
            raise RuntimeError(f"Demand {demand.demand_id} has level {demand.level} but no matching TensorProvider")
        occurrence_index = max(use_index, 0)
        replay_op = demand.occurrences[occurrence_index]
        return demand.provider.provide(demand.rank, replay_op, demand.comm_group, use_index)

    def _allocate_shared_persistent(self) -> None:
        device = f"cuda:{self.local_rank}"
        for plan in self.allocation_plans_by_size_asc():
            representative = self.demands[plan.representative_demand_id]
            cpu_tensor = self._provide_for_demand(representative, use_index=-1)

            pipe = self.redis.command.pipeline()
            pipe.set(redis_field(plan.key, "owner_kind"), plan.owner_kind)
            pipe.set(redis_field(plan.key, "num_elements"), plan.num_elements)
            pipe.set(redis_field(plan.key, "nbytes"), plan.nbytes)
            pipe.set(redis_field(plan.key, "version"), 0)

            if self.used_bytes + plan.nbytes <= self.allocatable_bytes:
                dtype = dtype_from_str(plan.dtype_str)
                tensor = torch.empty(plan.num_elements, dtype=dtype, device=device)
                tensor.copy_(cpu_tensor.reshape(-1))
                handle_bytes = export_cuda_ipc_handle(tensor)

                self._allocation_registry[plan.key] = tensor
                self.used_bytes += plan.nbytes

                pipe.set(redis_field(plan.key, "user_count"), 0)
                pipe.set(redis_field(plan.key, "cuda_ipc_handle"), handle_bytes)
                pipe.set(redis_field(plan.key, "state"), STATE_READY)
                logger.info("Allocated %s allocation %s (%d bytes) to GPU", plan.owner_kind, plan.key, plan.nbytes)
            else:
                self._cpu_backup_registry[plan.key] = cpu_tensor
                pipe.set(redis_field(plan.key, "user_count"), 0)
                pipe.set(redis_field(plan.key, "cuda_ipc_handle"), PLACEHOLDER)
                pipe.set(redis_field(plan.key, "state"), STATE_EVICTED)
                logger.info(
                    "Budget exceeded, keeping %s allocation %s (%d bytes) on CPU",
                    plan.owner_kind,
                    plan.key,
                    plan.nbytes,
                )

            pipe.execute()

    def write_scaffold_to_redis(self) -> None:
        """Write the CUDA-independent Redis scaffold: demand/instant/init keys.

        Must be called after :meth:`plan_resources`. Pool/allocation resource
        keys (state, cuda_ipc_handle, ...) are written separately once a real
        CUDA allocation attempt has been made (Stage D).
        """
        pipe = self.redis.command.pipeline()

        managed_ranks_key = key_init_managed_ranks(self.local_rank)
        pipe.delete(managed_ranks_key)
        if self.managed_ranks:
            pipe.rpush(managed_ranks_key, *self.managed_ranks)
        pipe.set(key_init_reserved_memory_ratio(self.local_rank), str(self.reserved_memory_ratio))

        for demand_id, demand in self.demands.items():
            pipe.set(key_demand_level(self.local_rank, demand_id), demand.level.name)
            pipe.set(key_demand_target_ref(self.local_rank, demand_id), self.demand_target_ref[demand_id])

        for demand_id, instant_plan in self.instant_plans.items():
            pipe.set(key_instant_next_use_index(self.local_rank, demand_id), 0)
            pipe.set(key_instant_total_use_count(self.local_rank, demand_id), instant_plan.total_use_count)
            for use_index in range(instant_plan.total_use_count):
                pipe.set(key_instant_use_allocation_ref(self.local_rank, demand_id, use_index), PLACEHOLDER)

        pipe.execute()
        logger.info("Wrote CUDA-independent Redis scaffold for local_rank=%d", self.local_rank)

    def mark_init_done(self) -> None:
        """Signal to waiting assistants that initialization and pre-allocation are complete."""
        self.redis.command.set(key_init_done(self.local_rank), "1")

    # -----------------------------------------------------------------
    # Stage E: runtime request-queue consumption and eviction cascade.
    # -----------------------------------------------------------------

    def _try_destroy_instant(self, key: str) -> bool:
        if self.redis.claim_for_release(key, STATE_DESTROYING) != CLAIM_CLAIMED:
            return False

        nbytes = int(self.redis.command.get(redis_field(key, "nbytes")))
        self._allocation_registry.pop(key, None)
        self._live_instant_allocations.discard(key)
        self.used_bytes -= nbytes

        pipe = self.redis.command.pipeline()
        pipe.set(redis_field(key, "cuda_ipc_handle"), PLACEHOLDER)
        pipe.set(redis_field(key, "state"), STATE_DESTROYED)
        pipe.execute()
        logger.info("Destroyed idle INSTANT allocation %s, freed %d bytes", key, nbytes)
        return True

    def _try_evict_allocation(self, key: str, nbytes: int) -> bool:
        if self.redis.claim_for_release(key, STATE_EVICTING) != CLAIM_CLAIMED:
            return False

        tensor = self._allocation_registry.pop(key)
        self._cpu_backup_registry[key] = tensor.detach().cpu().clone()
        del tensor
        self.used_bytes -= nbytes

        pipe = self.redis.command.pipeline()
        pipe.set(redis_field(key, "cuda_ipc_handle"), PLACEHOLDER)
        pipe.set(redis_field(key, "state"), STATE_EVICTED)
        pipe.execute()
        logger.info("Evicted allocation %s to CPU, freed %d bytes", key, nbytes)
        return True

    def _ensure_budget(self, nbytes_needed: int) -> None:
        """Free GPU memory until ``nbytes_needed`` fits.

        Eviction order is INSTANT destroy -> PERSISTENT evict -> SHARED evict.
        Raises RuntimeError if enough memory is still unreachable.
        """
        if self.used_bytes + nbytes_needed <= self.allocatable_bytes:
            return

        for key in list(self._live_instant_allocations):
            if self.used_bytes + nbytes_needed <= self.allocatable_bytes:
                return
            self._try_destroy_instant(key)

        for owner_kind in (OWNER_KIND_PERSISTENT, OWNER_KIND_SHARED):
            candidates = sorted(
                (p for p in self.allocation_plans if p.owner_kind == owner_kind),
                key=lambda p: -p.nbytes,
            )
            for plan in candidates:
                if self.used_bytes + nbytes_needed <= self.allocatable_bytes:
                    return
                self._try_evict_allocation(plan.key, plan.nbytes)

        if self.used_bytes + nbytes_needed > self.allocatable_bytes:
            raise RuntimeError(
                f"Cannot free enough GPU memory: need {nbytes_needed} bytes, "
                f"only {self.allocatable_bytes - self.used_bytes} available after evicting everything evictable"
            )

    def _process_shared_persistent_request(self, target_ref: str) -> None:
        state = self.redis.command.get(redis_field(target_ref, "state")).decode()
        if state == STATE_READY:
            self.redis.command.incrby(redis_field(target_ref, "user_count"), 1)
            return
        if state != STATE_EVICTED:
            raise RuntimeError(f"Unexpected state {state!r} for {target_ref} while processing a request")

        plan = self._allocation_plan_by_key[target_ref]
        self._ensure_budget(plan.nbytes)

        cpu_tensor = self._cpu_backup_registry.pop(target_ref)
        dtype = dtype_from_str(plan.dtype_str)
        tensor = torch.empty(plan.num_elements, dtype=dtype, device=f"cuda:{self.local_rank}")
        tensor.copy_(cpu_tensor.reshape(-1))
        handle_bytes = export_cuda_ipc_handle(tensor)

        self._allocation_registry[target_ref] = tensor
        self.used_bytes += plan.nbytes
        new_version = int(self.redis.command.get(redis_field(target_ref, "version"))) + 1

        pipe = self.redis.command.pipeline()
        pipe.set(redis_field(target_ref, "cuda_ipc_handle"), handle_bytes)
        pipe.set(redis_field(target_ref, "version"), new_version)
        pipe.set(redis_field(target_ref, "user_count"), 1)
        pipe.set(redis_field(target_ref, "state"), STATE_READY)
        pipe.execute()
        logger.info("Reloaded %s allocation %s from CPU backup (version=%d)", plan.owner_kind, target_ref, new_version)

    def _process_instant_request(self, demand_id: str, use_index: int) -> str:
        next_use_index_key = key_instant_next_use_index(self.local_rank, demand_id)
        next_use_index = int(self.redis.command.get(next_use_index_key))
        if use_index != next_use_index:
            raise RuntimeError(
                f"INSTANT use_index mismatch for demand {demand_id}: request={use_index}, expected={next_use_index}"
            )

        demand = self.demands[demand_id]
        num_elements = shape_numel(demand.shape)
        nbytes = num_elements * dtype_itemsize(demand.dtype)
        self._ensure_budget(nbytes)

        cpu_tensor = self._provide_for_demand(demand, use_index)
        dtype = dtype_from_str(demand.dtype)
        tensor = torch.empty(num_elements, dtype=dtype, device=f"cuda:{self.local_rank}")
        tensor.copy_(cpu_tensor.reshape(-1))
        handle_bytes = export_cuda_ipc_handle(tensor)

        alloc_key = key_allocation(self.local_rank, "instant", self._next_instant_seq)
        self._next_instant_seq += 1

        self._allocation_registry[alloc_key] = tensor
        self._live_instant_allocations.add(alloc_key)
        self.used_bytes += nbytes

        pipe = self.redis.command.pipeline()
        pipe.set(redis_field(alloc_key, "owner_kind"), OWNER_KIND_INSTANT)
        pipe.set(redis_field(alloc_key, "num_elements"), num_elements)
        pipe.set(redis_field(alloc_key, "nbytes"), nbytes)
        pipe.set(redis_field(alloc_key, "version"), 0)
        pipe.set(redis_field(alloc_key, "user_count"), 1)
        pipe.set(redis_field(alloc_key, "cuda_ipc_handle"), handle_bytes)
        pipe.set(redis_field(alloc_key, "state"), STATE_READY)
        pipe.set(key_instant_use_allocation_ref(self.local_rank, demand_id, use_index), alloc_key)
        pipe.incrby(next_use_index_key, 1)
        pipe.execute()
        logger.info("Created INSTANT allocation %s for demand %s use_index=%d", alloc_key, demand_id, use_index)
        return alloc_key

    def process_request(self, request_id: str) -> None:
        """Process a single tensor-use request dequeued from the request queue.

        Mirrors the manager-side steps of the Tensor Acquire protocol: resolve
        the demand's level/target_ref, satisfy the request (fast path, reload
        from CPU backup, or create a new INSTANT allocation), then publish a
        wake-up event on the requesting rank's channel.
        """
        demand_id = self.redis.command.get(key_request_demand_id(self.local_rank, request_id)).decode()
        level = TensorMemoryLevel[self.redis.command.get(key_demand_level(self.local_rank, demand_id)).decode()]
        target_ref = self.redis.command.get(key_demand_target_ref(self.local_rank, demand_id)).decode()

        if level in {TensorMemoryLevel.SHARED, TensorMemoryLevel.PERSISTENT}:
            self._process_shared_persistent_request(target_ref)
            resolved_ref = target_ref
        elif level == TensorMemoryLevel.INSTANT:
            use_index = int(self.redis.command.get(key_request_use_index(self.local_rank, request_id)))
            resolved_ref = self._process_instant_request(demand_id, use_index)
        else:
            raise RuntimeError(f"Demand {demand_id} has level {level}, which should never reach the request queue")

        self.redis.command.set(key_request_target_ref(self.local_rank, request_id), resolved_ref)

        rank = parse_request_id_rank(request_id)
        self.redis.command.publish(key_rank_events_channel(self.local_rank, rank), request_id)

    def run_request_loop(self, poll_timeout: float = 1.0) -> None:
        """Continuously consume the request queue until :meth:`stop_request_loop` is called."""
        self._stop_request_loop = False
        queue_key = key_request_queue(self.local_rank)
        while not self._stop_request_loop:
            item = self.redis.queue.brpop(queue_key, timeout=poll_timeout)
            if item is None:
                continue
            _, request_id_bytes = item
            self.process_request(request_id_bytes.decode())

    def stop_request_loop(self) -> None:
        """Signal :meth:`run_request_loop` to exit after its current poll timeout."""
        self._stop_request_loop = True


def submit_request(
    redis_client: MemoryManagerRedisClient,
    local_rank: int,
    rank: int,
    request_seq: int,
    demand_id: str,
    use_index: int = -1,
) -> str:
    """Assistant-side helper: enqueue a tensor-use request for the manager to process.

    Not used by MemoryManager itself (the manager only consumes the queue); this
    is provided for the Stage F test driver and for a future assistant.py
    integration to reuse the exact wire protocol.
    """
    request_id = make_request_id(rank, request_seq)
    pipe = redis_client.command.pipeline()
    pipe.set(key_request_demand_id(local_rank, request_id), demand_id)
    pipe.set(key_request_use_index(local_rank, request_id), use_index)
    pipe.set(key_request_target_ref(local_rank, request_id), PLACEHOLDER)
    pipe.lpush(key_request_queue(local_rank), request_id)
    pipe.execute()
    return request_id


def launch_memory_manager(
    local_rank: int,
    world_size: int,
    sandbox_ranks: list[int],
    managed_ranks: list[int],
    trace_dir: str,
    tensor_provider: str | None,
    reserved_memory_ratio: float = RESERVED_MEMORY_RATIO_DEFAULT,
) -> MemoryManager:
    """Parse each managed rank's trace file and run the manager through full init.

    Blocks forever (via :meth:`MemoryManager.run_request_loop`) once initialization
    completes; the manager's lifecycle is expected to be terminated by its runner.
    """
    from .parser import CrystalTraceParser

    rank_comm_groups: dict[int, list[ReplayerCommGroupInfo]] = {}
    rank_replay_ops: dict[int, list[ReplayerReplayOpInfo]] = {}
    for rank in managed_ranks:
        trace_parser = CrystalTraceParser(f"{trace_dir}/trace_{rank}.log.gz")
        rank_comm_groups[rank] = trace_parser.comm_groups_ordered_list
        rank_replay_ops[rank] = trace_parser.replay_ops_ordered_list

    tensor_providers = load_tensor_providers(tensor_provider)

    manager = MemoryManager(
        local_rank=local_rank,
        world_size=world_size,
        sandbox_ranks=sandbox_ranks,
        rank_comm_groups=rank_comm_groups,
        rank_replay_ops=rank_replay_ops,
        tensor_providers=tensor_providers,
        reserved_memory_ratio=reserved_memory_ratio,
    )
    manager.build_demands()
    manager.plan_resources()
    manager.allocate()
    manager.write_scaffold_to_redis()
    manager.mark_init_done()

    manager.run_request_loop()
    return manager


if __name__ == "__main__":
    import argparse

    LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))

    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--sandbox-ranks", required=True, help="Sandbox ranks", type=json.loads)
    cli_parser.add_argument(
        "--managed-ranks", required=True, help="Global ranks managed by this GPU's manager", type=json.loads
    )
    cli_parser.add_argument("--trace-dir", required=True, help="Crystal C10D Trace directory", type=str)
    cli_parser.add_argument(
        "--tensor-provider",
        help="Path to Python module containing TensorProvider implementations (TENSOR_PROVIDERS list)",
        type=str,
        default=None,
    )
    cli_parser.add_argument(
        "--reserved-memory-ratio",
        help="Fraction of GPU memory to never use",
        type=float,
        default=RESERVED_MEMORY_RATIO_DEFAULT,
    )
    cli_args = cli_parser.parse_args()

    launch_memory_manager(
        local_rank=LOCAL_RANK,
        world_size=WORLD_SIZE,
        sandbox_ranks=cli_args.sandbox_ranks,
        managed_ranks=cli_args.managed_ranks,
        trace_dir=cli_args.trace_dir,
        tensor_provider=cli_args.tensor_provider,
        reserved_memory_ratio=cli_args.reserved_memory_ratio,
    )
