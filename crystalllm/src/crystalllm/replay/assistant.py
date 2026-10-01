# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Entry Point for Crystal Assistant Process."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import pickle
import threading
import time
from queue import Empty, Queue

import redis
import torch
from torch.distributed import _coalescing_manager  # noqa: PLC2701

from ..planning import (
    CrystalGDurationNotIncreasedError,
    CrystalGSpanId,
    CrystalGSpanPoint,
    SpanPointType,
)
from ..runtime.rendezvous import replay_rendezvous
from ..util import is_truthy_value
from .data import (
    ReplayerCommGroupInfo,
    ReplayerReplayOpInfo,
    ReplayerReplayOpType,
    TensorMemoryLevel,
)
from .graph_bundle import COMM_TYPE_FIELD, CrystalGraphBundle
from .memory_manager import (
    ACQUIRE_NOT_READY,
    ACQUIRE_READY,
    ACQUIRE_VERSION_MISMATCH,
    MEMORY_TENSOR_INPUT,
    MEMORY_TENSOR_OUTPUT,
    MemoryManagerRedisClient,
    dtype_to_str,
    encode_demand_id,
    key_demand_level,
    key_demand_target_ref,
    key_init_done,
    key_instant_next_use_index,
    key_instant_use_allocation_ref,
    key_rank_events_channel,
    make_request_id,
    redis_field,
    shape_numel,
    submit_request,
)
from .parser import CrystalTraceParser
from .util.graph_cache import cached_op_timings, load_graph_cache, validate_graph_cache

# Configure logging with default StreamHandler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Launch topology injected by the runner through environment variables. RANK/WORLD_SIZE
# address the emulated world, while LOCAL_RANK selects this process's CUDA device.
WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
RANK = int(os.getenv("RANK"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK"))

# Concurrency cap for asynchronous replay communications. It sizes the borrowable
# side-stream pool (see _init_timing_workers) and bounds the number of in-flight
# async comms: once all streams are borrowed, _borrow_async_stream_id() blocks.
MAX_PARALLEL_COMM = 8

# Time conversion constants
MS_TO_NS = 1_000_000
S_TO_NS = 1_000_000_000


@dataclasses.dataclass
class TimingContext:
    """CUDA timing context attached to a planned replay operation.

    Created only with online updates on: the completion worker reads
    elapsed_time() after synchronizing the end event and broadcasts the measured
    duration to every other assistant rank through Redis.

    Attributes:
        trace_comm_index: Identity of the timed op; used as the broadcast key so
            remote ranks can apply the update to the right span.
        start_event: Records the op launch instant on the timing stream.
        end_event: Records the completion instant, after work.wait() on the same
            stream, so elapsed_time() covers launch-to-completion of the comm.
        stream_id: Id of the borrowed side stream when the op is async (returned
            to the pool by the worker); None when timed on the current stream.
    """

    trace_comm_index: int
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    stream_id: int | None = None


@dataclasses.dataclass
class CompletionTask:
    """Worker task describing what to do once a replay op completes on the GPU.

    Produced by _replay_tracked_op() and consumed by the timing workers: the
    completion event marks the op's completion point on the stream where it was
    waited on, so resource release, duration broadcast, and stream return all
    happen strictly after the GPU has finished the communication.

    Attributes:
        resource_refs: MemoryManager resource refs acquired for the op; the worker
            decrements each one's user_count to release it.
        completion_event: Event recorded at the op's completion point. None when a
            timing_context is attached, in which case its end_event plays this role.
        timing_context: Timing context for online-update ops; when present the worker
            also broadcasts the measured duration.
        stream_id: Async stream borrowed from _async_timing_streams for this op; the
            worker returns it to the pool after the completion event is synchronized.
    """

    resource_refs: list[str]
    completion_event: torch.cuda.Event | None
    timing_context: TimingContext | None
    stream_id: int | None = None


class AssistantProcess:
    """Assistant process for replaying distributed training traces.

    Emulates inter-rank collectives and computation timing by replaying recorded traces.
    """

    def __init__(
        self,
        sandbox_ranks: list[int],
    ):
        """Initialize the AssistantProcess replayer.

        Args:
            sandbox_ranks: List of ranks running in sandbox (not emulated by assistant).
        """
        self.sandbox_ranks = sandbox_ranks
        self.emulated_ranks = [rank for rank in range(WORLD_SIZE) if rank not in sandbox_ranks]

        # group_name -> torch ProcessGroup handle; the entry "0" maps to the default
        # WORLD process group created by init_process_group() and stores None, because
        # collectives select it by passing group=None. A group this rank is not a member
        # of is never inserted (see comm_groups_init).
        self.comm_groups: dict[str, torch.distributed.ProcessGroup | None] = {}
        # group_name -> member ranks of every traced group (membership of this rank is
        # irrelevant); used by _intra_assistant_only() and as the new_group_table of
        # CrystalGraphBundle.load() to resolve graph channels.
        self.comm_groups_ranks: dict[str, list[int]] = {}

        self.op_queue: list[ReplayerReplayOpInfo] = []
        # demand_id -> target_ref
        # a demand's target_ref never changes after the MemoryManager finished planning,
        # so it is safe to memoize locally.
        self.demand_target_refs: dict[str, str] = {}
        # demand_id -> (tensor, version)
        self.tensor_cache: dict[str, tuple[torch.Tensor, int]] = {}

        # Launch-timing source of planned replay. Either the merged per-step graphs of
        # replay_with_graph() or the trace_comm_index -> launch_time_ns records of a
        # prebuilt cache armed by replay_with_graph_cache(); the two never coexist, and
        # _op_launch_times is not None exactly in cache mode.
        self._bundle: CrystalGraphBundle | None = None
        self._op_launch_times: dict[int, int] | None = None
        self._enable_online_update = False

        # (group_name, peer) P2P pairs whose first isend/irecv has already been executed
        # once to establish the NCCL peer connection (see _execute_op).
        self.pp_group_has_init = set()

        # MemoryManager client and the wake-up machinery used while waiting for the
        # manager to prepare a resource: requests are announced on a Redis list and the
        # completion notification arrives on a per-rank pub/sub channel, which a daemon
        # thread turns into local threading.Event objects.
        self.memory_redis: MemoryManagerRedisClient | None = None
        self._memory_pubsub = None
        self._memory_listener_thread: threading.Thread | None = None
        self._memory_request_events: dict[str, threading.Event] = {}
        self._memory_request_lock = threading.Lock()
        self._memory_request_seq = 0

    def _intra_assistant_only(self, peers: list[int]) -> bool:
        """Check if all peers are within emulated ranks (intra-assistant only).

        A collective whose every member is emulated never crosses the sandbox
        boundary, so replaying it would only generate useless traffic between
        assistant processes: _execute_op() skips it entirely. Collectives with at
        least one sandbox member are really issued; the redundant assistant-side
        traffic is eliminated at the NCCL layer (mock neighbour / active proxy).

        Args:
            peers: List of rank IDs to check (the members of one process group).

        Returns:
            True if all peers are in emulated_ranks, False otherwise.
        """
        return all(peer in self.emulated_ranks for peer in peers)

    @staticmethod
    def default_group_init():
        """Initialize the default (WORLD) NCCL process group.

        Uses the exact same env-based rendezvous as the training program:
        MASTER_ADDR/MASTER_PORT (under mock NCCL bootstrap these point at the
        bootstrap planner's store) are read by init_process_group itself. The CUDA
        device is pinned to LOCAL_RANK so every subsequent collective and CUDA IPC
        handle uses the right GPU.
        """
        torch.cuda.set_device(LOCAL_RANK)
        # Use the same init_process_group as the training program.
        init_process_group_kwargs = {
            "backend": "nccl",
            "world_size": WORLD_SIZE,
            "rank": RANK,
            "timeout": datetime.timedelta(minutes=180),
        }

        print(f"init_process_group_kwargs: {init_process_group_kwargs}", flush=True)
        torch.distributed.init_process_group(**init_process_group_kwargs)
        print("Complete init_process_group", flush=True)

    def comm_groups_init(self, comm_groups: list[ReplayerCommGroupInfo]):
        """Create the trace's process groups, in trace order.

        The default WORLD group is registered under the reserved name "0" with a
        None handle (collectives select it via group=None); every traced new_group()
        call is replayed through torch.distributed.new_group with its recorded
        ranks/backend/timeout/group_desc. Group creation is collective across the
        WORLD, so every rank calls new_group for every traced group and only the
        membership decides whether a usable handle comes back.

        Args:
            comm_groups: Traced group configurations, in the order the training
                run created them.
        """
        self.comm_groups["0"] = None
        self.comm_groups_ranks["0"] = list(range(WORLD_SIZE))

        for comm_group in comm_groups:
            timeout = datetime.timedelta(seconds=comm_group.timeout) if comm_group.timeout is not None else None

            new_group_kwargs = {
                "ranks": comm_group.ranks,
                "timeout": timeout,
                "backend": comm_group.backend,
                "group_desc": comm_group.group_desc,
            }
            pg = torch.distributed.new_group(**new_group_kwargs)
            # The rank table must record every traced group, not only the ones this rank
            # belongs to: it feeds CrystalGraphBundle.load(), which resolves the channels
            # of every merged rank's graph, including groups this rank is not a member of.
            self.comm_groups_ranks[comm_group.group_name] = comm_group.ranks
            # Skip PGs this rank is not a member of; NON_GROUP_MEMBER is an int sentinel that would
            # crash downstream collective code.
            if pg is torch.distributed.GroupMember.NON_GROUP_MEMBER:
                continue
            self.comm_groups[comm_group.group_name] = pg

            logger.info("Complete new_group with kwargs %s", new_group_kwargs)

    def setup_memory_manager_client(self) -> None:
        """Connect to the local MemoryManager Redis namespace and subscribe to wake-up events.

        Blocks until the MemoryManager of this node (identified by LOCAL_RANK) has
        published its init-done key, then subscribes to this rank's per-(node, rank)
        events channel and starts the daemon thread that turns pub/sub completion
        notifications into local threading.Event objects (see _memory_event_listener).
        """
        self.memory_redis = MemoryManagerRedisClient(LOCAL_RANK)
        logger.info("Waiting for MemoryManager init key %s", key_init_done(LOCAL_RANK))
        while not self.memory_redis.command.get(key_init_done(LOCAL_RANK)):
            time.sleep(0.5)
        logger.info("MemoryManager init completed for local_rank=%d", LOCAL_RANK)

        self._memory_pubsub = self.memory_redis.new_pubsub()
        self._memory_pubsub.subscribe(key_rank_events_channel(LOCAL_RANK, RANK))
        self._memory_listener_thread = threading.Thread(
            target=self._memory_event_listener,
            daemon=True,
            name=f"memory_event_listener_rank_{RANK}",
        )
        self._memory_listener_thread.start()

    def _memory_event_listener(self) -> None:
        """Daemon thread: translate MemoryManager pub/sub replies into local events.

        The MemoryManager publishes the request_id on this rank's events channel once
        the requested resource has been prepared. Messages other than "message"
        (e.g. the initial subscribe ack) are ignored; the request_id is looked up in
        _memory_request_events and the waiting thread is woken via event.set().
        """
        for message in self._memory_pubsub.listen():
            if message["type"] != "message":
                continue
            request_id = message["data"].decode() if isinstance(message["data"], bytes) else message["data"]
            with self._memory_request_lock:
                event = self._memory_request_events.get(request_id)
            if event is not None:
                event.set()

    def _next_memory_request_seq(self) -> int:
        """Advance and return the process-local request sequence number.

        Combined with RANK by make_request_id() this yields a globally unique
        request_id, even though the sequence restarts at 1 in every assistant rank.
        """
        self._memory_request_seq += 1
        return self._memory_request_seq

    def _wait_for_memory_request(self, request_id: str) -> None:
        """Block until the listener thread signals that the request was served.

        The event was registered by _submit_memory_request() before the request was
        submitted, so a reply arriving before this call cannot be lost. The entry is
        removed afterwards to keep _memory_request_events bounded.
        """
        with self._memory_request_lock:
            event = self._memory_request_events[request_id]
        event.wait()
        with self._memory_request_lock:
            self._memory_request_events.pop(request_id, None)

    def _submit_memory_request(self, demand_id: str, use_index: int = -1) -> str:
        """Ask the MemoryManager to prepare the resource for one tensor demand.

        Registers the wake-up event before enqueueing the request on the Redis
        request queue; this ordering guarantees the listener can always find the
        event when the (potentially immediate) reply arrives. INSTANT demands pass
        their use_index so the manager serves exactly that use slot.

        Args:
            demand_id: Identity of the tensor demand (see _demand_id_for_op).
            use_index: 0-based use index for INSTANT demands, -1 otherwise.

        Returns:
            The unique request_id to wait on via _wait_for_memory_request().
        """
        request_seq = self._next_memory_request_seq()
        request_id = make_request_id(RANK, request_seq)
        with self._memory_request_lock:
            self._memory_request_events[request_id] = threading.Event()
        submit_request(self.memory_redis, LOCAL_RANK, RANK, request_seq, demand_id, use_index)
        return request_id

    @staticmethod
    def _direction_shape_dtype(op: ReplayerReplayOpInfo, direction: str) -> tuple[list[int], torch.dtype]:
        """Return the traced shape/dtype of an op's tensor on the given side.

        INPUT describes the tensor sent into a collective, OUTPUT the tensor it
        writes into; the pair identifies the MemoryManager demand.
        """
        if direction == MEMORY_TENSOR_INPUT:
            return op.info.input_shape, op.info.input_dtype
        if direction == MEMORY_TENSOR_OUTPUT:
            return op.info.output_shape, op.info.output_dtype
        raise ValueError(f"Unsupported tensor direction: {direction}")

    def _demand_id_for_op(self, op: ReplayerReplayOpInfo, direction: str) -> str:
        """Compute the MemoryManager demand_id of one op's INPUT/OUTPUT tensor.

        The id encodes (rank, c10d_func_name, signature, direction, shape, dtype),
        exactly the identity the MemoryManager derived its planning from, so both
        sides agree on the demand without any extra coordination.
        """
        shape, dtype = self._direction_shape_dtype(op, direction)
        return encode_demand_id(RANK, op.info.c10d_func_name, op.info.signature, direction, shape, dtype_to_str(dtype))

    @staticmethod
    def _build_tensor_from_cuda_ipc_handle(target_ref: str, shape: list[int], dtype: torch.dtype, handle_bytes: bytes):
        """Open the MemoryManager's CUDA allocation locally as a zero-copy tensor.

        The CUDA IPC handle (pickled tuple from Redis) names a device allocation
        owned by the MemoryManager process. ``_new_shared_cuda`` opens it in this
        process without copying any device memory; an empty local tensor is then
        rebound onto that shared storage with ``set_``. The view starts at offset 0
        and covers ``shape_numel(shape)`` elements, which is what the ANYTHING
        per-dtype pool expects: the pool holds the max element count over all its
        demands, and each demand slices its own shape from the front.

        Args:
            target_ref: Resource key prefix, used for logging only.
            shape: Traced tensor shape to view the storage as.
            dtype: Traced tensor dtype.
            handle_bytes: Pickled CUDA IPC handle tuple fetched from Redis.

        Returns:
            A tensor backed by the MemoryManager's device memory.
        """
        handle = pickle.loads(handle_bytes)
        storage = torch.UntypedStorage._new_shared_cuda(*handle)  # noqa: SLF001
        tensor = torch.empty(0, dtype=dtype, device=f"cuda:{LOCAL_RANK}")
        tensor.set_(storage, 0, [shape_numel(shape)])
        logger.debug("Built tensor from MemoryManager resource %s", target_ref)
        return tensor.reshape(shape)

    def _acquire_tensor_from_memory_manager(
        self,
        op: ReplayerReplayOpInfo,
        direction: str,
        acquired_refs: list[str],
    ) -> torch.Tensor:
        """Obtain the op's tensor from the MemoryManager, blocking until usable.

        Every acquire appends the resource ref to ``acquired_refs`` so the timing
        workers can release it (user_count -1) once the GPU finishes the op.

        The strategy differs per tensor memory level:

        - ANYTHING: a per-dtype pool that is never evicted (state is pinned READY,
          version pinned 0). Any acquire failure therefore means broken Redis state
          and raises immediately instead of waiting on the manager.
        - SHARED / PERSISTENT: evictable allocations. A local cache avoids repeating
          the IPC-handle dance, but a cache hit must still atomically re-acquire
          with the cached version: VERSION_MISMATCH means the resource was evicted
          and re-exposed as a new CUDA object generation (the cached tensor points
          at stale device memory), NOT_READY means it is currently evicted — both
          drop the cache. When the uncached acquire also fails, the manager is asked
          (via request queue + pub/sub wake-up) to reload the allocation; the manager
          has already accounted for this use in user_count, so the wake-up path must
          NOT call acquire_if_ready again (that would double-count).
        - INSTANT: no caching — each use gets its own throwaway allocation. The
          next use index is read from the manager, the request names that exact
          use_index, and the wake-up path resolves the per-use allocation_ref.

        Args:
            op: The replay operation needing the tensor.
            direction: MEMORY_TENSOR_INPUT or MEMORY_TENSOR_OUTPUT.
            acquired_refs: Collected resource refs of this op, appended in place.

        Returns:
            A tensor backed by MemoryManager device memory, ready for the collective.
        """
        if self.memory_redis is None:
            raise RuntimeError("MemoryManager client is not initialized")

        shape, dtype = self._direction_shape_dtype(op, direction)
        demand_id = self._demand_id_for_op(op, direction)
        level = TensorMemoryLevel[self.memory_redis.command.get(key_demand_level(LOCAL_RANK, demand_id)).decode()]
        target_ref = self.demand_target_refs.get(demand_id)
        if target_ref is None:
            target_ref = self.memory_redis.command.get(key_demand_target_ref(LOCAL_RANK, demand_id)).decode()
            self.demand_target_refs[demand_id] = target_ref

        if level in {TensorMemoryLevel.ANYTHING, TensorMemoryLevel.SHARED, TensorMemoryLevel.PERSISTENT}:
            cached = self.tensor_cache.get(demand_id)
            if cached is not None:
                cached_tensor, cached_version = cached
                acquire_result = self.memory_redis.acquire_if_ready(target_ref, cached_version)
                if acquire_result == ACQUIRE_READY:
                    acquired_refs.append(target_ref)
                    return cached_tensor
                if acquire_result in {ACQUIRE_NOT_READY, ACQUIRE_VERSION_MISMATCH}:
                    self.tensor_cache.pop(demand_id, None)
                if level == TensorMemoryLevel.ANYTHING:
                    raise RuntimeError(f"ANYTHING resource {target_ref} failed cached acquire: {acquire_result}")

            acquire_result = self.memory_redis.acquire_if_ready(target_ref)
            if acquire_result == ACQUIRE_READY:
                version = int(self.memory_redis.command.get(redis_field(target_ref, "version")))
                handle = self.memory_redis.command.get(redis_field(target_ref, "cuda_ipc_handle"))
                tensor = self._build_tensor_from_cuda_ipc_handle(target_ref, shape, dtype, handle)
                self.tensor_cache[demand_id] = (tensor, version)
                acquired_refs.append(target_ref)
                return tensor

            if level == TensorMemoryLevel.ANYTHING:
                raise RuntimeError(f"ANYTHING resource {target_ref} unexpectedly failed acquire: {acquire_result}")

            # Ask the manager to reload the (evicted) allocation and wait for the
            # wake-up. The manager bumps user_count itself while serving the
            # request, so read version/handle directly below without re-acquiring.
            request_id = self._submit_memory_request(demand_id)
            self._wait_for_memory_request(request_id)
            target_ref = self.memory_redis.command.get(key_demand_target_ref(LOCAL_RANK, demand_id)).decode()
            self.demand_target_refs[demand_id] = target_ref
            version = int(self.memory_redis.command.get(redis_field(target_ref, "version")))
            handle = self.memory_redis.command.get(redis_field(target_ref, "cuda_ipc_handle"))
            tensor = self._build_tensor_from_cuda_ipc_handle(target_ref, shape, dtype, handle)
            self.tensor_cache[demand_id] = (tensor, version)
            acquired_refs.append(target_ref)
            return tensor

        if level == TensorMemoryLevel.INSTANT:
            use_index = int(self.memory_redis.command.get(key_instant_next_use_index(LOCAL_RANK, demand_id)))
            request_id = self._submit_memory_request(demand_id, use_index)
            self._wait_for_memory_request(request_id)
            allocation_ref = self.memory_redis.command.get(
                key_instant_use_allocation_ref(LOCAL_RANK, demand_id, use_index)
            ).decode()
            handle = self.memory_redis.command.get(redis_field(allocation_ref, "cuda_ipc_handle"))
            tensor = self._build_tensor_from_cuda_ipc_handle(allocation_ref, shape, dtype, handle)
            acquired_refs.append(allocation_ref)
            return tensor

        raise RuntimeError(f"Unsupported tensor memory level: {level}")

    def _execute_op(self, op: ReplayerReplayOpInfo, acquired_refs: list[str]) -> torch.distributed.Work | None:
        """Execute a single communication operation.

        Skipping rules, in order to keep the assistant node free of useless traffic:

        - Ops whose group was never created on this rank are dropped.
        - Collectives whose whole group is emulated (_intra_assistant_only) never
          cross the sandbox boundary and are dropped; collectives with at least one
          sandbox member are really issued.
        - P2P isend/irecv towards an emulated peer are dropped as well, EXCEPT the
          first one per (group, peer) pair: NCCL builds the peer connection lazily
          on the first paired send/recv, and skipping that first op would leave the
          connection unbuilt and hang later communication on it.
        - all_to_all_single may never cross the sandbox boundary at all; a partial
          overlap raises instead of silently mis-replaying (the graph/planning side
          guarantees sandbox layout keeps every all-to-all fully on one side).

        Tensors come from the MemoryManager via _acquire_tensor_from_memory_manager;
        INPUT tensors for data-flowing ops, OUTPUT tensors for the writing ones.

        Args:
            op: The replay operation to execute.
            acquired_refs: Resource refs acquired for this op, appended in place.

        Returns:
            Work object for async operations, None for sync or skipped operations.
        """
        logger.debug("_execute_op() handling op: %s", op)

        # Skip ops for groups that were not created
        if op.info.group_name not in self.comm_groups:
            logger.debug("Skipping op for non-existent group: %s", op.info.group_name)
            return None

        # ----- ISEND -----
        if op.info.c10d_func_name == "isend":
            # batch_isend_irecv traces record the peer as a group-relative rank;
            # translate it to a global rank once before use.
            if "group_dst" in op.info.meta and "dst" not in op.info.meta:
                op.info.meta["dst"] = torch.distributed.get_global_rank(
                    self.comm_groups[op.info.group_name], op.info.meta["group_dst"]
                )
            if op.info.meta["dst"] in self.emulated_ranks:
                logger.debug("%s peer is emulated", op.info.c10d_func_name)
                # Pure assistant-side P2P: execute only the first op of this
                # (group, peer) pair to establish the NCCL peer connection —
                # skipping the first one would make NCCL hang — then skip the rest.
                if f"{op.info.group_name}-{op.info.meta['dst']}" not in self.pp_group_has_init:
                    self.pp_group_has_init.add(f"{op.info.group_name}-{op.info.meta['dst']}")
                    logger.debug("executed first time, init pp group")
                else:
                    logger.debug("skipping")
                    return None
            tensor_var = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_INPUT, acquired_refs)
            return torch.distributed.isend(
                tensor=tensor_var, dst=op.info.meta["dst"], group=self.comm_groups[op.info.group_name]
            )
        # ----- IRECV -----
        if op.info.c10d_func_name == "irecv":
            # See isend: group-relative peer rank to global rank.
            if "group_src" in op.info.meta and "src" not in op.info.meta:
                op.info.meta["src"] = torch.distributed.get_global_rank(
                    self.comm_groups[op.info.group_name], op.info.meta["group_src"]
                )
            if op.info.meta["src"] in self.emulated_ranks:
                logger.debug("%s peer is emulated", op.info.c10d_func_name)
                # See isend: first op of the (group, peer) pair builds the connection.
                if f"{op.info.group_name}-{op.info.meta['src']}" not in self.pp_group_has_init:
                    self.pp_group_has_init.add(f"{op.info.group_name}-{op.info.meta['src']}")
                    logger.debug("executed first time, init pp group")
                else:
                    logger.debug("skipping")
                    return None
            output_tensor = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_OUTPUT, acquired_refs)
            return torch.distributed.irecv(
                tensor=output_tensor, src=op.info.meta["src"], group=self.comm_groups[op.info.group_name]
            )
        # ----- BROADCAST -----
        if op.info.c10d_func_name == "broadcast":
            if self._intra_assistant_only(self.comm_groups_ranks[op.info.group_name]):
                logger.debug("%s is intra only, skipping", op.info.c10d_func_name)
                return None
            tensor_var = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_INPUT, acquired_refs)
            return torch.distributed.broadcast(
                tensor_var,
                op.info.meta["src"],
                group=self.comm_groups[op.info.group_name],
                async_op=op.info.async_op,
            )
        # ----- ALL REDUCE -----
        if op.info.c10d_func_name == "all_reduce":
            if self._intra_assistant_only(self.comm_groups_ranks[op.info.group_name]):
                logger.debug("%s is intra only, skipping", op.info.c10d_func_name)
                return None
            tensor_var = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_INPUT, acquired_refs)
            return torch.distributed.all_reduce(
                tensor_var,
                op=op.info.meta["reduce_op"],
                group=self.comm_groups[op.info.group_name],
                async_op=op.info.async_op,
            )
        # ----- ALL GATHER -----
        if op.info.c10d_func_name in {"all_gather_into_tensor", "_all_gather_base"}:
            if self._intra_assistant_only(self.comm_groups_ranks[op.info.group_name]):
                logger.debug("%s is intra only, skipping", op.info.c10d_func_name)
                return None
            tensor_var = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_INPUT, acquired_refs)
            output_tensor = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_OUTPUT, acquired_refs)
            return torch.distributed.all_gather_into_tensor(
                output_tensor,
                tensor_var,
                group=self.comm_groups[op.info.group_name],
                async_op=op.info.async_op,
            )
        # ----- REDUCE SCATTER -----
        if op.info.c10d_func_name in {"reduce_scatter_tensor", "_reduce_scatter_base"}:
            if self._intra_assistant_only(self.comm_groups_ranks[op.info.group_name]):
                logger.debug("%s is intra only, skipping", op.info.c10d_func_name)
                return None
            tensor_var = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_INPUT, acquired_refs)
            output_tensor = self._acquire_tensor_from_memory_manager(op, MEMORY_TENSOR_OUTPUT, acquired_refs)
            return torch.distributed.reduce_scatter_tensor(
                output_tensor,
                tensor_var,
                op=op.info.meta["reduce_op"],
                group=self.comm_groups[op.info.group_name],
                async_op=op.info.async_op,
            )
        # ----- ALL TO ALL -----
        if op.info.c10d_func_name == "all_to_all_single":
            if self._intra_assistant_only(self.comm_groups_ranks[op.info.group_name]):
                logger.debug("%s is intra only, skipping", op.info.c10d_func_name)
                return None
            # all_to_all exchanges data with every member simultaneously, so a group
            # overlapping the sandbox cannot be split into real and mocked halves.
            raise ValueError("all-to-all should not cross the boundary of sandbox")
        # ----- BARRIER -----
        if op.info.c10d_func_name == "barrier":
            return torch.distributed.barrier(
                group=self.comm_groups[op.info.group_name],
                async_op=op.info.async_op,
            )
        raise ValueError(f"Unsupported op: {op}")

    def _replay_op(
        self, op: ReplayerReplayOpInfo, acquired_refs: list[str]
    ) -> torch.distributed.distributed_c10d._CoalescingManager | torch.distributed.Work | None:
        """Replay a single operation, handling coalescing and communication ops.

        A COALESCING_START op consumes every following op up to (and including) the
        matching COALESCING_END from the queue and executes them inside one
        torch _coalescing_manager context, mirroring the traced coalescing block:
        the works of the individual collectives are collected by the manager, so
        their return values are intentionally discarded here, and the returned
        manager's wait() waits for all of them at once. A COMM op is a plain
        single-op replay.

        When the op's group does not exist on this rank, the op (and, for a
        coalescing block, every op up to COALESCING_END) is dropped so the queue
        stays consistent.

        Args:
            op: The replay operation to replay.
            acquired_refs: Resource refs acquired by this op/coalescing block.

        Returns:
            CoalescingManager for coalescing operations, Work object for async ops,
            or None for sync/skipped operations.
        """
        logger.debug("_replay_op() handling op: %s", op)

        # Skip if group was not created
        if op.info.group_name not in self.comm_groups:
            logger.debug("Skipping op for non-existent group: %s", op.info.group_name)
            if op.type == ReplayerReplayOpType.COALESCING_START:
                # Consume all ops until COALESCING_END so they are not mistaken
                # for top-level ops by the replay loops.
                while self.op_queue:
                    coalesced_ops = self.op_queue.pop(0)
                    if coalesced_ops.type == ReplayerReplayOpType.COALESCING_END:
                        break
            return None

        if op.type == ReplayerReplayOpType.COALESCING_START:
            logger.debug("Coalescing start: %s", op)
            # device_set mirrors the traced call: only when the original context
            # manager bound a device does the replay bind one (the current CUDA
            # device) as well; async_ops decides whether the block is awaited lazily.
            with _coalescing_manager(
                group=self.comm_groups[op.info.group_name],
                device=torch.device(torch.cuda.current_device()) if op.info.device_set else None,
                async_ops=op.info.async_ops,
            ) as cm:
                while self.op_queue:
                    coalesced_ops = self.op_queue.pop(0)
                    if coalesced_ops.type == ReplayerReplayOpType.COALESCING_END:
                        break
                    self._execute_op(coalesced_ops, acquired_refs)
            return cm
        if op.type == ReplayerReplayOpType.COMM:
            return self._execute_op(op, acquired_refs)
        return None

    def _borrow_async_stream_id(self, warn_interval_s: float = 1.0) -> int:
        """Borrow a stream id from the async stream pool.

        Blocks until a worker returns a stream, but logs a warning every
        ``warn_interval_s`` seconds so pool exhaustion (in-flight async comms
        reaching MAX_PARALLEL_COMM) stalls the replay visibly, never silently.
        """
        wait_start = time.perf_counter()
        has_warned = False
        while True:
            try:
                stream_id = self._async_stream_available.get(timeout=warn_interval_s)
            except Empty:
                has_warned = True
                logger.warning(
                    "Async stream pool exhausted: all %d streams busy, waited %.1f s for a return",
                    MAX_PARALLEL_COMM,
                    time.perf_counter() - wait_start,
                )
                continue
            if has_warned:
                logger.warning(
                    "Borrowed async stream %d after waiting %.1f s",
                    stream_id,
                    time.perf_counter() - wait_start,
                )
            return stream_id

    def _replay_tracked_op(self, op: ReplayerReplayOpInfo, timing_context: TimingContext | None = None) -> None:
        """Execute an op while tracking MemoryManager resource lifetime.

        Pipeline: acquire resources inside _replay_op(), record CUDA events that
        mark the op's completion on the stream it is waited on, then hand a
        CompletionTask to the timing workers — only they, after synchronizing the
        event, release the resources, broadcast the measured duration (when a
        timing_context is attached), and return the borrowed stream to the pool.

        Stream selection for the wait:

        - timing + async: wait on the borrowed side stream, so both timing events
          live on it and elapsed_time() covers launch-to-completion of the comm.
        - timing + sync: the op already blocked its own stream, wait on the current
          stream so the end event lands right after the blocking call.
        - no timing + async: wait on a borrowed side stream, mirroring the planned-op
          path, so the main stream is never serialized behind the async communication.
        - no timing + sync (incl. sync coalescing blocks): the wait already happened
          on the current stream.
        - skipped op that still acquired resources (e.g. a sync collective): record
          the completion event on the current stream so release stays ordered.

        Args:
            op: The COMM or COALESCING_START op to execute.
            timing_context: Timing context when the op must broadcast its duration.
        """
        logger.debug("_replay_tracked_op() handling op: %s", op)

        acquired_refs: list[str] = []
        timing_stream = None
        borrowed_stream_id: int | None = None
        # Coalescing blocks carry their async flag on the COALESCING_START op.
        is_async = op.info.async_op if op.type == ReplayerReplayOpType.COMM else op.info.async_ops

        if timing_context is not None:
            timing_context.start_event = torch.cuda.Event(enable_timing=True)
            timing_context.end_event = torch.cuda.Event(enable_timing=True)
            if is_async:
                borrowed_stream_id = self._borrow_async_stream_id()
                timing_context.stream_id = borrowed_stream_id
                timing_stream = self._async_timing_streams[borrowed_stream_id]
            else:
                timing_context.stream_id = None
                timing_stream = torch.cuda.current_stream()
            timing_context.start_event.record(stream=timing_stream)

        # Both Work and _CoalescingManager expose wait(): _CoalescingManager.wait()
        # waits every coalesced work, and is a no-op for sync coalescing blocks
        # (empty works), so the completion handle can be used uniformly below.
        work = self._replay_op(op, acquired_refs)

        # Nothing was acquired and nothing is timed: no completion work to delegate.
        if not acquired_refs and timing_context is None:
            return

        completion_event = None
        if timing_context is not None:
            if work is not None:
                with torch.cuda.stream(timing_stream):
                    work.wait()
            timing_context.end_event.record(stream=timing_stream)
        elif work is not None:
            if is_async:
                # Wait on a borrowed side stream, mirroring the planned-op path, so
                # the main stream is never serialized behind the async communication.
                borrowed_stream_id = self._borrow_async_stream_id()
                completion_stream = self._async_timing_streams[borrowed_stream_id]
            else:
                # Sync coalescing block: _coalescing_manager already made the current
                # stream wait, so the completion point stays on the current stream.
                completion_stream = torch.cuda.current_stream()
            with torch.cuda.stream(completion_stream):
                work.wait()
            completion_event = torch.cuda.Event(enable_timing=False)
            completion_event.record(stream=completion_stream)
        elif acquired_refs:
            current_stream = torch.cuda.current_stream()
            completion_event = torch.cuda.Event(enable_timing=False)
            completion_event.record(stream=current_stream)

        self._timing_task_queue.put(
            CompletionTask(
                resource_refs=acquired_refs,
                completion_event=completion_event,
                timing_context=timing_context,
                stream_id=borrowed_stream_id,
            )
        )

    def replay_immediate(self):
        """Replay operations immediately without any plan-based timing control.

        Every op fires as soon as it is dequeued — no launch-time sleeps, no
        predecessor waits, no Redis broadcasting. Resource lifetime is still
        tracked (tensors come from the MemoryManager and are released by the
        timing workers), so this path exercises the real communication and memory
        machinery and is used when no replay plan (graph) is provided.
        """
        current_iteration = 0
        self._init_timing_workers()
        try:
            while self.op_queue:
                op = self.op_queue.pop(0)
                if op.type == ReplayerReplayOpType.ITERATION_BEGIN:
                    current_iteration += 1
                    # Keep the per-iteration rendezvous so all ranks advance
                    # through iterations together even without a graph.
                    replay_rendezvous(current_iteration)
                elif op.type in {ReplayerReplayOpType.COMM, ReplayerReplayOpType.COALESCING_START}:
                    self._replay_tracked_op(op)
                elif op.type == ReplayerReplayOpType.COALESCING_END:
                    logger.error("Unexpected COALESCING_END op without matching COALESCING_START")
        finally:
            self._cleanup_timing_workers()

    def replay_with_graph(
        self,
        trace_dir: str,
        graph_dir: str,
        profiler_dir: str | None,
        pp_rank_stride: int,
        enable_online_update: bool = False,
    ):
        """Replay operations with CrystalGraph-based timing control and cross-rank synchronization.

        Args:
            trace_dir: Directory containing trace files (trace_{rank}.log.gz).
            graph_dir: Directory containing graph files (graph_{rank}.json.gz).
            profiler_dir: Directory containing runtime_profiler_{rank}.json.gz. If None,
                all span durations are left as 0.
            pp_rank_stride: Rank distance between adjacent PP stages, used to enumerate PP chains.
            enable_online_update: If True, set up the full Redis broadcast system and apply
                online duration updates. If False (default), no broadcast infrastructure is
                created at all and every launch time comes solely from the loaded graph.
                Forced to False when this process runs with CUDA_DEVICE_MAX_CONNECTIONS=1 or
                the graphs were captured under it: the timing CUDA events online updates
                rely on would serialize the emulated communications.
        """
        self._bundle = CrystalGraphBundle.load(
            graph_dir=graph_dir,
            trace_dir=trace_dir,
            profiler_dir=profiler_dir,
            pp_rank_stride=pp_rank_stride,
            sandbox_ranks=self.sandbox_ranks,
            world_size=WORLD_SIZE,
            own_rank=RANK,
            launched_ranks=self._launched_emulated_ranks(),
            new_group_table=self.comm_groups_ranks,
        )

        # Online updates measure every op with timing-enabled CUDA events; under
        # max_connections == 1 those events serialize the overlapped communications
        # (same interaction the runtime profiler avoids through issue-wait durations).
        # Neither this assistant process nor the captured graphs may ask for 1.
        env_max_connections = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "8"))
        if enable_online_update and (env_max_connections == 1 or self._bundle.max_connections == 1):
            logger.warning(
                "Disabling online updates: max_connections is 1 (environment=%d, graphs=%d); "
                "their timing CUDA events would serialize the emulated communications",
                env_max_connections,
                self._bundle.max_connections,
            )
            enable_online_update = False

        self._enable_online_update = enable_online_update

        if enable_online_update:
            # Broadcast infrastructure first: the listener must be up before any
            # op can complete and broadcast a duration.
            self._setup_redis_listener()
        else:
            # No duration ever changes after load, so no rank needs to announce an arrival
            # or hear one: no event table, listener thread, or publisher is created, and
            # every launch time comes straight from the loaded graph. Precompute all
            # step intervals once so replay-time queries only read the cache.
            self._precompute_graph_times()

        self._replay_online()

    def replay_with_graph_cache(self, graph_cache_path: str):
        """Replay operations from a prebuilt launch-timing cache, without any graph.

        The cache replaces the merged CrystalGraphBundle of replay_with_graph()
        entirely: per (rank, trace_comm_index) it holds only the anchorable
        operations and their synchronization groups' launch times, all extracted
        from the very graphs and runtime profiler data this scenario would load.
        An operation absent from the cache is itself an answer — outside the
        analysis window, or on a purely emulated synchronization group — and
        replays immediately, exactly as the graph path would decide for it.
        Only this rank's records are kept in memory, so the footprint is
        proportional to the rank's own trace instead of the whole scenario's
        merged graphs, and the startup cost is one small JSON file read instead
        of loading, merging and analyzing every PP chain.

        Online updates cannot exist here: durations never change, so no Redis
        broadcast, no predecessor wait, and no TimingContext is ever created —
        the caller-side mutual exclusion with --enable-online-update enforces it.

        Args:
            graph_cache_path: Path of the graph_cache_{timestamp}.json.gz file
                generated by generate_graph_cache().
        """
        cache = load_graph_cache(graph_cache_path)
        validate_graph_cache(cache, world_size=WORLD_SIZE, sandbox_ranks=self.sandbox_ranks)
        self._op_launch_times = cached_op_timings(cache, RANK)
        self._bundle = None
        self._enable_online_update = False
        logger.info(
            "Loaded graph cache %s: %d anchorable ops of rank %d over steps %s",
            graph_cache_path,
            len(self._op_launch_times),
            RANK,
            cache["steps"],
        )
        self._replay_online()

    def _launched_emulated_ranks(self) -> list[int]:
        """Return the emulated ranks actually started for this run.

        With mock NCCL bootstrap only the member ranks derived by the bootstrap planner are
        started; without it every emulated rank runs, which makes every PP chain mandatory
        and turns chain pruning into a no-op.
        """
        member_ranks = os.getenv("CRYSTAL_MOCK_NCCL_MEMBER_RANKS")
        if not member_ranks:
            return list(self.emulated_ranks)
        return [int(rank) for rank in member_ranks.split(",") if rank.strip()]

    def _precompute_graph_times(self) -> None:
        """Relax every step's graph once so replay-time queries only read the cache.

        get_points_interval() without an end point sweeps the whole condensed graph,
        which fills the incremental longest-path cache of that step completely. Online
        updates are off, so no duration changes afterwards and no cache rewind ever
        happens: every launch-time lookup during replay is then a final cache hit
        instead of the first relaxation of its prefix — the per-query cost drops from
        a partial longest-path relaxation to a table read. With online updates on,
        each increase_span_duration() rewinds the frontier, so precomputing would be
        thrown away almost immediately and is therefore skipped in replay_with_graph().
        """
        for step, graph in self._bundle.step_graphs.items():
            interval_ns = graph.get_points_interval()
            logger.info(
                "Precomputed all graph times of step %s: %s ns (%.3f ms)",
                step,
                interval_ns,
                interval_ns / MS_TO_NS,
            )

    def _setup_redis_listener(self) -> None:
        """Set up the Redis broadcast system for online duration updates.

        Only called with online updates on. Builds three pieces:

        - _pred_op_events: the (rank, trace_comm_index) -> Event table the replay
          side waits on. Only started assistant ranks ever execute an op and thus
          broadcast an arrival, so the bundle drops predecessors living on any
          other rank (waiting on a never-started proxy rank would deadlock).
        - _redis_client: the publishing client the timing workers broadcast
          measured durations through.
        - the subscription thread (_redis_broadcast_listener) applying incoming
          updates and signalling arrivals.
        """
        event_keys = self._bundle.predecessor_event_keys(set(self._launched_emulated_ranks()))
        self._pred_op_events: dict[tuple[int, int], threading.Event] = {key: threading.Event() for key in event_keys}

        redis_host = os.getenv("CRYSTAL_REDIS_HOST", "localhost")
        redis_port = int(os.getenv("CRYSTAL_REDIS_PORT", "6379"))
        self._redis_client = redis.Redis(host=redis_host, port=redis_port, db=1)
        logger.info("Redis client initialized for broadcasting (host=%s, port=%s)", redis_host, redis_port)

        # Start Redis listener thread. It owns a separate connection on purpose:
        # the blocking pubsub listen() would monopolize any shared connection.
        listener_thread = threading.Thread(
            target=self._redis_broadcast_listener,
            daemon=True,
        )
        listener_thread.start()
        logger.info("Started Redis broadcast listener with %d predecessor events", len(self._pred_op_events))

    def _redis_broadcast_listener(self) -> None:
        """Listen for Redis broadcasts and update graph durations.

        The thread only exists with online updates on, so every message received
        here is applied, never merely observed. Each message does two things:

        1. increase_span_duration() on the sender's measured duration. Durations
           only ever grow; a measurement below the runtime profiler value the
           graph was loaded with raises CrystalGDurationNotIncreasedError, which
           is expected and logged, keeping the graph as it is.
        2. Set the (rank, trace_comm_index) arrival event. This must happen even
           when the update itself was skipped: a waiter blocked on this arrival
           would otherwise deadlock on a harmless no-op.
        """
        redis_host = os.getenv("CRYSTAL_REDIS_HOST", "localhost")
        redis_port = int(os.getenv("CRYSTAL_REDIS_PORT", "6379"))
        rc = redis.Redis(host=redis_host, port=redis_port, db=1)
        pubsub = rc.pubsub()
        pubsub.subscribe("crystal_broadcast")

        logger.info("Redis listener connected to %s:%s db=1", redis_host, redis_port)

        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                rank = data["rank"]
                trace_comm_index = data["trace_comm_index"]
                duration_ns = data["duration_ns"]

                logger.debug(
                    "Received Redis broadcast: rank=%s, trace_comm_index=%s, duration_ns=%s",
                    rank,
                    trace_comm_index,
                    duration_ns,
                )

                step = self._bundle.step_of_trace_comm_index(rank, trace_comm_index)
                if step is not None:
                    try:
                        self._bundle.step_graphs[step].increase_span_duration(
                            rank, CrystalGSpanId(COMM_TYPE_FIELD, trace_comm_index), duration_ns
                        )
                    except CrystalGDurationNotIncreasedError as e:
                        # Expected whenever the measurement undercuts the runtime profiler value
                        # this graph was loaded with; keep the graph as it is and carry on.
                        logger.info("Skipping online update: %s", e)

                # Signal arrival even when the update above was skipped (see docstring).
                key = (rank, trace_comm_index)
                if key in self._pred_op_events:
                    self._pred_op_events[key].set()

    def _broadcast_duration(self, rank: int, trace_comm_index: int, duration_ns: int) -> None:
        """Broadcast a measured duration to all assistant ranks via Redis.

        Publishes on the shared "crystal_broadcast" channel; every rank's
        _redis_broadcast_listener applies the update to its own copy of the step
        graphs (all ranks hold the same merged graph), so the launch-time
        recomputation stays consistent across ranks.

        Args:
            rank: The rank that executed the operation.
            trace_comm_index: The communication trace index.
            duration_ns: The duration in nanoseconds.
        """
        message = json.dumps({"rank": rank, "trace_comm_index": trace_comm_index, "duration_ns": duration_ns})
        self._redis_client.publish("crystal_broadcast", message)
        logger.info(
            "Broadcasted duration for rank %s, trace_comm_index %s: %s ns (%.3f ms)",
            rank,
            trace_comm_index,
            duration_ns,
            duration_ns / MS_TO_NS,
        )

    def _replay_online(self) -> None:
        """Replay operations with timing control based on CrystalGraph analysis.

        Main loop over the op queue:

        - ITERATION_BEGIN: step the profilers (PyTorch profiler schedule and the
          optional Nsight Systems range are aligned to CRYSTAL_ANALYSIS_STEP_START/
          END), run the cross-rank rendezvous, and stamp this iteration's start
          time — the base every planned launch time is measured against.
        - COMM / COALESCING_START: anchorable ops (their sync group crosses the
          sandbox boundary) go to _replay_planned_op() for graph-derived launch
          timing; everything else goes to the immediate _replay_tracked_op() path.
        - COALESCING_END at top level marks a malformed trace and is only logged.
        """
        # Initialize PyTorch profiler if enabled. The schedule maps the analysis
        # window [STEP_START, STEP_END) onto wait/warmup/active steps of the
        # profiler: everything before the window is skipped, one warmup step
        # precedes the active window (only when the window starts after step 0).
        pytorch_profiler = None
        if is_truthy_value(os.getenv("ENABLE_CRYSTAL_ASSISTANT_PYTORCH_PROFILER")):
            pytorch_profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=max(int(os.getenv("CRYSTAL_ANALYSIS_STEP_START")) - 1, 0),
                    warmup=1 if int(os.getenv("CRYSTAL_ANALYSIS_STEP_START")) > 0 else 0,
                    active=int(os.getenv("CRYSTAL_ANALYSIS_STEP_END")) - int(os.getenv("CRYSTAL_ANALYSIS_STEP_START")),
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=str(pathlib.Path(os.getenv("CRYSTAL_LOG_DIR_ASSISTANT")) / "pytorch_profiler"),
                    worker_name="rank_" + os.getenv("RANK"),
                    use_gzip=True,
                ),
                with_stack=True,
            )
            pytorch_profiler.start()

        use_nsight_system = is_truthy_value(os.getenv("ENABLE_CRYSTAL_ASSISTANT_NSIGHT_SYSTEM"))
        if use_nsight_system:
            (pathlib.Path(os.getenv("CRYSTAL_LOG_DIR_ASSISTANT")) / "nsys").mkdir(parents=True, exist_ok=True)

        iteration_timer_start = None
        current_iteration = 0
        # iteration number -> perf_counter_ns at its ITERATION_BEGIN; the base of
        # the sleep-until calculations in _replay_planned_op().
        self._iteration_start_times: dict[int, int] = {}

        # Initialize timing workers for planned ops
        self._init_timing_workers()

        while self.op_queue:
            op = self.op_queue.pop(0)

            if op.type == ReplayerReplayOpType.ITERATION_BEGIN:
                # Log elapsed time for previous iteration
                if iteration_timer_start is not None and current_iteration > 0:
                    elapsed_ns = time.perf_counter_ns() - iteration_timer_start
                    logger.info("Iteration %s elapsed time: %s ms", current_iteration, elapsed_ns / MS_TO_NS)

                if pytorch_profiler is not None:
                    pytorch_profiler.step()

                # current_iteration is still the previous (0-based) iteration
                # count here, so +1 is the 1-based number of the iteration that
                # is about to start: bracket the analysis window with it.
                if use_nsight_system:
                    if (current_iteration + 1) == int(os.getenv("CRYSTAL_ANALYSIS_STEP_START")):
                        torch.cuda.cudart().cudaProfilerStart()
                    if (current_iteration + 1) == int(os.getenv("CRYSTAL_ANALYSIS_STEP_END")):
                        torch.cuda.cudart().cudaProfilerStop()

                current_iteration += 1
                replay_rendezvous(current_iteration)

                iteration_timer_start = time.perf_counter_ns()
                self._iteration_start_times[current_iteration] = iteration_timer_start

            elif op.type in {ReplayerReplayOpType.COMM, ReplayerReplayOpType.COALESCING_START}:
                # Handle both COMM and COALESCING_START ops
                if self._is_anchorable_op(op):
                    self._replay_planned_op(op, current_iteration)
                else:
                    # No anchorable synchronization group: execute immediately with resource
                    # tracking and let _execute_op()'s _intra_assistant_only() decide whether
                    # the operation is skipped altogether.
                    self._replay_tracked_op(op)

            elif op.type == ReplayerReplayOpType.COALESCING_END:
                # COALESCING_END is consumed by _replay_op() while processing its
                # COALESCING_START; seeing one here means the trace is malformed.
                logger.error("Unexpected COALESCING_END op without matching COALESCING_START")

        # Cleanup timing workers
        self._cleanup_timing_workers()

        # Save PyTorch Profiler data if enabled
        if pytorch_profiler is not None:
            pytorch_profiler.stop()

    def _is_anchorable_op(self, op: ReplayerReplayOpInfo) -> bool:
        """Decide whether an operation must be launched at its graph-derived time.

        Being present in the graph is weaker than being anchorable: the merged graph also
        contains synchronization groups between purely emulated ranks, which are found here
        but must fall through to the immediate path so _intra_assistant_only() skips them.

        The answer comes from the loaded bundle in graph mode and from this rank's cached
        records in cache mode — an absent cache record is itself the answer, covering
        both the operations outside the analysis window and the non-anchorable ones, see
        replay_with_graph_cache().

        Args:
            op: The COMM or COALESCING_START operation.

        Returns:
            True when the operation's synchronization group crosses the sandbox boundary.
        """
        trace_comm_index = op.info.trace_comm_index
        if self._op_launch_times is not None:
            return trace_comm_index in self._op_launch_times

        step = self._bundle.step_of_trace_comm_index(RANK, trace_comm_index)
        if step is None:
            # Outside the analysis window: the graph knows nothing about this op,
            # so no launch timing exists for it either.
            return False

        return self._bundle.is_anchorable_point(
            step, CrystalGSpanPoint(RANK, CrystalGSpanId(COMM_TYPE_FIELD, trace_comm_index), SpanPointType.STOP)
        )

    def _init_timing_workers(self) -> None:
        """Initialize timing infrastructure for planned op execution.

        Creates MAX_PARALLEL_COMM + 1 timing worker threads (one more than the
        stream count) for parallel completion synchronization, resource release,
        and duration broadcasting, plus MAX_PARALLEL_COMM dedicated CUDA streams
        on which async ops are waited, so the main replay stream is never
        serialized behind an in-flight communication.

        Stream bookkeeping uses two queues: _async_stream_available hands out and
        takes back stream ids (borrow in _borrow_async_stream_id, return in the
        worker), and _timing_task_queue feeds the workers, whose join() in
        _cleanup_timing_workers waits for task_done() of every queued task.
        """
        self._timing_workers: list[threading.Thread] = []

        # Dedicated CUDA streams async ops are waited on
        self._async_timing_streams: list[torch.cuda.Stream] = [
            torch.cuda.Stream(device=f"cuda:{LOCAL_RANK}") for _ in range(MAX_PARALLEL_COMM)
        ]

        # Pool of free stream ids; borrowing blocks when all streams are busy
        self._async_stream_available: Queue = Queue()
        for i in range(MAX_PARALLEL_COMM):
            self._async_stream_available.put(i)

        # Shared task queue for all timing workers
        self._timing_task_queue: Queue = Queue()

        # Stop event checked by the workers between queue timeouts
        self._timing_worker_stop_event: threading.Event = threading.Event()

        # Start MAX_PARALLEL_COMM + 1 timing worker threads
        for i in range(MAX_PARALLEL_COMM + 1):
            t = threading.Thread(
                target=self._timing_worker_loop,
                daemon=True,
                name=f"timing_worker_{i}",
            )
            t.start()
            self._timing_workers.append(t)

        logger.info(
            "Initialized timing infrastructure: %d timing workers, %d async streams",
            MAX_PARALLEL_COMM + 1,
            MAX_PARALLEL_COMM,
        )

    def _cleanup_timing_workers(self) -> None:
        """Drain the task queue, then stop and join every timing worker.

        join() on the task queue blocks until each queued CompletionTask has been
        task_done()'d, i.e. every op's completion was synchronized, its resources
        released, and its stream returned — only then are the workers signalled
        to stop (they wake from their 1 s queue timeout at most a second later)
        and the stream list is dropped.
        """
        self._timing_task_queue.join()
        self._timing_worker_stop_event.set()

        # Wait for all timing workers to finish
        for t in self._timing_workers:
            t.join(timeout=5.0)

        self._timing_workers = []
        self._async_timing_streams = []

        logger.info("Cleaned up timing workers")

    def _timing_worker_loop(self) -> None:
        """Persistent worker loop for completion synchronization, release, and optional timing.

        Per CompletionTask, strictly in order:

        1. Synchronize the completion (or timing end) event, i.e. block until the
           GPU finished the op — releasing a resource before this point could
           evict/destroy memory a communication is still reading or writing.
        2. Release every acquired resource ref by decrementing user_count.
        3. With a timing context attached, read elapsed_time() between the two
           timing events and broadcast the measured duration.
        4. Return the borrowed async stream (if any) to the availability queue.

        The 1 s get() timeout keeps the stop event responsive even when the queue
        is idle; task_done() runs in finally so a failing task cannot wedge
        cleanup's join().
        """
        while not self._timing_worker_stop_event.is_set():
            try:
                task: CompletionTask = self._timing_task_queue.get(timeout=1)
            except Empty:
                continue

            try:
                if task.timing_context is not None:
                    task.timing_context.end_event.synchronize()
                elif task.completion_event is not None:
                    task.completion_event.synchronize()

                logger.info(
                    "Completion sync done: trace_comm_index=%s(if None, no timing_context), "
                    "stream_id=%s, releasing %d resource refs: %s",
                    task.timing_context.trace_comm_index if task.timing_context is not None else None,
                    task.timing_context.stream_id if task.timing_context is not None else None,
                    len(task.resource_refs),
                    task.resource_refs,
                )

                for resource_ref in task.resource_refs:
                    self.memory_redis.command.incrby(redis_field(resource_ref, "user_count"), -1)

                if task.timing_context is not None:
                    elapsed_ms = task.timing_context.start_event.elapsed_time(task.timing_context.end_event)
                    duration_ns = int(elapsed_ms * MS_TO_NS)
                    self._broadcast_duration(RANK, task.timing_context.trace_comm_index, duration_ns)

                if task.stream_id is not None:
                    self._async_stream_available.put(task.stream_id)
            finally:
                self._timing_task_queue.task_done()

    def _replay_planned_op(
        self,
        op: ReplayerReplayOpInfo,
        iteration: int,
    ) -> None:
        """Execute an operation at the launch time derived from its synchronization group.

        Sequence per op:

        1. (online updates only) Wait until every direct anchor predecessor of
           this op's group has completed on its own rank, so cross-rank ordering
           discovered by the graph is respected.
        2. Query the group's launch time: a multi-party communication starts
           when its last participant arrives, so this is the latest START over
           all group members (computed from the group's join component). In
           cache mode the value comes from this rank's cached record instead,
           and no predecessor wait can exist there, see replay_with_graph_cache().
        3. Busy-wait-free sleep until iteration_start + launch_time, then hand
           the op to _replay_tracked_op() — with a TimingContext when the
           measured duration must be broadcast, without one otherwise.

        Args:
            op: The operation to execute.
            iteration: Current iteration number.
        """
        trace_comm_index = op.info.trace_comm_index
        logger.debug(
            "Executing planned op: rank=%s, trace_comm_index=%s, iteration=%s", RANK, trace_comm_index, iteration
        )

        if self._op_launch_times is not None:
            # Cache mode: online updates are off by construction, so no predecessor
            # wait exists and the launch time is this rank's cached record.
            interval_ns = self._op_launch_times[trace_comm_index]
            logger.info(
                "Launch time for op %s: %s ns (%.6f ms) from graph cache",
                trace_comm_index,
                interval_ns,
                interval_ns / MS_TO_NS,
            )
        else:
            graph = self._bundle.step_graphs[iteration]
            span_id = CrystalGSpanId(COMM_TYPE_FIELD, trace_comm_index)
            start_point = CrystalGSpanPoint(RANK, span_id, SpanPointType.START)

            # Wait for all predecessor events (pred_anchorable_ops to complete). With online
            # updates off no rank broadcasts, so no event table exists and the launch time is
            # taken straight from the loaded graph instead.
            if self._enable_online_update:
                for pred_point in graph.get_direct_anchor_predecessors(start_point):
                    pred_key = (pred_point.rank, pred_point.span_id.count_field)
                    if pred_key in self._pred_op_events:
                        logger.info("Waiting for pred %s", pred_key)
                        self._pred_op_events[pred_key].wait()

            group = graph.sync_span_group_of_point(start_point)
            interval_ns = self._bundle.sync_span_group_launch_time_ns(iteration, group)
            logger.info(
                "Launch time for op %s: %s ns (%.6f ms) over %d group members",
                trace_comm_index,
                interval_ns,
                interval_ns / MS_TO_NS,
                len(group.start_points),
            )
        target_time_ns = self._iteration_start_times[iteration] + interval_ns

        # Sleep until the planned instant; a negative remainder means the op is
        # already behind schedule and fires immediately.
        sleep_ns = target_time_ns - time.perf_counter_ns()
        if sleep_ns > 0:
            time.sleep(sleep_ns / S_TO_NS)
            logger.info("Slept %s ms for op %s", sleep_ns / MS_TO_NS, trace_comm_index)

        if self._enable_online_update:
            # The timing context measures the op so the completion worker can broadcast
            # its real duration; with online updates off nobody listens, so a plain
            # tracked op (completion event only) is enough.
            self._replay_tracked_op(op, TimingContext(trace_comm_index=trace_comm_index))
        else:
            self._replay_tracked_op(op)


def launch_replayer(
    sandbox_ranks: list[int],
    trace_dir: str,
    graph_dir: str | None,
    profiler_dir: str | None,
    pp_rank_stride: int | None = None,
    enable_online_update: bool = False,
    graph_cache_path: str | None = None,
):
    """Launch the assistant replayer on modified NCCL.

    Wiring order matters: the default group is initialized first (NCCL bootstrap
    needs every rank to arrive), then the traced groups are replayed in order,
    then the MemoryManager client connects (it waits for the manager's init-done
    key), and only then is the trace parsed into the replay op queue. Replaying
    goes through replay_immediate() when neither a graph nor a cache is given,
    through replay_with_graph() for graph-derived launch timing, or through
    replay_with_graph_cache() to read the same launch timing from a prebuilt
    cache while loading no graph at all.

    Args:
        sandbox_ranks: List of ranks running in sandbox (not emulated by assistant).
        trace_dir: Directory containing trace files.
        graph_dir: Directory containing graph files for timing control. If None,
            replay immediately without timing control.
        profiler_dir: Directory containing runtime_profiler_{rank}.json.gz. If None,
            durations are left as 0.
        pp_rank_stride: Rank distance between adjacent PP stages, used to enumerate PP chains.
        enable_online_update: If True, apply online duration updates from Redis broadcasts.
            Mutually exclusive with graph_cache_path: online updates recompute launch
            times from the live graphs, which a static cache cannot represent.
        graph_cache_path: Path of a prebuilt launch-timing cache. Mutually exclusive
            with enable_online_update; replaces loading the graphs entirely.

    Raises:
        ValueError: If graph_dir and graph_cache_path are both given, or neither
            enable_online_update nor graph_cache_path selects a timing source when a
            graph or cache path is present, or online updates and the cache are
            requested together.
    """
    if graph_cache_path is not None and enable_online_update:
        raise ValueError(
            "--graph-cache-path and --enable-online-update are mutually exclusive: online updates "
            "recompute launch times from the live graphs, which a static cache cannot represent"
        )

    if graph_cache_path is not None and graph_dir is not None:
        raise ValueError("Give either --graph-dir or --graph-cache-path, not both")

    # Parse trace file to get comm groups and operations to replay
    trace_parser = CrystalTraceParser(f"{trace_dir}/trace_{RANK}.log.gz")

    assistant_process = AssistantProcess(sandbox_ranks)
    assistant_process.default_group_init()
    assistant_process.comm_groups_init(trace_parser.comm_groups_ordered_list)
    assistant_process.setup_memory_manager_client()
    assistant_process.op_queue = list(trace_parser.replay_ops_ordered_list)
    logger.info("Initialized replay op queue with %d ops", len(assistant_process.op_queue))

    if graph_cache_path is not None:
        assistant_process.replay_with_graph_cache(graph_cache_path)
    elif graph_dir is not None:
        assistant_process.replay_with_graph(trace_dir, graph_dir, profiler_dir, pp_rank_stride, enable_online_update)
    else:
        assistant_process.replay_immediate()

    # Keep the process alive: ranks finish their queues at different times, and a
    # rank exiting early would tear down its NCCL/Redis endpoints while other
    # ranks are still replaying towards (or waiting on) it.
    threading.Event().wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox-ranks", required=True, help="Sandbox ranks", type=json.loads)
    parser.add_argument("--trace-dir", required=True, help="Crystal C10D Trace directory", type=str)
    parser.add_argument("--graph-dir", help="Crystal Graph directory", type=str, default=None)
    parser.add_argument(
        "--graph-cache-path",
        help="Prebuilt launch-timing cache file (graph_cache_{timestamp}.json.gz); replaces --graph-dir",
        type=str,
        default=None,
    )
    parser.add_argument("--pp-rank-stride", help="Rank distance between adjacent PP stages", type=int, default=None)
    parser.add_argument(
        "--enable-online-update",
        help="Enable online duration updates from Redis broadcasts",
        action="store_true",
        default=False,
    )
    parser.add_argument("--profiler-dir", help="Crystal runtime profiler directory", type=str, default=None)
    args = parser.parse_args()

    launch_replayer(
        args.sandbox_ranks,
        args.trace_dir,
        args.graph_dir,
        args.profiler_dir,
        args.pp_rank_stride,
        args.enable_online_update,
        args.graph_cache_path,
    )
