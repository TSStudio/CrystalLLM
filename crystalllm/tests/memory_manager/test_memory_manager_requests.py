# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Unit tests for MemoryManager runtime request processing.

These tests cover request handling and eviction cascade behavior (Stage E).
Requires a CUDA device and a local Redis server.
"""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import redis
import torch

from crystalllm.replay import memory_manager as mm
from crystalllm.replay.data import (
    ReplayerCommGroupInfo,
    ReplayerCommOpInfo,
    ReplayerReplayOpInfo,
    ReplayerReplayOpType,
    TensorMemoryLevel,
    TensorProvider,
)
from crystalllm.replay.memory_manager import MemoryManager


# Local copies of the trace-building helpers (duplicated across the
# test_memory_manager_* files on purpose) so this file runs standalone from any
# working directory, without importing from sibling test files.
def comm_op(
    trace_comm_index,
    c10d_func_name,
    group_name="0",
    group_desc="0",
    output_shape=None,
    output_dtype=None,
    input_shape=None,
    input_dtype=None,
    async_op=False,
    meta=None,
    signature=None,
):
    return ReplayerCommOpInfo(
        trace_comm_index=trace_comm_index,
        c10d_func_name=c10d_func_name,
        signature=signature if signature is not None else f"{c10d_func_name}()",
        group_name=group_name,
        group_desc=group_desc,
        output_shape=output_shape,
        output_dtype=output_dtype,
        input_shape=input_shape,
        input_dtype=input_dtype,
        async_op=async_op,
        meta=meta or {},
    )


def comm(op_info):
    return ReplayerReplayOpInfo(ReplayerReplayOpType.COMM, op_info)


def make_manager(world_size, sandbox_ranks, rank_comm_groups, rank_replay_ops, tensor_providers=None):
    return MemoryManager(
        local_rank=0,
        world_size=world_size,
        sandbox_ranks=sandbox_ranks,
        rank_comm_groups=rank_comm_groups,
        rank_replay_ops=rank_replay_ops,
        tensor_providers=tensor_providers,
    )


def _import_ipc_tensor_in_subprocess(local_rank: int, resource_key: str, shape: list[int], dtype_str: str) -> list:
    """Import a CUDA IPC handle from Redis in a fresh process and return its values.

    CUDA IPC handles cannot be re-imported within the exporting process itself,
    so this spawns a genuinely separate python process (mirroring what a real
    assistant process would do) and returns the tensor contents via stdout.
    """
    script = textwrap.dedent(f"""
        import pickle
        import redis
        import torch

        torch.cuda.set_device({local_rank})
        rc = redis.Redis(host="localhost", port=6379, db={mm.MEMORY_MANAGER_REDIS_DB})
        handle_bytes = rc.get("{resource_key}:cuda_ipc_handle")
        handle = pickle.loads(handle_bytes)
        storage = torch.UntypedStorage._new_shared_cuda(*handle)
        t = torch.empty(0, dtype={dtype_str}, device="cuda:{local_rank}")
        t.set_(storage, 0, {shape!r})
        torch.cuda.synchronize()
        print(t.cpu().tolist())
    """)
    with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=60, check=False)
    finally:
        Path(script_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"IPC import subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return eval(result.stdout.strip())


def _redis_available() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB)
        client.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False


REDIS_AVAILABLE = _redis_available()
CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "requires CUDA + local redis-server on localhost:6379"
SMALL_PERSISTENT_NUM_ELEMENTS = 4
BIG_PERSISTENT_NUM_ELEMENTS = 1000


class _ConstantProvider(TensorProvider):
    def __init__(self, func_names, level, fill_value=7.0):
        self.func_names = set(func_names)
        self.level = level
        self.fill_value = fill_value
        self.provide_calls = []

    def can_handle(self, rank, replay_op, comm_group):
        return replay_op.c10d_func_name in self.func_names

    def tensor_memory_level(self, rank, replay_op, comm_group):
        return self.level

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        self.provide_calls.append((rank, replay_op.trace_comm_index, use_index))
        return torch.full(replay_op.input_shape, self.fill_value, dtype=replay_op.input_dtype)


class _UseIndexValueProvider(TensorProvider):
    """INSTANT provider whose value depends on use_index, to verify per-use fidelity."""

    def can_handle(self, rank, replay_op, comm_group):
        return replay_op.c10d_func_name == "all_reduce"

    def tensor_memory_level(self, rank, replay_op, comm_group):
        return TensorMemoryLevel.INSTANT

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        return torch.full(replay_op.input_shape, float(use_index), dtype=replay_op.input_dtype)


def _wait_for_pubsub_message(pubsub, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
        if msg is not None:
            return msg
    return None


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestReadyFastPath(unittest.TestCase):
    def setUp(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def test_ready_persistent_request_increments_user_count_without_realloc(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "all_reduce",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            )
        ]
        provider = _ConstantProvider(["all_reduce"], TensorMemoryLevel.PERSISTENT, fill_value=5.0)
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[provider])
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)
        mgr.write_scaffold_to_redis()
        mgr.mark_init_done()

        plan = mgr.allocation_plans[0]
        original_tensor = mgr._allocation_registry[plan.key]

        pubsub = mgr.redis.new_pubsub()
        pubsub.subscribe(mm.key_rank_events_channel(mgr.local_rank, 0))
        pubsub.get_message(timeout=1)  # consume subscribe confirmation

        demand_id = plan.representative_demand_id
        request_id = mm.submit_request(mgr.redis, mgr.local_rank, rank=0, request_seq=1, demand_id=demand_id)
        mgr.process_request(request_id)

        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(plan.key, "user_count"))), 1)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(plan.key, "version"))), 0)
        self.assertIs(mgr._allocation_registry[plan.key], original_tensor)  # no realloc

        target_ref_debug = mgr.redis.command.get(mm.key_request_target_ref(mgr.local_rank, request_id)).decode()
        self.assertEqual(target_ref_debug, plan.key)

        msg = _wait_for_pubsub_message(pubsub)
        self.assertIsNotNone(msg)
        self.assertEqual(msg["data"].decode(), request_id)


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestEvictedReload(unittest.TestCase):
    def setUp(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def test_evicted_persistent_reload_restores_value_and_bumps_version(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "all_reduce",
                    group_name="g",
                    input_shape=[1_000_000],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            )
        ]
        provider = _ConstantProvider(["all_reduce"], TensorMemoryLevel.PERSISTENT, fill_value=4.5)
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[provider])
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=1024)  # too small: allocation lands on CPU (EVICTED) immediately

        plan = mgr.allocation_plans[0]
        self.assertEqual(mgr.redis.command.get(mm.redis_field(plan.key, "state")).decode(), mm.STATE_EVICTED)
        mgr.write_scaffold_to_redis()
        mgr.mark_init_done()

        # Give it a real budget for the runtime reload to succeed.
        mgr.allocatable_bytes = 10 * 1024**3

        request_id = mm.submit_request(
            mgr.redis, mgr.local_rank, rank=0, request_seq=1, demand_id=plan.representative_demand_id
        )
        mgr.process_request(request_id)

        self.assertEqual(mgr.redis.command.get(mm.redis_field(plan.key, "state")).decode(), mm.STATE_READY)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(plan.key, "version"))), 1)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(plan.key, "user_count"))), 1)
        self.assertNotIn(plan.key, mgr._cpu_backup_registry)
        self.assertIn(plan.key, mgr._allocation_registry)

        imported = _import_ipc_tensor_in_subprocess(mgr.local_rank, plan.key, [1_000_000], "torch.float32")
        self.assertEqual(imported[:5], [4.5] * 5)
        self.assertEqual(len(imported), 1_000_000)


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestInstantLifecycle(unittest.TestCase):
    def setUp(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def _build_instant_manager(self, total_use_count):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    i,
                    "all_reduce",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            )
            for i in range(1, total_use_count + 1)
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[_UseIndexValueProvider()])
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)
        mgr.write_scaffold_to_redis()
        mgr.mark_init_done()
        return mgr

    def test_each_use_creates_distinct_allocation_with_correct_value(self):
        mgr = self._build_instant_manager(total_use_count=3)
        demand_id = next(iter(mgr.demands))

        alloc_keys = []
        for use_index in range(3):
            request_id = mm.submit_request(
                mgr.redis, mgr.local_rank, rank=0, request_seq=use_index, demand_id=demand_id, use_index=use_index
            )
            mgr.process_request(request_id)
            alloc_ref = mgr.redis.command.get(
                mm.key_instant_use_allocation_ref(mgr.local_rank, demand_id, use_index)
            ).decode()
            alloc_keys.append(alloc_ref)

            tensor = mgr._allocation_registry[alloc_ref]
            self.assertTrue(torch.equal(tensor, torch.full([4], float(use_index), device=tensor.device)))
            self.assertEqual(mgr.redis.command.get(mm.redis_field(alloc_ref, "state")).decode(), mm.STATE_READY)
            self.assertEqual(int(mgr.redis.command.get(mm.redis_field(alloc_ref, "user_count"))), 1)

        self.assertEqual(len(set(alloc_keys)), 3)  # all distinct
        self.assertEqual(int(mgr.redis.command.get(mm.key_instant_next_use_index(mgr.local_rank, demand_id))), 3)

    def test_use_index_mismatch_raises(self):
        mgr = self._build_instant_manager(total_use_count=2)
        demand_id = next(iter(mgr.demands))
        # Skip ahead to use_index=1 without having consumed use_index=0 first.
        request_id = mm.submit_request(
            mgr.redis, mgr.local_rank, rank=0, request_seq=1, demand_id=demand_id, use_index=1
        )
        with self.assertRaises(RuntimeError):
            mgr.process_request(request_id)


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestEvictionCascadeOrder(unittest.TestCase):
    def setUp(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def _build_manager_with_two_persistent(self, small_elems, big_elems):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        small_op = comm_op(
            1,
            "all_reduce",
            group_name="g",
            input_shape=[small_elems],
            input_dtype=torch.float32,
            meta={"reduce_op": torch.distributed.ReduceOp.SUM},
            signature="small",
        )
        big_op = comm_op(
            2,
            "all_reduce",
            group_name="g",
            input_shape=[big_elems],
            input_dtype=torch.float32,
            meta={"reduce_op": torch.distributed.ReduceOp.SUM},
            signature="big",
        )
        provider = _ConstantProvider(["all_reduce"], TensorMemoryLevel.PERSISTENT, fill_value=1.0)
        mgr = make_manager(2, [1], {0: [group]}, {0: [comm(small_op), comm(big_op)]}, tensor_providers=[provider])
        mgr.build_demands()
        mgr.plan_resources()
        return mgr

    def test_idle_instant_destroyed_before_touching_persistent(self):
        mgr = self._build_manager_with_two_persistent(small_elems=4, big_elems=8)
        mgr.allocate(total_bytes=10 * 1024**3)
        mgr.write_scaffold_to_redis()

        # Manually create an idle INSTANT allocation to simulate one already used
        # and released (assistant-side completion tracking isn't implemented yet).
        instant_key = mm.key_allocation(mgr.local_rank, "instant", 999)
        instant_tensor = torch.empty(10_000_000, dtype=torch.float32, device=f"cuda:{mgr.local_rank}")
        mgr._allocation_registry[instant_key] = instant_tensor
        mgr._live_instant_allocations.add(instant_key)
        instant_nbytes = 10_000_000 * 4
        mgr.used_bytes += instant_nbytes
        mgr.redis.command.set(mm.redis_field(instant_key, "nbytes"), instant_nbytes)
        mgr.redis.command.set(mm.redis_field(instant_key, "user_count"), 0)
        mgr.redis.command.set(mm.redis_field(instant_key, "state"), mm.STATE_READY)

        persistent_plans = {p.num_elements: p for p in mgr.allocation_plans}
        for plan in mgr.allocation_plans:
            mgr.redis.command.set(mm.redis_field(plan.key, "user_count"), 0)

        # Cap the budget so used_bytes currently sits right at the limit, forcing
        # eviction to free exactly the instant allocation's worth of memory.
        mgr.allocatable_bytes = mgr.used_bytes

        mgr._ensure_budget(instant_nbytes)

        self.assertEqual(mgr.redis.command.get(mm.redis_field(instant_key, "state")).decode(), mm.STATE_DESTROYED)
        self.assertNotIn(instant_key, mgr._live_instant_allocations)
        for plan in persistent_plans.values():
            self.assertEqual(mgr.redis.command.get(mm.redis_field(plan.key, "state")).decode(), mm.STATE_READY)

    def test_persistent_evicted_largest_first(self):
        mgr = self._build_manager_with_two_persistent(
            small_elems=SMALL_PERSISTENT_NUM_ELEMENTS, big_elems=BIG_PERSISTENT_NUM_ELEMENTS
        )
        mgr.allocate(total_bytes=10 * 1024**3)
        mgr.write_scaffold_to_redis()

        small_plan = next(p for p in mgr.allocation_plans if p.num_elements == SMALL_PERSISTENT_NUM_ELEMENTS)
        big_plan = next(p for p in mgr.allocation_plans if p.num_elements == BIG_PERSISTENT_NUM_ELEMENTS)
        for plan in mgr.allocation_plans:
            mgr.redis.command.set(mm.redis_field(plan.key, "user_count"), 0)

        # Need exactly big_plan's worth of memory freed; evicting the big one alone
        # must be enough (evict-largest-first should never touch the small one).
        mgr.allocatable_bytes = mgr.used_bytes

        mgr._ensure_budget(big_plan.nbytes)

        self.assertEqual(mgr.redis.command.get(mm.redis_field(big_plan.key, "state")).decode(), mm.STATE_EVICTED)
        self.assertEqual(mgr.redis.command.get(mm.redis_field(small_plan.key, "state")).decode(), mm.STATE_READY)
        self.assertIn(big_plan.key, mgr._cpu_backup_registry)

    def test_busy_persistent_is_skipped_in_favor_of_idle_shared(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)
        persistent_op = comm_op(
            1,
            "all_reduce",
            group_name="g",
            input_shape=[1000],
            input_dtype=torch.float32,
            meta={"reduce_op": torch.distributed.ReduceOp.SUM},
        )
        shared_op = comm_op(
            2,
            "broadcast",
            group_name="g",
            input_shape=[1000],
            input_dtype=torch.float32,
            meta={"src": 0},
        )
        persistent_provider = _ConstantProvider(["all_reduce"], TensorMemoryLevel.PERSISTENT, fill_value=1.0)
        shared_provider = _ConstantProvider(["broadcast"], TensorMemoryLevel.SHARED, fill_value=2.0)
        mgr = MemoryManager(
            local_rank=0,
            world_size=3,
            sandbox_ranks=[2],
            rank_comm_groups={0: [group]},
            rank_replay_ops={0: [comm(persistent_op), comm(shared_op)]},
            tensor_providers=[persistent_provider, shared_provider],
        )
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)
        mgr.write_scaffold_to_redis()

        persistent_plan = next(p for p in mgr.allocation_plans if p.owner_kind == mm.OWNER_KIND_PERSISTENT)
        shared_plan = next(p for p in mgr.allocation_plans if p.owner_kind == mm.OWNER_KIND_SHARED)

        # Persistent resource is busy (in use); only the shared one is eligible.
        mgr.redis.command.set(mm.redis_field(persistent_plan.key, "user_count"), 1)
        mgr.redis.command.set(mm.redis_field(shared_plan.key, "user_count"), 0)
        mgr.allocatable_bytes = mgr.used_bytes

        mgr._ensure_budget(shared_plan.nbytes)

        self.assertEqual(mgr.redis.command.get(mm.redis_field(persistent_plan.key, "state")).decode(), mm.STATE_READY)
        self.assertEqual(mgr.redis.command.get(mm.redis_field(shared_plan.key, "state")).decode(), mm.STATE_EVICTED)

    def test_raises_when_nothing_left_to_evict(self):
        mgr = self._build_manager_with_two_persistent(small_elems=4, big_elems=8)
        mgr.allocate(total_bytes=10 * 1024**3)
        mgr.write_scaffold_to_redis()
        for plan in mgr.allocation_plans:
            mgr.redis.command.set(mm.redis_field(plan.key, "user_count"), 1)  # everything busy
        mgr.allocatable_bytes = mgr.used_bytes

        with self.assertRaises(RuntimeError):
            mgr._ensure_budget(1024 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
