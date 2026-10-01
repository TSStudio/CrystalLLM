# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Unit tests for MemoryManager.allocate() (Stage D).

These tests cover real CUDA pool/allocation creation, CUDA IPC export,
dtype-legal random init, and CPU fallback under a tight memory budget. Requires
a CUDA device and a local Redis server.
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
BIG_PERSISTENT_NUM_ELEMENTS = 2_000_000


class _ConstantProvider(TensorProvider):
    """Provides a deterministic, recognizable CPU tensor for every demand it handles."""

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


def _prep_manager_with_persistent_demand(shape, dtype, fill_value=7.0, extra_ops=None):
    group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
    ops = [
        comm(
            comm_op(
                1,
                "all_reduce",
                group_name="g",
                input_shape=shape,
                input_dtype=dtype,
                meta={"reduce_op": torch.distributed.ReduceOp.SUM},
            )
        )
    ]
    if extra_ops:
        ops += extra_ops
    provider = _ConstantProvider(["all_reduce"], TensorMemoryLevel.PERSISTENT, fill_value)
    mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[provider])
    return mgr, provider


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestAnythingPoolAllocation(unittest.TestCase):
    def setUp(self):
        self.local_rank = 0
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def test_float_pool_has_no_nan_or_inf(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[1000], output_dtype=torch.float16, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)

        tensor = mgr._pool_registry["torch.float16"]
        self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertEqual(tensor.device.type, "cuda")

    def test_integer_pool_within_dtype_range(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[1000], output_dtype=torch.int64, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)

        tensor = mgr._pool_registry["torch.int64"]
        info = torch.iinfo(torch.int64)
        self.assertTrue((tensor >= info.min).all().item())
        self.assertTrue((tensor <= info.max).all().item())

    def test_pool_redis_fields_and_ipc_handle_importable(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[16], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)

        pool_key = mm.key_pool_anything(mgr.local_rank, "torch.float32")
        self.assertEqual(mgr.redis.command.get(mm.redis_field(pool_key, "state")).decode(), mm.STATE_READY)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(pool_key, "version"))), 0)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(pool_key, "user_count"))), 0)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(pool_key, "max_elements"))), 16)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(pool_key, "nbytes"))), 16 * 4)

        imported = _import_ipc_tensor_in_subprocess(mgr.local_rank, pool_key, [16], "torch.float32")
        self.assertEqual(imported, mgr._pool_registry["torch.float32"].cpu().tolist())

    def test_pool_allocation_fails_hard_when_budget_too_small(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[1_000_000], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        with self.assertRaises(RuntimeError):
            mgr.allocate(total_bytes=1024)  # way too small


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestPersistentSharedAllocation(unittest.TestCase):
    def setUp(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def test_persistent_allocation_has_provided_value_on_gpu(self):
        mgr, provider = _prep_manager_with_persistent_demand([8], torch.float32, fill_value=3.5)
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)

        self.assertEqual(len(mgr.allocation_plans), 1)
        plan = mgr.allocation_plans[0]
        tensor = mgr._allocation_registry[plan.key]
        self.assertTrue(torch.equal(tensor, torch.full([8], 3.5, device=tensor.device)))
        self.assertEqual(len(provider.provide_calls), 1)

        state = mgr.redis.command.get(mm.redis_field(plan.key, "state")).decode()
        self.assertEqual(state, mm.STATE_READY)

    def test_shared_allocation_calls_provide_only_once(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)
        op_info = comm_op(1, "broadcast", group_name="g", input_shape=[4], input_dtype=torch.float32, meta={"src": 0})
        provider = _ConstantProvider(["broadcast"], TensorMemoryLevel.SHARED, fill_value=9.0)
        mgr = MemoryManager(
            local_rank=0,
            world_size=3,
            sandbox_ranks=[2],
            rank_comm_groups={0: [group], 1: [group]},
            rank_replay_ops={0: [comm(op_info)], 1: [comm(op_info)]},
            tensor_providers=[provider],
        )
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)

        self.assertEqual(len(provider.provide_calls), 1)
        self.assertEqual(len(mgr.allocation_plans), 1)
        tensor = mgr._allocation_registry[mgr.allocation_plans[0].key]
        self.assertTrue(torch.equal(tensor, torch.full([4], 9.0, device=tensor.device)))

    def test_allocation_ipc_handle_importable_with_correct_value(self):
        mgr, provider = _prep_manager_with_persistent_demand([4, 4], torch.float16, fill_value=1.25)
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=10 * 1024**3)

        plan = mgr.allocation_plans[0]
        imported = _import_ipc_tensor_in_subprocess(mgr.local_rank, plan.key, [16], "torch.float16")
        self.assertEqual(imported, [1.25] * 16)

    def test_budget_exceeded_falls_back_to_cpu_backup(self):
        # A single large PERSISTENT allocation that cannot possibly fit in the
        # (artificially tiny) budget must be kept on CPU instead of crashing.
        mgr, provider = _prep_manager_with_persistent_demand([1_000_000], torch.float32, fill_value=2.0)
        mgr.build_demands()
        mgr.plan_resources()
        mgr.allocate(total_bytes=1024)  # tiny budget: even reserved ratio leaves ~ nothing

        plan = mgr.allocation_plans[0]
        self.assertNotIn(plan.key, mgr._allocation_registry)
        self.assertIn(plan.key, mgr._cpu_backup_registry)
        cpu_tensor = mgr._cpu_backup_registry[plan.key]
        self.assertEqual(cpu_tensor.device.type, "cpu")
        self.assertTrue(torch.equal(cpu_tensor, torch.full([1_000_000], 2.0)))

        state = mgr.redis.command.get(mm.redis_field(plan.key, "state")).decode()
        handle = mgr.redis.command.get(mm.redis_field(plan.key, "cuda_ipc_handle")).decode()
        self.assertEqual(state, mm.STATE_EVICTED)
        self.assertEqual(handle, mm.PLACEHOLDER)

    def test_small_allocations_fit_even_when_large_one_does_not(self):
        # Allocation order is smallest-first, so a small PERSISTENT demand should
        # land on GPU even if a much larger one (planned earlier by seq) must fall
        # back to CPU due to the shared budget.
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        small_op = comm_op(
            1,
            "all_reduce",
            group_name="g",
            input_shape=[SMALL_PERSISTENT_NUM_ELEMENTS],
            input_dtype=torch.float32,
            meta={"reduce_op": torch.distributed.ReduceOp.SUM},
            signature="small",
        )
        big_op = comm_op(
            2,
            "all_reduce",
            group_name="g",
            input_shape=[BIG_PERSISTENT_NUM_ELEMENTS],
            input_dtype=torch.float32,
            meta={"reduce_op": torch.distributed.ReduceOp.SUM},
            signature="big",
        )
        provider = _ConstantProvider(["all_reduce"], TensorMemoryLevel.PERSISTENT, fill_value=1.0)
        mgr = make_manager(2, [1], {0: [group]}, {0: [comm(big_op), comm(small_op)]}, tensor_providers=[provider])
        mgr.build_demands()
        mgr.plan_resources()

        # Budget fits the small allocation plus headroom, but not the big one.
        small_plan = next(p for p in mgr.allocation_plans if p.num_elements == SMALL_PERSISTENT_NUM_ELEMENTS)
        big_plan = next(p for p in mgr.allocation_plans if p.num_elements == BIG_PERSISTENT_NUM_ELEMENTS)
        budget_total_bytes = int((small_plan.nbytes + 4096) / (1 - mgr.reserved_memory_ratio))
        mgr.allocate(total_bytes=budget_total_bytes)

        self.assertIn(small_plan.key, mgr._allocation_registry)
        self.assertIn(big_plan.key, mgr._cpu_backup_registry)


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, SKIP_REASON)
class TestMarkInitDone(unittest.TestCase):
    def test_init_done_set_after_allocate(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()
        ops = [comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        mgr.write_scaffold_to_redis()
        mgr.allocate(total_bytes=10 * 1024**3)
        self.assertIsNone(mgr.redis.command.get(mm.key_init_done(mgr.local_rank)))
        mgr.mark_init_done()
        self.assertEqual(mgr.redis.command.get(mm.key_init_done(mgr.local_rank)).decode(), "1")
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()


if __name__ == "__main__":
    unittest.main()
