# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Unit tests for MemoryManager resource planning and Redis scaffold (Stage C).

Pure Python planning tests need no external services. Scaffold-writing tests
need a local Redis server (same as test_memory_manager.py) but never touch CUDA.
"""

import unittest

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


def _redis_available() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB)
        client.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False


REDIS_AVAILABLE = _redis_available()


class _AllReduceInstantProvider(TensorProvider):
    def can_handle(self, rank, replay_op, comm_group):
        return replay_op.c10d_func_name == "all_reduce"

    def tensor_memory_level(self, rank, replay_op, comm_group):
        return TensorMemoryLevel.INSTANT

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        raise NotImplementedError


class _BroadcastSharedProvider(TensorProvider):
    def can_handle(self, rank, replay_op, comm_group):
        return replay_op.c10d_func_name == "broadcast"

    def tensor_memory_level(self, rank, replay_op, comm_group):
        return TensorMemoryLevel.SHARED

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        raise NotImplementedError


class _PersistentProvider(TensorProvider):
    def can_handle(self, rank, replay_op, comm_group):
        return replay_op.c10d_func_name == "all_reduce"

    def tensor_memory_level(self, rank, replay_op, comm_group):
        return TensorMemoryLevel.PERSISTENT

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        raise NotImplementedError


class TestDtypeHelpers(unittest.TestCase):
    def test_dtype_round_trip(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.uint8):
            dtype_str = mm.dtype_to_str(dtype)
            self.assertEqual(mm.dtype_from_str(dtype_str), dtype)

    def test_itemsize_matches_torch(self):
        self.assertEqual(mm.dtype_itemsize("torch.float32"), 4)
        self.assertEqual(mm.dtype_itemsize("torch.float16"), 2)
        self.assertEqual(mm.dtype_itemsize("torch.bfloat16"), 2)
        self.assertEqual(mm.dtype_itemsize("torch.int64"), 8)
        self.assertEqual(mm.dtype_itemsize("torch.bool"), 1)
        self.assertEqual(mm.dtype_itemsize("torch.uint8"), 1)

    def test_shape_numel(self):
        self.assertEqual(mm.shape_numel([]), 1)
        self.assertEqual(mm.shape_numel([4]), 4)
        self.assertEqual(mm.shape_numel([4, 8]), 32)
        self.assertEqual(mm.shape_numel([1, 1, 1]), 1)


class TestAnythingPoolPlanning(unittest.TestCase):
    def test_pool_sized_to_max_elements_across_demands(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(comm_op(1, "irecv", group_name="g", output_shape=[4, 4], output_dtype=torch.float32, meta={"src": 1})),
            comm(comm_op(2, "irecv", group_name="g", output_shape=[10], output_dtype=torch.float32, meta={"src": 1})),
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()

        self.assertEqual(len(mgr.pool_plans), 1)
        pool = mgr.pool_plans["torch.float32"]
        self.assertEqual(pool.max_elements, 16)  # max(4*4, 10)
        self.assertEqual(pool.nbytes, 16 * 4)
        self.assertEqual(len(pool.demand_ids), 2)

    def test_different_dtypes_get_isolated_pools(self):
        ops = [
            comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1})),
            comm(comm_op(2, "irecv", output_shape=[4], output_dtype=torch.float16, meta={"src": 1})),
        ]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        self.assertEqual(set(mgr.pool_plans.keys()), {"torch.float32", "torch.float16"})

    def test_all_demands_point_to_pool_key(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        mgr.plan_resources()
        demand_id = next(iter(mgr.demands))
        self.assertEqual(mgr.demand_target_ref[demand_id], mm.key_pool_anything(mgr.local_rank, "torch.float32"))


class TestSharedAllocationPlanning(unittest.TestCase):
    def test_demands_with_same_share_key_map_to_one_allocation(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)
        op_info = comm_op(1, "broadcast", group_name="g", input_shape=[4], input_dtype=torch.float32, meta={"src": 0})
        mgr = MemoryManager(
            local_rank=0,
            world_size=3,
            sandbox_ranks=[2],
            rank_comm_groups={0: [group], 1: [group]},
            rank_replay_ops={0: [comm(op_info)], 1: [comm(op_info)]},
            tensor_providers=[_BroadcastSharedProvider()],
        )
        mgr.build_demands()
        mgr.plan_resources()

        self.assertEqual(len(mgr.allocation_plans), 1)
        plan = mgr.allocation_plans[0]
        self.assertEqual(plan.owner_kind, mm.OWNER_KIND_SHARED)
        self.assertEqual(len(plan.demand_ids), 2)
        self.assertEqual(plan.num_elements, 4)

        # Representative is the lowest-rank demand.
        rank_of_representative, *_ = mm.decode_demand_id(plan.representative_demand_id)
        self.assertEqual(rank_of_representative, 0)

        for demand_id in plan.demand_ids:
            self.assertEqual(mgr.demand_target_ref[demand_id], plan.key)

    def test_distinct_share_keys_get_distinct_allocations(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "broadcast",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"src": 0},
                    signature="broadcast(shape=4)",
                )
            ),
            comm(
                comm_op(
                    2,
                    "broadcast",
                    group_name="g",
                    input_shape=[8],
                    input_dtype=torch.float32,
                    meta={"src": 0},
                    signature="broadcast(shape=8)",
                )
            ),
        ]
        mgr = make_manager(3, [2], {0: [group]}, {0: ops}, tensor_providers=[_BroadcastSharedProvider()])
        mgr.build_demands()
        mgr.plan_resources()
        self.assertEqual(len(mgr.allocation_plans), 2)
        keys = {p.key for p in mgr.allocation_plans}
        self.assertEqual(len(keys), 2)


class TestPersistentAllocationPlanning(unittest.TestCase):
    def test_each_persistent_demand_gets_its_own_allocation(self):
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
                    signature=f"all_reduce#{i}",
                )
            )
            for i in range(1, 3)
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[_PersistentProvider()])
        mgr.build_demands()
        mgr.plan_resources()

        self.assertEqual(len(mgr.allocation_plans), 2)
        for plan in mgr.allocation_plans:
            self.assertEqual(plan.owner_kind, mm.OWNER_KIND_PERSISTENT)
            self.assertEqual(len(plan.demand_ids), 1)


class TestAllocationSizeOrdering(unittest.TestCase):
    def test_allocation_plans_by_size_asc_orders_smallest_first(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "all_reduce",
                    group_name="g",
                    input_shape=[100],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="big",
                )
            ),
            comm(
                comm_op(
                    2,
                    "all_reduce",
                    group_name="g",
                    input_shape=[2],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="small",
                )
            ),
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[_PersistentProvider()])
        mgr.build_demands()
        mgr.plan_resources()
        ordered = mgr.allocation_plans_by_size_asc()
        self.assertEqual([p.nbytes for p in ordered], sorted(p.nbytes for p in mgr.allocation_plans))
        self.assertEqual(ordered[0].nbytes, 2 * 4)
        self.assertEqual(ordered[-1].nbytes, 100 * 4)


class TestInstantPlanning(unittest.TestCase):
    def test_total_use_count_matches_occurrence_count(self):
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
            for i in range(1, 6)
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[_AllReduceInstantProvider()])
        mgr.build_demands()
        mgr.plan_resources()

        self.assertEqual(len(mgr.instant_plans), 1)
        plan = next(iter(mgr.instant_plans.values()))
        self.assertEqual(plan.total_use_count, 5)

        demand_id = next(iter(mgr.demands))
        self.assertEqual(mgr.demand_target_ref[demand_id], mm.key_instant_prefix(mgr.local_rank, demand_id))

    def test_no_pool_or_allocation_created_for_instant(self):
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
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[_AllReduceInstantProvider()])
        mgr.build_demands()
        mgr.plan_resources()
        self.assertEqual(len(mgr.pool_plans), 0)
        self.assertEqual(len(mgr.allocation_plans), 0)


class TestAllocatableBytes(unittest.TestCase):
    def test_default_reserved_ratio(self):
        mgr = make_manager(2, [], {0: []}, {0: []})
        self.assertEqual(mgr.reserved_memory_ratio, mm.RESERVED_MEMORY_RATIO_DEFAULT)
        self.assertEqual(mgr.compute_allocatable_bytes(1000), 900)

    def test_custom_reserved_ratio(self):
        mgr = MemoryManager(
            local_rank=0,
            world_size=2,
            sandbox_ranks=[],
            rank_comm_groups={0: []},
            rank_replay_ops={0: []},
            reserved_memory_ratio=0.25,
        )
        self.assertEqual(mgr.compute_allocatable_bytes(1000), 750)


@unittest.skipUnless(REDIS_AVAILABLE, "requires a local redis-server on localhost:6379")
class TestWriteScaffoldToRedis(unittest.TestCase):
    def setUp(self):
        self.group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)

    def _build_manager(self, ops, tensor_providers=None, sandbox_ranks=None):
        mgr = make_manager(
            2,
            sandbox_ranks if sandbox_ranks is not None else [1],
            {0: [self.group]},
            {0: ops},
            tensor_providers=tensor_providers,
        )
        mgr.redis.command.flushdb()
        return mgr

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def test_demand_level_and_target_ref_written(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = self._build_manager(ops)
        mgr.build_demands()
        mgr.plan_resources()
        mgr.write_scaffold_to_redis()

        demand_id = next(iter(mgr.demands))
        level = mgr.redis.command.get(mm.key_demand_level(mgr.local_rank, demand_id)).decode()
        target_ref = mgr.redis.command.get(mm.key_demand_target_ref(mgr.local_rank, demand_id)).decode()
        self.assertEqual(level, "ANYTHING")
        self.assertEqual(target_ref, mm.key_pool_anything(mgr.local_rank, "torch.float32"))

    def test_instant_use_placeholders_written(self):
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
            for i in range(1, 4)
        ]
        mgr = self._build_manager(ops, tensor_providers=[_AllReduceInstantProvider()])
        mgr.build_demands()
        mgr.plan_resources()
        mgr.write_scaffold_to_redis()

        demand_id = next(iter(mgr.demands))
        self.assertEqual(int(mgr.redis.command.get(mm.key_instant_next_use_index(mgr.local_rank, demand_id))), 0)
        self.assertEqual(int(mgr.redis.command.get(mm.key_instant_total_use_count(mgr.local_rank, demand_id))), 3)
        for use_index in range(3):
            value = mgr.redis.command.get(mm.key_instant_use_allocation_ref(mgr.local_rank, demand_id, use_index))
            self.assertEqual(value.decode(), mm.PLACEHOLDER)

    def test_init_managed_ranks_and_ratio_written(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = self._build_manager(ops)
        mgr.build_demands()
        mgr.plan_resources()
        mgr.write_scaffold_to_redis()

        ranks = [int(x) for x in mgr.redis.command.lrange(mm.key_init_managed_ranks(mgr.local_rank), 0, -1)]
        self.assertEqual(ranks, [0])
        ratio = float(mgr.redis.command.get(mm.key_init_reserved_memory_ratio(mgr.local_rank)))
        self.assertAlmostEqual(ratio, mm.RESERVED_MEMORY_RATIO_DEFAULT)

    def test_init_done_not_set_by_scaffold_alone(self):
        ops = [comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = self._build_manager(ops)
        mgr.build_demands()
        mgr.plan_resources()
        mgr.write_scaffold_to_redis()
        self.assertIsNone(mgr.redis.command.get(mm.key_init_done(mgr.local_rank)))

    def test_no_demands_writes_empty_scaffold_without_error(self):
        mgr = self._build_manager([])
        mgr.build_demands()
        mgr.plan_resources()
        mgr.write_scaffold_to_redis()
        ranks = mgr.redis.command.lrange(mm.key_init_managed_ranks(mgr.local_rank), 0, -1)
        self.assertEqual(ranks, [b"0"])


if __name__ == "__main__":
    unittest.main()
