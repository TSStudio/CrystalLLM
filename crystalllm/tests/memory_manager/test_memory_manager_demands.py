# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Unit tests for MemoryManager.build_demands() (Stage B).

These tests cover op filtering, direction/level assignment, and TensorProvider
matching. No CUDA/Redis needed.
"""

import unittest

import torch

from crystalllm.replay.data import (
    ReplayerCommGroupInfo,
    ReplayerCommOpInfo,
    ReplayerReplayOpInfo,
    ReplayerReplayOpType,
    TensorMemoryLevel,
    TensorProvider,
)
from crystalllm.replay.memory_manager import MEMORY_TENSOR_INPUT, MEMORY_TENSOR_OUTPUT, MemoryManager


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


def iteration_begin(iteration=1):
    from crystalllm.replay.data import ReplayerScheduleInfo

    return ReplayerReplayOpInfo(ReplayerReplayOpType.ITERATION_BEGIN, ReplayerScheduleInfo(iteration))


def coalescing_start(trace_comm_index, group_name="0", group_desc="0", device_set=False, async_ops=False):
    from crystalllm.replay.data import ReplayerCoalescingInfo

    return ReplayerReplayOpInfo(
        ReplayerReplayOpType.COALESCING_START,
        ReplayerCoalescingInfo(trace_comm_index, group_name, group_desc, device_set, async_ops),
    )


def coalescing_end(trace_comm_index, group_name="0", group_desc="0", device_set=False, async_ops=False):
    from crystalllm.replay.data import ReplayerCoalescingInfo

    return ReplayerReplayOpInfo(
        ReplayerReplayOpType.COALESCING_END,
        ReplayerCoalescingInfo(trace_comm_index, group_name, group_desc, device_set, async_ops),
    )


def make_manager(world_size, sandbox_ranks, rank_comm_groups, rank_replay_ops, tensor_providers=None):
    return MemoryManager(
        local_rank=0,
        world_size=world_size,
        sandbox_ranks=sandbox_ranks,
        rank_comm_groups=rank_comm_groups,
        rank_replay_ops=rank_replay_ops,
        tensor_providers=tensor_providers,
    )


class TestUnknownGroupSkip(unittest.TestCase):
    def test_op_for_unknown_group_generates_no_demand(self):
        ops = [
            comm(
                comm_op(
                    1,
                    "all_reduce",
                    group_name="unknown_group",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            )
        ]
        mgr = make_manager(2, [], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 0)

    def test_op_for_group_rank_is_not_member_of_generates_no_demand(self):
        # group "g" exists in the trace but rank 0 is not one of its members.
        group = ReplayerCommGroupInfo(ranks=[1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)
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
        mgr = make_manager(3, [], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 0)


class TestIntraAssistantSkip(unittest.TestCase):
    def _run(self, func_name, sandbox_ranks, extra_meta=None, output_shape=None, output_dtype=None):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    func_name,
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    output_shape=output_shape,
                    output_dtype=output_dtype,
                    meta=extra_meta or {},
                )
            )
        ]
        mgr = make_manager(2, sandbox_ranks, {0: [group]}, {0: ops})
        mgr.build_demands()
        return mgr

    def test_broadcast_intra_only_skipped(self):
        mgr = self._run("broadcast", sandbox_ranks=[], extra_meta={"src": 0})
        self.assertEqual(len(mgr.demands), 0)

    def test_broadcast_crossing_boundary_generates_demand(self):
        # rank 1 is sandbox (not emulated), so group [0,1] is not intra-assistant-only.
        mgr = self._run("broadcast", sandbox_ranks=[1], extra_meta={"src": 0})
        self.assertEqual(len(mgr.demands), 1)
        demand = next(iter(mgr.demands.values()))
        self.assertEqual(demand.direction, MEMORY_TENSOR_INPUT)

    def test_all_reduce_intra_only_skipped(self):
        mgr = self._run("all_reduce", sandbox_ranks=[], extra_meta={"reduce_op": torch.distributed.ReduceOp.SUM})
        self.assertEqual(len(mgr.demands), 0)

    def test_all_gather_intra_only_skipped(self):
        mgr = self._run("all_gather_into_tensor", sandbox_ranks=[], output_shape=[8], output_dtype=torch.float32)
        self.assertEqual(len(mgr.demands), 0)

    def test_reduce_scatter_intra_only_skipped(self):
        mgr = self._run(
            "reduce_scatter_tensor",
            sandbox_ranks=[],
            extra_meta={"reduce_op": torch.distributed.ReduceOp.SUM},
            output_shape=[2],
            output_dtype=torch.float32,
        )
        self.assertEqual(len(mgr.demands), 0)

    def test_all_to_all_intra_only_skipped_without_raising(self):
        mgr = self._run("all_to_all_single", sandbox_ranks=[])
        self.assertEqual(len(mgr.demands), 0)

    def test_all_to_all_crossing_boundary_raises(self):
        with self.assertRaises(ValueError):
            self._run("all_to_all_single", sandbox_ranks=[1])

    def test_barrier_never_generates_demand(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [comm(comm_op(1, "barrier"))]
        mgr = make_manager(2, [], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 0)


class TestOutputGeneration(unittest.TestCase):
    def test_irecv_generates_output_anything_demand(self):
        ops = [
            comm(
                comm_op(
                    1,
                    "irecv",
                    output_shape=[4],
                    output_dtype=torch.float32,
                    meta={"src": 1},
                )
            )
        ]
        # rank 1 is sandbox -> src not in emulated_ranks -> no PP-init skip logic applies.
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)
        demand = next(iter(mgr.demands.values()))
        self.assertEqual(demand.direction, MEMORY_TENSOR_OUTPUT)
        self.assertEqual(demand.level, TensorMemoryLevel.ANYTHING)

    def test_all_gather_generates_input_and_output_demands(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "all_gather_into_tensor",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    output_shape=[8],
                    output_dtype=torch.float32,
                )
            )
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        directions = sorted(d.direction for d in mgr.demands.values())
        self.assertEqual(directions, [MEMORY_TENSOR_INPUT, MEMORY_TENSOR_OUTPUT])
        for d in mgr.demands.values():
            if d.direction == MEMORY_TENSOR_OUTPUT:
                self.assertEqual(d.level, TensorMemoryLevel.ANYTHING)

    def test_reduce_scatter_generates_input_and_output_demands(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "reduce_scatter_tensor",
                    group_name="g",
                    input_shape=[8],
                    input_dtype=torch.float32,
                    output_shape=[4],
                    output_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            )
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        directions = sorted(d.direction for d in mgr.demands.values())
        self.assertEqual(directions, [MEMORY_TENSOR_INPUT, MEMORY_TENSOR_OUTPUT])

    def test_all_reduce_generates_only_input_demand(self):
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
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)
        self.assertEqual(next(iter(mgr.demands.values())).direction, MEMORY_TENSOR_INPUT)


class TestIsendIrecvPPInit(unittest.TestCase):
    def test_isend_to_intra_emulated_dst_only_fires_once(self):
        # dst=1 is emulated (not sandboxed): only the first occurrence should fire.
        ops = [
            comm(comm_op(1, "isend", input_shape=[4], input_dtype=torch.float32, meta={"dst": 1})),
            comm(comm_op(2, "isend", input_shape=[4], input_dtype=torch.float32, meta={"dst": 1})),
            comm(comm_op(3, "isend", input_shape=[4], input_dtype=torch.float32, meta={"dst": 1})),
        ]
        mgr = make_manager(2, [], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)
        demand = next(iter(mgr.demands.values()))
        self.assertEqual(len(demand.occurrences), 1)

    def test_isend_to_sandbox_dst_fires_every_time(self):
        # dst=1 is sandboxed (not emulated): every occurrence should fire, feeding
        # the same demand_id's occurrences list (relevant for INSTANT counting).
        # Same call site -> identical signature across repeated trace occurrences.
        ops = [
            comm(comm_op(i, "isend", input_shape=[4], input_dtype=torch.float32, meta={"dst": 1})) for i in range(1, 4)
        ]
        mgr = make_manager(2, [1], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)
        demand = next(iter(mgr.demands.values()))
        self.assertEqual(len(demand.occurrences), 3)

    def test_irecv_to_intra_emulated_src_only_fires_once(self):
        ops = [
            comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1})),
            comm(comm_op(2, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1})),
        ]
        mgr = make_manager(2, [], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)
        self.assertEqual(len(next(iter(mgr.demands.values())).occurrences), 1)

    def test_isend_group_dst_resolved_via_comm_group_ranks(self):
        # batch_isend_irecv style meta uses group_dst (index within group.ranks).
        # rank 0 (the sending/owning rank) must itself be a member of the group.
        group = ReplayerCommGroupInfo(ranks=[0, 3, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            comm(
                comm_op(
                    1,
                    "isend",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"group_dst": 2},
                )
            )
        ]
        # global dst = group.ranks[2] == 1, which is sandboxed -> fires unconditionally (no PP-init skip).
        mgr = make_manager(4, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)

    def test_repeated_isend_different_dst_are_independent_demands(self):
        # In real traces, differing arguments (e.g. dst) show up as differing
        # signature text; two independent call sites are independent demands.
        ops = [
            comm(
                comm_op(
                    1, "isend", input_shape=[4], input_dtype=torch.float32, meta={"dst": 1}, signature="isend(dst=1)"
                )
            ),
            comm(
                comm_op(
                    2, "isend", input_shape=[4], input_dtype=torch.float32, meta={"dst": 2}, signature="isend(dst=2)"
                )
            ),
        ]
        mgr = make_manager(3, [], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 2)


class TestCoalescing(unittest.TestCase):
    def test_coalescing_block_skipped_when_group_unknown(self):
        ops = [
            coalescing_start(1, group_name="unknown"),
            comm(
                comm_op(
                    1,
                    "all_reduce",
                    group_name="unknown",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            ),
            coalescing_end(1, group_name="unknown"),
        ]
        mgr = make_manager(2, [], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 0)

    def test_coalescing_block_processed_when_group_known(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        ops = [
            coalescing_start(1, group_name="g"),
            comm(
                comm_op(
                    1,
                    "all_reduce",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                )
            ),
            coalescing_end(1, group_name="g"),
        ]
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)

    def test_iteration_begin_ignored(self):
        ops = [iteration_begin(1), comm(comm_op(1, "barrier"))]
        mgr = make_manager(2, [], {0: []}, {0: ops})
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 0)


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


class _NeverMatchProvider(TensorProvider):
    def can_handle(self, rank, replay_op, comm_group):
        return False

    def tensor_memory_level(self, rank, replay_op, comm_group):
        raise AssertionError("should never be called")

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        raise NotImplementedError


class TestProviderMatching(unittest.TestCase):
    def test_no_providers_defaults_to_anything(self):
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
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertEqual(next(iter(mgr.demands.values())).level, TensorMemoryLevel.ANYTHING)

    def test_provider_order_first_match_wins(self):
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
        # _NeverMatchProvider first (never matches), then the real one; ensures
        # ordering is respected and non-matching providers are skipped over.
        mgr = make_manager(
            2, [1], {0: [group]}, {0: ops}, tensor_providers=[_NeverMatchProvider(), _AllReduceInstantProvider()]
        )
        mgr.build_demands()
        self.assertEqual(next(iter(mgr.demands.values())).level, TensorMemoryLevel.INSTANT)

    def test_output_demand_never_consults_providers(self):
        # A provider that matches everything must still not affect OUTPUT demands.
        class _MatchAllProvider(TensorProvider):
            def can_handle(self, rank, replay_op, comm_group):
                return True

            def tensor_memory_level(self, rank, replay_op, comm_group):
                return TensorMemoryLevel.PERSISTENT

            def provide(self, rank, replay_op, comm_group, use_index=-1):
                raise NotImplementedError

        ops = [comm(comm_op(1, "irecv", output_shape=[4], output_dtype=torch.float32, meta={"src": 1}))]
        mgr = make_manager(2, [1], {0: []}, {0: ops}, tensor_providers=[_MatchAllProvider()])
        mgr.build_demands()
        demand = next(iter(mgr.demands.values()))
        self.assertEqual(demand.direction, MEMORY_TENSOR_OUTPUT)
        self.assertEqual(demand.level, TensorMemoryLevel.ANYTHING)

    def test_instant_level_accumulates_occurrences_for_total_use_count(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1], backend="nccl", group_name="g", group_desc="g", timeout=None)
        # Same call site repeated across iterations -> identical signature -> one
        # demand_id whose occurrences list grows; total_use_count is len(occurrences).
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
        mgr = make_manager(2, [1], {0: [group]}, {0: ops}, tensor_providers=[_AllReduceInstantProvider()])
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 1)
        demand = next(iter(mgr.demands.values()))
        self.assertEqual(demand.level, TensorMemoryLevel.INSTANT)
        self.assertEqual(len(demand.occurrences), 3)


class TestShareKeyGroupingAcrossRanks(unittest.TestCase):
    def test_same_signature_shape_dtype_different_rank_share_same_share_key(self):
        # rank 2 is sandboxed so the group is not intra-assistant-only, letting the
        # broadcast demand fire for both managed ranks 0 and 1.
        group = ReplayerCommGroupInfo(ranks=[0, 1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)
        op_info = comm_op(
            1,
            "broadcast",
            group_name="g",
            input_shape=[4],
            input_dtype=torch.float32,
            meta={"src": 0},
        )
        mgr = MemoryManager(
            local_rank=0,
            world_size=3,
            sandbox_ranks=[2],
            rank_comm_groups={0: [group], 1: [group]},
            rank_replay_ops={0: [comm(op_info)], 1: [comm(op_info)]},
            tensor_providers=[_BroadcastSharedProvider()],
        )
        mgr.build_demands()
        self.assertEqual(len(mgr.demands), 2)
        share_keys = {d.share_key for d in mgr.demands.values()}
        self.assertEqual(len(share_keys), 1)
        self.assertNotIn(None, share_keys)

    def test_non_shared_levels_have_no_share_key(self):
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
        mgr = make_manager(2, [1], {0: [group]}, {0: ops})
        mgr.build_demands()
        self.assertIsNone(next(iter(mgr.demands.values())).share_key)


class TestUnsupportedOp(unittest.TestCase):
    def test_unsupported_func_name_raises(self):
        ops = [comm(comm_op(1, "some_future_collective"))]
        mgr = make_manager(2, [], {0: []}, {0: ops})
        with self.assertRaises(ValueError):
            mgr.build_demands()


if __name__ == "__main__":
    unittest.main()
