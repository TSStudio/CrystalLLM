# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Stage F end-to-end MemoryManager protocol test.

This test runs a real MemoryManager in-process with a background request-processing
thread plus separate OS subprocess fake assistant drivers
(`tests/memory/e2e_fake_assistant.py`). It exercises the full Tensor Acquire protocol
over real Redis and real CUDA IPC, across all four TensorMemoryLevel kinds,
including an init-time CPU fallback and a runtime EVICTED -> READY reload.

No real collective communication is issued anywhere in this test.
"""

import json
import math
import subprocess
import sys
import tempfile
import threading
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


DRIVER_SCRIPT = str(Path(__file__).parent / "e2e_fake_assistant.py")

LEVEL_MAP = {
    "persistent_op": TensorMemoryLevel.PERSISTENT,
    "shared_op": TensorMemoryLevel.SHARED,
    "instant_op": TensorMemoryLevel.INSTANT,
}
FILL_MAP = {
    "persistent_op": 3.0,
    "shared_op": 9.0,
}


class _E2EProvider(TensorProvider):
    def __init__(self):
        self.provide_calls = []

    def can_handle(self, rank, replay_op, comm_group):
        return replay_op.signature in LEVEL_MAP

    def tensor_memory_level(self, rank, replay_op, comm_group):
        return LEVEL_MAP[replay_op.signature]

    def provide(self, rank, replay_op, comm_group, use_index=-1):
        self.provide_calls.append((rank, replay_op.signature, use_index))
        value = float(use_index) if replay_op.signature == "instant_op" else FILL_MAP[replay_op.signature]
        return torch.full(replay_op.input_shape, value, dtype=replay_op.input_dtype)


def _redis_available() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB)
        client.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False


REDIS_AVAILABLE = _redis_available()
CUDA_AVAILABLE = torch.cuda.is_available()


def _run_driver(local_rank: int, rank: int, plan: list, tmp_dir: Path) -> list:
    plan_file = tmp_dir / f"plan_{rank}.json"
    output_file = tmp_dir / f"output_{rank}.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            DRIVER_SCRIPT,
            "--local-rank",
            str(local_rank),
            "--rank",
            str(rank),
            "--plan-file",
            str(plan_file),
            "--output-file",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"driver rank={rank} failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(output_file.read_text(encoding="utf-8"))


@unittest.skipUnless(CUDA_AVAILABLE and REDIS_AVAILABLE, "requires CUDA + local redis-server on localhost:6379")
class TestMemoryManagerEndToEnd(unittest.TestCase):
    def setUp(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def tearDown(self):
        redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB).flushdb()

    def test_full_protocol_across_all_levels_with_init_eviction_and_runtime_reload(self):
        group = ReplayerCommGroupInfo(ranks=[0, 1, 2], backend="nccl", group_name="g", group_desc="g", timeout=None)

        rank0_ops = [
            comm(comm_op(1, "irecv", output_shape=[100], output_dtype=torch.float32, meta={"src": 2})),
            comm(
                comm_op(
                    2,
                    "all_reduce",
                    group_name="g",
                    input_shape=[50],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="persistent_op",
                )
            ),
            comm(
                comm_op(
                    3,
                    "broadcast",
                    group_name="g",
                    input_shape=[20],
                    input_dtype=torch.float32,
                    meta={"src": 0},
                    signature="shared_op",
                )
            ),
            comm(
                comm_op(
                    4,
                    "all_reduce",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="instant_op",
                )
            ),
            comm(
                comm_op(
                    5,
                    "all_reduce",
                    group_name="g",
                    input_shape=[50],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="persistent_op",
                )
            ),
            comm(
                comm_op(
                    6,
                    "all_reduce",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="instant_op",
                )
            ),
            comm(
                comm_op(
                    7,
                    "all_reduce",
                    group_name="g",
                    input_shape=[4],
                    input_dtype=torch.float32,
                    meta={"reduce_op": torch.distributed.ReduceOp.SUM},
                    signature="instant_op",
                )
            ),
        ]
        rank1_ops = [
            comm(
                comm_op(
                    1,
                    "broadcast",
                    group_name="g",
                    input_shape=[20],
                    input_dtype=torch.float32,
                    meta={"src": 0},
                    signature="shared_op",
                )
            )
        ]

        provider = _E2EProvider()
        mgr = MemoryManager(
            local_rank=0,
            world_size=3,
            sandbox_ranks=[2],
            rank_comm_groups={0: [group], 1: [group]},
            rank_replay_ops={0: rank0_ops, 1: rank1_ops},
            tensor_providers=[provider],
            reserved_memory_ratio=0.0,
        )
        mgr.build_demands()
        mgr.plan_resources()

        # pool=400B (100*f32), shared=80B (20*f32), persistent=200B (50*f32).
        # Budget fits pool+shared (480) but not +persistent (680) -> persistent
        # is deliberately left on CPU (EVICTED) after init.
        mgr.allocate(total_bytes=600)

        persistent_plan = next(p for p in mgr.allocation_plans if p.owner_kind == mm.OWNER_KIND_PERSISTENT)
        shared_plan = next(p for p in mgr.allocation_plans if p.owner_kind == mm.OWNER_KIND_SHARED)
        self.assertEqual(mgr.redis.command.get(mm.redis_field(persistent_plan.key, "state")).decode(), mm.STATE_EVICTED)
        self.assertEqual(mgr.redis.command.get(mm.redis_field(shared_plan.key, "state")).decode(), mm.STATE_READY)

        # Simulate more memory becoming available before the runtime phase, so the
        # request-queue-driven reload path is exercised without needing a second,
        # racy eviction cascade across two concurrent driver processes (that
        # cascade ordering is already covered deterministically by
        # tests/test_memory_manager_requests.py).
        mgr.allocatable_bytes = 10 * 1024**3

        mgr.write_scaffold_to_redis()
        mgr.mark_init_done()

        loop_thread = threading.Thread(target=mgr.run_request_loop, kwargs={"poll_timeout": 0.2}, daemon=True)
        loop_thread.start()

        try:
            with tempfile.TemporaryDirectory() as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)

                rank0_plan = [
                    {
                        "c10d_func_name": "irecv",
                        "signature": "irecv()",
                        "direction": mm.MEMORY_TENSOR_OUTPUT,
                        "shape": [100],
                        "dtype": "torch.float32",
                        "use_index": -1,
                    },
                    {
                        "c10d_func_name": "all_reduce",
                        "signature": "persistent_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [50],
                        "dtype": "torch.float32",
                        "use_index": -1,
                    },
                    {
                        "c10d_func_name": "broadcast",
                        "signature": "shared_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [20],
                        "dtype": "torch.float32",
                        "use_index": -1,
                    },
                    {
                        "c10d_func_name": "all_reduce",
                        "signature": "instant_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [4],
                        "dtype": "torch.float32",
                        "use_index": 0,
                    },
                    {
                        "c10d_func_name": "all_reduce",
                        "signature": "persistent_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [50],
                        "dtype": "torch.float32",
                        "use_index": -1,
                    },
                    {
                        "c10d_func_name": "all_reduce",
                        "signature": "instant_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [4],
                        "dtype": "torch.float32",
                        "use_index": 1,
                    },
                    {
                        "c10d_func_name": "all_reduce",
                        "signature": "instant_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [4],
                        "dtype": "torch.float32",
                        "use_index": 2,
                    },
                ]
                rank1_plan = [
                    {
                        "c10d_func_name": "broadcast",
                        "signature": "shared_op",
                        "direction": mm.MEMORY_TENSOR_INPUT,
                        "shape": [20],
                        "dtype": "torch.float32",
                        "use_index": -1,
                    },
                ]

                # Run sequentially: deterministic, and cross-rank contention on the
                # SHARED resource is already covered by test_shared_allocation_calls_provide_only_once.
                rank0_results = _run_driver(local_rank=0, rank=0, plan=rank0_plan, tmp_dir=tmp_dir)
                rank1_results = _run_driver(local_rank=0, rank=1, plan=rank1_plan, tmp_dir=tmp_dir)
        finally:
            mgr.stop_request_loop()
            loop_thread.join(timeout=5)

        # --- ANYTHING (irecv) ---
        irecv_result = rank0_results[0]
        self.assertEqual(irecv_result["level"], "ANYTHING")
        self.assertEqual(len(irecv_result["values"]), 100)
        self.assertTrue(all(math.isfinite(v) for v in irecv_result["values"]))

        # --- PERSISTENT (evicted at init, reloaded at runtime; reused from cache 2nd time) ---
        persistent_1st = rank0_results[1]
        persistent_2nd = rank0_results[4]
        self.assertEqual(persistent_1st["level"], "PERSISTENT")
        self.assertEqual(persistent_1st["values"], [3.0] * 50)
        self.assertEqual(persistent_2nd["values"], [3.0] * 50)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(persistent_plan.key, "version"))), 1)

        # --- SHARED (cross-rank, single provide() call) ---
        rank0_shared = rank0_results[2]
        rank1_shared = rank1_results[0]
        self.assertEqual(rank0_shared["level"], "SHARED")
        self.assertEqual(rank0_shared["values"], [9.0] * 20)
        self.assertEqual(rank1_shared["values"], [9.0] * 20)
        self.assertEqual(rank0_shared["target_ref"], rank1_shared["target_ref"])
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(shared_plan.key, "version"))), 0)

        # --- INSTANT (fresh allocation + correct per-use value each time) ---
        instant_results = [rank0_results[3], rank0_results[5], rank0_results[6]]
        for use_index, result in enumerate(instant_results):
            self.assertEqual(result["level"], "INSTANT")
            self.assertEqual(result["values"], [float(use_index)] * 4)
        self.assertEqual(len({r["target_ref"] for r in instant_results}), 3)  # distinct allocations

        # --- provide() call counts: persistent+shared once each (at init), instant 3x (runtime) ---
        signatures_called = [c[1] for c in provider.provide_calls]
        self.assertEqual(signatures_called.count("persistent_op"), 1)
        self.assertEqual(signatures_called.count("shared_op"), 1)
        self.assertEqual(signatures_called.count("instant_op"), 3)
        instant_use_indices = sorted(c[2] for c in provider.provide_calls if c[1] == "instant_op")
        self.assertEqual(instant_use_indices, [0, 1, 2])

        # --- final state: everything released, no leaked user_count ---
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(persistent_plan.key, "user_count"))), 0)
        self.assertEqual(int(mgr.redis.command.get(mm.redis_field(shared_plan.key, "user_count"))), 0)
        for result in instant_results:
            self.assertEqual(int(mgr.redis.command.get(mm.redis_field(result["target_ref"], "user_count"))), 0)
            self.assertEqual(
                mgr.redis.command.get(mm.redis_field(result["target_ref"], "state")).decode(), mm.STATE_READY
            )


if __name__ == "__main__":
    unittest.main()
