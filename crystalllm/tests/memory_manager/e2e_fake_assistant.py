# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Standalone fake-assistant driver used by the Stage F end-to-end test.

Implements the assistant-side Tensor Acquire protocol from
ASSISTANT_MEMORY_MANAGER.md (local tensor cache with version check via
acquire_if_ready, falling back to the request queue + pub/sub wake-up, then
building the tensor from the CUDA IPC handle). Does NOT issue any real
collective communication -- it only proves tensors can be acquired, read, and
released with the values a real assistant.py would see.

Must run as a separate OS process from the MemoryManager (CUDA IPC handles
cannot be re-imported within the exporting process).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import torch

from crystalllm.replay import memory_manager as mm


def _build_tensor_from_handle(
    local_rank: int, handle_bytes: bytes, dtype: torch.dtype, num_elements: int
) -> torch.Tensor:
    handle = pickle.loads(handle_bytes)
    storage = torch.UntypedStorage._new_shared_cuda(*handle)
    tensor = torch.empty(0, dtype=dtype, device=f"cuda:{local_rank}")
    tensor.set_(storage, 0, [num_elements])
    return tensor


def _wait_for_wakeup(pubsub, request_id: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
        if msg is not None and msg["data"].decode() == request_id:
            return
    raise TimeoutError(f"Timed out waiting for wake-up of request {request_id}")


def run(local_rank: int, rank: int, plan: list[dict], output_file: str) -> None:
    torch.cuda.set_device(local_rank)

    redis_client = mm.MemoryManagerRedisClient(local_rank)

    while not redis_client.command.get(mm.key_init_done(local_rank)):
        time.sleep(0.05)

    pubsub = redis_client.new_pubsub()
    pubsub.subscribe(mm.key_rank_events_channel(local_rank, rank))
    pubsub.get_message(timeout=5)  # consume the subscribe confirmation

    cache: dict[str, tuple[torch.Tensor, int]] = {}
    request_seq = 0
    results = []

    for use in plan:
        dtype = mm.dtype_from_str(use["dtype"])
        shape = use["shape"]
        num_elements = mm.shape_numel(shape)
        demand_id = mm.encode_demand_id(
            rank, use["c10d_func_name"], use["signature"], use["direction"], shape, use["dtype"]
        )
        level = mm.TensorMemoryLevel[redis_client.command.get(mm.key_demand_level(local_rank, demand_id)).decode()]
        target_ref = redis_client.command.get(mm.key_demand_target_ref(local_rank, demand_id)).decode()

        tensor = None
        resolved_ref = target_ref

        if level != mm.TensorMemoryLevel.INSTANT:
            cached = cache.get(target_ref)
            if cached is not None:
                cached_tensor, cached_version = cached
                result = redis_client.acquire_if_ready(target_ref, expected_version=cached_version)
                if result == mm.ACQUIRE_READY:
                    tensor = cached_tensor
                elif result == mm.ACQUIRE_NOT_READY and level == mm.TensorMemoryLevel.ANYTHING:
                    raise RuntimeError(f"ANYTHING pool {target_ref} unexpectedly NOT_READY")
                else:
                    del cache[target_ref]

            if tensor is None:
                result = redis_client.acquire_if_ready(target_ref)
                if result == mm.ACQUIRE_READY:
                    version = int(redis_client.command.get(mm.redis_field(target_ref, "version")))
                    handle_bytes = redis_client.command.get(mm.redis_field(target_ref, "cuda_ipc_handle"))
                    tensor = _build_tensor_from_handle(local_rank, handle_bytes, dtype, num_elements)
                    cache[target_ref] = (tensor, version)
                elif level == mm.TensorMemoryLevel.ANYTHING:
                    raise RuntimeError(f"ANYTHING pool {target_ref} unexpectedly not ready: {result}")
                else:
                    request_seq += 1
                    request_id = mm.submit_request(redis_client, local_rank, rank, request_seq, demand_id)
                    _wait_for_wakeup(pubsub, request_id)
                    resolved_ref = redis_client.command.get(mm.key_demand_target_ref(local_rank, demand_id)).decode()
                    version = int(redis_client.command.get(mm.redis_field(resolved_ref, "version")))
                    handle_bytes = redis_client.command.get(mm.redis_field(resolved_ref, "cuda_ipc_handle"))
                    tensor = _build_tensor_from_handle(local_rank, handle_bytes, dtype, num_elements)
                    cache[resolved_ref] = (tensor, version)
        else:
            use_index = use["use_index"]
            request_seq += 1
            request_id = mm.submit_request(redis_client, local_rank, rank, request_seq, demand_id, use_index=use_index)
            _wait_for_wakeup(pubsub, request_id)
            resolved_ref = redis_client.command.get(
                mm.key_instant_use_allocation_ref(local_rank, demand_id, use_index)
            ).decode()
            handle_bytes = redis_client.command.get(mm.redis_field(resolved_ref, "cuda_ipc_handle"))
            tensor = _build_tensor_from_handle(local_rank, handle_bytes, dtype, num_elements)

        torch.cuda.synchronize()
        values = tensor.detach().cpu().tolist()
        results.append({"demand_id": demand_id, "level": level.name, "target_ref": resolved_ref, "values": values})

        redis_client.command.decrby(mm.redis_field(resolved_ref, "user_count"), 1)

    with pathlib.Path(output_file).open("w", encoding="utf-8") as f:
        json.dump(results, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--plan-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    with pathlib.Path(args.plan_file).open(encoding="utf-8") as f:
        plan = json.load(f)

    run(args.local_rank, args.rank, plan, args.output_file)


if __name__ == "__main__":
    main()
