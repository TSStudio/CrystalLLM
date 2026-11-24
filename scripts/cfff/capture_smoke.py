#!/usr/bin/env python3
"""Capture a small synthetic training workload; not an emulation benchmark."""
import os

import torch
import torch.distributed as dist
from crystalllm.runtime import crystal_runtime_step_start, crystal_runtime_step_stop, destroy_crystal_runtime


def main():
    if os.getenv("CRYSTAL_MODE") != "CAPTURE":
        raise RuntimeError("This workload is only validated for Crystal capture mode")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="nccl", group_desc=f"DATA_PARALLEL_GROUP-{rank}-1-nccl")
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 128)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(int(os.environ["CRYSTAL_TRAIN_ITERS"])):
        crystal_runtime_step_start()
        optimizer.zero_grad()
        loss = model(torch.randn(8, 128, device="cuda")).square().mean()
        loss.backward()
        for parameter in model.parameters():
            dist.all_reduce(parameter.grad, group=group)
        optimizer.step()
        crystal_runtime_step_stop()
    destroy_crystal_runtime()
    dist.destroy_process_group()
    print(f"CAPTURE_RANK_OK {rank}", flush=True)


if __name__ == "__main__":
    main()
