#!/usr/bin/env python3
"""Exercise real multi-node DDP with a small synthetic training workload."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 1024)
    ).cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(10):
        inputs = torch.randn(16, 1024, device=local_rank)
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        print(f"rank={dist.get_rank()} step={step} loss={loss.item():.6f}", flush=True)
    dist.barrier()
    print(f"rank={dist.get_rank()} DDP_SMOKE_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
