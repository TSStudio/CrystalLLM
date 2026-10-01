# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Runtime Rendezvous for Replay."""

from __future__ import annotations

import os
import time
from typing import Any

import torch

# Global cache for the replay rendezvous process group
_replay_rendezvous_pg: Any = None

_DEBUG = False


def _get_replay_rendezvous_pg() -> Any:
    """Get or create the replay rendezvous process group.

    Returns:
        The process group for replay rendezvous, or None if not in torch.distributedributed mode.
    """
    global _replay_rendezvous_pg  # noqa: PLW0603

    if _replay_rendezvous_pg is not None:
        return _replay_rendezvous_pg

    if not torch.distributed.is_initialized():
        return None

    if (
        os.getenv("CRYSTAL_MOCK_NCCL_SANDBOX_RANKS") is not None
        and os.getenv("CRYSTAL_MOCK_NCCL_MEMBER_RANKS") is not None
    ):
        sandbox_ranks = [int(r.strip()) for r in os.getenv("CRYSTAL_MOCK_NCCL_SANDBOX_RANKS").split(",") if r.strip()]
        member_ranks = [int(r.strip()) for r in os.getenv("CRYSTAL_MOCK_NCCL_MEMBER_RANKS").split(",") if r.strip()]
        rendezvous_ranks = sandbox_ranks + member_ranks
    else:
        rendezvous_ranks = list(range(torch.distributed.get_world_size()))

    # Create a new group with all ranks using gloo backend
    _replay_rendezvous_pg = torch.distributed.new_group(
        ranks=rendezvous_ranks, backend="gloo", group_desc="_crystal_replay_rendezvous_pg"
    )

    return _replay_rendezvous_pg


def replay_rendezvous(step: int) -> None:
    """Replay rendezvous.

    This function performs a barrier synchronization across all member ranks
    using the gloo communication backend.
    """
    # Skip if distributed is not initialized
    if not torch.distributed.is_initialized():
        return

    # Get or create the replay rendezvous process group
    pg = _get_replay_rendezvous_pg()

    if pg is not None:
        # Perform barrier synchronization using the dedicated group
        if _DEBUG:
            print(f"[rank {os.getenv('RANK')}] Replay Rendezvous for step {step} entered", flush=True)
            start_t = time.perf_counter()
        torch.distributed.barrier(group=pg)
        if _DEBUG:
            end_t = time.perf_counter()
            print(
                f"[rank {os.getenv('RANK')}] Replay Rendezvous for step {step} completed after {end_t - start_t}s ",
                flush=True,
            )
