# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Optional CrystalLLM hooks for NVIDIA Megatron Lite."""

from __future__ import annotations

import atexit
import os
from typing import Any

import torch
import torch.distributed as dist

from crystalllm.patch import BehaviorType, CrystalPatchPlan, CrystalPatchRequest, PatchType, cond_crystal_patch_request
from crystalllm.runtime import (
    crystal_runtime_step_start,
    crystal_runtime_step_stop,
    destroy_crystal_runtime,
    initialize_crystal_runtime,
)

_PACKED_INPUT_DIM = 1
_DENSE_INPUT_DIM = 2
_ATEXIT_REGISTERED = False


def _is_crystal_mode_enabled() -> bool:
    """Keep optional-mode handling local to the Lite adapter."""
    return os.getenv("CRYSTAL_MODE") in {"CAPTURE", "EMULATE"}


def _enable_transformer_engine_fake_process_groups() -> None:
    """Let Transformer Engine recognize Crystal's capture process groups."""
    from transformer_engine.pytorch.attention.dot_product_attention import (
        backends,
        context_parallel,
        dot_product_attention,
    )

    from crystalllm.torch import CrystalFakeProcessGroup

    group_types = (dist.ProcessGroup, CrystalFakeProcessGroup)
    for module in (dot_product_attention, backends, context_parallel):
        module.dist_group_type = group_types


def initialize() -> None:
    """Initialize Crystal before Megatron Lite creates process groups."""
    global _ATEXIT_REGISTERED  # noqa: PLW0603
    if not _is_crystal_mode_enabled():
        return
    initialize_crystal_runtime()
    if not _ATEXIT_REGISTERED:
        atexit.register(destroy_crystal_runtime)
        _ATEXIT_REGISTERED = True
    _enable_transformer_engine_fake_process_groups()


def step_start() -> None:
    """Start the Lite step before forward/backward and optimizer communication."""
    if not _is_crystal_mode_enabled():
        return
    crystal_runtime_step_start()


def step_stop() -> None:
    """Close the Lite step after optimizer work (or forward-only evaluation)."""
    if _is_crystal_mode_enabled():
        crystal_runtime_step_stop()


def non_thd_pipeline_tensor_shape(batch: Any, shape: tuple[int, ...], ps: Any) -> tuple[int, ...]:
    """Return the fixed PP shape used by Crystal's non-THD replay path."""
    input_ids = batch.input_ids
    if input_ids.dim() == _PACKED_INPUT_DIM:
        sequence = int(input_ids.size(0))
    elif input_ids.dim() == _DENSE_INPUT_DIM:
        sequence = int(input_ids.size(1))
    else:
        raise ValueError(f"Unsupported input_ids rank for pipeline runtime: {input_ids.dim()}.")

    tp_size = int(getattr(ps, "tp_size", 1) or 1)
    if sequence % tp_size != 0:
        raise ValueError(
            f"Pipeline tensor sequence length {sequence} is not divisible by TP={tp_size}."
        )
    return (sequence // tp_size, *shape[1:])


def all_gather_ep_token_counts(
    output: torch.Tensor,
    local_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> Any:
    """Gather EP token counts while keeping capture-time split sizes valid.

    Crystal capture does not execute the remote ranks' real collective work.  The
    receiver therefore gets deterministic zero counts for remote EP ranks and the
    real local row for itself. Emulation executes the real all-gather within the
    Sandbox-local EP group, without patching its result.
    """
    if not _is_crystal_mode_enabled():
        return dist.all_gather_into_tensor(output, local_counts, group=group)

    def set_local_row(tensor: torch.Tensor) -> torch.Tensor:
        tensor.zero_()
        group_rank = dist.get_rank(group=group)
        group_size = dist.get_world_size(group=group)
        tensor.view(group_size, -1)[group_rank].copy_(local_counts)
        return tensor

    request = CrystalPatchRequest(
        patch_func=dist.all_gather_into_tensor,
        capture_behavior=BehaviorType.ONLY_PATCH,
        emulate_behavior=BehaviorType.ONLY_EXEC,
        patch_plan=CrystalPatchPlan(
            patch_type=PatchType.ARGS,
            position=0,
            action=set_local_row,
        ),
        patch_res_length=0,
    )
    with cond_crystal_patch_request(request):
        return dist.all_gather_into_tensor(output, local_counts, group=group)


__all__ = [
    "all_gather_ep_token_counts",
    "initialize",
    "non_thd_pipeline_tensor_shape",
    "step_start",
    "step_stop",
]
