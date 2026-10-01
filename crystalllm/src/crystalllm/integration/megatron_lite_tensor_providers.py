# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tensor providers used by CrystalLLM replay for Megatron Lite."""

from __future__ import annotations

from typing import override

import torch

from crystalllm.replay.data import ReplayerCommGroupInfo, ReplayerCommOpInfo, TensorMemoryLevel, TensorProvider


def _input_spec(replay_op: ReplayerCommOpInfo) -> tuple[list[int], torch.dtype]:
    if replay_op.input_shape is None or replay_op.input_dtype is None:
        raise ValueError(f"Communication input metadata is incomplete: {replay_op}")
    return replay_op.input_shape, replay_op.input_dtype


def _normal_tensor(
    replay_op: ReplayerCommOpInfo,
    scale: float,
) -> torch.Tensor:
    """Create synthetic CPU data using the reference provider's RNG convention."""
    shape, dtype = _input_spec(replay_op)
    return torch.randn(shape, dtype=dtype, device="cpu") * scale


def _is_forward_pipeline_send(
    rank: int,
    replay_op: ReplayerCommOpInfo,
    comm_group: ReplayerCommGroupInfo,
) -> bool:
    """Return whether an isend moves from one PP stage to the next stage."""
    ranks = comm_group.ranks
    source_stage = ranks.index(rank)
    if "group_dst" in replay_op.meta:
        destination_stage = int(replay_op.meta["group_dst"])
    elif "dst" in replay_op.meta:
        destination_stage = ranks.index(int(replay_op.meta["dst"]))
    else:
        raise ValueError(f"P2P destination metadata is missing: {replay_op}")

    # Pipeline traffic is not cyclic: the last stage does not send an activation
    # to the first stage.  This also distinguishes activation and gradient sends
    # when PP=2, where both messages otherwise have the same peer.
    return destination_stage == source_stage + 1


class LogicalAndProvider(TensorProvider):
    """Keep model-parallel logical-AND collectives true during replay."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return "logical_and_across_model_parallel_group" in replay_op.signature

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.PERSISTENT

    @override
    def provide(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        shape, dtype = _input_spec(replay_op)
        return torch.ones(shape, dtype=dtype)


class P2PActivationOrGradientProvider(TensorProvider):
    """Supply stable activations and small gradients to pipeline assistants."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return (
            replay_op.c10d_func_name == "isend"
            and comm_group.group_desc.startswith("PIPELINE-")
        )

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.PERSISTENT

    @override
    def provide(
        self,
        rank: int,
        replay_op: ReplayerCommOpInfo,
        comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        scale = 0.02 if _is_forward_pipeline_send(rank, replay_op, comm_group) else 3.6e-8
        return _normal_tensor(replay_op, scale)


class GradSyncProvider(TensorProvider):
    """Supply representative gradient buffers before optimizer synchronization."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return "start_grad_sync" in replay_op.signature

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.SHARED

    @override
    def provide(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        return _normal_tensor(replay_op, 7.9e-6)


class ParamSyncProvider(TensorProvider):
    """Supply representative parameter buffers after optimizer updates."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return "start_param_sync" in replay_op.signature

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.SHARED

    @override
    def provide(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        return _normal_tensor(replay_op, 0.012)


class GradStatsProvider(TensorProvider):
    """Keep gradient-norm reductions non-negative during replay."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return "get_grad_norm_fp32" in replay_op.signature

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.SHARED

    @override
    def provide(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        return _normal_tensor(replay_op, 1e-8).abs_()


class GradNormGroupFlagProvider(TensorProvider):
    """Keep disabled-MTP grad-group detection false during replay."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return "has_grad_norm_group" in replay_op.signature

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.PERSISTENT

    @override
    def provide(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        shape, dtype = _input_spec(replay_op)
        return torch.zeros(shape, dtype=dtype)


class ExpertTensorParallelMaxLenProvider(TensorProvider):
    """Keep assistant ranks neutral in expert-padding MAX reductions."""

    @override
    def can_handle(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        return (
            replay_op.c10d_func_name == "all_reduce"
            and "primitive/modules/experts.py:" in replay_op.signature
            and replay_op.meta.get("reduce_op") == torch.distributed.ReduceOp.MAX
            and replay_op.input_shape == [1]
            and replay_op.input_dtype == torch.int64
            and comm_group.group_desc.startswith("EXPERT_TENSOR-")
        )

    @override
    def tensor_memory_level(
        self,
        _rank: int,
        _replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        return TensorMemoryLevel.PERSISTENT

    @override
    def provide(
        self,
        _rank: int,
        replay_op: ReplayerCommOpInfo,
        _comm_group: ReplayerCommGroupInfo,
        _use_index: int = -1,
    ) -> torch.Tensor:
        shape, dtype = _input_spec(replay_op)
        return torch.zeros(shape, dtype=dtype)


TENSOR_PROVIDERS: list[TensorProvider] = [
    LogicalAndProvider(),
    GradNormGroupFlagProvider(),
    ExpertTensorParallelMaxLenProvider(),
    GradStatsProvider(),
    P2PActivationOrGradientProvider(),
    GradSyncProvider(),
    ParamSyncProvider(),
]


__all__ = ["TENSOR_PROVIDERS"]
