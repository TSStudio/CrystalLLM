# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for the optional Megatron Lite integration."""

from __future__ import annotations

import pytest
import torch

from crystalllm.integration import megatron_lite
from crystalllm.integration.megatron_lite_tensor_providers import (
    ExpertTensorParallelMaxLenProvider,
    GradNormGroupFlagProvider,
    GradStatsProvider,
    P2PActivationOrGradientProvider,
    _is_forward_pipeline_send,
)
from crystalllm.patch import crystal_patch_response
from crystalllm.replay.data import ReplayerCommGroupInfo, ReplayerCommOpInfo, TensorMemoryLevel
from crystalllm.torch import CrystalFakeProcessGroup


def _pipeline_group() -> ReplayerCommGroupInfo:
    return ReplayerCommGroupInfo(
        ranks=[0, 8],
        backend="nccl",
        group_name="33",
        group_desc="PIPELINE-0-33-nccl",
        timeout=None,
    )


def _pipeline_send(rank: int, destination: int) -> ReplayerCommOpInfo:
    return ReplayerCommOpInfo(
        trace_comm_index=rank + 7,
        c10d_func_name="isend",
        signature="torch.distributed.distributed_c10d.batch_isend_irecv",
        group_name="33",
        group_desc="PIPELINE-0-33-nccl",
        output_shape=None,
        output_dtype=None,
        input_shape=[1024],
        input_dtype=torch.float32,
        async_op=True,
        meta={"group_dst": destination},
    )


def test_non_thd_pipeline_shape_is_not_partitioned_by_cp():
    """Crystal's fixed non-THD PP buffer is partitioned by TP only."""
    batch = type("Batch", (), {"input_ids": torch.zeros(128, dtype=torch.long)})()
    ps = type("ParallelState", (), {"tp_size": 8, "cp_size": 2})()

    shape = megatron_lite.non_thd_pipeline_tensor_shape(batch, (8, 1, 4096), ps)

    assert shape == (16, 1, 4096)


def test_transformer_engine_recognizes_crystal_context_parallel_group(monkeypatch):
    """TE must keep CP enabled when capture replaces its process group."""
    from transformer_engine.pytorch.attention.dot_product_attention import (
        backends,
        context_parallel,
        dot_product_attention,
    )

    group = CrystalFakeProcessGroup([0, 1], 0, "7", "CONTEXT-0-7-nccl")
    for module in (dot_product_attention, backends, context_parallel):
        monkeypatch.setattr(module, "dist_group_type", module.dist_group_type)
    megatron_lite._enable_transformer_engine_fake_process_groups()

    for module in (dot_product_attention, backends, context_parallel):
        assert isinstance(group, module.dist_group_type)


def test_ep_token_count_capture_keeps_only_the_local_row(monkeypatch):
    """Capture metadata must not derive allocation sizes from junk rows."""
    monkeypatch.setenv("CRYSTAL_MODE", "CAPTURE")
    calls = []

    @crystal_patch_response
    def fake_all_gather(output, local_counts, group=None):
        calls.append((output, local_counts, group))

    monkeypatch.setattr(megatron_lite.dist, "all_gather_into_tensor", fake_all_gather)
    monkeypatch.setattr(megatron_lite.dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(megatron_lite.dist, "get_world_size", lambda group=None: 3)

    local_counts = torch.tensor([2, 5])
    output = torch.full((6,), -1, dtype=torch.int64)
    megatron_lite.all_gather_ep_token_counts(output, local_counts, object())

    assert calls == []
    assert output.view(3, 2).tolist() == [[0, 0], [2, 5], [0, 0]]


@pytest.mark.parametrize("mode", ["EMULATE", "BASELINE", None])
def test_ep_token_count_outside_capture_executes_collective_without_patching(monkeypatch, mode):
    """Sandbox-local EP gathers real remote counts in emulation and normal runs."""
    if mode is None:
        monkeypatch.delenv("CRYSTAL_MODE", raising=False)
    else:
        monkeypatch.setenv("CRYSTAL_MODE", mode)
    calls = []
    gathered_counts = torch.tensor([11, 13, 3, 7], dtype=torch.int64)

    @crystal_patch_response
    def fake_all_gather(output, local_counts, group=None):
        calls.append((output, local_counts, group))
        output.copy_(gathered_counts)
        return "work"

    monkeypatch.setattr(megatron_lite.dist, "all_gather_into_tensor", fake_all_gather)
    monkeypatch.setattr(megatron_lite.dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(megatron_lite.dist, "get_world_size", lambda group=None: 2)

    local_counts = torch.tensor([3, 7])
    output = torch.empty(4, dtype=torch.int64)
    group = object()
    result = megatron_lite.all_gather_ep_token_counts(output, local_counts, group)

    assert result == "work"
    assert len(calls) == 1
    assert calls[0][0] is output
    assert calls[0][1] is local_counts
    assert calls[0][2] is group
    assert torch.equal(output, gathered_counts)
    assert local_counts.tolist() == [3, 7]


def test_grad_stats_provider_uses_shared_memory():
    """Gradient statistics use the reference integration's shared buffer."""
    provider = GradStatsProvider()

    assert provider.tensor_memory_level(0, None, None) is TensorMemoryLevel.SHARED


def test_grad_norm_group_flag_provider_keeps_disabled_mtp_false():
    """Synthetic ranks must not make the disabled-MTP control flag true."""
    provider = GradNormGroupFlagProvider()
    replay_op = ReplayerCommOpInfo(
        trace_comm_index=18,
        c10d_func_name="all_reduce",
        signature="megatron/core/optimizer/optimizer.py:386:has_grad_norm_group",
        group_name="190",
        group_desc="INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP-0-190-nccl",
        output_shape=None,
        output_dtype=None,
        input_shape=[1],
        input_dtype=torch.int32,
        async_op=False,
        meta={"reduce_op": torch.distributed.ReduceOp.MAX},
    )

    assert provider.can_handle(8, replay_op, None)
    assert provider.tensor_memory_level(8, replay_op, None) is TensorMemoryLevel.PERSISTENT
    assert torch.equal(provider.provide(8, replay_op, None), torch.zeros(1, dtype=torch.int32))


def test_etp_max_len_provider_keeps_assistant_rank_neutral():
    """Assistant ranks must not inflate a host-consumed expert padding length."""
    provider = ExpertTensorParallelMaxLenProvider()
    replay_op = ReplayerCommOpInfo(
        trace_comm_index=5,
        c10d_func_name="all_reduce",
        signature=(
            "/repo/experimental/lite/megatron/lite/primitive/modules/experts.py:156:forward"
        ),
        group_name="1057",
        group_desc="EXPERT_TENSOR-0-1057-nccl",
        output_shape=None,
        output_dtype=None,
        input_shape=[1],
        input_dtype=torch.int64,
        async_op=False,
        meta={"reduce_op": torch.distributed.ReduceOp.MAX},
    )
    group = ReplayerCommGroupInfo(
        ranks=list(range(16)),
        backend="nccl",
        group_name="1057",
        group_desc="EXPERT_TENSOR-0-1057-nccl",
        timeout=None,
    )

    assert provider.can_handle(8, replay_op, group)
    assert provider.tensor_memory_level(8, replay_op, group) is TensorMemoryLevel.PERSISTENT
    assert torch.equal(provider.provide(8, replay_op, group), torch.zeros(1, dtype=torch.int64))


@pytest.mark.parametrize("peer_key", ["group_dst", "dst"])
def test_pp2_direction_uses_ordered_stages(peer_key):
    """Classify PP2 sends by stage order, not by synthetic tensor magnitudes."""
    provider = P2PActivationOrGradientProvider()
    group = _pipeline_group()
    forward_op = _pipeline_send(0, 1)
    backward_op = _pipeline_send(8, 0)
    forward_op.meta.clear()
    forward_op.meta[peer_key] = 1 if peer_key == "group_dst" else 8
    backward_op.meta.clear()
    backward_op.meta[peer_key] = 0

    assert _is_forward_pipeline_send(0, forward_op, group) is True
    assert _is_forward_pipeline_send(8, backward_op, group) is False
    assert provider.can_handle(0, forward_op, group)
    assert provider.can_handle(8, backward_op, group)
    assert provider.tensor_memory_level(0, forward_op, group) is TensorMemoryLevel.PERSISTENT
