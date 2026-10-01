# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Regression tests for Crystal communication trace parsing."""

from __future__ import annotations

import gzip

import pytest
import torch

from crystalllm.replay.data import ReplayerReplayOpType
from crystalllm.replay.parser import CrystalTraceParser


def test_broadcast_src_can_be_read_from_current_c10d_kwargs(tmp_path):
    """Megatron Lite capture records broadcast ``src`` in trace kwargs."""
    trace_file = tmp_path / "trace_0.log.gz"
    lines = [
        "NEW_GROUP#{'ranks': [0, 8], 'timeout': None, 'backend': None, "
        "'group_name': '33', 'group_desc': 'PIPELINE-0-33-nccl'}\n",
        "SCHEDULE#iteration 1 begin\n",
        "0#broadcast#runtime.py:553:forward_backward#33#PIPELINE-0-33-nccl#"
        "['tensor-torch.Size([2])-torch.dtype(torch.float32)']#"
        "{'src': 8, 'group': CrystalFakeProcessGroup(...)}\n",
    ]
    with gzip.open(trace_file, "wt") as stream:
        stream.writelines(lines)

    parser = CrystalTraceParser(str(trace_file))
    comm_ops = [
        replay_op.info
        for replay_op in parser.replay_ops_ordered_list
        if replay_op.type is ReplayerReplayOpType.COMM
    ]

    assert len(comm_ops) == 1
    assert comm_ops[0].meta == {"src": 8}


@pytest.mark.parametrize("style", ["positional", "keyword", "batch"])
@pytest.mark.parametrize(
    ("op_name", "peer", "expected_meta", "expected_input", "expected_output"),
    [
        ("isend", 1, {"dst": 1}, [128, 1, 512], None),
        ("irecv", 0, {"src": 0}, None, [128, 1, 512]),
    ],
)
def test_p2p_supports_lite_positional_and_reference_keyword_or_batch_calls(
    tmp_path, style, op_name, peer, expected_meta, expected_input, expected_output
):
    """Support Lite's positional P2P while retaining reference and batch parsing."""
    trace_file = tmp_path / f"trace_{op_name}.log.gz"
    tensor = "tensor-torch.Size([128, 1, 512])-torch.dtype(torch.bfloat16)"
    peer_name = "dst" if op_name == "isend" else "src"
    signature = "pipeline.py:123:run_microbatch_loop"
    if style == "positional":
        args = f"['{tensor}', {peer}, CrystalFakeProcessGroup(...)]"
        kwargs = "{}"
    elif style == "keyword":
        args = "[]"
        kwargs = f"{{'tensor': '{tensor}', '{peer_name}': {peer}, 'group': CrystalFakeProcessGroup(...)}}"
    else:
        signature = "distributed_c10d.py:2887:batch_isend_irecv"
        args = f"['{tensor}']"
        kwargs = f"{{'group': CrystalFakeProcessGroup(...), 'group_{peer_name}': {peer}}}"
        expected_meta = {f"group_{peer_name}": peer}
    lines = [
        "NEW_GROUP#{'ranks': [0, 1], 'timeout': None, 'backend': None, "
        "'group_name': '7', 'group_desc': 'PIPELINE-0-7-nccl'}\n",
        "SCHEDULE#iteration 1 begin\n",
        f"0#{op_name}#{signature}#7#PIPELINE-0-7-nccl#{args}#{kwargs}\n",
    ]
    with gzip.open(trace_file, "wt") as stream:
        stream.writelines(lines)

    parser = CrystalTraceParser(str(trace_file))
    comm_ops = [
        replay_op.info
        for replay_op in parser.replay_ops_ordered_list
        if replay_op.type is ReplayerReplayOpType.COMM
    ]

    assert len(comm_ops) == 1
    assert comm_ops[0].meta == expected_meta
    assert comm_ops[0].input_shape == expected_input
    assert comm_ops[0].output_shape == expected_output
    if expected_input is not None:
        assert comm_ops[0].input_dtype is torch.bfloat16
    if expected_output is not None:
        assert comm_ops[0].output_dtype is torch.bfloat16


@pytest.mark.parametrize("coalesced", [False, True])
def test_cp_filter_preserves_iteration_boundaries_trace_ids_and_ep(tmp_path, coalesced):
    """Skip CP replay records without renumbering PP/EP or iteration boundaries."""
    lines = []
    for name, desc in [("17", "CONTEXT-0-17-nccl"), ("33", "PIPELINE-0-33-nccl"),
                       ("44", "EXPERT-0-44-nccl")]:
        lines.append("NEW_GROUP#" + str({"ranks": [0, 8], "backend": "nccl",
                                      "group_name": name, "group_desc": desc}))

    def cp(index):
        comm = f"{index}#isend#te.py:1:forward#17#CONTEXT-0-17-nccl#[]#{{}}"
        # CP tensor/peer metadata must not even be parsed for replay.
        if not coalesced:
            return [comm]
        return [f"COALESCING#START#17#CONTEXT-0-17-nccl#device_set=True#async_ops=True#{index}",
                comm,
                f"COALESCING#END#17#CONTEXT-0-17-nccl#device_set=True#async_ops=True#{index}"]

    lines += ["SCHEDULE#iteration 1 begin",
              "0#all_to_all_single#moe.py:1:forward#44#EXPERT-0-44-nccl#[]#{}"]
    lines += cp(1)
    lines += ["SCHEDULE#iteration 2 begin",
              "2#isend#pp.py:1:forward#33#PIPELINE-0-33-nccl#"
              "['tensor-torch.Size([4])-torch.dtype(torch.float32)', 8]#{}"]
    lines += cp(3)
    trace_file = tmp_path / "trace_0.log.gz"
    with gzip.open(trace_file, "wt") as stream:
        stream.write("\n".join(lines) + "\n")

    parser = CrystalTraceParser(str(trace_file))
    assert set(parser.comm_groups_dict) == {"0", "17", "33", "44"}
    assert not parser.replay_comm_ops_dict["17"]
    assert parser.trace_comm_index_dict == {1: 0, 2: 1, 3: 3}
    assert [op.type for op in parser.replay_ops_ordered_list] == [
        ReplayerReplayOpType.ITERATION_BEGIN, ReplayerReplayOpType.COMM,
        ReplayerReplayOpType.ITERATION_BEGIN, ReplayerReplayOpType.COMM,
    ]
    comms = [op.info for op in parser.replay_ops_ordered_list if op.type == ReplayerReplayOpType.COMM]
    assert [(op.group_name, op.trace_comm_index) for op in comms] == [("44", 0), ("33", 2)]
