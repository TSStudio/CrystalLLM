# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Sandbox-local CP stays in raw traces, but not graphs or replay operations."""

import functools
import gzip
from unittest import mock

import pytest
import torch

from crystalllm import runtime
from crystalllm.planning import SpanType
from crystalllm.replay.data import ReplayerReplayOpType
from crystalllm.replay.parser import CrystalTraceParser
from crystalllm.runtime.context import CrystalRuntimeContext
from crystalllm.torch import collective_handler, fake, wrapper
from crystalllm.torch.c10d_trace import CrystalC10DTracer


@pytest.fixture
def capture(monkeypatch, tmp_path):
    """Use the real 26.8.27 CPU capture state machine in its analysis window."""
    for name, value in {
        "RANK": "9", "LOCAL_WORLD_SIZE": "8", "CRYSTAL_MODE": "CAPTURE",
        "CRYSTAL_LOG_DIR": str(tmp_path), "CRYSTAL_GRAPH_CAPTURING": "true",
        "ENABLE_CRYSTAL_C10D_TRACE": "true", "CRYSTAL_C10D_CAPTURING": "true",
        "CRYSTAL_ANALYSIS_STEP_START": "1", "CRYSTAL_ANALYSIS_STEP_END": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ENABLE_CRYSTAL_RUNTIME_PROFILER", raising=False)
    for singleton in (CrystalRuntimeContext, CrystalC10DTracer):
        monkeypatch.setattr(singleton, "_instance", None)
        monkeypatch.setattr(singleton, "_initialized", False)
    ctx = CrystalRuntimeContext()
    for module in (runtime, collective_handler):
        monkeypatch.setattr(module, "get_context", lambda: ctx)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    monkeypatch.setattr(torch.distributed, "get_group_rank",
                        lambda group, rank: group._global_ranks_in_group.index(rank))
    monkeypatch.setattr(torch.distributed, "get_global_rank",
                        lambda group, rank: group._global_ranks_in_group[rank])
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks",
                        lambda group: group._global_ranks_in_group)
    yield ctx
    ctx.destroy()


def comm_spans(ctx):
    """Ignore computation spans introduced by the new communication boundaries."""
    graph = ctx._captured_graph_per_steps[1]
    return [span for rank in graph.ranks.values() for span in rank.spans.values()
            if span.span_type == SpanType.COMM]


def p2p(name, group, keyword_group=False):
    """Invoke the capture wrappers using the real c10d signature metadata."""
    @functools.wraps(getattr(torch.distributed, name))
    def operation(*args, **kwargs):
        raise AssertionError("Capture must not execute real communication")

    operation.__name__ = name
    operation = wrapper.cond_crystal_c10d_trace_writer(
        wrapper.cond_crystal_collectives_handler(operation)
    )
    if keyword_group:
        return operation(torch.empty(4), 1, group=group)
    return operation(torch.empty(4), 1, group)


@pytest.mark.parametrize("op", ["isend", "irecv"])
@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("keyword_group", [False, True])
def test_cp_is_traced_but_not_replayed_and_pp_keeps_trace_graph_ids(
    capture, tmp_path, op, coalesced, keyword_group
):
    """Retain raw CP records while keeping only PP spans and replay operations."""
    ctx = capture
    cp = fake.CrystalFakeProcessGroup([9, 1], 0, "17", "CONTEXT-9-17-nccl")
    pp = fake.CrystalFakeProcessGroup([9, 1], 0, "33", "PIPELINE-9-33-nccl")
    for name, desc in [("17", "CONTEXT-9-17-nccl"), ("33", "PIPELINE-9-33-nccl")]:
        ctx.cond_custom_write_c10d_trace("NEW_GROUP", str({
            "ranks": [9, 1], "backend": "nccl", "group_name": name, "group_desc": desc,
        }))
    ctx.step_start()
    if coalesced:
        wrapper.cond_crystal_c10d_start_coalescing(cp, True, True)
    work = p2p(op, cp, keyword_group)
    if coalesced:
        assert isinstance(work, fake.CrystalFakeWork)
        work = wrapper.cond_crystal_c10d_end_coalescing(cp, True, True)
    work.wait()
    assert not ctx.is_comm_span_coalescing
    assert not comm_spans(ctx)
    assert any(line.startswith(f"0#{op}#") and "#17#CONTEXT-" in line
               for line in ctx._c10d_trace._trace_buffer)
    assert ctx._comm_trace_counter == 1
    assert ctx.last_trace_comm_index == 0
    assert not ctx.is_comm_trace_counter_frozen

    # A subsequent PP operation must still create its span and wait dependency.
    if coalesced:
        wrapper.cond_crystal_c10d_start_coalescing(pp, True, True)
    work = p2p(op, pp, keyword_group)
    if coalesced:
        assert isinstance(work, fake.CrystalFakeWork)
        work = wrapper.cond_crystal_c10d_end_coalescing(pp, True, True)
    work.wait()
    spans = comm_spans(ctx)
    assert len(spans) == 1
    assert spans[0].comm_tasks.channel == ("33" if coalesced else "33_0:1")
    assert len(spans[0].comm_tasks.tasks) == 1
    assert spans[0].span_id.count_field == 1
    assert ctx._comm_trace_counter == spans[0].span_id.count_field + 1
    assert any(line.startswith(f"1#{op}#") and "#33#PIPELINE-" in line
               for line in ctx._c10d_trace._trace_buffer)

    trace_file = tmp_path / "trace_9.log.gz"
    with gzip.open(trace_file, "wt") as stream:
        stream.write("\n".join(ctx._c10d_trace._trace_buffer) + "\n")
    parser = CrystalTraceParser(str(trace_file))
    assert set(parser.comm_groups_dict) == {"0", "17", "33"}
    assert not parser.replay_comm_ops_dict["17"]
    replay = parser.replay_ops_ordered_list[1:]
    expected_types = [ReplayerReplayOpType.COMM]
    if coalesced:
        expected_types = [ReplayerReplayOpType.COALESCING_START, *expected_types,
                          ReplayerReplayOpType.COALESCING_END]
    assert [item.type for item in replay] == expected_types
    assert all(item.info.group_name == "33" for item in replay)
    assert all(item.info.trace_comm_index == 1 for item in replay)


@pytest.mark.parametrize(
    ("op", "group_position"),
    [("all_gather_into_tensor", 2), ("all_to_all_single", 4), ("all_reduce", 2), ("barrier", 0)],
)
def test_cp_collectives_are_traced_without_graph_spans(capture, op, group_position):
    """Preserve CP collective/coalescing records without cross-Sandbox spans."""
    ctx = capture
    cp = fake.CrystalFakeProcessGroup([9, 1], 0, "17", "CONTEXT-9-17-nccl")
    metadata = "{'ranks': [9, 1], 'group_name': '17', 'group_desc': 'CONTEXT-9-17-nccl'}"
    ctx.cond_custom_write_c10d_trace("NEW_GROUP", metadata)
    ctx.step_start()

    @functools.wraps(getattr(torch.distributed, op))
    def operation(*args, **kwargs):
        raise AssertionError("Capture must not execute real communication")

    operation.__name__ = op
    operation = wrapper.cond_crystal_c10d_trace_writer(
        wrapper.cond_crystal_collectives_handler(operation)
    )
    args = [None] * group_position + [cp]
    wrapper.cond_crystal_c10d_start_coalescing(cp, True, True)
    operation(*args, async_op=True)
    wrapper.cond_crystal_c10d_end_coalescing(cp, True, True).wait()

    lines = ctx._c10d_trace._trace_buffer
    expected_records = ("NEW_GROUP", "SCHEDULE", "COALESCING", "0", "COALESCING")
    assert tuple(line.split("#", 1)[0] for line in lines) == expected_records
    assert lines[0] == "NEW_GROUP#" + metadata
    assert lines[2].startswith("COALESCING#START#17#")
    assert lines[3].startswith(f"0#{op}#")
    assert lines[4].startswith("COALESCING#END#17#")
    assert ctx._comm_trace_counter == 1
    assert not ctx.is_comm_trace_counter_frozen
    assert not ctx.is_comm_span_coalescing
    assert not comm_spans(ctx)


def test_non_capture_cp_is_not_suppressed(monkeypatch):
    """Pass non-capture CP calls through without invoking the graph filter."""
    monkeypatch.delenv("CRYSTAL_C10D_CAPTURING", raising=False)
    calls = []

    @wrapper.cond_crystal_collectives_handler
    def isend(*args):
        calls.append(args)

    cp = fake.CrystalFakeProcessGroup([0, 8], 0, "17", "CONTEXT-0-17-nccl")
    isend(None, 8, cp)
    assert calls == [(None, 8, cp)]


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("op", ["isend", "irecv", "all_reduce"])
def test_cp_profiler_executes_real_work_without_cp_graph_spans(capture, monkeypatch, op, coalesced):
    """Keep CP real and untargeted while preserving profiler completion slots."""
    ctx = capture
    ctx.step_start()
    cp = fake.CrystalFakeProcessGroup([9, 1], 0, "17", "CONTEXT-9-17-nccl")
    real_work = mock.Mock()
    operation = mock.Mock(return_value=real_work, __name__=op)
    slot = mock.Mock()
    monkeypatch.setattr(ctx, "cond_record_profiler_comm_slot", slot)
    if coalesced:
        wrapper.cond_crystal_runtime_profiler_start_coalescing(cp, True)
    work = collective_handler.crystal_runtime_profiler_collectives_handler(
        operation, [torch.empty(4), 1, cp], {} if op != "all_reduce" else {"async_op": True}
    )
    operation.assert_called_once()
    assert work is real_work
    if coalesced:
        slot.assert_not_called()
        launch = mock.Mock(return_value=real_work)
        work = wrapper.cond_crystal_runtime_profiler_end_coalescing(cp, True, launch)
        launch.assert_called_once_with()
    assert work is real_work
    slot.assert_called_once_with(cp, "17")
    assert not comm_spans(ctx)
    assert not ctx.is_comm_span_coalescing
