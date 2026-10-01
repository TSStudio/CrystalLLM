# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Megatron 1F1B Pipeline Parallelism Execution Graph Builder.

This module provides functions to build execution graphs for Megatron's
non-interleaved 1F1B pipeline parallelism schedule. Spans are cut and named
exactly like a runtime graph capture (``_proc``/``_comm`` counters with the
schedule semantics in ``span_desc``), so the built graph matches what capturing
the same schedule records.
"""

from __future__ import annotations

from ..graph import (
    CrystalGCommPattern,
    CrystalGCommTasks,
    CrystalGraph,
)
from .parallel_layout import (
    DP_ALL_REDUCE,
    GLOBAL_ALL_REDUCE,
    GLOBAL_CHANNEL,
    dp_and_global_channel_members,
    dp_channel_of,
    rank_of,
)
from .span_builder import RankSpanBuilder

FORWARD = "FWD"
BACKWARD = "BWD"
# FORWARD/BACKWARD label the computation spans' span_desc entries written by the
# graph marks, mirroring the runtime's cond_crystal_graph_mark("FWD"/"BWD").

# Communication type fields: {U|D}{R|S}{F|B} = {Up|Down}{Recv|Send}{Forward|Backward}
# Multiple operations in one span are concatenated. These label the comm spans'
# span_desc entries; the span identities are the runtime capture's _comm counters.
RECV_FORWARD = "URF"  # Recv forward activation
SEND_FORWARD = "DSF"  # Send forward activation
RECV_BACKWARD = "DRB"  # Recv backward gradient
SEND_BACKWARD = "USB"  # Send backward gradient
SEND_FORWARD_RECV_BACKWARD = "DSFRB"  # Send forward + recv backward
SEND_BACKWARD_RECV_FORWARD = "USBRF"  # Send backward + recv forward

# Channel base for PP communication tasks; one channel per PP chain
PP_CHANNEL_BASE = "PP"


def pp_channel_of(dp_index: int) -> str:
    """Return the PP channel name of a PP chain."""
    return f"{PP_CHANNEL_BASE}_{dp_index}"


def get_megatron_1f1b_channel_members(
    pipeline_parallel_size: int,
    data_parallel_size: int = 1,
) -> dict[str, list[int]]:
    """Return the channel member tables of a Megatron 1F1B graph.

    A PP channel spans the whole PP chain, so its members are the chain's ranks in stage
    order and the adjacent member pairs form the pipeline's P2P connections.

    Args:
        pipeline_parallel_size: Number of pipeline stages.
        data_parallel_size: Number of DP positions per pipeline stage.

    Returns:
        Mapping from channel name to member global ranks, ordered by group rank.
    """
    members = {
        pp_channel_of(dp_index): [
            rank_of(stage, dp_index, data_parallel_size) for stage in range(pipeline_parallel_size)
        ]
        for dp_index in range(data_parallel_size)
    }
    members.update(dp_and_global_channel_members(pipeline_parallel_size, data_parallel_size))
    return members


def _p2p_comm_tasks(comm_type: str, channel: str) -> CrystalGCommTasks:
    """Return the comm tasks of one p2p exchange, in issue order.

    The two operations of a composite exchange ride one _communicate() call that
    waits for both before returning, so they become one span with two tasks,
    like the runtime's coalescing manager collapses a batch of operations.

    Args:
        comm_type: Communication type field of the exchange.
        channel: PP channel the exchange runs on.

    Returns:
        The comm tasks of the exchange.

    Raises:
        ValueError: If the comm_type is unknown.
    """
    if comm_type in {RECV_FORWARD, SEND_BACKWARD}:
        # Recv forward activation from previous rank (upward receive)
        # Send backward gradient to previous rank (upward send)
        tasks = [CrystalGCommPattern.P2P_UP]
    elif comm_type in {SEND_FORWARD, RECV_BACKWARD}:
        # Send forward activation to next rank (downward send)
        # Recv backward gradient from next rank (downward receive)
        tasks = [CrystalGCommPattern.P2P_DN]
    elif comm_type == SEND_FORWARD_RECV_BACKWARD:
        # Composite: send_forward (P2P_DN) + recv_backward (P2P_DN)
        tasks = [CrystalGCommPattern.P2P_DN, CrystalGCommPattern.P2P_DN]
    elif comm_type == SEND_BACKWARD_RECV_FORWARD:
        # Composite: send_backward (P2P_UP) + recv_forward (P2P_UP)
        tasks = [CrystalGCommPattern.P2P_UP, CrystalGCommPattern.P2P_UP]
    else:
        raise ValueError(f"Unknown comm_type: {comm_type}")
    return CrystalGCommTasks(channel=channel, tasks=tasks)


def _build_graph(
    num_microbatches: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    max_connections: int,
) -> CrystalGraph:
    """Build a 1F1B execution graph with duration_ns=0 for all spans.

    Args:
        num_microbatches: Total number of microbatches
        pipeline_parallel_size: Number of pipeline stages
        data_parallel_size: Number of DP positions per pipeline stage
        max_connections: If 1, enable START-to-START dependency between consecutive spans

    Returns:
        CrystalGraph: The constructed execution graph
    """
    graph = CrystalGraph()
    # A capture of the same schedule records the value on its graph; the built
    # graph matches it, and a set value keeps export and merge available.
    graph.set_max_connections(max_connections)

    def _build_rank_schedule(rank: int, stage: int, dp_index: int) -> None:
        """Build the whole 1F1B schedule of one rank.

        Spans go through the RankSpanBuilder state machine in the order
        forward_backward_pipelining_without_interleaving() executes them. Every
        p2p exchange of the non-interleaved schedule is synchronous
        (_communicate() leaves wait_on_reqs True), so each comm span is added
        with async_op=False and the next computation span starts after it.
        """
        builder = RankSpanBuilder(graph, rank, max_connections)
        channel = pp_channel_of(dp_index)

        def communicate(comm_type: str) -> None:
            """Run one synchronous p2p exchange of the schedule."""
            builder.add_comm_span(_p2p_comm_tasks(comm_type, channel), async_op=False, description=comm_type)

        num_warmup = min(pipeline_parallel_size - stage - 1, num_microbatches)
        num_steady = max(0, num_microbatches - num_warmup)
        is_pipeline_first_stage = stage == 0
        is_pipeline_last_stage = stage == pipeline_parallel_size - 1

        # Computation span opened by the step start
        builder.step_start()

        # Warmup
        for _ in range(num_warmup):
            # recv_forward
            if not is_pipeline_first_stage:
                communicate(RECV_FORWARD)

            # forward_step
            builder.graph_mark(FORWARD)

            # send_forward
            if not is_pipeline_last_stage:
                communicate(SEND_FORWARD)

        # Additional recv for steady state transition
        # recv_forward
        if not is_pipeline_first_stage and num_steady != 0:
            communicate(RECV_FORWARD)

        # Steady
        for i in range(num_steady):
            # forward_step
            builder.graph_mark(FORWARD)

            # send_forward_recv_backward
            if not is_pipeline_last_stage:
                communicate(SEND_FORWARD_RECV_BACKWARD)

            # backward_step
            builder.graph_mark(BACKWARD)

            # send_backward_recv_forward / send_backward
            if not is_pipeline_first_stage:
                if i < num_steady - 1:
                    # Not the last steady iteration: send_backward_recv_forward
                    communicate(SEND_BACKWARD_RECV_FORWARD)
                else:
                    # Last steady iteration: only send_backward (no recv_forward)
                    communicate(SEND_BACKWARD)

        # Cooldown
        for _ in range(num_warmup):
            # recv_backward
            if not is_pipeline_last_stage:
                communicate(RECV_BACKWARD)

            # backward_step
            builder.graph_mark(BACKWARD)

            # send_backward
            if not is_pipeline_first_stage:
                communicate(SEND_BACKWARD)

        # Close the iteration like finalize_model_grads_func(): the DP gradient
        # all-reduce and the global all-reduce, two synchronous collectives that
        # run after the pipeline's last (already synchronous) p2p exchange.
        builder.add_comm_span(
            CrystalGCommTasks(channel=dp_channel_of(stage), tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
            description=DP_ALL_REDUCE,
        )
        builder.add_comm_span(
            CrystalGCommTasks(channel=GLOBAL_CHANNEL, tasks=[CrystalGCommPattern.SYMM]),
            async_op=False,
            description=GLOBAL_ALL_REDUCE,
        )

        # Trailing computation span materialized by the step stop
        builder.step_stop()

    for dp_index in range(data_parallel_size):
        for stage in range(pipeline_parallel_size):
            _build_rank_schedule(rank_of(stage, dp_index, data_parallel_size), stage, dp_index)

    return graph


def build_megatron_1f1b_graph(
    num_microbatches: int,
    pipeline_parallel_size: int,
    data_parallel_size: int = 1,
    max_connections: int = 1,
) -> CrystalGraph:
    """Build the complete Megatron 1F1B graph of every rank, with channel members registered.

    Args:
        num_microbatches: Total number of microbatches
        pipeline_parallel_size: Number of pipeline stages
        data_parallel_size: Number of DP positions per pipeline stage
        max_connections: If 1, enable START-to-START dependency between consecutive spans

    Returns:
        CrystalGraph: The constructed execution graph, ready for analyze()
    """
    assert pipeline_parallel_size > 1
    assert data_parallel_size >= 1
    assert max_connections >= 1
    graph = _build_graph(num_microbatches, pipeline_parallel_size, data_parallel_size, max_connections)
    for channel, members in get_megatron_1f1b_channel_members(pipeline_parallel_size, data_parallel_size).items():
        graph.set_channel_members(channel, members)
    return graph
