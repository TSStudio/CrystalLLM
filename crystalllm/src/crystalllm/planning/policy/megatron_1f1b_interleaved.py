# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Megatron 1F1B Interleaved Pipeline Parallelism Execution Graph Builder.

This module provides functions to build execution graphs for Megatron's
interleaved 1F1B pipeline parallelism schedule. Spans are cut and named
exactly like a runtime graph capture (``_proc``/``_comm`` counters with the
schedule semantics in ``span_desc``), so the built graph matches what capturing
the same schedule records.
"""

from __future__ import annotations

from ..graph import (
    CrystalGCommPattern,
    CrystalGCommTasks,
    CrystalGraph,
    CrystalGSpanId,
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

# Communication type: {U|D}{R|S}{F|B} = {Up|Down}{Recv|Send}{Forward|Backward}
# In interleaved mode with overlap_p2p_comm=True, we use independent send/recv
# operations. These label the comm spans' span_desc entries; the span identities
# are the runtime capture's _comm counters.
RECV_FORWARD = "URF"  # Recv forward activation
SEND_FORWARD = "DSF"  # Send forward activation
RECV_BACKWARD = "DRB"  # Recv backward gradient
SEND_BACKWARD = "USB"  # Send backward gradient

# Channel base for PP communication tasks; one channel per adjacent stage pair of a PP chain
PP_CHANNEL_BASE = "PP"


def pp_channel_of(dp_index: int, first_stage: int, second_stage: int) -> str:
    """Return the PP channel name of an adjacent stage pair inside a PP chain."""
    return f"{PP_CHANNEL_BASE}_{dp_index}_{first_stage}:{second_stage}"


def get_megatron_1f1b_interleaved_channel_members(
    pipeline_parallel_size: int,
    data_parallel_size: int = 1,
) -> dict[str, list[int]]:
    """Return the channel member tables of a Megatron 1F1B interleaved graph.

    Interleaved 1F1B issues independent send/recv operations, so each adjacent stage pair
    of a PP chain gets its own two-member channel.

    Args:
        pipeline_parallel_size: Number of pipeline stages.
        data_parallel_size: Number of DP positions per pipeline stage.

    Returns:
        Mapping from channel name to member global ranks, ordered by group rank.
    """
    members: dict[str, list[int]] = {}
    for dp_index in range(data_parallel_size):
        for stage in range(pipeline_parallel_size):
            next_stage = (stage + 1) % pipeline_parallel_size
            first_stage, second_stage = min(stage, next_stage), max(stage, next_stage)
            channel = pp_channel_of(dp_index, first_stage, second_stage)
            members[channel] = [
                rank_of(first_stage, dp_index, data_parallel_size),
                rank_of(second_stage, dp_index, data_parallel_size),
            ]
    members.update(dp_and_global_channel_members(pipeline_parallel_size, data_parallel_size))
    return members


def _get_schedule_table(
    num_microbatches: int,
    num_model_chunks: int,
    pipeline_parallel_size: int,
) -> list[tuple[int, int]]:
    """Get the schedule table for interleaved PP scheduling.

    The schedule table maps virtual_microbatch_id to (microbatch_id, model_chunk_id).
    For example, for PP2 M5 with VP2:
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1

    microbatch_group_size_per_vp_stage is forced to pipeline_parallel_size

    Args:
        num_microbatches: Total number of microbatches
        num_model_chunks: Number of model chunks (virtual pipeline size)
        pipeline_parallel_size: Number of pipeline stages (equals microbatch_group_size_per_vp_stage)

    Returns:
        List of (microbatch_id, model_chunk_id) tuples indexed by virtual_microbatch_id
    """
    schedule_table = []
    for min_microbatch_id_in_group in range(0, num_microbatches, pipeline_parallel_size):
        if min_microbatch_id_in_group + pipeline_parallel_size >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend([
                (microbatch_id, model_chunk_id)
                for model_chunk_id in range(num_model_chunks)
                for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
            ])
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend([
                (microbatch_id, model_chunk_id)
                for model_chunk_id in range(num_model_chunks)
                for microbatch_id in range(
                    min_microbatch_id_in_group,
                    min_microbatch_id_in_group + pipeline_parallel_size,
                )
            ])
    return schedule_table


def _get_pp_rank_microbatches(
    num_microbatches: int,
    num_model_chunks: int,
    pipeline_parallel_size: int,
    pipeline_parallel_rank: int,
) -> tuple[int, bool, int, int]:
    """Get the number of total, warmup, and remaining microbatches in PP scheduling.

    Args:
        num_microbatches: Total number of microbatches
        num_model_chunks: Number of model chunks (virtual pipeline size)
        pipeline_parallel_size: Number of pipeline stages
        pipeline_parallel_rank: Current pipeline rank

    Returns:
        (total_num_microbatches, are_all_microbatches_in_warmup, num_warmup_microbatches, num_microbatches_remaining)
    """
    total_num_microbatches = num_microbatches * num_model_chunks

    # For interleaved schedule:
    # Run (num_model_chunks-1)*pipeline_parallel_size on all workers,
    # followed by more microbatches after depending on stage ID
    num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
    num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size

    are_all_microbatches_in_warmup = False
    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def _get_model_chunk_id(
    virtual_microbatch_id: int, forward: bool, num_model_chunks: int, model_chunk_id_table: list[int]
) -> int:
    """Get the model chunk ID given the virtual microbatch ID."""
    total_num_microbatches = len(model_chunk_id_table)
    model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
    # For backward, the model_chunk_id is in reversed order of forward
    if not forward:
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id


def _recv_tensor_from_previous_stage(
    virtual_microbatch_id: int,
    forward: bool,
    pipeline_parallel_size: int,
    pipeline_parallel_rank: int,
    num_model_chunks: int,
    model_chunk_id_table: list[int],
) -> tuple[bool, int]:
    """Determine if current rank should receive tensor from previous stage.

    Args:
        virtual_microbatch_id: The virtual microbatch ID
        forward: Whether in forward pass
        pipeline_parallel_size: Number of pipeline stages
        pipeline_parallel_rank: Current pipeline rank
        num_model_chunks: Number of model chunks
        model_chunk_id_table: Schedule table mapping virtual_microbatch_id to model_chunk_id

    Returns:
        (recv, next_model_chunk_id) where recv indicates whether to receive,
        and next_model_chunk_id is the model chunk ID for the next microbatch
    """
    recv = True
    # The leading pipeline stage is the first rank in fwd and the last rank in bwd
    is_leading_pipeline_stage = (
        pipeline_parallel_rank == 0 if forward else pipeline_parallel_rank == pipeline_parallel_size - 1
    )

    last_model_chunk = (num_model_chunks - 1) if forward else 0

    if is_leading_pipeline_stage:
        # The leading pipeline stage is ahead of the ending stage
        # by (pipeline_parallel_size - 1)
        if virtual_microbatch_id < (pipeline_parallel_size - 1):
            # The ending stage has not produced any tensors, so no recv will be initiated
            recv = False
            next_model_chunk_id = _get_model_chunk_id(
                virtual_microbatch_id + 1, forward, num_model_chunks, model_chunk_id_table
            )
        else:
            # Find the model chunk of the aligned microbatches in the ending stage
            next_model_chunk_id = _get_model_chunk_id(
                virtual_microbatch_id - (pipeline_parallel_size - 1), forward, num_model_chunks, model_chunk_id_table
            )
        # Last model chunk in the final stage does not produce tensors
        if next_model_chunk_id == last_model_chunk:
            recv = False
        if forward:
            # Model chunk id increases in forward
            next_model_chunk_id += 1
        else:
            # Model chunk id decreases in backward
            next_model_chunk_id -= 1
    else:
        next_model_chunk_id = _get_model_chunk_id(
            virtual_microbatch_id + 1, forward, num_model_chunks, model_chunk_id_table
        )

    return recv, next_model_chunk_id


def _build_graph(
    num_microbatches: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    num_model_chunks: int,
    max_connections: int,
) -> CrystalGraph:
    """Build an interleaved 1F1B execution graph with duration_ns=0 for all spans.

    This implements the overlap_p2p_comm=True and overlap_p2p_comm_warmup_flush=False path.

    Args:
        num_microbatches: Number of microbatches per pipeline stage
        pipeline_parallel_size: Number of pipeline stages
        data_parallel_size: Number of DP positions per pipeline stage
        num_model_chunks: Number of model chunks (virtual pipeline size)
        max_connections: If 1, enable START-to-START dependency between consecutive spans

    Returns:
        CrystalGraph: The constructed execution graph
    """
    graph = CrystalGraph()
    # A capture of the same schedule records the value on its graph; the built
    # graph matches it, and a set value keeps export and merge available.
    graph.set_max_connections(max_connections)

    # Build schedule table
    schedule_table = _get_schedule_table(num_microbatches, num_model_chunks, pipeline_parallel_size)
    _, model_chunk_id_table = zip(*schedule_table, strict=True)

    def _build_rank_schedule(rank: int, stage: int, dp_index: int) -> None:
        """Build the whole interleaved schedule of one rank.

        Spans go through the RankSpanBuilder state machine in the order
        forward_backward_pipelining_with_interleaving() executes them under
        overlap_p2p_comm=True: the warmup exchanges stay synchronous (the
        non-warmup-flush path leaves wait_on_reqs True), the steady-state
        exchanges are issued asynchronously and waited at the exact points the
        schedule calls wait() on its saved handles, and the cooldown exchanges
        are synchronous again.
        """
        builder = RankSpanBuilder(graph, rank, max_connections)

        def channel_of(peer_stage: int) -> str:
            """Return the channel of the stage pair this rank and its peer form."""
            return pp_channel_of(dp_index, min(stage, peer_stage), max(stage, peer_stage))

        def communicate(comm_type: str, *, async_op: bool) -> CrystalGSpanId:
            """Run one p2p operation (isend/irecv) of the schedule."""
            if comm_type in {RECV_FORWARD, SEND_BACKWARD}:
                # Recv forward activation from previous rank (upward receive)
                # Send backward gradient to previous rank (upward send)
                tasks = [CrystalGCommPattern.P2P_UP]
                channel = channel_of((stage - 1) % pipeline_parallel_size)
            elif comm_type in {SEND_FORWARD, RECV_BACKWARD}:
                # Send forward activation to next rank (downward send)
                # Recv backward gradient from next rank (downward receive)
                tasks = [CrystalGCommPattern.P2P_DN]
                channel = channel_of((stage + 1) % pipeline_parallel_size)
            else:
                raise ValueError(f"Unknown comm_type: {comm_type}")
            return builder.add_comm_span(
                CrystalGCommTasks(channel=channel, tasks=tasks), async_op=async_op, description=comm_type
            )

        def send_forward_recv_forward(
            send_forward: bool, recv_forward: bool, *, async_op: bool
        ) -> dict[str, CrystalGSpanId]:
            """Run send_forward_recv_forward, one comm span per operation.

            _p2p_ops() issues the operations of one exchange in an order fixed
            by the stage's parity and the schedule keeps the returned wait
            handles in a dict keyed by operation, so the builder returns the
            operation's comm span IDs keyed the same way.
            """
            wait_handles: dict[str, CrystalGSpanId] = {}
            if stage % 2 == 0:
                # Even stages: send_forward then recv_forward
                ordered_ops = [(SEND_FORWARD, send_forward), (RECV_FORWARD, recv_forward)]
            else:
                # Odd stages: recv_forward then send_forward
                ordered_ops = [(RECV_FORWARD, recv_forward), (SEND_FORWARD, send_forward)]
            for comm_type, enabled in ordered_ops:
                if enabled:
                    wait_handles[comm_type] = communicate(comm_type, async_op=async_op)
            return wait_handles

        def send_backward_recv_backward(
            send_backward: bool, recv_backward: bool, *, async_op: bool
        ) -> dict[str, CrystalGSpanId]:
            """Run send_backward_recv_backward, one comm span per operation."""
            wait_handles: dict[str, CrystalGSpanId] = {}
            if stage % 2 == 0:
                # Even stages: send_backward then recv_backward
                ordered_ops = [(SEND_BACKWARD, send_backward), (RECV_BACKWARD, recv_backward)]
            else:
                # Odd stages: recv_backward then send_backward
                ordered_ops = [(RECV_BACKWARD, recv_backward), (SEND_BACKWARD, send_backward)]
            for comm_type, enabled in ordered_ops:
                if enabled:
                    wait_handles[comm_type] = communicate(comm_type, async_op=async_op)
            return wait_handles

        # Calculate phase sizes for this pipeline stage
        (
            total_num_microbatches,
            are_all_microbatches_in_warmup,
            num_warmup_microbatches,
            num_microbatches_remaining,
        ) = _get_pp_rank_microbatches(
            num_microbatches,
            num_model_chunks,
            pipeline_parallel_size,
            stage,
        )

        is_pipeline_first_stage = stage == 0
        is_pipeline_last_stage = stage == pipeline_parallel_size - 1

        def is_vp_first_stage(vp_stage: int) -> bool:
            return is_pipeline_first_stage and vp_stage == 0

        def is_vp_last_stage(vp_stage: int) -> bool:
            return is_pipeline_last_stage and vp_stage == num_model_chunks - 1

        # Wait handles of the async p2p operations, kept exactly like the
        # recv_prev / recv_next handle lists and the send_prev / send_next
        # handle slots of the Megatron schedule; comm_span_wait() draws the
        # wait boundary each handle's wait() draws at runtime.
        recv_prev_wait_handles: list[CrystalGSpanId] = []
        recv_next_wait_handles: list[CrystalGSpanId] = []
        send_prev_wait_handle: CrystalGSpanId | None = None
        send_next_wait_handle: CrystalGSpanId | None = None

        # Computation span opened by the step start
        builder.step_start()

        # ===== Warmup phase =====
        # Initial recv_forward before warmup loop
        if not is_vp_first_stage(0):
            communicate(RECV_FORWARD, async_op=False)

        for k in range(num_warmup_microbatches):
            cur_model_chunk_id = _get_model_chunk_id(k, True, num_model_chunks, model_chunk_id_table)

            # Determine if tensor should be received from previous stage
            recv_prev, _ = _recv_tensor_from_previous_stage(
                k, True, pipeline_parallel_size, stage, num_model_chunks, model_chunk_id_table
            )

            # No receive in last iteration when recv iteration k+1
            if k == (total_num_microbatches - 1):
                recv_prev = False

            # Forward step
            builder.graph_mark(FORWARD)

            # Don't send tensor downstream if on last stage.
            send_next = not is_vp_last_stage(cur_model_chunk_id)

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            send_forward_recv_forward(send_next, recv_prev, async_op=False)

            if k == (num_warmup_microbatches - 1) and not are_all_microbatches_in_warmup:
                # Last warmup iteration: receive the backward gradient the
                # steady state consumes first (nothing to send yet), the
                # schedule's first exchange issued with overlap_p2p_comm.
                recv_next = not is_pipeline_last_stage
                bwd_wait_handles = send_backward_recv_backward(False, recv_next, async_op=True)
                send_prev_wait_handle = bwd_wait_handles.get(SEND_BACKWARD)
                if RECV_BACKWARD in bwd_wait_handles:
                    recv_next_wait_handles.append(bwd_wait_handles[RECV_BACKWARD])

        # ===== Steady state (1F1B) =====
        for k in range(num_microbatches_remaining):
            # Forward pass
            forward_k = k + num_warmup_microbatches
            forward_model_chunk_id = _get_model_chunk_id(forward_k, True, num_model_chunks, model_chunk_id_table)

            if not is_vp_first_stage(forward_model_chunk_id) and recv_prev_wait_handles:
                # Wait for the activation tensor this forward step consumes
                builder.comm_span_wait(recv_prev_wait_handles.pop(0))

            # Forward step
            builder.graph_mark(FORWARD)

            # Last virtual stage no activation tensor to send.
            send_next = not is_vp_last_stage(forward_model_chunk_id)

            recv_prev, _ = _recv_tensor_from_previous_stage(
                forward_k, True, pipeline_parallel_size, stage, num_model_chunks, model_chunk_id_table
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation
            # tensor from the previous stage, without waiting.
            fwd_wait_handles = send_forward_recv_forward(send_next, recv_prev, async_op=True)

            # Wait for the previous iteration's send to complete.
            builder.comm_span_wait(send_next_wait_handle)
            send_next_wait_handle = fwd_wait_handles.get(SEND_FORWARD)
            if RECV_FORWARD in fwd_wait_handles:
                recv_prev_wait_handles.append(fwd_wait_handles[RECV_FORWARD])

            # Backward pass
            backward_k = k
            backward_model_chunk_id = _get_model_chunk_id(backward_k, False, num_model_chunks, model_chunk_id_table)

            if not is_vp_last_stage(backward_model_chunk_id) and recv_next_wait_handles:
                # Wait for the gradient tensor this backward step consumes
                builder.comm_span_wait(recv_next_wait_handles.pop(0))

            # Backward step
            builder.graph_mark(BACKWARD)

            # First virtual stage no activation gradient tensor to send.
            send_prev = not is_vp_first_stage(backward_model_chunk_id)

            recv_next, _ = _recv_tensor_from_previous_stage(
                backward_k, False, pipeline_parallel_size, stage, num_model_chunks, model_chunk_id_table
            )

            # Send gradient tensor to the previous stage and receive gradient
            # tensor from the next stage, without waiting.
            bwd_wait_handles = send_backward_recv_backward(send_prev, recv_next, async_op=True)

            # Wait for the previous iteration's send to complete.
            builder.comm_span_wait(send_prev_wait_handle)
            send_prev_wait_handle = bwd_wait_handles.get(SEND_BACKWARD)
            if RECV_BACKWARD in bwd_wait_handles:
                recv_next_wait_handles.append(bwd_wait_handles[RECV_BACKWARD])

        # ===== Cooldown phase =====
        if are_all_microbatches_in_warmup and not is_vp_last_stage(num_model_chunks - 1):
            communicate(RECV_BACKWARD, async_op=False)

        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = _get_model_chunk_id(k, False, num_model_chunks, model_chunk_id_table)

            if not is_vp_last_stage(cur_model_chunk_id) and k != 0 and recv_next_wait_handles:
                # Wait for the gradient tensor this backward step consumes
                builder.comm_span_wait(recv_next_wait_handles.pop(0))

            recv_next, _ = _recv_tensor_from_previous_stage(
                k, False, pipeline_parallel_size, stage, num_model_chunks, model_chunk_id_table
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Backward step
            builder.graph_mark(BACKWARD)

            # First virtual stage no activation gradient tensor to send.
            send_prev = not is_vp_first_stage(cur_model_chunk_id)

            # Synchronous exchange: the cooldown path does not pass
            # overlap_p2p_comm, so _communicate() waits before returning.
            send_backward_recv_backward(send_prev, recv_next, async_op=False)

        # Wait for the last async backward send, like send_prev_wait_handle.wait()
        # right before the schedule enables the grad reductions.
        builder.comm_span_wait(send_prev_wait_handle)

        assert not recv_prev_wait_handles, "recv_prev wait handles should be cleared at the end of a step"
        assert not recv_next_wait_handles, "recv_next wait handles should be cleared at the end of a step"

        # Close the iteration like finalize_model_grads_func(): the DP gradient
        # all-reduce and the global all-reduce, two synchronous collectives
        # that run after the pipeline's last p2p wait.
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


def build_megatron_1f1b_interleaved_graph(
    num_microbatches: int,
    pipeline_parallel_size: int,
    num_model_chunks: int,
    data_parallel_size: int = 1,
    max_connections: int = 1,
) -> CrystalGraph:
    """Build the complete interleaved 1F1B graph of every rank, with channel members registered.

    This function models the interleaved 1F1B schedule with:
    - overlap_p2p_comm=True: Independent send/recv operations
    - overlap_p2p_comm_warmup_flush=False: No warmup flush optimization
    - microbatch_group_size_per_vp_stage = pipeline_parallel_size

    Args:
        num_microbatches: Total number of microbatches
        pipeline_parallel_size: Number of pipeline stages
        num_model_chunks: Number of model chunks (virtual pipeline size)
        data_parallel_size: Number of DP positions per pipeline stage
        max_connections: If 1, enable START-to-START dependency between consecutive spans

    Returns:
        CrystalGraph: The constructed execution graph, ready for analyze()
    """
    assert pipeline_parallel_size > 1
    assert num_model_chunks > 1
    assert num_microbatches % pipeline_parallel_size == 0
    assert data_parallel_size >= 1
    assert max_connections >= 1
    graph = _build_graph(
        num_microbatches,
        pipeline_parallel_size,
        data_parallel_size,
        num_model_chunks,
        max_connections,
    )
    channel_members = get_megatron_1f1b_interleaved_channel_members(pipeline_parallel_size, data_parallel_size)
    for channel, members in channel_members.items():
        graph.set_channel_members(channel, members)
    return graph
