# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for Megatron 1F1B Interleaved pipeline parallelism policy."""

import unittest

from crystalllm.planning import CrystalGSpanPoint, SpanPointType, SpanType
from crystalllm.planning.policy.megatron_1f1b_interleaved import (
    BACKWARD,
    FORWARD,
    RECV_BACKWARD,
    RECV_FORWARD,
    SEND_BACKWARD,
    SEND_FORWARD,
    build_megatron_1f1b_interleaved_graph,
)
from crystalllm.planning.policy.parallel_layout import DP_ALL_REDUCE, GLOBAL_ALL_REDUCE

# Time scale factor: convert test time units to nanoseconds
TIME_SCALE = 1_000_000

# span_desc labels of the four P2P communication span kinds
P2P_DESCRIPTIONS = {RECV_FORWARD, SEND_FORWARD, RECV_BACKWARD, SEND_BACKWARD}


def ms(duration_ms: int) -> int:
    """Convert milliseconds to nanoseconds for test data."""
    return duration_ms * TIME_SCALE


def span_descriptions(graph, rank_id: int, span) -> list[str]:
    """Return the span_desc labels of one span of a rank."""
    return graph.ranks[rank_id].span_desc.get(span.span_id, [])


def count_comm_spans(graph, rank_id: int) -> int:
    """Count the P2P communication spans of a rank, excluding the closing collectives."""
    return sum(
        1
        for span in graph.ranks[rank_id].spans.values()
        if span.span_type == SpanType.COMM
        and any(description in P2P_DESCRIPTIONS for description in span_descriptions(graph, rank_id, span))
    )


def set_span_durations(graph, forward_duration_ns: int, backward_duration_ns: int) -> None:
    """Set uniform durations for the marked computation spans of every rank.

    The computation spans are named like a runtime capture, so the schedule
    semantics live in span_desc and the duration lists are indexed by the
    count_field of the rank's ``_proc`` spans. A span carrying both marks (two
    steps that no communication boundary separates) gets the sum of both
    durations.
    """
    duration_dict: dict[int, dict[str, list[int]]] = {}
    for rank_id, rank in graph.ranks.items():
        proc_spans = sorted(
            (span for span in rank.spans.values() if span.span_type == SpanType.PROC),
            key=lambda span: span.span_id.count_field,
        )
        durations = []
        for span in proc_spans:
            descriptions = rank.span_desc.get(span.span_id, [])
            duration_ns = 0
            if FORWARD in descriptions:
                duration_ns += forward_duration_ns
            if BACKWARD in descriptions:
                duration_ns += backward_duration_ns
            durations.append(duration_ns)
        duration_dict[rank_id] = {proc_spans[0].span_id.type_field: durations}
    graph.load_span_duration_dict(duration_dict)


def interleaved_num_warmup(pipeline_parallel_size: int, stage: int, num_model_chunks: int) -> int:
    """Warmup virtual-microbatch count of a stage in the interleaved schedule."""
    return (pipeline_parallel_size - stage - 1) * 2 + (num_model_chunks - 1) * pipeline_parallel_size


def proc_execution_pattern(graph, rank_id: int) -> str:
    """Return a rank's computation steps as an "F"/"B" string in launch order.

    The computation spans of a rank form a dependency chain, so their earliest
    start times are strictly increasing: sorting by them recovers the order the
    schedule executes. Each span contributes one character per marked step, so
    two steps that no communication boundary separates appear as "FB" on their
    shared span, exactly like in a runtime capture of the same schedule.
    """
    rank = graph.ranks[rank_id]
    marked_spans = [
        span
        for span in rank.spans.values()
        if span.span_type == SpanType.PROC
        and any(description in {FORWARD, BACKWARD} for description in rank.span_desc.get(span.span_id, []))
    ]
    marked_spans.sort(
        key=lambda span: graph.get_points_interval(
            end_point=CrystalGSpanPoint(rank_id, span.span_id, SpanPointType.START)
        )
    )
    pattern = ""
    for span in marked_spans:
        for description in rank.span_desc[span.span_id]:
            if description in {FORWARD, BACKWARD}:
                pattern += "F" if description == FORWARD else "B"
    return pattern


def expected_1f1b_pattern(num_warmup: int, num_steady: int) -> str:
    """Expected PROC pattern of a 1F1B rank: F^warmup (FB)^steady B^warmup."""
    return "F" * num_warmup + "FB" * num_steady + "B" * num_warmup


class TestMegatron1F1BInterleaved(unittest.TestCase):
    """Test the Megatron 1F1B interleaved execution graph policy.

    These tests verify the schedule modeling of build_megatron_1f1b_interleaved_graph():
    - Phase structure per rank, checked through the launch-order computation
      pattern: warmup forwards, strictly alternating steady state, cooldown
      backwards, with the interleaved warmup 2*(PP - stage - 1) + (chunks - 1)*PP
    - max_connections only tightening dependencies, never changing the schedule
    - Pipeline bubble behavior of the total execution time
    - Closing collectives: DP and global all-reduce synchronization groups
    """

    def test_basic_interleaved_two_ranks(self):
        """Test the basic 2-rank, 4-microbatch, 2-chunk schedule.

        The interleaved warmup is 2*(PP - stage - 1) + (chunks - 1)*PP virtual
        microbatches, after which steady state alternates 1F1B. The schedule is
        identical for max_connections 1 and 4, since extra connections only
        tighten dependencies, never reorder spans.
        """
        num_microbatches, pipeline_parallel_size, num_model_chunks = 4, 2, 2
        total_virtual = num_microbatches * num_model_chunks

        for max_connections in (1, 4):
            with self.subTest(max_connections=max_connections):
                graph = build_megatron_1f1b_interleaved_graph(
                    num_microbatches=num_microbatches,
                    pipeline_parallel_size=pipeline_parallel_size,
                    num_model_chunks=num_model_chunks,
                    max_connections=max_connections,
                )
                set_span_durations(graph, forward_duration_ns=ms(30), backward_duration_ns=ms(60))
                graph.analyze()

                # Independent send/recv per direction on every chunk boundary:
                # rank0: 8 send_forward + 4 recv_forward + 8 recv_backward + 4 send_backward
                # rank1: mirrored (8 recv_forward + 4 send_forward + 8 send_backward
                #        + 4 recv_backward)
                self.assertEqual(count_comm_spans(graph, 0), 24, "Rank 0 comm span count mismatch")
                self.assertEqual(count_comm_spans(graph, 1), 24, "Rank 1 comm span count mismatch")

                # The built graph carries what a capture of the same schedule
                # records: the max_connections it was built under and the launch
                # shape of every exchange, with the steady-state async ones
                # distinguished from the synchronous warmup/cooldown ones.
                self.assertEqual(graph.max_connections, max_connections)
                async_flags = {
                    span.comm_tasks.async_op
                    for rank in graph.ranks.values()
                    for span in rank.spans.values()
                    if span.span_type == SpanType.COMM
                }
                self.assertEqual(async_flags, {True, False})

                # Launch-order computation pattern
                for stage in range(pipeline_parallel_size):
                    num_warmup = interleaved_num_warmup(pipeline_parallel_size, stage, num_model_chunks)
                    expected = expected_1f1b_pattern(num_warmup, total_virtual - num_warmup)
                    self.assertEqual(
                        proc_execution_pattern(graph, stage),
                        expected,
                        f"Rank {stage} execution pattern mismatch",
                    )

    def test_interleaved_four_ranks(self):
        """Test the 4-rank, 32-microbatch, 2-chunk schedule: staggered warmup and bubble.

        Every stage runs 2*(PP - stage - 1) + (chunks - 1)*PP warmup forwards,
        and the total time stays between the serial lower bound over all 64
        virtual microbatches and twice it.
        """
        num_microbatches, pipeline_parallel_size, num_model_chunks = 32, 4, 2
        total_virtual = num_microbatches * num_model_chunks  # 64

        for max_connections in (1, 4):
            with self.subTest(max_connections=max_connections):
                graph = build_megatron_1f1b_interleaved_graph(
                    num_microbatches=num_microbatches,
                    pipeline_parallel_size=pipeline_parallel_size,
                    num_model_chunks=num_model_chunks,
                    max_connections=max_connections,
                )
                set_span_durations(graph, forward_duration_ns=ms(30), backward_duration_ns=ms(60))
                graph.analyze()

                for stage in range(pipeline_parallel_size):
                    num_warmup = interleaved_num_warmup(pipeline_parallel_size, stage, num_model_chunks)
                    expected = expected_1f1b_pattern(num_warmup, total_virtual - num_warmup)
                    self.assertEqual(
                        proc_execution_pattern(graph, stage),
                        expected,
                        f"Rank {stage} execution pattern mismatch",
                    )

                # Pipeline bubble over the 64 virtual microbatches
                total_time = graph.get_points_interval()
                theoretical_min = ms(total_virtual * 90)  # 64 * (30ms fwd + 60ms bwd)
                self.assertGreater(total_time, theoretical_min)
                self.assertLess(total_time, theoretical_min * 2)

    def test_interleaved_different_configurations(self):
        """Test step counts across configurations with varying chunk counts.

        Every rank executes exactly one forward and one backward step per virtual
        microbatch (num_microbatches * num_model_chunks), for chunk counts from 2
        to 4 and both max_connections modes. A step is counted through its
        span_desc mark, wherever the communication boundaries of the stage put it.
        """
        configs = [
            (4, 2, 2),
            (8, 4, 2),
            (8, 4, 4),
            (16, 4, 4),
        ]

        for num_mb, pp_size, num_chunks in configs:
            for max_conn in (1, 4):
                with self.subTest(
                    num_microbatches=num_mb,
                    pipeline_parallel_size=pp_size,
                    num_model_chunks=num_chunks,
                    max_connections=max_conn,
                ):
                    graph = build_megatron_1f1b_interleaved_graph(
                        num_microbatches=num_mb,
                        pipeline_parallel_size=pp_size,
                        num_model_chunks=num_chunks,
                        max_connections=max_conn,
                    )
                    set_span_durations(graph, forward_duration_ns=ms(30), backward_duration_ns=ms(60))

                    total_virtual_microbatches = num_mb * num_chunks

                    for rank_id in range(pp_size):
                        rank = graph.ranks[rank_id]
                        fwd_count = sum(
                            1
                            for span in rank.spans.values()
                            if span.span_type == SpanType.PROC and FORWARD in rank.span_desc.get(span.span_id, [])
                        )
                        bwd_count = sum(
                            1
                            for span in rank.spans.values()
                            if span.span_type == SpanType.PROC and BACKWARD in rank.span_desc.get(span.span_id, [])
                        )
                        self.assertEqual(
                            fwd_count, total_virtual_microbatches, f"Rank {rank_id} forward count mismatch"
                        )
                        self.assertEqual(
                            bwd_count, total_virtual_microbatches, f"Rank {rank_id} backward count mismatch"
                        )

                    # The topological sort inside analyze() rejects dependency cycles
                    graph.analyze()

    def test_interleaved_closing_collectives(self):
        """Test that every rank ends with one DP group all-reduce and one global all-reduce.

        The DP all-reduce members are the ranks of the same pipeline stage, so they sit in
        different PP chains, which is exactly the synchronization a single loaded chain
        cannot express.
        """
        pipeline_parallel_size, data_parallel_size = 4, 3
        graph = build_megatron_1f1b_interleaved_graph(
            num_microbatches=4,
            pipeline_parallel_size=pipeline_parallel_size,
            num_model_chunks=2,
            data_parallel_size=data_parallel_size,
        )

        self.assertEqual(len(graph.ranks), pipeline_parallel_size * data_parallel_size)
        for rank_id, rank in graph.ranks.items():
            dp_spans = [
                span
                for span in rank.spans.values()
                if span.span_type == SpanType.COMM and DP_ALL_REDUCE in rank.span_desc.get(span.span_id, [])
            ]
            global_spans = [
                span
                for span in rank.spans.values()
                if span.span_type == SpanType.COMM and GLOBAL_ALL_REDUCE in rank.span_desc.get(span.span_id, [])
            ]
            self.assertEqual(len(dp_spans), 1, f"Rank {rank_id} DP all-reduce count mismatch")
            self.assertEqual(len(global_spans), 1, f"Rank {rank_id} global all-reduce count mismatch")

        graph.analyze()

        dp_group_sizes = set()
        global_group_sizes = set()
        for group in graph.sync_span_groups:
            point = group.stop_points[0]
            descriptions = graph.ranks[point.rank].span_desc.get(point.span_id, [])
            if DP_ALL_REDUCE in descriptions:
                dp_group_sizes.add(len(group.stop_points))
            elif GLOBAL_ALL_REDUCE in descriptions:
                global_group_sizes.add(len(group.stop_points))

        self.assertEqual(dp_group_sizes, {data_parallel_size})
        self.assertEqual(global_group_sizes, {pipeline_parallel_size * data_parallel_size})


if __name__ == "__main__":
    unittest.main()
