# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for Megatron 1F1B (non-interleaved) pipeline parallelism policy."""

import unittest

from crystalllm.planning import CrystalGSpanPoint, SpanPointType, SpanType
from crystalllm.planning.policy.megatron_1f1b import (
    BACKWARD,
    FORWARD,
    RECV_BACKWARD,
    RECV_FORWARD,
    SEND_BACKWARD,
    SEND_BACKWARD_RECV_FORWARD,
    SEND_FORWARD,
    SEND_FORWARD_RECV_BACKWARD,
    build_megatron_1f1b_graph,
)
from crystalllm.planning.policy.parallel_layout import DP_ALL_REDUCE, GLOBAL_ALL_REDUCE

# Time scale factor: convert test time units to nanoseconds
TIME_SCALE = 1_000_000

# span_desc labels of the six P2P communication span kinds
P2P_DESCRIPTIONS = {
    RECV_FORWARD,
    SEND_FORWARD,
    RECV_BACKWARD,
    SEND_BACKWARD,
    SEND_FORWARD_RECV_BACKWARD,
    SEND_BACKWARD_RECV_FORWARD,
}


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
    steps that no communication boundary separates, e.g. the steady-state
    forward and backward of the last stage) gets the sum of both durations.
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


class TestMegatron1F1B(unittest.TestCase):
    """Test the Megatron 1F1B (non-interleaved) execution graph policy.

    These tests verify the schedule modeling of build_megatron_1f1b_graph():
    - Phase structure per rank, checked through the launch-order computation
      pattern: warmup forwards, strictly alternating steady state, cooldown
      backwards
    - Communication span specialization of the first and last pipeline stages
    - Pipeline bubble behavior of the total execution time
    - Closing collectives: DP and global all-reduce synchronization groups
    """

    def test_basic_1f1b_two_ranks(self):
        """Test the basic 2-rank, 4-microbatch schedule.

        Rank 0 warms up with one forward before entering 1F1B, while rank 1 (the
        last stage) alternates from the very first microbatch.
        """
        graph = build_megatron_1f1b_graph(num_microbatches=4, pipeline_parallel_size=2)
        set_span_durations(graph, forward_duration_ns=ms(30), backward_duration_ns=ms(60))
        graph.analyze()

        # Edge-stage communication specialization: the first stage neither
        # recv-forwards nor send-backwards, the last stage neither send-forwards
        # nor recv-backwards.
        # rank0: 1 send_forward (warmup) + 3 send_forward_recv_backward (steady)
        #        + 1 recv_backward (cooldown)
        # rank1: 1 recv_forward (steady transition) + 3 send_backward_recv_forward
        #        + 1 send_backward (steady)
        self.assertEqual(count_comm_spans(graph, 0), 5, "Rank 0 comm span count mismatch")
        self.assertEqual(count_comm_spans(graph, 1), 5, "Rank 1 comm span count mismatch")

        # The built graph carries what a capture of the same schedule records:
        # the max_connections it was built under and, on every comm tasks, the
        # synchronous launch shape of the non-interleaved schedule.
        self.assertEqual(graph.max_connections, 1)
        for rank in graph.ranks.values():
            for span in rank.spans.values():
                if span.span_type == SpanType.COMM:
                    self.assertFalse(span.comm_tasks.async_op)

        # Launch-order computation pattern, warmup = PP - stage - 1
        self.assertEqual(proc_execution_pattern(graph, 0), expected_1f1b_pattern(num_warmup=1, num_steady=3))
        self.assertEqual(proc_execution_pattern(graph, 1), expected_1f1b_pattern(num_warmup=0, num_steady=4))

    def test_1f1b_four_ranks(self):
        """Test the 4-rank, 32-microbatch schedule: staggered warmup and bubble.

        The warmup count decreases by one on every stage, the total time stays
        between the serial lower bound and twice it, and the edge stages carry
        the communication specialization of a pipeline boundary.
        """
        num_microbatches, pipeline_parallel_size = 32, 4
        graph = build_megatron_1f1b_graph(
            num_microbatches=num_microbatches,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        set_span_durations(graph, forward_duration_ns=ms(30), backward_duration_ns=ms(60))
        graph.analyze()

        # Warmup is staggered across the stages: PP - stage - 1 forwards each
        for stage in range(pipeline_parallel_size):
            num_warmup = pipeline_parallel_size - stage - 1
            expected = expected_1f1b_pattern(num_warmup, num_microbatches - num_warmup)
            self.assertEqual(
                proc_execution_pattern(graph, stage),
                expected,
                f"Rank {stage} execution pattern mismatch",
            )

        # Pipeline bubble: never faster than one rank doing all the work serially,
        # but well below twice that
        total_time = graph.get_points_interval()
        theoretical_min = ms(num_microbatches * 90)  # 32 microbatches * (30ms fwd + 60ms bwd)
        self.assertGreater(total_time, theoretical_min)
        self.assertLess(total_time, theoretical_min * 2)

        # Edge-stage communication specialization:
        # rank0 (first): 3 send_forward + 29 send_forward_recv_backward + 3 recv_backward
        # rank3 (last): 1 recv_forward + 31 send_backward_recv_forward + 1 send_backward
        self.assertEqual(count_comm_spans(graph, 0), 35, "Rank 0 comm span count mismatch")
        self.assertEqual(count_comm_spans(graph, 3), 33, "Rank 3 comm span count mismatch")

    def test_1f1b_different_configurations(self):
        """Test step counts across configurations, including edge cases.

        Every rank executes exactly one forward and one backward step per
        microbatch, from the single-microbatch edge case up to an 8-stage
        pipeline. A step is counted through its span_desc mark, wherever the
        communication boundaries of the stage put it.
        """
        configs = [
            (1, 2),  # Single microbatch edge case
            (4, 2),
            (8, 4),
            (16, 8),  # Large pipeline: 16 microbatches, 8 ranks
        ]

        for num_mb, pp_size in configs:
            with self.subTest(num_microbatches=num_mb, pipeline_parallel_size=pp_size):
                graph = build_megatron_1f1b_graph(
                    num_microbatches=num_mb,
                    pipeline_parallel_size=pp_size,
                )
                set_span_durations(graph, forward_duration_ns=ms(30), backward_duration_ns=ms(60))

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
                    self.assertEqual(fwd_count, num_mb, f"Rank {rank_id} forward count mismatch")
                    self.assertEqual(bwd_count, num_mb, f"Rank {rank_id} backward count mismatch")

                # The topological sort inside analyze() rejects dependency cycles
                graph.analyze()

    def test_1f1b_closing_collectives(self):
        """Test that every rank ends with one DP group all-reduce and one global all-reduce.

        With a data parallel size above 1 the DP all-reduce members are the ranks of the same
        pipeline stage, so they span several PP chains, while the global all-reduce spans all.
        """
        pipeline_parallel_size, data_parallel_size = 4, 3
        graph = build_megatron_1f1b_graph(
            num_microbatches=4,
            pipeline_parallel_size=pipeline_parallel_size,
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
