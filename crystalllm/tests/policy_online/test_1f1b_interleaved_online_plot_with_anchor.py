# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Visualization tests for the Megatron 1F1B interleaved graph under anchor online adjustment.

Unlike tests/policy_online/test_1f1b_interleaved_online_performance.py, which measures the
per-query cost of the online path, this file exercises the adjustment itself: after
analyze(), every anchorable communication span grows its duration by a fixed increment
through increase_span_duration(), and a plot_with_runtime_profiler() PNG is rendered after
every step so the stretched timeline, the anchor stars and the anchor predecessor arrows
can be inspected together. The images and the printed intervals are the artifact under
review; the assertions only guard that the artifact is produced and that the whole-graph
interval never shrinks, which is the contract increase_span_duration() builds on.
"""

import unittest
from pathlib import Path

from crystalllm.planning import CrystalGSpanPoint, SpanPointType, SpanType, plot_with_runtime_profiler
from crystalllm.planning.policy.megatron_1f1b_interleaved import (
    BACKWARD,
    FORWARD,
    build_megatron_1f1b_interleaved_graph,
)

# Time scale factor: convert test time units to nanoseconds
TIME_SCALE = 1_000_000


def ms(duration_ms: int) -> int:
    """Convert milliseconds to nanoseconds for test data."""
    return duration_ms * TIME_SCALE


def set_span_durations_by_rank(graph) -> None:
    """Set forward/backward step durations based on rank.

    FWD duration: rank0=25ms, rank1&rank2=20ms, rank3=30ms
    BWD duration: FWD * 2 (rank0=50ms, rank1&rank2=40ms, rank3=60ms)

    The uneven ranks force a non-trivial longest path, so every +10ms growth of an
    anchor span below has a visible effect on the interval and the timeline. The
    computation spans are named like a runtime capture, so the semantics live in
    span_desc and the durations are one positional list over the rank's _proc spans.
    """
    fwd_durations = {0: ms(25), 1: ms(20), 2: ms(20), 3: ms(30)}
    bwd_durations = {0: ms(50), 1: ms(40), 2: ms(40), 3: ms(60)}

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
                duration_ns += fwd_durations[rank_id]
            if BACKWARD in descriptions:
                duration_ns += bwd_durations[rank_id]
            durations.append(duration_ns)
        duration_dict[rank_id] = {proc_spans[0].span_id.type_field: durations}

    graph.load_span_duration_dict(duration_dict)


class TestMegatron1F1BInterleavedAnchorOnline(unittest.TestCase):
    """Render the anchor online adjustment of the interleaved graph step by step."""

    def setUp(self):
        """Set up test output directory."""
        self.test_output_dir = Path(__file__).parent / "test_output_1f1b_interleaved_online_plot_with_anchor"
        self.test_output_dir.mkdir(exist_ok=True)

    def test_interleaved_four_ranks_by_sandbox(self):
        """Grow the anchors under a sandbox on the first, a middle and the last stage.

        Steps, for each sandbox split:
        1. Build the graph, set the per-rank FWD/BWD durations of the computation
           spans, then register the sandbox ranks and analyze(): sandbox_ranks are an
           input to analyze().
        2. Render step_0.png under the case directory, the state before any online
           update.
        3. Walk the anchorable points sorted by (rank, count_field), the issue order
           inside each rank (anchorable_points itself is an unordered set; only
           get_direct_anchor_predecessors() lists anchors topologically). Each step
           grows the COMM span an anchorable STOP point belongs to by 10 ms and
           renders step_<n>.png. The whole-graph interval is asserted to never
           shrink, because increase_span_duration() may only push the longest-path
           times up.
        """
        for sandbox_ranks in ([0], [1], [3]):
            with self.subTest(sandbox_ranks=sandbox_ranks):
                self._grow_anchors_step_by_step(sandbox_ranks)

    def _grow_anchors_step_by_step(self, sandbox_ranks: list[int]) -> None:
        """Run the step-by-step anchor growth of one sandbox split, see the test method."""
        graph = build_megatron_1f1b_interleaved_graph(
            num_microbatches=8,
            pipeline_parallel_size=4,
            num_model_chunks=2,
            max_connections=1,
        )

        set_span_durations_by_rank(graph)

        # Sandbox ranks are an input to analyze(), so they have to be registered first.
        graph.set_sandbox_ranks(sandbox_ranks)
        graph.analyze()

        # Without an anchorable point the figure would silently degrade to a plain
        # timeline, so the anchor state is checked before plotting.
        self.assertTrue(graph.anchorable_points, "No anchorable point for this sandbox split")

        suffix = "rank" + "_".join(str(rank) for rank in sandbox_ranks)
        case_output_dir = self.test_output_dir / suffix
        case_output_dir.mkdir(exist_ok=True)

        total_time = graph.get_points_interval()
        print(f"\n=== Sandbox ranks {sandbox_ranks}: {len(graph.anchorable_points)} anchorable points ===")
        print(f"Step 0 (before any update): total time = {total_time / TIME_SCALE:.2f} ms")

        # anchorable_points holds only the STOP points of the COMM spans whose
        # synchronization group crosses the sandbox boundary, and the red stars are
        # drawn right there. The queried points are their START twins, because the
        # START query is the launch-time predecessor view (the anchors replay waits
        # for before issuing the span), drawn with the light magenta arrows.
        show_anchor_points = [
            CrystalGSpanPoint(point.rank, point.span_id, SpanPointType.START) for point in graph.anchorable_points
        ]

        output_path = case_output_dir / "step_0.png"
        plot_with_runtime_profiler(
            graph, draw_anchor_preds_points=show_anchor_points, output_path=str(output_path), show=False
        )
        self.assertTrue(output_path.exists(), f"Plot was not written to {output_path}")

        duration_adjustment = ms(10)

        for step, anchor_point in enumerate(
            sorted(graph.anchorable_points, key=lambda p: (p.rank, p.span_id.count_field)), start=1
        ):
            print(f"\n--- Step {step} ---")
            print(
                f"Growing the span of the anchorable point rank={anchor_point.rank}, "
                f"span_id={anchor_point.span_id}, point_type={anchor_point.point_type}"
            )

            span = graph.ranks[anchor_point.rank].spans[anchor_point.span_id]
            new_duration = span.duration_ns + duration_adjustment
            print(
                f"Updating span {anchor_point.span_id} on rank {anchor_point.rank}: "
                f"duration {span.duration_ns / TIME_SCALE:.2f}ms -> {new_duration / TIME_SCALE:.2f}ms"
            )
            graph.increase_span_duration(anchor_point.rank, anchor_point.span_id, new_duration)
            # increase_span_duration() rewrites only the START -> STOP edge weight of
            # the condensed graph and leaves the static span.duration_ns at its planning
            # value, while plot_with_runtime_profiler() sizes every bar from that static
            # field. Updating it in lockstep keeps each step's image aligned with the
            # printed interval, and a later analyze() would rebuild the edge from it, so
            # the two stay consistent. Every COMM span contributes at most one anchorable
            # STOP point, hence each span is grown exactly once, from its planning value.
            span.duration_ns = new_duration

            previous_time = total_time
            total_time = graph.get_points_interval()
            print(f"After the update: total time = {total_time / TIME_SCALE:.2f} ms")
            # A longer edge can only push the longest-path times up, never down.
            self.assertGreaterEqual(total_time, previous_time)

            output_path = case_output_dir / f"step_{step}.png"
            plot_with_runtime_profiler(
                graph, draw_anchor_preds_points=show_anchor_points, output_path=str(output_path), show=False
            )
            self.assertTrue(output_path.exists(), f"Plot was not written to {output_path}")

        print(f"\n=== Completed {len(graph.anchorable_points)} steps of anchor growth ===")


if __name__ == "__main__":
    unittest.main()
