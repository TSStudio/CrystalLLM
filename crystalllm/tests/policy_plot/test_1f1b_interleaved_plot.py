# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Visualization tests for the Megatron 1F1B interleaved execution graph policy.

Unlike tests/policy, which checks the schedule semantics, these tests render PNG images
of plot_with_runtime_profiler() for visual inspection: the chunk-interleaved schedule
structure of the smallest graph, the pipeline depth at different max_connections, and
the anchor coverage under different sandbox splits, including a split with two sandbox
stages. The tests only assert that each image is written and that the anchor figures
have anchorable points to draw; the image itself is the artifact under review.
"""

import random
import unittest
from pathlib import Path

from crystalllm.planning import CrystalGSpanPoint, SpanPointType, SpanType, plot_with_runtime_profiler
from crystalllm.planning.policy.megatron_1f1b_interleaved import (
    BACKWARD,
    FORWARD,
    build_megatron_1f1b_interleaved_graph,
)
from crystalllm.planning.policy.parallel_layout import rank_of

# Time scale factor: convert test time units to nanoseconds
TIME_SCALE = 1_000_000


def ms(duration_ms: int) -> int:
    """Convert milliseconds to nanoseconds for test data."""
    return duration_ms * TIME_SCALE


def set_proc_span_durations(
    graph,
    *,
    forward_ms: int,
    backward_ms: int,
    perturbation_ms: float = 5.0,
    seed: int = 0,
) -> None:
    """Set the marked computation span durations with a seeded random perturbation.

    The perturbation varies same-mark spans so the plots resemble runtime profiler
    data instead of perfectly uniform bars, and the seed keeps the images
    reproducible across runs. The computation spans are named like a runtime
    capture, so the semantics live in span_desc and the duration lists are
    indexed by the count_field of the rank's ``_proc`` spans; a span carrying
    both marks gets both perturbed bases summed.
    """
    rng = random.Random(seed)
    duration_dict: dict[int, dict[str, list[int]]] = {}
    for rank_id, rank in graph.ranks.items():
        proc_spans = sorted(
            (span for span in rank.spans.values() if span.span_type == SpanType.PROC),
            key=lambda span: span.span_id.count_field,
        )
        durations: list[int] = []
        for span in proc_spans:
            descriptions = rank.span_desc.get(span.span_id, [])
            duration_ns = 0
            if FORWARD in descriptions:
                duration_ns += ms(forward_ms) + int(rng.uniform(-perturbation_ms, perturbation_ms) * TIME_SCALE)
            if BACKWARD in descriptions:
                duration_ns += ms(backward_ms) + int(rng.uniform(-perturbation_ms, perturbation_ms) * TIME_SCALE)
            durations.append(duration_ns)
        duration_dict[rank_id] = {proc_spans[0].span_id.type_field: durations}
    graph.load_span_duration_dict(duration_dict)


class TestMegatron1F1BInterleavedPlot(unittest.TestCase):
    """Render the Megatron 1F1B interleaved graph for visual inspection."""

    def setUp(self):
        """Set up test output directory."""
        self.test_output_dir = Path(__file__).parent / "test_output_1f1b_interleaved_plot"
        self.test_output_dir.mkdir(exist_ok=True)

    def _save_plot(self, graph, output_path: Path, ranks: list[int] | None = None, **plot_kwargs) -> None:
        """Render the graph and assert that the PNG file was written."""
        plot_with_runtime_profiler(graph, ranks, output_path=str(output_path), show=False, **plot_kwargs)
        self.assertTrue(output_path.exists(), f"Plot was not written to {output_path}")

    def test_dp_group_view(self):
        """Plot only the ranks of one pipeline stage, i.e. the DP-group view.

        Those ranks sit in different PP chains in the interleaved setting and are tied
        together by the DP and global all-reduce synchronization groups, so the figure
        shows the SYMM relay chains across PP chains rather than the P2P chain of a
        single pipeline.
        """
        data_parallel_size, stage = 3, 1
        graph = build_megatron_1f1b_interleaved_graph(
            num_microbatches=4,
            pipeline_parallel_size=2,
            num_model_chunks=2,
            data_parallel_size=data_parallel_size,
        )
        set_proc_span_durations(graph, forward_ms=30, backward_ms=60)
        graph.analyze()

        dp_group = [rank_of(stage, dp_index, data_parallel_size) for dp_index in range(data_parallel_size)]
        self._save_plot(graph, self.test_output_dir / "test_interleaved_dp_group.png", dp_group)

    def test_two_ranks_plot(self):
        """Plot the smallest interleaved graph (2 stages, 4 microbatches, 2 model chunks).

        The small figure keeps the chunk-interleaved warmup pattern and the per-stage-pair
        channels readable at a glance.
        """
        graph = build_megatron_1f1b_interleaved_graph(
            num_microbatches=4,
            pipeline_parallel_size=2,
            num_model_chunks=2,
        )
        set_proc_span_durations(graph, forward_ms=30, backward_ms=60)
        graph.analyze()
        self._save_plot(graph, self.test_output_dir / "test_interleaved_two_ranks.png")

    def test_four_ranks_max_connections(self):
        """Plot the 4-stage, 16-microbatch graph at max_connections 1 and 4.

        The two figures differ in how many microbatches a stage may keep in flight, which
        shows up as the depth of the pipeline bubble.
        """
        for max_connections in (1, 4):
            with self.subTest(max_connections=max_connections):
                graph = build_megatron_1f1b_interleaved_graph(
                    num_microbatches=16,
                    pipeline_parallel_size=4,
                    num_model_chunks=2,
                    max_connections=max_connections,
                )
                set_proc_span_durations(graph, forward_ms=30, backward_ms=60)
                graph.analyze()
                self._save_plot(graph, self.test_output_dir / f"test_interleaved_four_ranks_mc{max_connections}.png")

    def test_four_ranks_anchor_by_sandbox(self):
        """Plot the anchors for a sandbox on the first, a middle, the last and two non-adjacent stages.

        draw_anchor_preds_points implies draw_anchor, which requires set_sandbox_ranks()
        before analyze(). The queried points are only the START twins of the
        anchorable points: anchorable_points itself holds only STOP points, and the
        START query is the launch-time predecessor view, drawn with the light magenta
        arrows. STOP queries stay supported by the plot (black arrows) but are not
        exercised here.
        """
        for sandbox_ranks in ([0], [1], [3], [0, 2]):
            with self.subTest(sandbox_ranks=sandbox_ranks):
                graph = build_megatron_1f1b_interleaved_graph(
                    num_microbatches=16,
                    pipeline_parallel_size=4,
                    num_model_chunks=2,
                    max_connections=4,
                )
                set_proc_span_durations(graph, forward_ms=30, backward_ms=60)

                # Sandbox ranks are an input to analyze(), so they have to be registered first.
                graph.set_sandbox_ranks(sandbox_ranks)
                graph.analyze()

                # Without an anchorable point the figure would silently degrade to a plain
                # timeline, so the anchor state is checked before plotting.
                self.assertTrue(graph.anchorable_points, "No anchorable point for this sandbox split")

                anchorable_points = sorted(graph.anchorable_points, key=lambda p: (p.rank, p.span_id.count_field))
                show_anchor_points = [
                    CrystalGSpanPoint(point.rank, point.span_id, SpanPointType.START) for point in anchorable_points
                ]

                suffix = "_".join(str(rank) for rank in sandbox_ranks)
                self._save_plot(
                    graph,
                    self.test_output_dir / f"test_interleaved_four_ranks_sandbox_{suffix}.png",
                    draw_anchor_preds_points=show_anchor_points,
                )


if __name__ == "__main__":
    unittest.main()
