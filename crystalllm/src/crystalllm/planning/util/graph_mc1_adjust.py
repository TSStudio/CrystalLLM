# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Adjust captured max_connections==1 graphs into a DP-replay friendly form.

Old captures ran under ``CUDA_DEVICE_MAX_CONNECTIONS=1`` without COMM span
durations.  Two per-step edits make such graphs usable by the current tooling
while leaving ``max_connections`` itself untouched:

1. Every ``comm_tasks`` with ``async_op == True`` is rewritten to
   ``async_op == False``.  ``async_op`` is read at exactly one place — the
   cross-rank dependency builder downgrades an async pair to WEAK_BIDIRECT only
   under ``max_connections == 1``; clearing the flag therefore lets every pair
   build a BIDIRECT edge (joining the timing model) exactly like a graph
   captured with overlapping connections, without altering anything else.
2. Every communication channel regains its STOP(prev) -> START(next) ordering
   dependency between consecutive comm spans of the same channel (in issue
   order, i.e. ascending count_field).  Under max_connections == 1 those edges
   were deliberately not recorded; restoring them keeps the channel serialized.

The adjustment is incremental: a rank file whose adjusted copy is missing, or
older than the source file, or no longer source-aligned (different step count
or max_connections) is recomputed; every other copy is reused as is.

Script usage (run the module file directly; the module has no package-internal
imports, so no ``-m`` is needed and the working directory does not matter)::

    python3 <repo>/src/crystalllm/planning/util/graph_mc1_adjust.py \
        --cases <graph_dir> [<graph_dir> ...] [--jobs N]

Library usage::

    from crystalllm.planning.util import adjust_graph_dir
    output_dir = adjust_graph_dir(Path("/path/to/capture/graph"))

Each adjusted ``graph_{rank}.json.gz`` keeps the exact input file naming and
JSON layout (gzip + json indent=2) and is written to the sibling directory
``graph_mc1_adjust`` next to the input graph directory.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

COMM_TYPE_FIELD = "_comm"
DEFAULT_OUTPUT_NAME = "graph_mc1_adjust"


@dataclass(frozen=True)
class AdjustStats:
    """What one adjustment run changed."""

    steps: int
    async_flipped: int
    deps_added: int


def adjust_graph_dict(data: dict) -> AdjustStats:
    """Adjust one rank's whole graph payload (every step) in place.

    ``data`` is the parsed content of one ``graph_{rank}.json.gz`` file, i.e.
    ``{step: {"max_connections", "rank_id", "spans", "span_desc",
    "dependencies"}}``; ``max_connections`` is never modified here.
    """
    steps = 0
    async_flipped = 0
    deps_added = 0
    for step_data in data.values():  # keep the original step order
        flipped, added = _adjust_step(step_data)
        steps += 1
        async_flipped += flipped
        deps_added += added
    return AdjustStats(steps=steps, async_flipped=async_flipped, deps_added=deps_added)


def adjust_graph_dir(
    graph_dir: Path | str,
    output_dir: Path | str | None = None,
    jobs: int = 16,
) -> Path:
    r"""Adjust every rank graph of one graph directory, incrementally.

    A rank file is recomputed when its adjusted copy is missing, is older than
    the source file, or holds a different step count / max_connections than the
    source (a leftover from an earlier run against other data).  Copies that
    still match are reused, so repeated calls never redo the whole directory.

    Args:
        graph_dir: Directory holding ``graph_{rank}.json.gz`` files.
        output_dir: Directory receiving the adjusted copies; defaults to the
            sibling directory ``graph_mc1_adjust`` next to ``graph_dir``.
        jobs: Parallel worker processes.

    Returns:
        The output directory path.

    Raises:
        FileNotFoundError: If ``graph_dir`` holds no graph files.
    """
    graph_dir = Path(graph_dir)
    output_dir = Path(output_dir) if output_dir is not None else graph_dir.parent / DEFAULT_OUTPUT_NAME

    graph_files = sorted(graph_dir.glob("graph_*.json.gz"), key=lambda p: _rank_of_file(p.name))
    if not graph_files:
        raise FileNotFoundError(f"No graph_*.json.gz under {graph_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(source), str(output_dir / source.name))
        for source in graph_files
        if _needs_adjust(source, output_dir / source.name)
    ]

    totals = AdjustStats(steps=0, async_flipped=0, deps_added=0)
    if tasks:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_adjust_graph_file, graph_path, out_path) for graph_path, out_path in tasks]
            for future in as_completed(futures):
                stats = future.result()  # propagate worker failures immediately
                totals = AdjustStats(
                    steps=totals.steps + stats.steps,
                    async_flipped=totals.async_flipped + stats.async_flipped,
                    deps_added=totals.deps_added + stats.deps_added,
                )
    reused = len(graph_files) - len(tasks)
    print(
        f"[{graph_dir}] adjusted {len(tasks)}/{len(graph_files)} ranks "
        f"({reused} reused): {totals.steps} steps, "
        f"{totals.async_flipped} async_op flags cleared, +{totals.deps_added} channel deps -> {output_dir}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _rank_of_file(name: str) -> int:
    """Rank number of a ``graph_{rank}.json.gz`` style file name."""
    return int(name.split("_", maxsplit=1)[1].split(".", maxsplit=1)[0])


def _dep_key(dep: dict) -> tuple:
    """Hashable identity of one serialized dependency, for deduplication."""
    return (
        dep["dep_type"],
        dep["v1"]["rank"],
        dep["v1"]["span_id"]["type_field"],
        dep["v1"]["span_id"]["count_field"],
        dep["v1"]["point_type"],
        dep["v2"]["rank"],
        dep["v2"]["span_id"]["type_field"],
        dep["v2"]["span_id"]["count_field"],
        dep["v2"]["point_type"],
    )


def _make_channel_dep(rank: int, prev_count: int, next_count: int) -> dict:
    """Serialize one STOP(prev) -> START(next) UNIDIRECT dependency on ``rank``."""
    return {
        "dep_type": "unidirect",
        "v1": {
            "rank": rank,
            "span_id": {"type_field": COMM_TYPE_FIELD, "count_field": prev_count},
            "point_type": "stop",
        },
        "v2": {
            "rank": rank,
            "span_id": {"type_field": COMM_TYPE_FIELD, "count_field": next_count},
            "point_type": "start",
        },
    }


def _adjust_step(step_data: dict) -> tuple[int, int]:
    """Apply both edits to one step payload in place.

    Returns (async_op flags cleared, dependencies added).
    """
    rank = step_data["rank_id"]
    spans = step_data["spans"]
    deps = step_data["dependencies"]

    # 1. Clear every async_op flag so the merge-time cross-rank dependency
    #    builder pairs peers with BIDIRECT instead of WEAK_BIDIRECT.
    async_flipped = 0
    for span in spans.values():
        if span["span_type"] == "comm" and span["comm_tasks"].get("async_op"):
            span["comm_tasks"]["async_op"] = False
            async_flipped += 1

    # 2. Group the comm spans of each channel in issue order (ascending
    #    count_field) and chain consecutive spans STOP -> START.
    channel_counts: dict[str, list[int]] = {}
    for span in spans.values():
        if span["span_type"] == "comm":
            channel = span["comm_tasks"]["channel"]
            channel_counts.setdefault(channel, []).append(span["span_id"]["count_field"])
    for counts in channel_counts.values():
        counts.sort()

    existing = {_dep_key(dep) for dep in deps}
    deps_added = 0
    for counts in channel_counts.values():
        for prev_count, next_count in itertools.pairwise(counts):
            dep = _make_channel_dep(rank, prev_count, next_count)
            key = _dep_key(dep)
            if key in existing:
                continue
            deps.append(dep)
            existing.add(key)
            deps_added += 1
    return async_flipped, deps_added


def _adjust_graph_file(graph_path: str, out_path: str) -> AdjustStats:
    """Read one graph file, apply the edits, and write the adjusted copy."""
    with gzip.open(graph_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    stats = adjust_graph_dict(data)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return stats


def _source_fingerprint(data: dict) -> tuple:
    """Cheap source identity of one parsed graph payload.

    Combining the step count and the max_connections of the first step catches
    a leftover adjusted copy written against a different capture (different
    analysis window or device setting) without re-parsing both files deeply.
    """
    first_step = next(iter(data.values()), None)
    max_connections = first_step["max_connections"] if first_step else None
    return (len(data), max_connections)


def _needs_adjust(source: Path, target: Path) -> bool:
    """Whether the adjusted copy of one rank file must be recomputed."""
    if not target.exists():
        return True
    if target.stat().st_mtime < source.stat().st_mtime:
        return True
    try:
        with gzip.open(source, "rt", encoding="utf-8") as f:
            source_data = json.load(f)
        with gzip.open(target, "rt", encoding="utf-8") as f:
            target_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True
    return _source_fingerprint(target_data) != _source_fingerprint(source_data)


def main() -> None:
    """Parse the graph directories from the command line and adjust each one."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        type=Path,
        help="one or more capture graph directories holding graph_{rank}.json.gz",
    )
    parser.add_argument("--jobs", type=int, default=16, help="parallel worker processes")
    args = parser.parse_args()

    for graph_dir in args.cases:
        if not graph_dir.is_dir():
            sys.exit(f"Not a directory: {graph_dir}")
        adjust_graph_dir(graph_dir, jobs=args.jobs)


if __name__ == "__main__":
    main()
