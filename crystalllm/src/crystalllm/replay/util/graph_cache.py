# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Replay Graph Cache.

Builds and reads the launch-timing cache that lets an assistant replay without
online updates while loading none of the merged CrystalGraph bundles.

What goes into the cache: replay_with_graph() with online updates off asks the
bundle, per communication operation, only whether the operation is anchorable
(its synchronization group crosses the sandbox boundary) and, when it is, when
the group may be launched. Both answers are frozen once the graphs are loaded,
because no duration ever changes afterwards, so the cache stores one record per
anchorable operation: keyed by (rank, trace_comm_index), it holds the launch
time of the operation's synchronization group - the latest START over the
group's members. An operation without a record (outside the analysis window, or
inside it on a purely emulated group) replays immediately, which is exactly
what the graph path decides for it too, so an absent record is an answer rather
than missing data.

The trace_comm_index of a record is the same counter the c10d trace numbers its
COMM and COALESCING lines with, which is what ties a cache record to the
operation the replay loop is holding; every other attribute of that operation
still comes from the trace itself.

Every assistant rank derives the same merged graphs - the mandatory chains come
from the sandbox and the launched ranks, and a rank's own chain is always among
the launched ones - so one cache file serves them all and each assistant keeps
only its own rank's records in memory.

The cache is built from the same bundle the started assistants load: the caller
passes the launched ranks the mock NCCL bootstrap planner decided - the member
ranks list every assistant receives through CRYSTAL_MOCK_NCCL_MEMBER_RANKS, or
every emulated rank when no planner ran - and the own rank is drawn from that
set, so the mandatory chains, and therefore every queried value, coincide with
what replay_with_graph() computes for a started rank. The process group table
is never handed in: the bundle falls back to parsing it from the own rank trace
(group creation is world-wide, so any rank's trace carries the full table), the
same source a started assistant feeds it from.

Every cache file also records the invocation it was built from - the absolute
paths of the graph, trace and profiler directories, the strides, the sandbox
split and the launched ranks - and generate_graph_cache() returns the newest
cache of output_dir unchanged when that recorded invocation matches the current
call and no file of the input directories has been modified since the cache was
written; only then is a fresh cache skipped, otherwise one is built and written.

Script usage (run the module file directly, or through ``python -m``)::

    python3 src/crystalllm/replay/util/graph_cache.py \
        --graph-dir <dir> --trace-dir <dir> --profiler-dir <dir> \
        --pp-rank-stride 8 --sandbox-ranks '[0, 8]' --world-size 32 \
        [--launched-ranks '[9, 17]'] --output-dir <dir>

Library usage::

    from crystalllm.replay.util import generate_graph_cache
    cache_path = generate_graph_cache(
        graph_dir=..., trace_dir=..., profiler_dir=..., pp_rank_stride=8,
        sandbox_ranks=[0, 8], world_size=32, output_dir=...,
        launched_ranks=[9, 17],
    )
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import operator
import time
from pathlib import Path

from ...planning.graph import CrystalGSpanPoint, SpanPointType
from ..graph_bundle import COMM_TYPE_FIELD, CrystalGraphBundle

logger = logging.getLogger(__name__)

CACHE_FILE_PREFIX = "graph_cache_"


def _normalize_launched_ranks(
    launched_ranks: list[int] | None, *, sandbox_ranks: list[int], world_size: int
) -> list[int]:
    """Resolve the launched ranks the way an assistant's replay would see them.

    None means every emulated rank is started, which is also what an assistant
    concludes when no mock bootstrap planner ran.
    """
    if launched_ranks is None:
        return [rank for rank in range(world_size) if rank not in set(sandbox_ranks)]
    return sorted(launched_ranks)


def _scenario_fingerprint(
    *,
    graph_dir: str,
    trace_dir: str,
    profiler_dir: str | None,
    pp_rank_stride: int,
    sandbox_ranks: list[int],
    world_size: int,
    launched_ranks: list[int],
) -> dict:
    """Record the invocation a cache is built from, for the reuse check."""
    return {
        "graph_dir": str(Path(graph_dir).resolve()),
        "trace_dir": str(Path(trace_dir).resolve()),
        "profiler_dir": str(Path(profiler_dir).resolve()) if profiler_dir is not None else None,
        "pp_rank_stride": pp_rank_stride,
        "sandbox_ranks": sorted(sandbox_ranks),
        "world_size": world_size,
        "launched_ranks": list(launched_ranks),
    }


def build_graph_cache_dict(
    *,
    graph_dir: str,
    trace_dir: str,
    profiler_dir: str | None,
    pp_rank_stride: int,
    sandbox_ranks: list[int],
    world_size: int,
    launched_ranks: list[int] | None = None,
) -> dict:
    """Build the launch-timing cache of a scenario and return it as a dictionary.

    Loads one CrystalGraphBundle the way a started assistant would - same graphs,
    traces, profiler data, sandbox split and launched ranks, with the process
    group table parsed from the own rank trace - and records, for every COMM span
    of every loaded emulated rank and step that is anchorable, the launch time of
    its synchronization group. Sandbox ranks are skipped because no assistant
    ever replays them, and non-anchorable spans because they replay immediately
    with no cached data at all. The own rank only decides which chain is fatal
    when unusable, so the first launched rank is as good a choice as any.

    Args:
        graph_dir: Directory holding graph_{rank}.json.gz.
        trace_dir: Directory holding trace_{rank}.log.gz.
        profiler_dir: Directory holding runtime_profiler_{rank}.json.gz, or None to
            keep every span duration at 0.
        pp_rank_stride: Rank distance between adjacent PP stages.
        sandbox_ranks: Ranks running the real training program.
        world_size: Total number of ranks in the target scenario.
        launched_ranks: Emulated ranks actually started for this run, i.e. the
            member ranks every assistant receives; None means every emulated rank
            is started.

    Returns:
        The cache dictionary, ready for generate_graph_cache() to write and
        load_graph_cache() to read back.

    Raises:
        ValueError: If the scenario has no launched rank to replay, or the bundle
            load rejects the inputs (see CrystalGraphBundle.load()).
    """
    launched_ranks = _normalize_launched_ranks(launched_ranks, sandbox_ranks=sandbox_ranks, world_size=world_size)
    if not launched_ranks:
        raise ValueError("A graph cache needs at least one launched rank to replay")

    bundle = CrystalGraphBundle.load(
        graph_dir=graph_dir,
        trace_dir=trace_dir,
        profiler_dir=profiler_dir,
        pp_rank_stride=pp_rank_stride,
        sandbox_ranks=sandbox_ranks,
        world_size=world_size,
        own_rank=launched_ranks[0],
        launched_ranks=launched_ranks,
    )

    ops = []
    sandbox_set = set(sandbox_ranks)
    for step in sorted(bundle.step_graphs):
        graph = bundle.step_graphs[step]
        for rank_id in sorted(graph.ranks):
            if rank_id in sandbox_set:
                # No assistant ever replays a sandbox rank, so its records would
                # only inflate the shared cache file.
                continue
            for span_id in graph.ranks[rank_id].spans:
                if span_id.type_field != COMM_TYPE_FIELD:
                    continue
                stop_point = CrystalGSpanPoint(rank_id, span_id, SpanPointType.STOP)
                if not bundle.is_anchorable_point(step, stop_point):
                    # Not launched at a planned time, so the immediate path
                    # handles it without any cached data.
                    continue
                group = bundle.sync_span_group_of_point(rank_id, span_id.count_field)
                ops.append({
                    "rank": rank_id,
                    "trace_comm_index": span_id.count_field,
                    "launch_time_ns": bundle.sync_span_group_launch_time_ns(step, group),
                })
    ops.sort(key=operator.itemgetter("rank", "trace_comm_index"))

    return {
        **_scenario_fingerprint(
            graph_dir=graph_dir,
            trace_dir=trace_dir,
            profiler_dir=profiler_dir,
            pp_rank_stride=pp_rank_stride,
            sandbox_ranks=sandbox_ranks,
            world_size=world_size,
            launched_ranks=launched_ranks,
        ),
        "steps": sorted(bundle.step_graphs),
        "max_connections": bundle.max_connections,
        "op_count": len(ops),
        "ops": ops,
    }


def generate_graph_cache(
    *,
    graph_dir: str,
    trace_dir: str,
    profiler_dir: str | None,
    pp_rank_stride: int,
    sandbox_ranks: list[int],
    world_size: int,
    output_dir: str | Path,
    launched_ranks: list[int] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Build a cache and write it to output_dir/graph_cache_{timestamp}.json.gz.

    The newest cache already sitting in output_dir is returned unchanged when it
    is still valid - its recorded invocation matches this call and no file of the
    graph, trace or profiler directories was modified after it was written.
    Otherwise a fresh cache is built: the launch times depend on the sandbox
    split and on the runtime profiler data of the moment, so replaying against a
    stale file would silently time another run.

    Args:
        graph_dir: Directory holding graph_{rank}.json.gz.
        trace_dir: Directory holding trace_{rank}.log.gz.
        profiler_dir: Directory holding runtime_profiler_{rank}.json.gz, or None to
            keep every span duration at 0.
        pp_rank_stride: Rank distance between adjacent PP stages.
        sandbox_ranks: Ranks running the real training program.
        world_size: Total number of ranks in the target scenario.
        output_dir: Directory receiving the cache file, created when missing.
        launched_ranks: Emulated ranks actually started for this run; None means
            every emulated rank is started.
        timestamp: File name timestamp of a newly written cache; defaults to the
            current local time with microseconds, so successive builds never
            overwrite each other.

    Returns:
        The path of the reused or newly written cache file.
    """
    launched_ranks = _normalize_launched_ranks(launched_ranks, sandbox_ranks=sandbox_ranks, world_size=world_size)
    fingerprint = _scenario_fingerprint(
        graph_dir=graph_dir,
        trace_dir=trace_dir,
        profiler_dir=profiler_dir,
        pp_rank_stride=pp_rank_stride,
        sandbox_ranks=sandbox_ranks,
        world_size=world_size,
        launched_ranks=launched_ranks,
    )
    output_dir = Path(output_dir)

    reusable = _reusable_cache_path(output_dir, fingerprint)
    if reusable is not None:
        logger.info(
            "Reusing graph cache %s: invocation matches and no input file changed since it was written",
            reusable,
        )
        return reusable

    build_start = time.perf_counter()
    cache = build_graph_cache_dict(
        graph_dir=graph_dir,
        trace_dir=trace_dir,
        profiler_dir=profiler_dir,
        pp_rank_stride=pp_rank_stride,
        sandbox_ranks=sandbox_ranks,
        world_size=world_size,
        launched_ranks=launched_ranks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    path = output_dir / f"{CACHE_FILE_PREFIX}{timestamp}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(cache, f)

    logger.info(
        "Generated graph cache %s: %d anchorable ops over steps %s in %.1fs",
        path,
        cache["op_count"],
        cache["steps"],
        time.perf_counter() - build_start,
    )
    return path


def load_graph_cache(path: str | Path) -> dict:
    """Read a cache file written by generate_graph_cache() and check its shape.

    The file is written and read within the same _run_emulate call, so no version
    guard is needed: a shape check against the current layout is all that can
    ever fail.

    Args:
        path: Path of the graph_cache_{timestamp}.json.gz file.

    Returns:
        The cache dictionary.

    Raises:
        ValueError: If the file misses an entry of the cache layout.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    for key in ("world_size", "pp_rank_stride", "sandbox_ranks", "steps", "ops"):
        if not isinstance(cache, dict) or key not in cache:
            raise ValueError(f"Graph cache {path} misses the '{key}' entry")
    return cache


def _reusable_cache_path(output_dir: Path, fingerprint: dict) -> Path | None:
    """Return the newest cache of output_dir when it is still valid, else None.

    Valid means the cache's recorded invocation matches the fingerprint and no
    file of the input directories has an mtime at or after the cache file's own
    mtime: the bundle read exactly those files when building the cache, so an
    unchanged set of files would yield the very same cache.

    Args:
        output_dir: Directory holding the graph_cache_{timestamp}.json.gz files.
        fingerprint: Invocation record of the current call, from
            _scenario_fingerprint().

    Returns:
        The path of the reusable cache, or None when output_dir holds no valid
        newest cache.
    """
    caches = [path for path in output_dir.glob(f"{CACHE_FILE_PREFIX}*.json.gz") if path.is_file()]
    if not caches:
        return None
    newest = max(caches, key=lambda path: path.stat().st_mtime)
    try:
        cache = load_graph_cache(newest)
    except (OSError, ValueError):
        return None
    if any(cache.get(key) != value for key, value in fingerprint.items()):
        return None

    cache_mtime = newest.stat().st_mtime
    for directory in (fingerprint["graph_dir"], fingerprint["trace_dir"], fingerprint["profiler_dir"]):
        if directory is None:
            continue
        for file in Path(directory).rglob("*"):
            if file.is_file() and file.stat().st_mtime >= cache_mtime:
                return None
    return newest


def validate_graph_cache(cache: dict, *, world_size: int, sandbox_ranks: list[int]) -> None:
    """Check that a cache was built for the scenario the caller is about to replay.

    Args:
        cache: Cache dictionary, e.g. from load_graph_cache().
        world_size: Total number of ranks of the caller.
        sandbox_ranks: Sandbox ranks of the caller.

    Raises:
        ValueError: If the world size or the sandbox split differ, either of which
            would make every cached launch time time a different scenario.
    """
    if cache["world_size"] != world_size:
        raise ValueError(
            f"Graph cache was built for world size {cache['world_size']}, but this process replays "
            f"world size {world_size}"
        )
    if sorted(cache["sandbox_ranks"]) != sorted(sandbox_ranks):
        raise ValueError(
            f"Graph cache was built for sandbox ranks {sorted(cache['sandbox_ranks'])}, but this process "
            f"replays sandbox ranks {sorted(sandbox_ranks)}"
        )


def cached_op_timings(cache: dict, rank: int) -> dict[int, int]:
    """Extract the launch times of one rank's anchorable operations from a cache.

    Args:
        cache: Cache dictionary, e.g. from load_graph_cache().
        rank: The rank whose records to keep; every other rank's records are
            dropped, keeping the memory of an assistant proportional to its own
            trace rather than the whole scenario.

    Returns:
        Mapping from trace_comm_index to the operation's launch time in
        nanoseconds; an index absent from the mapping replays immediately.
    """
    return {op["trace_comm_index"]: op["launch_time_ns"] for op in cache["ops"] if op["rank"] == rank}


def main() -> None:
    """Parse a scenario from the command line and generate its graph cache."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-dir", required=True, type=str, help="Crystal Graph directory")
    parser.add_argument("--trace-dir", required=True, type=str, help="Crystal C10D Trace directory")
    parser.add_argument("--profiler-dir", type=str, default=None, help="Crystal runtime profiler directory")
    parser.add_argument("--pp-rank-stride", required=True, type=int, help="Rank distance between adjacent PP stages")
    parser.add_argument("--sandbox-ranks", required=True, type=json.loads, help="Sandbox ranks, as a JSON list")
    parser.add_argument("--world-size", required=True, type=int, help="Total number of ranks in the scenario")
    parser.add_argument(
        "--launched-ranks",
        type=json.loads,
        default=None,
        help="Emulated ranks actually started, as a JSON list; defaults to every emulated rank",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Directory receiving graph_cache_{timestamp}.json.gz"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    path = generate_graph_cache(
        graph_dir=args.graph_dir,
        trace_dir=args.trace_dir,
        profiler_dir=args.profiler_dir,
        pp_rank_stride=args.pp_rank_stride,
        sandbox_ranks=args.sandbox_ranks,
        world_size=args.world_size,
        output_dir=args.output_dir,
        launched_ranks=args.launched_ranks,
    )
    print(path)


if __name__ == "__main__":
    main()
