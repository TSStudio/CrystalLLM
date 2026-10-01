# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Entry Point for Distributed Training Emulation.

This module provides the command-line interface for running CrystalLLM
in different modes: baseline, capture, emulate and planning.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ..coordinator import CrystalServer
from ..coordinator.protocol import ReplyCode, describe_reply_code
from ..planning.util import adjust_graph_dir
from ..replay.util import generate_graph_cache

logger = logging.getLogger(__name__)

MOCK_NCCL_BOOTSTRAP_NODES_THRESHOLD = 4
MIN_TRAIN_ITERS = 3


def parse_args() -> argparse.Namespace:
    r"""Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="crystalllm run",
        description="CrystalLLM run entry point for distributed training emulation",
    )

    # Positional argument: mode
    parser.add_argument(
        "mode",
        type=str,
        choices=["baseline", "capture", "emulate", "planning"],
        help=(
            "Run mode: "
            "baseline (full training for baseline), "
            "capture (capture communication traces), "
            "emulate (emulate training with assistants), "
            "planning (planning emulation with assistants)"
        ),
    )

    # Required arguments
    parser.add_argument(
        "--crystal-workspace",
        type=str,
        default=".",
        help="Crystal workspace directory",
    )
    parser.add_argument(
        "--log-dir-name",
        type=str,
        required=True,
        help="Log directory name",
    )
    parser.add_argument(
        "--target-nodes",
        type=int,
        required=True,
        help="Total number of nodes in the target emulated scenario (e.g., 2048)",
    )
    parser.add_argument(
        "--available-nodes",
        type=int,
        required=True,
        help="Number of physically available nodes",
    )
    parser.add_argument(
        "--train-script-path",
        type=str,
        required=True,
        help="Path to the training script",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        required=True,
        help="Master port for training environment variables (MASTER_PORT)",
    )
    parser.add_argument(
        "--server-addr",
        type=str,
        default=None,
        help="Coordinator server address for client-server communication",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        required=True,
        help="Coordinator server port for client-server communication",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        required=True,
        help="Number of warmup training iterations before analysis",
    )
    parser.add_argument(
        "--analysis-iters",
        type=int,
        required=True,
        help="Number of analysis iterations",
    )

    # Optional arguments
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="Number of GPUs per node (default: 8)",
    )
    parser.add_argument(
        "--pretrain-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional pretrain arguments as command line options",
    )
    parser.add_argument(
        "--pretrain-args-json",
        type=str,
        nargs="+",
        help="Path(s) to JSON file(s) with additional training arguments. Multiple files can be specified.",
    )
    parser.add_argument(
        "--pretrain-envs-json",
        type=str,
        nargs="+",
        help="Path(s) to JSON file(s) with additional environment variables. Multiple files can be specified.",
    )
    parser.add_argument(
        "--shutdown-client",
        action="store_true",
        help="Shutdown clients after training completes",
    )

    # Feature switches
    parser.add_argument(
        "--enable-mock-nccl-bootstrap",
        action="store_true",
        help="Enable mock NCCL bootstrap",
    )
    parser.add_argument(
        "--enable-crystal-c10d-trace",
        action="store_true",
        help="Enable Crystal C10D trace",
    )
    parser.add_argument(
        "--crystal-c10d-printing",
        action="store_true",
        help="Enable Crystal C10D printing",
    )
    parser.add_argument(
        "--enable-crystal-timer",
        action="store_true",
        help="Enable Crystal timer",
    )
    parser.add_argument(
        "--enable-crystal-monitor",
        action="store_true",
        help="Enable Crystal monitor",
    )
    parser.add_argument(
        "--enable-crystal-runtime-profiler",
        action="store_true",
        help="Enable Crystal runtime profiler",
    )

    # Emulate/Planning mode specific arguments
    parser.add_argument(
        "--sandbox-nodes",
        type=int,
        nargs="+",
        help=(
            "Node IDs in the target emulated scenario that run the real training program "
            "(e.g., --sandbox-nodes 0 1). Required for emulate mode; optional for planning mode, "
            "where it only selects which DP positions to cover"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        help="Path to trace directory (for emulate/planning mode)",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        help="Path to graph directory (for emulate/planning mode)",
    )
    parser.add_argument(
        "--profiler-dir",
        type=str,
        help="Path to Crystal runtime profiler directory (for emulate mode only)",
    )
    parser.add_argument(
        "--pp-rank-stride",
        type=int,
        help=(
            "Rank distance between adjacent PP stages, i.e. the number of ranks in one PP stage "
            "(for emulate/planning mode)"
        ),
    )
    parser.add_argument(
        "--dp-rank-stride",
        type=int,
        help=(
            "Rank distance between adjacent DP positions inside a PP stage, i.e. the number of "
            "consecutive ranks one DP position occupies (for emulate/planning mode)"
        ),
    )
    parser.add_argument(
        "--enable-assistant-online-update",
        action="store_true",
        help="Enable online duration updates for assistant processes (for emulate/planning mode)",
    )
    parser.add_argument(
        "--tensor-provider",
        type=str,
        help="Path to tensor provider module (for emulate/planning mode)",
    )
    parser.add_argument(
        "--enable-assistant-pytorch-profiler",
        action="store_true",
        help="Enable PyTorch profiler for assistant processes (for emulate/planning mode)",
    )
    parser.add_argument(
        "--enable-assistant-nsight-system",
        action="store_true",
        help="Enable Nsight Systems profiling for assistant processes (for emulate/planning mode)",
    )
    parser.add_argument(
        "--assistant-nsight-system-ranks",
        type=int,
        nargs="+",
        default=[],
        help="List of ranks to profile with Nsight Systems (for emulate/planning mode)",
    )
    parser.add_argument(
        "--planning-epochs",
        type=int,
        help="Number of planning epochs (for planning mode only)",
        default=1,
    )
    parser.add_argument(
        "--planning-sandbox-nodes-count",
        type=int,
        help="Number of nodes used as sandbox nodes to collect information during planning (for planning mode only)",
        default=1,
    )
    planning_dp_coverage = parser.add_mutually_exclusive_group()
    planning_dp_coverage.add_argument(
        "--planning-dp-indices",
        type=int,
        nargs="+",
        help="DP position indices to cover during planning (for planning mode only)",
    )
    planning_dp_coverage.add_argument(
        "--planning-dp-indices-num",
        type=int,
        help="Cover the first K DP positions during planning, i.e. indices 0..K-1 (for planning mode only)",
    )

    return parser.parse_args()


def node_dp_index(node: int, dp_node_stride: int, dp_size: int) -> int:
    """Return the DP position index of a node inside its PP stage."""
    return (node // dp_node_stride) % dp_size


def _validate_rank_layout(args: argparse.Namespace) -> None:
    """Validate the PP/DP stride constraints that make the rank layout well formed."""
    world_size = args.target_nodes * args.gpus_per_node

    # A stride below the node size cannot be honored (the sandbox is allocated per
    # node), but one that divides the node size evenly expresses "a fraction of a
    # node per DP position": raise it to the whole node instead of failing, and let
    # the following checks run against the raised value.
    if args.dp_rank_stride < args.gpus_per_node and args.gpus_per_node % args.dp_rank_stride == 0:
        logger.warning(
            "--dp-rank-stride %d is below --gpus-per-node %d but divides it evenly; because the sandbox "
            "is allocated per node, --dp-rank-stride is raised to %d so a DP position occupies whole nodes",
            args.dp_rank_stride,
            args.gpus_per_node,
            args.gpus_per_node,
        )
        args.dp_rank_stride = args.gpus_per_node

    if args.dp_rank_stride % args.gpus_per_node != 0:
        raise ValueError(
            "--dp-rank-stride must be a multiple of --gpus-per-node because the sandbox is allocated "
            "per node, so a DP position must contain a whole number of nodes; a stride below "
            "--gpus-per-node that divides it evenly would be raised to --gpus-per-node, but this one "
            f"does not divide it; got --dp-rank-stride={args.dp_rank_stride}, "
            f"--gpus-per-node={args.gpus_per_node}"
        )
    if args.pp_rank_stride % args.dp_rank_stride != 0:
        raise ValueError(
            "--dp-rank-stride must divide --pp-rank-stride so that a PP stage contains a whole number of "
            "DP positions and no DP position crosses a PP stage boundary; got "
            f"--pp-rank-stride={args.pp_rank_stride}, --dp-rank-stride={args.dp_rank_stride}"
        )
    if world_size % args.pp_rank_stride != 0:
        raise ValueError(
            "--pp-rank-stride must divide the world size so that every PP chain has the same number of "
            f"stages; got --pp-rank-stride={args.pp_rank_stride}, world size={world_size}"
        )


def _validate_sandbox_units(sandbox_nodes: list[int], dp_node_stride: int) -> None:
    """Validate that the sandbox nodes consist of whole DP positions.

    A DP position is the smallest sandbox unit: its internal communication stays inside
    the sandbox, it holds exactly one member of every DP group on its stage, and it never
    crosses a PP stage boundary.
    """
    units: dict[int, set[int]] = {}
    for node in sandbox_nodes:
        units.setdefault(node // dp_node_stride, set()).add(node)

    for unit, nodes in sorted(units.items()):
        expected = set(range(unit * dp_node_stride, (unit + 1) * dp_node_stride))
        if nodes != expected:
            raise ValueError(
                f"--sandbox-nodes must consist of whole DP positions of {dp_node_stride} nodes each, but "
                f"the position covering nodes {sorted(expected)} is incomplete: got {sorted(nodes)}, "
                f"expected {sorted(expected)}"
            )


def _resolve_planning_dp_indices(args: argparse.Namespace) -> list[int]:
    """Resolve which DP positions planning must cover.

    Falls back to the positions the sandbox nodes belong to, and refuses to guess when
    nothing at all is given so planning never silently covers a single DP position.
    """
    dp_size = args.pp_rank_stride // args.dp_rank_stride
    dp_node_stride = args.dp_rank_stride // args.gpus_per_node

    if args.planning_dp_indices:
        for dp_index in args.planning_dp_indices:
            if dp_index < 0 or dp_index >= dp_size:
                raise ValueError(
                    f"--planning-dp-indices contains invalid DP position {dp_index}; expected each index "
                    f"to be in [0, {dp_size}) because a PP stage holds {dp_size} DP positions"
                )
        return sorted(set(args.planning_dp_indices))

    if args.planning_dp_indices_num is not None:
        if args.planning_dp_indices_num < 1 or args.planning_dp_indices_num > dp_size:
            raise ValueError(
                f"--planning-dp-indices-num must be in [1, {dp_size}] because a PP stage holds {dp_size} "
                f"DP positions; got {args.planning_dp_indices_num}"
            )
        return list(range(args.planning_dp_indices_num))

    if args.sandbox_nodes:
        return sorted({node_dp_index(node, dp_node_stride, dp_size) for node in args.sandbox_nodes})

    raise ValueError(
        "planning mode requires one of --planning-dp-indices, --planning-dp-indices-num or --sandbox-nodes "
        "to determine which DP positions to cover"
    )


def _validate_emulate_profiler_coverage(args: argparse.Namespace) -> None:
    """Validate that every rank of the covered DP positions has runtime profiler data.

    This prevents an emulate sandbox from landing on a DP position that planning never
    covered, where the graph would only ever see durations of 0.
    """
    if not args.profiler_dir:
        return

    if not (args.graph_dir and args.pp_rank_stride is not None and args.dp_rank_stride is not None):
        raise ValueError(
            "emulate mode requires --graph-dir, --pp-rank-stride and --dp-rank-stride when --profiler-dir "
            "is provided, because the profiler coverage check needs the rank layout"
        )
    if not Path(args.profiler_dir).exists():
        raise ValueError(f"--profiler-dir does not exist: {args.profiler_dir}")

    world_size = args.target_nodes * args.gpus_per_node
    dp_size = args.pp_rank_stride // args.dp_rank_stride
    dp_node_stride = args.dp_rank_stride // args.gpus_per_node
    covered = {node_dp_index(node, dp_node_stride, dp_size) for node in args.sandbox_nodes}

    profiler_dir = Path(args.profiler_dir)
    missing = [
        rank
        for rank in range(world_size)
        if (rank % args.pp_rank_stride) // args.dp_rank_stride in covered
        and not (profiler_dir / f"runtime_profiler_{rank}.json.gz").exists()
    ]
    if missing:
        raise ValueError(
            f"--profiler-dir is missing runtime profiler data for {len(missing)} ranks of the DP positions "
            f"{sorted(covered)} covered by --sandbox-nodes, for example rank {missing[0]}; run planning for "
            "these DP positions first"
        )


def validate_args(args: argparse.Namespace) -> None:
    r"""Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If validation fails
    """
    try:
        _validate_args(args)
    except ValueError as error:
        logger.error("Invalid arguments: %s", error)
        sys.exit(1)


def _validate_args(args: argparse.Namespace) -> None:
    r"""Validate command line arguments and raise ValueError on the first failure.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If validation fails
    """
    # Validate paths
    if not Path(args.train_script_path).exists():
        raise ValueError(f"--train-script-path does not exist: {args.train_script_path}")

    # Three warmup iterations leave room for lazily initialized NCCL connections to
    # be established and for every module to settle before the analysis window.
    if args.train_iters < MIN_TRAIN_ITERS:
        raise ValueError(
            f"--train-iters must be >= {MIN_TRAIN_ITERS} so that lazily initialized NCCL connections are "
            "established and every module is fully warmed up before the analysis window; got "
            f"--train-iters={args.train_iters}"
        )

    # Create workspace and log directories
    workspace = Path(args.crystal_workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    log_dir = workspace / args.log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Validate mode specific arguments
    if args.mode == "baseline":
        if args.target_nodes > args.available_nodes:
            raise ValueError(
                "baseline mode requires --target-nodes <= --available-nodes, "
                f"but got --target-nodes={args.target_nodes} and --available-nodes={args.available_nodes}"
            )
    elif args.mode == "capture":
        if args.available_nodes <= 0:
            raise ValueError(f"capture mode requires --available-nodes > 0, but got {args.available_nodes}")
    elif args.mode in {"emulate", "planning"}:
        if args.mode == "emulate" and not args.sandbox_nodes:
            raise ValueError("emulate mode requires --sandbox-nodes")

        # Validate that all sandbox nodes are within valid range [0, target_nodes)
        for node_id in args.sandbox_nodes or []:
            if node_id < 0 or node_id >= args.target_nodes:
                raise ValueError(
                    f"--sandbox-nodes contains invalid node id {node_id}; expected each node id to be in "
                    f"[0, {args.target_nodes}) because --target-nodes={args.target_nodes}"
                )

        # --trace-dir is required in both emulate and planning modes.
        if not args.trace_dir:
            raise ValueError(f"{args.mode} mode requires --trace-dir")
        if not Path(args.trace_dir).exists():
            raise ValueError(f"--trace-dir does not exist: {args.trace_dir}")

        has_graph_dir = bool(args.graph_dir)
        has_pp_rank_stride = args.pp_rank_stride is not None
        has_dp_rank_stride = args.dp_rank_stride is not None

        if args.mode == "emulate":
            # The graph directory and both strides are optional, but only as a whole:
            # every cross-rank derivation needs all three.
            if len({has_graph_dir, has_pp_rank_stride, has_dp_rank_stride}) != 1:
                raise ValueError(
                    "emulate mode requires --graph-dir, --pp-rank-stride and --dp-rank-stride to be "
                    "given together or omitted together; got "
                    f"--graph-dir={args.graph_dir}, --pp-rank-stride={args.pp_rank_stride}, "
                    f"--dp-rank-stride={args.dp_rank_stride}"
                )
        elif args.mode == "planning":
            if not has_graph_dir:
                raise ValueError("planning mode requires --graph-dir")
            if not has_pp_rank_stride:
                raise ValueError("planning mode requires --pp-rank-stride")
            if not has_dp_rank_stride:
                raise ValueError("planning mode requires --dp-rank-stride")

        if has_graph_dir and not Path(args.graph_dir).exists():
            raise ValueError(f"--graph-dir does not exist: {args.graph_dir}")

        if has_pp_rank_stride and has_dp_rank_stride:
            _validate_rank_layout(args)
            dp_node_stride = args.dp_rank_stride // args.gpus_per_node
            _validate_sandbox_units(args.sandbox_nodes or [], dp_node_stride)
        else:
            dp_node_stride = 1

        if args.mode == "emulate":
            required_available_nodes = 2 * len(args.sandbox_nodes)
            if args.available_nodes < required_available_nodes:
                raise ValueError(
                    "emulate mode requires --available-nodes >= 2 * len(--sandbox-nodes) because each "
                    "sandbox node needs a dedicated peer node for PP communication to avoid bandwidth contention; "
                    f"got --available-nodes={args.available_nodes}, "
                    f"len(--sandbox-nodes)={len(args.sandbox_nodes)}, required={required_available_nodes}"
                )
            _validate_emulate_profiler_coverage(args)
        elif args.mode == "planning":
            if args.planning_sandbox_nodes_count % dp_node_stride != 0:
                raise ValueError(
                    "planning mode requires --planning-sandbox-nodes-count to consist of whole DP positions, "
                    f"so it must be a positive multiple of the {dp_node_stride} nodes one DP position occupies; "
                    f"got --planning-sandbox-nodes-count={args.planning_sandbox_nodes_count}"
                )
            if args.planning_sandbox_nodes_count < dp_node_stride:
                raise ValueError(
                    f"planning mode requires --planning-sandbox-nodes-count >= {dp_node_stride}, the number "
                    f"of nodes one DP position occupies; got "
                    f"--planning-sandbox-nodes-count={args.planning_sandbox_nodes_count}"
                )

            required_available_nodes = args.planning_sandbox_nodes_count + dp_node_stride
            if args.available_nodes < required_available_nodes:
                raise ValueError(
                    "planning mode requires --available-nodes >= --planning-sandbox-nodes-count + "
                    f"{dp_node_stride} so that assistant nodes are reserved for the sandbox and PP "
                    "communication does not contend for bandwidth; got "
                    f"--available-nodes={args.available_nodes}, "
                    f"--planning-sandbox-nodes-count={args.planning_sandbox_nodes_count}, "
                    f"required={required_available_nodes}"
                )
            # Validate that the planning DP coverage can be resolved.
            _resolve_planning_dp_indices(args)

    # Validate JSON files if provided
    if args.pretrain_envs_json:
        for file_path in args.pretrain_envs_json:
            if not Path(file_path).exists():
                raise ValueError(f"--pretrain-envs-json file does not exist: {file_path}")
            try:
                with Path(file_path).open(encoding="utf-8") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"--pretrain-envs-json file is not valid JSON: {file_path} ({e})") from e

    if args.pretrain_args_json:
        for file_path in args.pretrain_args_json:
            if not Path(file_path).exists():
                raise ValueError(f"--pretrain-args-json file does not exist: {file_path}")
            try:
                with Path(file_path).open(encoding="utf-8") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"--pretrain-args-json file is not valid JSON: {file_path} ({e})") from e

    if args.enable_mock_nccl_bootstrap and args.target_nodes < MOCK_NCCL_BOOTSTRAP_NODES_THRESHOLD:
        args.enable_mock_nccl_bootstrap = False

    # Validate assistant profiler arguments
    has_assistant_pytorch_profiler = args.enable_assistant_pytorch_profiler
    has_assistant_nsys_system = args.enable_assistant_nsight_system
    has_assistant_nsys_ranks = bool(args.assistant_nsight_system_ranks)

    if has_assistant_pytorch_profiler and (has_assistant_nsys_system or has_assistant_nsys_ranks):
        raise ValueError(
            "--enable-assistant-pytorch-profiler cannot be used together with "
            "--enable-assistant-nsight-system or --assistant-nsight-system-ranks"
        )

    if has_assistant_nsys_system != has_assistant_nsys_ranks:
        missing_arg = (
            "--assistant-nsight-system-ranks" if has_assistant_nsys_system else "--enable-assistant-nsight-system"
        )
        provided_arg = (
            "--enable-assistant-nsight-system" if has_assistant_nsys_system else "--assistant-nsight-system-ranks"
        )
        raise ValueError(f"{missing_arg} must be specified when {provided_arg} is provided")


def _load_json_dicts(file_paths: list[str] | None) -> dict[str, Any]:
    """Load and merge JSON dictionaries from the given files.

    Malformed files and non-dict payloads are skipped with a warning.

    Args:
        file_paths: List of paths to JSON files, or None if no files are provided.

    Returns:
        Dictionary containing merged entries from all JSON files.
    """
    merged_data: dict[str, Any] = {}
    for file_path in file_paths or []:
        try:
            with Path(file_path).open(encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged_data.update(data)
                else:
                    logger.warning("JSON file %s does not contain a dictionary, skipping", file_path)
        except Exception as e:
            logger.warning("Failed to load JSON file %s: %s", file_path, e)
    return merged_data


def _build_extra_args_str(args: argparse.Namespace) -> str:
    r"""Build extra arguments string from JSON files and CLI arguments.

    Args:
        args: Parsed command line arguments containing pretrain configuration.

    Returns:
        Formatted argument string combining CLI and JSON arguments.
    """
    # Load and merge JSON files
    merged_data = _load_json_dicts(args.pretrain_args_json)

    # Convert merged JSON data to argument string
    json_args_list = []
    for key, value in merged_data.items():
        arg_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                json_args_list.append(arg_key)
        else:
            json_args_list.append(f"{arg_key} {value}")
    json_args_str = " ".join(json_args_list)

    # Combine with CLI arguments
    cli_args_str = " ".join(args.pretrain_args) if args.pretrain_args else ""
    return " ".join(filter(None, [cli_args_str, json_args_str]))


def _build_envs_str(args: argparse.Namespace, *, for_assistant: bool = False) -> str:
    """Build training environment variables and return as formatted string.

    Args:
        args: Parsed command line arguments.
        for_assistant: Build envs for assistant processes instead of training processes.

    Returns:
        Formatted environment variables string ("KEY1=VALUE1 KEY2=VALUE2 ...")
    """
    analysis_step_start = args.train_iters + 1
    analysis_step_end = analysis_step_start + args.analysis_iters

    envs: dict[str, str] = {
        "GPUS_PER_NODE": str(args.gpus_per_node),
        "CRYSTAL_TRAIN_ITERS": str(args.train_iters + args.analysis_iters + 5),
        "CRYSTAL_ANALYSIS_STEP_START": str(analysis_step_start),
        "CRYSTAL_ANALYSIS_STEP_END": str(analysis_step_end),
    }

    # Add mode specific feature switches
    if for_assistant:
        if args.enable_assistant_pytorch_profiler:
            envs["ENABLE_CRYSTAL_ASSISTANT_PYTORCH_PROFILER"] = "true"
        if args.enable_assistant_nsight_system:
            envs["ENABLE_CRYSTAL_ASSISTANT_NSIGHT_SYSTEM"] = "true"
    else:
        envs["NNODES"] = str(args.target_nodes)
        if args.enable_crystal_c10d_trace:
            envs["ENABLE_CRYSTAL_C10D_TRACE"] = "true"
        if args.crystal_c10d_printing:
            envs["CRYSTAL_C10D_PRINTING"] = "true"
        if args.enable_crystal_timer and args.mode not in {"capture", "planning"}:
            envs["ENABLE_CRYSTAL_TIMER"] = "true"
        if args.enable_crystal_monitor and args.mode != "capture":
            envs["ENABLE_CRYSTAL_MONITOR"] = "true"
        if args.enable_crystal_runtime_profiler and args.mode not in {"capture", "planning"}:
            envs["ENABLE_CRYSTAL_RUNTIME_PROFILER"] = "true"

    # Load and merge additional envs from JSON files
    envs.update(_load_json_dicts(args.pretrain_envs_json))

    # Convert to string format
    return " ".join([f"{k}={v}" for k, v in envs.items()])


def baseline_mode(args: argparse.Namespace) -> int:
    r"""Run baseline mode.

    This mode runs full training on all nodes to establish a baseline.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success)
    """
    logger.info("Running baseline mode with %s nodes", args.target_nodes)

    # Start CrystalServer
    server = CrystalServer(host=args.server_addr, port=args.server_port)
    server.start_server()

    try:
        task_nodes = list(range(args.target_nodes))
        # Wait for all clients to connect
        server.control_client(task_nodes)

        # Build environment and arguments
        extra_args_str = _build_extra_args_str(args)
        extra_envs_str = _build_envs_str(args)

        extra_envs_str += " CRYSTAL_MODE=BASELINE"

        # Get the actual address of rank 0 node from coordinator server
        # This is used as the master address for distributed training
        rank0_address = server.get_client_address(0)
        if rank0_address is None:
            logger.error("Failed to get address of rank 0 node from coordinator server")
            raise RuntimeError("Rank 0 node address not available")
        logger.info("Using rank 0 node address as master_addr: %s", rank0_address)

        # Dispatch training tasks to all nodes and check start status
        dispatch_handles = []
        for node_rank in task_nodes:
            handle = server.dispatch_train_task(
                master_addr=rank0_address,
                master_port=args.master_port,
                node_rank=node_rank,
                log_dir_name=args.log_dir_name,
                train_script_path=args.train_script_path,
                extra_args_str=extra_args_str,
                extra_envs_str=extra_envs_str,
                msg_target_node_rank=node_rank,
            )
            dispatch_handles.append((node_rank, handle))

        # Check if all dispatch operations succeeded
        for node_rank, handle in dispatch_handles:
            results = server.wait_for_reply(handle)
            code = results.get(node_rank, ReplyCode.CLIENT_DISCONNECTED)
            if code != ReplyCode.SUCCESS:
                logger.error("Node %s failed to start training: %s", node_rank, describe_reply_code(code))
                raise RuntimeError(f"Node {node_rank} failed to start training: {describe_reply_code(code)}")
            logger.info("Node %s training started successfully", node_rank)

        # Send TRAIN_WAIT to all nodes and wait for completion
        wait_handle = server.dispatch_train_wait(task_nodes)
        results = server.wait_for_reply(wait_handle)
        logger.info("All nodes training completed with results: \n %s", results)

        # Check for any failures; a node missing from the results went offline
        # before the wait was even dispatched.
        for node_rank in task_nodes:
            code = results.get(node_rank, ReplyCode.CLIENT_DISCONNECTED)
            if code != ReplyCode.SUCCESS:
                logger.error("Node %s training failed: %s", node_rank, describe_reply_code(code))
                raise RuntimeError(f"Node {node_rank} training failed: {describe_reply_code(code)}")

        # Stop training on all nodes
        server.stop_train_on_nodes(task_nodes)

        logger.info("Baseline mode completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt, stopping all training...")
        server.stop_train_on_nodes(task_nodes)
        return 1
    except Exception as e:
        logger.exception("Error in baseline mode: %s", e)
        server.stop_train_on_nodes(task_nodes)
        return 1
    finally:
        # Shutdown all clients and stop server
        if args.shutdown_client:
            server.shutdown_client()
        server.stop_server()


def _node_log_has_success_markers(log_file: Path) -> bool:
    """Return whether a node log indicates all local processes finished successfully."""
    try:
        with log_file.open(encoding="utf-8", errors="ignore") as f:
            # Read the last portion of the file to check for completion markers.
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            # Read last 4KB (should be enough to find completion markers).
            read_size = min(4096, file_size)
            f.seek(max(0, file_size - read_size))
            tail_content = f.read()
    except OSError:
        return False

    success_marker = "All processes completed successfully"
    exit_success_marker = "Exit code: 0"
    success_idx = tail_content.rfind(success_marker)
    exit_success_idx = tail_content.rfind(exit_success_marker)
    last_exit_idx = tail_content.rfind("Exit code:")

    return (
        success_idx != -1
        and exit_success_idx != -1
        and success_idx < exit_success_idx
        and exit_success_idx == last_exit_idx
    )


def _collect_completed_capture_nodes(args: argparse.Namespace) -> set[int]:
    """Collect captured logical nodes with complete traces and successful node logs."""
    log_root = Path(args.crystal_workspace) / args.log_dir_name
    trace_dir = log_root / "trace"
    node_log_dir = log_root / "node"
    completed_ranks: set[int] = set()

    logger.info("Capture trace scan path: %s", trace_dir)
    logger.info("Capture node log scan path: %s", node_log_dir)
    if not trace_dir.exists() or not node_log_dir.exists():
        return set()

    for trace_file in trace_dir.glob("trace_*.log.gz"):
        file_name = trace_file.name
        rank_str = file_name[len("trace_") : -len(".log.gz")]
        if not rank_str.isdigit():
            continue
        completed_ranks.add(int(rank_str))

    completed_nodes: set[int] = set()
    for node_id in range(args.target_nodes):
        rank_start = node_id * args.gpus_per_node
        rank_end = rank_start + args.gpus_per_node
        has_complete_traces = all(global_rank in completed_ranks for global_rank in range(rank_start, rank_end))
        if has_complete_traces and _node_log_has_success_markers(node_log_dir / f"node_{node_id}.log"):
            completed_nodes.add(node_id)

    return completed_nodes


def capture_mode(args: argparse.Namespace) -> int:
    r"""Run capture mode.

    This mode captures communication traces from training by processing target nodes in batches.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info(
        "Running capture mode with %s target nodes using %s available nodes", args.target_nodes, args.available_nodes
    )

    # Start CrystalServer
    server = CrystalServer(host=args.server_addr, port=args.server_port)
    server.start_server()

    try:
        task_nodes = list(range(args.available_nodes))
        # Wait for all available clients to connect
        server.control_client(task_nodes)

        # Build environment and arguments
        extra_args_str = _build_extra_args_str(args)
        extra_envs_str = _build_envs_str(args)

        # Add capture-specific environment variables
        extra_envs_str += (
            " CRYSTAL_MODE=CAPTURE"
            " ENABLE_CRYSTAL_C10D_TRACE=true"
            " CRYSTAL_C10D_CAPTURING=true"
            " CRYSTAL_GRAPH_CAPTURING=true"
        )

        # Resume capture: skip logical nodes with complete existing traces and successful node logs.
        completed_nodes = _collect_completed_capture_nodes(args)
        pending_nodes = [node_id for node_id in range(args.target_nodes) if node_id not in completed_nodes]

        if completed_nodes:
            logger.info(
                "Detected %s/%s completed capture nodes from existing traces and successful node logs. Remaining: %s",
                len(completed_nodes),
                args.target_nodes,
                len(pending_nodes),
            )

        if not pending_nodes:
            logger.info(
                "All target nodes already have complete capture traces and successful node logs, nothing to do."
            )
            return 0

        logger.info("Pending target nodes to capture: %s", pending_nodes)

        # Process only pending logical nodes in batches.
        for batch_start_idx in range(0, len(pending_nodes), args.available_nodes):
            batch_nodes = pending_nodes[batch_start_idx : batch_start_idx + args.available_nodes]

            # Get the actual address of rank 0 node from coordinator server
            # This is used as the master address for distributed training
            rank0_address = server.get_client_address(0)
            if rank0_address is None:
                logger.error("Failed to get address of rank 0 node from coordinator server")
                raise RuntimeError("Rank 0 node address not available")
            logger.info("Using rank 0 node address as master_addr: %s", rank0_address)

            # Dispatch capturing tasks to nodes in this batch
            dispatch_handles = []
            for target_rank, node_rank in enumerate(batch_nodes):
                # Map each logical node in this batch to a unique physical node rank [0, batch_size-1].
                handle = server.dispatch_train_task(
                    master_addr=rank0_address,
                    master_port=args.master_port,
                    node_rank=node_rank,
                    log_dir_name=args.log_dir_name,
                    train_script_path=args.train_script_path,
                    extra_args_str=extra_args_str,
                    extra_envs_str=extra_envs_str,
                    msg_target_node_rank=target_rank,
                )
                dispatch_handles.append((node_rank, target_rank, handle))

            # Extract target ranks for batch operations
            batch_target_nodes = [target_rank for _, target_rank, _ in dispatch_handles]

            # Check if all dispatch operations succeeded
            for node_rank, target_rank, handle in dispatch_handles:
                results = server.wait_for_reply(handle)
                code = results.get(target_rank, ReplyCode.CLIENT_DISCONNECTED)
                if code != ReplyCode.SUCCESS:
                    detail = describe_reply_code(code)
                    logger.error(
                        "Node %s (target_rank=%s) failed to start capturing: %s", node_rank, target_rank, detail
                    )
                    raise RuntimeError(
                        f"Node {node_rank} (target_rank={target_rank}) failed to start capturing: {detail}"
                    )
                logger.info("Node %s (target_rank=%s) capturing started successfully", node_rank, target_rank)

            # Send TRAIN_WAIT to target ranks in this batch and wait for completion
            wait_handle = server.dispatch_train_wait(batch_target_nodes)
            results = server.wait_for_reply(wait_handle)
            logger.info("Batch with logical nodes %s capturing completed with results: %s", batch_nodes, results)

            # Check for any failures in this batch; a rank missing from the
            # results went offline before the wait was even dispatched.
            for target_rank in batch_target_nodes:
                code = results.get(target_rank, ReplyCode.CLIENT_DISCONNECTED)
                if code != ReplyCode.SUCCESS:
                    logger.error("Target rank %s capturing failed: %s", target_rank, describe_reply_code(code))
                    raise RuntimeError(f"Target rank {target_rank} capturing failed: {describe_reply_code(code)}")

            # Stop capturing on target ranks in this batch
            server.stop_train_on_nodes(batch_target_nodes)

            logger.info("Batch with logical nodes %s finished", batch_nodes)

            # Small delay between batches to ensure cleanup
            time.sleep(5)

        logger.info("Capture mode completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt, stopping all capturing...")
        server.stop_train_on_nodes(task_nodes)
        return 1
    except Exception as e:
        logger.exception("Error in capture mode: %s", e)
        server.stop_train_on_nodes(task_nodes)
        return 1
    finally:
        # Shutdown all clients and stop server
        if args.shutdown_client:
            server.shutdown_client()
        server.stop_server()


def _validate_mock_bootstrap_payload(
    payload: Any,
    sandbox_ranks: list[int],
    gpus_per_node: int,
) -> tuple[list[int], list[int]]:
    """Validate remote mock bootstrap payload and return required fields."""
    if not isinstance(payload, dict):
        raise RuntimeError("Mock bootstrap prepare reply payload is missing or invalid")

    extended_member_ranks = payload.get("extended_member_ranks")
    member_nodes = payload.get("member_nodes")
    if not isinstance(extended_member_ranks, list) or not all(isinstance(rank, int) for rank in extended_member_ranks):
        raise RuntimeError("Mock bootstrap prepare reply missing valid extended_member_ranks")
    if not isinstance(member_nodes, list) or not all(isinstance(node, int) for node in member_nodes):
        raise RuntimeError("Mock bootstrap prepare reply missing valid member_nodes")

    overlap_ranks = sorted(set(extended_member_ranks) & set(sandbox_ranks))
    if overlap_ranks:
        raise RuntimeError(f"Mock bootstrap extended_member_ranks overlap sandbox_ranks: {overlap_ranks}")

    expected_member_nodes = sorted({rank // gpus_per_node for rank in extended_member_ranks})
    if member_nodes != expected_member_nodes:
        raise RuntimeError(
            "Mock bootstrap member_nodes do not match extended_member_ranks: "
            f"member_nodes={member_nodes}, expected={expected_member_nodes}"
        )

    return extended_member_ranks, member_nodes


def _run_emulate(
    args: argparse.Namespace,
    server: CrystalServer,
    sandbox_nodes: list[int],
    log_dir_name: str,
    profiler_dir: str | None,
    additional_training_envs: str | None = None,
    sandbox_target_nodes: list[int] | None = None,
    assistant_target_nodes: list[int] | None = None,
    emulated_nodes_per_assistant: list[list[int]] | None = None,
):
    r"""Run emulation with sandbox nodes and assistant processes.

    This is a helper function that orchestrates the emulation process by dispatching
    training tasks to sandbox nodes and starting assistant processes on the assistant node(s).

    Args:
        args: Parsed command line arguments.
        server: CrystalServer instance for coordinating distributed tasks.
        sandbox_nodes: List of node IDs designated as sandbox nodes (running real training).
        log_dir_name: Name of the directory for logging.
        profiler_dir: Directory path for Crystal runtime profiler data.
        additional_training_envs: Additional environment variables for training processes.
            Defaults to None.
        sandbox_target_nodes: List of target node ranks for sandbox processes.
            If None, uses range(len(sandbox_nodes)).
        assistant_target_nodes: List of target node ranks for assistant processes.
            If None, uses single assistant mode at sandbox_target_nodes[-1] + 1.
        emulated_nodes_per_assistant: List of lists, where each inner list contains the
            node IDs that a specific assistant should emulate. If None, uses single
            assistant mode (backward compatible).
    """
    # Backward compatible defaults
    if sandbox_target_nodes is None:
        sandbox_target_nodes = list(range(len(sandbox_nodes)))
    if assistant_target_nodes is None:
        assistant_target_nodes = [len(sandbox_nodes)]
    if emulated_nodes_per_assistant is None:
        all_emulated_nodes = [n for n in range(args.target_nodes) if n not in sandbox_nodes]
        emulated_nodes_per_assistant = [all_emulated_nodes]

    # Build environment and arguments for training (sandbox nodes)
    extra_args_str = _build_extra_args_str(args)
    extra_envs_str = _build_envs_str(args)

    extra_envs_str += " CRYSTAL_MODE=EMULATE"
    if additional_training_envs:
        extra_envs_str += additional_training_envs

    # Build assistant environment variables
    # Note: LD_PRELOAD is automatically set by AssistantRunner based on mock NCCL bootstrap settings
    assistant_envs_str = _build_envs_str(args, for_assistant=True)

    # Get Redis host address before mock bootstrap planning so coordinator-side Redis writes use the same server.
    redis_host = server.get_client_address(assistant_target_nodes[0])
    if redis_host is None:
        logger.error(
            "Failed to get address of assistant node %s from coordinator server for Redis host",
            assistant_target_nodes[0],
        )
        raise RuntimeError("Assistant node address for Redis host not available")
    logger.info("Using Redis host: %s", redis_host)

    # Resolve the physical client that hosts logical rank 0, then use its address as MASTER_ADDR.
    if 0 in sandbox_nodes:
        rank0_target_node = sandbox_target_nodes[sandbox_nodes.index(0)]
    else:
        rank0_target_node = None
        for assistant_index, emulated_nodes in enumerate(emulated_nodes_per_assistant):
            if 0 in emulated_nodes:
                rank0_target_node = assistant_target_nodes[assistant_index]
                break
        if rank0_target_node is None:
            raise RuntimeError("Logical rank 0 is neither a sandbox node nor assigned to any assistant")

    rank0_address = server.get_client_address(rank0_target_node)
    if rank0_address is None:
        logger.error(
            "Failed to get address of physical node %s that hosts logical rank 0",
            rank0_target_node,
        )
        raise RuntimeError(f"Physical node {rank0_target_node} hosting logical rank 0 address not available")
    logger.info(
        "Using physical node %s address as master_addr for logical rank 0: %s",
        rank0_target_node,
        rank0_address,
    )

    required_assistant_nodes = None
    # Launched ranks the assistants replay with: the member ranks the mock NCCL
    # bootstrap planner decided, or every emulated rank when no planner ran. Also
    # the launched set the graph cache below is built from.
    launched_ranks: list[int] | None = None

    # Prepare mock NCCL bootstrap if enabled
    if args.enable_mock_nccl_bootstrap:
        sandbox_ranks = []
        for node_id in sandbox_nodes:
            sandbox_ranks.extend(list(range(node_id * args.gpus_per_node, (node_id + 1) * args.gpus_per_node)))

        handle = server.prepare_mock_bootstrap(
            msg_target_node_rank=assistant_target_nodes[0],
            redis_host=redis_host,
            trace_dir=args.trace_dir,
            nnodes=args.target_nodes,
            sandbox_nodes=sandbox_nodes,
            gpus_per_node=args.gpus_per_node,
        )
        prepare_results = server.wait_for_reply_with_payload(handle)
        prepare_result = prepare_results.get(
            assistant_target_nodes[0], {"code": ReplyCode.CLIENT_DISCONNECTED, "payload": None}
        )
        prepare_code = prepare_result.get("code", ReplyCode.CLIENT_DISCONNECTED)
        if prepare_code != ReplyCode.SUCCESS:
            raise RuntimeError(
                f"Mock bootstrap prepare failed on target node {assistant_target_nodes[0]}: "
                f"{describe_reply_code(prepare_code)}"
            )

        extended_member_ranks, member_nodes = _validate_mock_bootstrap_payload(
            prepare_result.get("payload"),
            sandbox_ranks,
            args.gpus_per_node,
        )
        required_assistant_nodes = set(member_nodes)

        init_store_size = len(extended_member_ranks) + len(sandbox_ranks)
        sandbox_ranks_str = ",".join(map(str, sandbox_ranks))
        member_ranks_str = ",".join(map(str, extended_member_ranks))

        # Build sandbox environment with mock NCCL bootstrap
        mock_bootstrap_envs_str = (
            f" CRYSTAL_MOCK_NCCL_INIT_STORE_SIZE={init_store_size}"
            f" CRYSTAL_MOCK_NCCL_SANDBOX_RANKS={sandbox_ranks_str}"
            f" CRYSTAL_MOCK_NCCL_MEMBER_RANKS={member_ranks_str}"
        )
        extra_envs_str += mock_bootstrap_envs_str
        assistant_envs_str += mock_bootstrap_envs_str
        logger.info(
            "Mock NCCL bootstrap enabled: init_store_size=%s, member_nodes=%s",
            init_store_size,
            member_nodes,
        )
        launched_ranks = extended_member_ranks

    # Generate the launch-timing cache for the assistants (on every _run_emulate
    # call): the cache must reflect this batch's sandbox split, and the timestamps
    # keep batches and planning epochs apart. It replaces loading the graphs in
    # every assistant process, so the cache path is passed instead of the graph
    # directory whenever online updates are off. Built after the mock bootstrap
    # planning so it sees exactly the launched ranks the assistants replay with,
    # making its bundle identical to the one a started assistant would load.
    graph_cache_path: str | None = None
    if args.graph_dir is not None and not args.enable_assistant_online_update:
        sandbox_ranks = []
        for node_id in sandbox_nodes:
            sandbox_ranks.extend(list(range(node_id * args.gpus_per_node, (node_id + 1) * args.gpus_per_node)))
        cache_dir = Path(args.graph_dir).parent / f"{Path(args.graph_dir).name}_cache"
        graph_cache_path = str(
            generate_graph_cache(
                graph_dir=args.graph_dir,
                trace_dir=args.trace_dir,
                profiler_dir=profiler_dir,
                pp_rank_stride=args.pp_rank_stride,
                sandbox_ranks=sandbox_ranks,
                world_size=args.target_nodes * args.gpus_per_node,
                output_dir=cache_dir,
                launched_ranks=launched_ranks,
            )
        )
        logger.info("Generated graph cache for assistants: %s", graph_cache_path)

    assistant_launch_plan: list[tuple[int, list[int]]] = []
    if not args.enable_mock_nccl_bootstrap:
        for i, assistant_target_node in enumerate(assistant_target_nodes):
            assistant_launch_plan.append((assistant_target_node, emulated_nodes_per_assistant[i]))
    elif required_assistant_nodes is None:
        raise RuntimeError("Mock NCCL bootstrap enabled but required assistant nodes were not computed")
    else:
        skipped_assistant_target_nodes = []
        for i, assistant_target_node in enumerate(assistant_target_nodes):
            nodes_for_this_assistant = emulated_nodes_per_assistant[i]
            required_emulated_nodes_for_this_assistant = [
                node for node in nodes_for_this_assistant if node in required_assistant_nodes
            ]
            if required_emulated_nodes_for_this_assistant:
                assistant_launch_plan.append((assistant_target_node, required_emulated_nodes_for_this_assistant))
            else:
                skipped_assistant_target_nodes.append((assistant_target_node, nodes_for_this_assistant))

        if not assistant_launch_plan:
            raise RuntimeError(
                "Mock NCCL bootstrap did not assign any member nodes to assistant launch plan: "
                f"member_nodes={sorted(required_assistant_nodes)}, "
                f"emulated_nodes_per_assistant={emulated_nodes_per_assistant}"
            )

        if skipped_assistant_target_nodes:
            logger.info(
                "Mock NCCL bootstrap skipped assistant target nodes without member nodes: %s",
                skipped_assistant_target_nodes,
            )

    # Dispatch training tasks to sandbox nodes
    # Use assistant node's address as master_addr for all training processes
    dispatch_handles = []
    for i, node_rank in enumerate(sandbox_nodes):
        target_rank = sandbox_target_nodes[i]
        handle = server.dispatch_train_task(
            master_addr=rank0_address,
            master_port=args.master_port,
            node_rank=node_rank,
            log_dir_name=log_dir_name,
            train_script_path=args.train_script_path,
            extra_args_str=extra_args_str,
            extra_envs_str=extra_envs_str,
            msg_target_node_rank=target_rank,
        )
        dispatch_handles.append((node_rank, target_rank, handle))

    # Check if all training dispatch operations succeeded
    for node_rank, target_rank, handle in dispatch_handles:
        results = server.wait_for_reply(handle)
        code = results.get(target_rank, ReplyCode.CLIENT_DISCONNECTED)
        if code != ReplyCode.SUCCESS:
            detail = describe_reply_code(code)
            logger.error("Node %s (target_rank=%s) failed to start training: %s", node_rank, target_rank, detail)
            raise RuntimeError(f"Node {node_rank} (target_rank={target_rank}) failed to start training: {detail}")
        logger.info("Node %s (target_rank=%s) training started successfully", node_rank, target_rank)

    # Start assistant processes on EACH assistant node
    assistant_handles = []
    for assistant_target_node, required_emulated_nodes_for_this_assistant in assistant_launch_plan:
        logger.info(
            "Starting assistant on physical node %s to emulate logical nodes %s",
            assistant_target_node,
            required_emulated_nodes_for_this_assistant,
        )

        # With online updates off the assistants read the prebuilt launch-timing
        # cache instead of loading the graphs themselves; with online updates on
        # they must hold the live graphs, so the cache path stays unused.
        assistant_graph_dir = None if graph_cache_path is not None else args.graph_dir
        handle = server.start_assistants(
            msg_target_node_rank=assistant_target_node,
            master_addr=rank0_address,
            master_port=args.master_port,
            redis_host=redis_host,
            log_dir_name=log_dir_name,
            emulated_nodes_for_this_assistant=required_emulated_nodes_for_this_assistant,
            nnodes=args.target_nodes,
            sandbox_nodes=sandbox_nodes,
            assistant_envs_str=assistant_envs_str,
            trace_dir=args.trace_dir,
            graph_dir=assistant_graph_dir,
            graph_cache_path=graph_cache_path,
            profiler_dir=profiler_dir,
            pp_rank_stride=args.pp_rank_stride,
            enable_assistant_online_update=args.enable_assistant_online_update,
            tensor_provider=args.tensor_provider,
            enable_mock_nccl_bootstrap=args.enable_mock_nccl_bootstrap,
            assistant_nsight_system_ranks=args.assistant_nsight_system_ranks,
            gpus_per_node=args.gpus_per_node,
        )
        assistant_handles.append((assistant_target_node, handle))

    # Wait for ALL assistant start confirmations
    for assistant_target_node, handle in assistant_handles:
        results = server.wait_for_reply(handle)
        code = results.get(assistant_target_node, ReplyCode.CLIENT_DISCONNECTED)
        if code != ReplyCode.SUCCESS:
            logger.error("Assistant node %s failed to start: %s", assistant_target_node, describe_reply_code(code))
            raise RuntimeError(f"Assistant node {assistant_target_node} failed to start: {describe_reply_code(code)}")
        logger.info("Assistant processes started successfully on node %s", assistant_target_node)

    # Wait for all training tasks to complete
    wait_handle = server.dispatch_train_wait(sandbox_target_nodes)
    results = server.wait_for_reply(wait_handle)
    logger.info("Training completed with results: %s", results)

    # Check for any failures; a rank missing from the results went offline
    # before the wait was even dispatched.
    for target_rank in sandbox_target_nodes:
        code = results.get(target_rank, ReplyCode.CLIENT_DISCONNECTED)
        if code != ReplyCode.SUCCESS:
            logger.error("Target rank %s training failed: %s", target_rank, describe_reply_code(code))
            raise RuntimeError(f"Target rank {target_rank} training failed: {describe_reply_code(code)}")

    # Stop training on sandbox nodes
    server.stop_train_on_nodes(sandbox_target_nodes)

    # Stop started assistant processes
    for assistant_target_node, _ in assistant_handles:
        server.stop_assistants(assistant_target_node)


def emulate_mode(args: argparse.Namespace) -> int:
    r"""Run emulate mode.

    This mode runs training on sandbox nodes with assistant processes
    emulating the remaining nodes. Supports distributing emulated nodes
    across multiple assistant nodes when available_nodes > sandbox_nodes + 1.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    sandbox_nodes = list(args.sandbox_nodes)
    logger.info("Emulate mode does not resume from partial logs; running all sandbox nodes: %s", sandbox_nodes)

    # Calculate emulated nodes (nodes that need assistant processes)
    emulated_nodes = [n for n in range(args.target_nodes) if n not in sandbox_nodes]

    # Calculate number of assistant nodes from available_nodes
    num_assistant_nodes = args.available_nodes - len(sandbox_nodes)

    logger.info(
        "Running emulate mode with sandbox_nodes=%s, emulated_nodes=%s, num_assistant_nodes=%s",
        sandbox_nodes,
        emulated_nodes,
        num_assistant_nodes,
    )

    # Distribute emulated nodes across assistant nodes (round-robin)
    emulated_nodes_per_assistant: list[list[int]] = [[] for _ in range(num_assistant_nodes)]
    for i, node_id in enumerate(emulated_nodes):
        assistant_idx = i % num_assistant_nodes
        emulated_nodes_per_assistant[assistant_idx].append(node_id)

    logger.info("Emulated nodes distribution across assistants: %s", emulated_nodes_per_assistant)

    # Use a stable physical layout: sandbox nodes first, then assistant nodes.
    # _run_emulate resolves the actual physical client hosting logical rank 0 from this mapping.
    sandbox_target_nodes = list(range(len(sandbox_nodes)))
    assistant_target_nodes = list(range(len(sandbox_nodes), len(sandbox_nodes) + num_assistant_nodes))

    logger.info(
        "Physical node assignment: sandbox_target_nodes=%s, assistant_target_nodes=%s",
        sandbox_target_nodes,
        assistant_target_nodes,
    )

    # Start CrystalServer
    server = CrystalServer(host=args.server_addr, port=args.server_port)
    server.start_server()

    try:
        # Determine controlled nodes (all physical nodes used)
        task_nodes = list(range(len(sandbox_nodes) + num_assistant_nodes))

        # Wait for all clients to connect
        server.control_client(task_nodes)

        _run_emulate(
            args,
            server,
            sandbox_nodes,
            args.log_dir_name,
            args.profiler_dir,
            sandbox_target_nodes=sandbox_target_nodes,
            assistant_target_nodes=assistant_target_nodes,
            emulated_nodes_per_assistant=emulated_nodes_per_assistant,
        )

        logger.info("Emulate mode completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt, stopping all training...")
        server.stop_train_on_nodes(sandbox_target_nodes)
        for assistant_target_node in assistant_target_nodes:
            server.stop_assistants(assistant_target_node)
        return 1
    except Exception as e:
        logger.exception("Error in emulate mode: %s", e)
        server.stop_train_on_nodes(sandbox_target_nodes)
        for assistant_target_node in assistant_target_nodes:
            server.stop_assistants(assistant_target_node)
        return 1
    finally:
        # Shutdown all clients and stop server
        if args.shutdown_client:
            server.shutdown_client()
        server.stop_server()


def _collect_completed_planning_sandbox_nodes(args: argparse.Namespace, epoch: int) -> set[int]:
    """Collect successfully completed sandbox nodes from the standard planning log directory.

    A sandbox node is considered completed if its node_X.log file exists and
    contains the success marker "All processes completed successfully" followed
    by "Exit code: 0".
    """
    log_dir = Path(args.crystal_workspace) / args.log_dir_name / f"epoch_{epoch}" / "node"
    completed_nodes: set[int] = set()

    logger.info("Planning log scan path: %s", log_dir)
    if not log_dir.exists():
        return completed_nodes

    for log_file in log_dir.glob("node_*.log"):
        file_name = log_file.name
        node_str = file_name[len("node_") : -len(".log")]
        if not node_str.isdigit():
            continue

        if _node_log_has_success_markers(log_file):
            completed_nodes.add(int(node_str))

    return completed_nodes


def _graph_max_connections(graph_dir: Path, rank: int = 0) -> int | None:
    """Read the max_connections of one rank's graph file; None when unreadable.

    Planning only needs the device overlap capability the graphs were captured
    under, which every step of every rank shares, so the first step of one rank
    (rank 0 by convention) decides for the whole capture.
    """
    graph_file = graph_dir / f"graph_{rank}.json.gz"
    try:
        with gzip.open(graph_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read graph file %s (%s); assuming non-max_connections==1", graph_file, exc)
        return None
    first_step = next(iter(data.values()), None)
    return first_step["max_connections"] if first_step else None


def planning_mode(args: argparse.Namespace) -> int:
    r"""Run planning mode.

    This mode runs multiple planning epochs with sandbox nodes to collect
    timing data for performance analysis and planning. Supports multiple
    assistant nodes when available_nodes > planning_sandbox_nodes_count + 1.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Running planning mode target sandbox_nodes=%s", args.sandbox_nodes)

    # Start CrystalServer
    server = CrystalServer(host=args.server_addr, port=args.server_port)
    server.start_server()

    # Calculate number of assistant nodes from available_nodes
    num_assistant_nodes = args.available_nodes - args.planning_sandbox_nodes_count

    # Use a stable physical layout: sandbox nodes first, then assistant nodes.
    # _run_emulate resolves the actual physical client hosting logical rank 0 from this mapping.
    sandbox_target_nodes = list(range(args.planning_sandbox_nodes_count))
    assistant_target_nodes = list(
        range(args.planning_sandbox_nodes_count, args.planning_sandbox_nodes_count + num_assistant_nodes)
    )

    workspace = Path(args.crystal_workspace)
    runtime_profiler_dir = workspace / args.log_dir_name / "profiler" / "crystal"
    runtime_profiler_dir.mkdir(parents=True, exist_ok=True)

    additional_training_envs = (
        f" ENABLE_CRYSTAL_RUNTIME_PROFILER=true CRYSTAL_RUNTIME_PROFILER_DIR={str(runtime_profiler_dir.absolute())}"
    )

    # A graph captured under max_connections == 1 pairs its cross-rank P2P peers with
    # WEAK_BIDIRECT edges that join no timing model, so replay could not calibrate
    # time through them. When rank 0's graph reports max_connections == 1, every rank
    # graph is adjusted incrementally into a sibling directory (async_op flags cleared
    # so the peers pair with BIDIRECT, channel STOP->START ordering restored), all
    # assistants are pointed at the adjusted copies, and the runtime profiler is told
    # to skip the issue-wait durations of asynchronous communications — their
    # overlapped computation would distort a graph whose async launches are now
    # treated as synchronous.
    if _graph_max_connections(Path(args.graph_dir), rank=0) == 1:
        adjusted_graph_dir = adjust_graph_dir(args.graph_dir)
        args.graph_dir = str(adjusted_graph_dir)
        additional_training_envs += " CRYSTAL_RUNTIME_PROFILER_IGNORE_ASYNC_ISSUE_WAIT=true"
        logger.info("max_connections==1 graph detected; using adjusted graphs at %s", args.graph_dir)

    # Nodes that planning must run as sandbox, ordered along PP first and DP second so that
    # finishing one DP position already yields complete PP chains usable for replay.
    dp_size = args.pp_rank_stride // args.dp_rank_stride
    dp_node_stride = args.dp_rank_stride // args.gpus_per_node
    covered_dp_indices = _resolve_planning_dp_indices(args)
    planning_nodes = sorted(
        (
            node
            for node in range(args.target_nodes)
            if node_dp_index(node, dp_node_stride, dp_size) in covered_dp_indices
        ),
        key=lambda node: (node_dp_index(node, dp_node_stride, dp_size), node),
    )
    logger.info("Planning covers DP positions %s over nodes: %s", covered_dp_indices, planning_nodes)

    try:
        # Determine controlled nodes (all physical nodes used)
        task_nodes = list(range(args.planning_sandbox_nodes_count + num_assistant_nodes))

        # Wait for all clients to connect
        server.control_client(task_nodes)

        for epoch in range(args.planning_epochs):
            epoch_log_dir_name = f"{args.log_dir_name}/epoch_{epoch}"
            epoch_log_dir = workspace / epoch_log_dir_name
            epoch_log_dir.mkdir(parents=True, exist_ok=True)

            # Resume planning: skip sandbox nodes that have already completed successfully in this epoch.
            completed_sandbox_nodes = _collect_completed_planning_sandbox_nodes(args, epoch)
            completed_planning_nodes = completed_sandbox_nodes & set(planning_nodes)
            if completed_planning_nodes:
                logger.info(
                    "Epoch %s: Detected %d/%d completed sandbox nodes from existing logs: %s",
                    epoch,
                    len(completed_planning_nodes),
                    len(planning_nodes),
                    sorted(completed_planning_nodes),
                )

            batch_size = args.planning_sandbox_nodes_count
            pending_batches = []
            for batch_start in range(0, len(planning_nodes), batch_size):
                batch_nodes = planning_nodes[batch_start : batch_start + batch_size]
                completed_batch_nodes = completed_sandbox_nodes & set(batch_nodes)
                batch_index = batch_start // batch_size

                if len(completed_batch_nodes) == len(batch_nodes):
                    logger.info(
                        "Epoch %s, batch %s already completed; skipping nodes: %s",
                        epoch,
                        batch_index,
                        batch_nodes,
                    )
                    continue

                if completed_batch_nodes:
                    logger.info(
                        "Epoch %s, batch %s is partially completed; rerunning entire batch. completed=%s, batch=%s",
                        epoch,
                        batch_index,
                        sorted(completed_batch_nodes),
                        batch_nodes,
                    )

                pending_batches.append((batch_index, batch_nodes))

            if not pending_batches:
                logger.info("Epoch %s: All batches already have successful logs, skipping this epoch.", epoch)
                continue

            logger.info("Epoch %s: Pending planning batches: %s", epoch, pending_batches)

            for batch_index, batch_nodes in pending_batches:
                logger.info("Epoch %s, running batch %s: nodes %s", epoch, batch_index, batch_nodes)
                batch_sandbox_target_nodes = sandbox_target_nodes[: len(batch_nodes)]

                # Calculate emulated nodes (nodes not in this batch)
                emulated_nodes = [n for n in range(args.target_nodes) if n not in batch_nodes]

                # Distribute emulated nodes across assistant nodes (round-robin)
                emulated_nodes_per_assistant: list[list[int]] = [[] for _ in range(num_assistant_nodes)]
                for j, node_id in enumerate(emulated_nodes):
                    assistant_idx = j % num_assistant_nodes
                    emulated_nodes_per_assistant[assistant_idx].append(node_id)

                _run_emulate(
                    args,
                    server,
                    batch_nodes,
                    epoch_log_dir_name,
                    str(runtime_profiler_dir.absolute()),
                    additional_training_envs=additional_training_envs,
                    sandbox_target_nodes=batch_sandbox_target_nodes,
                    assistant_target_nodes=assistant_target_nodes,
                    emulated_nodes_per_assistant=emulated_nodes_per_assistant,
                )

        logger.info("Planning mode completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt, stopping all training...")
        server.stop_train_on_nodes(sandbox_target_nodes)
        for assistant_target_node in assistant_target_nodes:
            server.stop_assistants(assistant_target_node)
        return 1
    except Exception as e:
        logger.exception("Error in planning mode: %s", e)
        server.stop_train_on_nodes(sandbox_target_nodes)
        for assistant_target_node in assistant_target_nodes:
            server.stop_assistants(assistant_target_node)
        return 1
    finally:
        # Shutdown all clients and stop server
        if args.shutdown_client:
            server.shutdown_client()
        server.stop_server()


def main() -> int:
    r"""Main entry point for crystalllm-run command.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Validate arguments
    validate_args(args)

    # Dispatch to appropriate mode handler
    if args.mode == "baseline":
        return baseline_mode(args)
    if args.mode == "capture":
        return capture_mode(args)
    if args.mode == "emulate":
        return emulate_mode(args)
    if args.mode == "planning":
        return planning_mode(args)
    logger.error("Unknown mode: %s", args.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
