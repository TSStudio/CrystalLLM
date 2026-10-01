# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Assistant Runner.

This module provides the AssistantRunner class for managing assistant
processes that emulate ranks during distributed training replay.
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess

import redis

from ..nccl import get_fake_lib_path, get_mock_lib_path
from ..nccl.mock_bootstrap import get_mock_node_hostid
from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class AssistantRunner(BaseRunner):
    """Runner for managing assistant processes on a node.

    This runner manages the execution of assistant processes that emulate
    ranks during distributed training replay. It handles the setup of
    NCCL mock bootstrap and launching of assistant processes.
    """

    def __init__(
        self,
        node_id: int,
        master_addr: str,
        master_port: int,
        redis_host: str,
        crystal_workspace: str,
        log_dir_name: str,
        emulated_nodes_for_this_assistant: list[int],
        nnodes: int,
        sandbox_nodes: list[int],
        assistant_envs_str: str,
        trace_dir: str,
        graph_dir: str | None = None,
        graph_cache_path: str | None = None,
        profiler_dir: str | None = None,
        pp_rank_stride: int | None = None,
        enable_assistant_online_update: bool = False,
        tensor_provider: str | None = None,
        enable_mock_nccl_bootstrap: bool = False,
        assistant_nsight_system_ranks: list[int] | None = None,
        gpus_per_node: int = 8,
    ) -> None:
        """Initialize the AssistantRunner.

        Args:
            node_id: ID of the physical node this runner is running on.
            master_addr: Master address for distributed training.
            master_port: Master port for distributed training.
            redis_host: Redis host address for coordination.
            crystal_workspace: Path to the crystal workspace directory.
            log_dir_name: Name of the log directory.
            emulated_nodes_for_this_assistant: Logical node IDs that this assistant should emulate.
            nnodes: Total number of nodes in the target cluster.
            sandbox_nodes: List of node IDs running in sandbox mode.
            assistant_envs_str: Environment variables string for assistant processes.
            trace_dir: Path to trace directory.
            graph_dir: Path to graph directory (optional).
            graph_cache_path: Path to a prebuilt launch-timing cache
                (graph_cache_{timestamp}.json.gz), replacing graph_dir (optional).
            profiler_dir: Path to the Crystal runtime profiler directory (optional).
            pp_rank_stride: Rank distance between adjacent PP stages (optional).
            enable_assistant_online_update: If True, apply online duration updates.
            tensor_provider: Path to tensor provider module (optional).
            enable_mock_nccl_bootstrap: If True, use mock NCCL bootstrap instead of fake NCCL.
            assistant_nsight_system_ranks: Ranks to enable Nsight Systems profiling (optional).
            gpus_per_node: Number of GPUs per node.
        """
        super().__init__()

        self.sandbox_nodes = sandbox_nodes
        self.node_id = node_id

        # This assistant emulates a precomputed subset of non-sandbox nodes.
        self.emulated_nodes = emulated_nodes_for_this_assistant

        self.gpus_per_node = gpus_per_node
        self.world_size = nnodes * gpus_per_node
        self.trace_dir = trace_dir
        self.sandbox_ranks = []
        for i in self.sandbox_nodes:
            self.sandbox_ranks.extend(list(range(i * gpus_per_node, (i + 1) * gpus_per_node)))

        # Ranks that THIS assistant will emulate
        self.emulated_ranks = []
        for i in self.emulated_nodes:
            self.emulated_ranks.extend(list(range(i * gpus_per_node, (i + 1) * gpus_per_node)))

        self.assistant_log_dir = (
            f"{crystal_workspace}/{log_dir_name}/assistant_for_{'_'.join(str(n) for n in sandbox_nodes)}"
        )
        pathlib.Path(self.assistant_log_dir).mkdir(parents=True, exist_ok=True)

        self.assistant_envs_str = (
            f"CRYSTAL_WORKSPACE_DIR={crystal_workspace} "
            f"CRYSTAL_LOG_DIR={crystal_workspace}/{log_dir_name} "
            f"CRYSTAL_LOG_DIR_ASSISTANT={self.assistant_log_dir} "
            f"CRYSTAL_REDIS_HOST={redis_host} "
            f"MASTER_ADDR={master_addr} "
            f"MASTER_PORT={master_port} "
            f"{assistant_envs_str}"
        )
        if enable_mock_nccl_bootstrap:
            self.assistant_envs_str += f" LD_PRELOAD={get_mock_lib_path()}"
        else:
            self.assistant_envs_str += f" LD_PRELOAD={get_fake_lib_path()}"

        self.assistant_args_str = (
            f"--sandbox-ranks {json.dumps((self.sandbox_ranks), separators=(',', ':'))} --trace-dir {trace_dir}"
        )
        if graph_dir is not None:
            self.assistant_args_str += f" --graph-dir {graph_dir}"
        if graph_cache_path is not None:
            self.assistant_args_str += f" --graph-cache-path {graph_cache_path}"
        if profiler_dir is not None:
            self.assistant_args_str += f" --profiler-dir {profiler_dir}"
        if pp_rank_stride is not None:
            self.assistant_args_str += f" --pp-rank-stride {pp_rank_stride}"
        if enable_assistant_online_update:
            self.assistant_args_str += " --enable-online-update"

        self.tensor_provider = tensor_provider

        self.assistant_nsight_system_ranks = (
            assistant_nsight_system_ranks if assistant_nsight_system_ranks is not None else []
        )
        # Log sandbox and emulated nodes and ranks
        logger.info(
            "AssistantRunner initialized: sandbox_nodes=%s, this_assistant_emulated_nodes=%s",
            self.sandbox_nodes,
            self.emulated_nodes,
        )
        logger.info(
            "AssistantRunner ranks: sandbox_ranks=%s, this_assistant_emulated_ranks=%s",
            self.sandbox_ranks,
            self.emulated_ranks,
        )

    def execute(self) -> None:
        """Launch MemoryManager processes and assistant processes on this node."""
        self._start_mps()
        self._clear_memory_manager_redis_db()

        # Launch one MemoryManager process for each local GPU that owns emulated ranks.
        ranks_by_local_rank: dict[int, list[int]] = {}
        for rank in self.emulated_ranks:
            ranks_by_local_rank.setdefault(rank % self.gpus_per_node, []).append(rank)

        for local_rank, managed_ranks in sorted(ranks_by_local_rank.items()):
            provider_arg = f" --tensor-provider {self.tensor_provider}" if self.tensor_provider is not None else ""
            cmd = (
                f"WORLD_SIZE={self.world_size} "
                f"LOCAL_WORLD_SIZE={self.gpus_per_node} "
                f"LOCAL_RANK={local_rank} "
                f"PYTORCH_CUDA_ALLOC_CONF='' "
                f"{self.assistant_envs_str} "
                f"python -u -m crystalllm.replay.memory_manager "
                f"--sandbox-ranks {json.dumps(self.sandbox_ranks, separators=(',', ':'))} "
                f"--managed-ranks {json.dumps(managed_ranks, separators=(',', ':'))} "
                f"--trace-dir {self.trace_dir}"
                f"{provider_arg} "
                f"> {self.assistant_log_dir}/memory_manager_node_{self.node_id}_local_rank_{local_rank}.log 2>&1"
            )

            logger.info("Launching MemoryManager for local_rank %s ranks %s: %s", local_rank, managed_ranks, cmd)
            process = subprocess.Popen(cmd, shell=True)
            self._add_process(process)
            logger.info("MemoryManager process started for local_rank %s with PID %s", local_rank, process.pid)

        # Use configured assistant_nsight_system_ranks for Nsight Systems profiling
        nsys_ranks = self.assistant_nsight_system_ranks

        # Launch assistant processes for each rank
        for i in self.emulated_ranks:
            nsys_command = (
                "nsys profile "
                f"-o {self.assistant_log_dir}/nsys/rank_{i} "
                "--force-overwrite=true "
                "--capture-range=cudaProfilerApi "
                "--capture-range-end=stop "
                "--trace=cuda,nvtx "
                "--cuda-event-trace=false "
                "--sample none "
            )

            cmd = (
                f"WORLD_SIZE={self.world_size} "
                f"LOCAL_WORLD_SIZE={self.gpus_per_node} "
                f"RANK={i} "
                f"LOCAL_RANK={i % self.gpus_per_node} "
                f"NCCL_HOSTID={get_mock_node_hostid(i // self.gpus_per_node)} "
                # Override expandable_segments (set by default_envs.json for sandbox) because assistant CUDA IPC
                # requires non-expandable allocator.
                f"PYTORCH_CUDA_ALLOC_CONF='' "
                f"{self.assistant_envs_str} "
                f"{nsys_command if i in nsys_ranks else ''}"
                f"python -u -m crystalllm.replay.assistant {self.assistant_args_str} "
                f"> {self.assistant_log_dir}/assistant_rank_{i}.log 2>&1"
            )

            logger.info("Launching assistant process for rank %s: %s", i, cmd)

            # Start the subprocess
            process = subprocess.Popen(cmd, shell=True)

            self._add_process(process)
            logger.info("Assistant process started for rank %s with PID %s", i, process.pid)

        logger.info("All assistant processes launched")

    def kill(self) -> None:
        """Force kill all assistant processes and stop MPS server."""
        self._kill_assistant()
        self._stop_mps()
        super().kill()

    @staticmethod
    def _start_mps() -> None:
        """Start the MPS (Multi-Process Service) server."""
        logger.info("Starting MPS server...")
        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info("MPS server started successfully")
            else:
                logger.error("Error starting MPS server: %s", result.stderr)
        except Exception as e:
            raise RuntimeError(f"Error starting MPS server: {e}") from e

    @staticmethod
    def _kill_assistant() -> None:
        """Kill assistant and memory-manager processes."""
        logger.info("Killing assistant processes...")
        # memory_manager is spawned through a shell and reparented to init once
        # the shell exits, so it must be matched and killed explicitly.
        for pattern in ("crystalllm.replay.assistant", "crystalllm.replay.memory_manager"):
            try:
                result = subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    logger.info("Processes matching %s killed successfully", pattern)
                else:
                    logger.info("No processes matching %s found or already stopped", pattern)
            except Exception as e:
                logger.error("Error killing processes matching %s: %s", pattern, e)

    @staticmethod
    def _stop_mps() -> None:
        """Stop the MPS (Multi-Process Service) server."""
        logger.info("Stopping MPS server...")
        try:
            result = subprocess.run(
                ["pkill", "-9", "-f", "nvidia-cuda-mps"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info("MPS server stopped successfully")
            else:
                logger.info("No MPS server found or already stopped")
        except Exception as e:
            logger.error("Error stopping MPS server: %s", e)

    @staticmethod
    def _clear_memory_manager_redis_db() -> None:
        """Clear local Redis DB 2 before starting per-GPU MemoryManager processes."""
        logger.info("Clearing local Redis DB 2 for MemoryManager")
        redis.Redis(host="localhost", port=6379, db=2).flushdb()
