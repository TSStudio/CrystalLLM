# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Train Runner.

This module provides the TrainRunner class for executing training scripts
on nodes using torchrun.
"""

from __future__ import annotations

import logging
import pathlib
import subprocess

from ..nccl.util import get_base_lib_path
from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class TrainRunner(BaseRunner):
    """Runner for executing training scripts on a node.

    This runner manages the execution of training scripts using bash.
    Each node runs a single training script that handles all ranks (GPUs)
    assigned to that node.
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        node_rank: int,
        crystal_workspace: str,
        log_dir_name: str,
        train_script_path: str,
        extra_args_str: str,
        extra_envs_str: str,
    ) -> None:
        """Initialize the TrainRunner."""
        super().__init__()

        log_dir = f"{crystal_workspace}/{log_dir_name}"
        node_log_dir = f"{log_dir}/node"
        pathlib.Path(node_log_dir).mkdir(parents=True, exist_ok=True)

        self.cmd = (
            f"CRYSTAL_WORKSPACE_DIR={crystal_workspace} "
            f"CRYSTAL_LOG_DIR={log_dir} "
            f"MASTER_ADDR={master_addr} "
            f"MASTER_PORT={master_port} "
            f"NODE_RANK={node_rank} "
            f"LD_PRELOAD={get_base_lib_path()} "
            f"{extra_envs_str} "
            f"bash {train_script_path} "
            f"{extra_args_str} "
            f"> {node_log_dir}/node_{node_rank}.log 2>&1"
        )

        logger.info("TrainRunner initialized for node %s", node_rank)
        logger.info("Command: %s", self.cmd)

    def execute(self) -> None:
        """Execute the training script on this node."""
        process = subprocess.Popen(self.cmd, shell=True)

        self._add_process(process)
        logger.info("Training started on with PID %s", process.pid)

    def kill(self) -> None:
        """Force kill all training processes."""
        self._kill_pretrain()
        super().kill()

    @staticmethod
    def _kill_pretrain() -> None:
        """Kill pretrain processes."""
        logger.info("Killing pretrain processes...")
        try:
            result = subprocess.run(
                ["pkill", "-9", "-f", "python -u pretrain"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info("Pretrain processes killed successfully")
            else:
                logger.info("No pretrain processes found or already stopped")
        except Exception as e:
            logger.error("Error killing pretrain processes: %s", e)
