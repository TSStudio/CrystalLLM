# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Base Runner.

This module provides the abstract base class for all runners.
A node can only hold one instance of a BaseRunner subclass at a time.

Resource Quota Management:
    - Each node can only run one BaseRunner instance at a time
    - TrainRunner and AssistantRunner share the same quota pool
    - Quota is acquired when a runner instance is created
    - Quota is released only when kill() is called
    - Attempting to create a new runner while another is active raises RunOutofRunner
"""

from __future__ import annotations

import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from typing_extensions import Self

logger = logging.getLogger(__name__)


class RunOutRunnerError(Exception):
    """Exception raised when runner quota is exhausted.

    This exception is raised when attempting to create a new BaseRunner instance
    while another instance is already active on the node. Each node can only
    run one BaseRunner instance (TrainRunner or AssistantRunner) at a time.

    Attributes:
        message: Explanation of the error.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception.

        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


class RunnerManager:
    """Global manager for BaseRunner instances.

    This class implements a singleton pattern to manage runner quotas across
    the entire node. It ensures that only one BaseRunner instance can be
    active at any given time.

    The manager is thread-safe and can be used across multiple threads.
    """

    _instance: RunnerManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> Self:
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.quota_lock = threading.Lock()
                    cls._instance.active_runner: BaseRunner | None = None

        return cls._instance

    def acquire_quota(self, runner: BaseRunner) -> bool:
        """Acquire quota for a runner instance.

        Args:
            runner: The runner instance requesting quota.

        Returns:
            True if quota was acquired, False if another runner is active.
        """
        with self.quota_lock:
            if self.active_runner is not None and self.active_runner is not runner:
                return False
            self.active_runner = runner
            logger.info("Runner quota acquired by %s", type(runner).__name__)
            return True

    def release_quota(self, runner: BaseRunner) -> bool:
        """Release quota held by a runner instance.

        Args:
            runner: The runner instance releasing quota.

        Returns:
            True if quota was released, False if this runner doesn't hold the quota.
        """
        with self.quota_lock:
            if self.active_runner is runner:
                self.active_runner = None
                logger.info("Runner quota released by %s", type(runner).__name__)
                return True
            return False

    def has_quota(self, runner: BaseRunner) -> bool:
        """Check if the given runner holds the quota.

        Args:
            runner: The runner instance to check.

        Returns:
            True if this runner holds the quota, False otherwise.
        """
        with self.quota_lock:
            return self.active_runner is runner

    def get_active_runner(self) -> BaseRunner | None:
        """Get the currently active runner instance.

        Returns:
            The active runner instance, or None if no runner is active.
        """
        with self.quota_lock:
            return self.active_runner


class BaseRunner(ABC):
    """Abstract base class for all runners.

    A node can only hold one instance of a BaseRunner subclass at a time.
    This ensures proper resource management and prevents conflicts.

    Resource Quota Management:
        - Each node can only run one BaseRunner instance at a time
        - TrainRunner and AssistantRunner share the same quota pool
        - Quota is checked and acquired during __init__
        - Quota is released only when kill() is called
        - Attempting to create a new runner while another is active raises RunOutofRunner

    Attributes:
        _processes: List of subprocesses managed by this runner.
        _quota_manager: The global quota manager instance.
    """

    def __init__(self) -> None:
        """Initialize the runner.

        Raises:
            RunOutRunnerError: If another runner instance is already active.
        """
        self._processes: list[subprocess.Popen[Any]] = []
        self._quota_manager = RunnerManager()

        # Check and acquire quota
        if not self._quota_manager.acquire_quota(self):
            active_runner = self._quota_manager.get_active_runner()
            active_type = type(active_runner).__name__ if active_runner else "Unknown"
            raise RunOutRunnerError(
                f"Cannot create new runner: {active_type} is already active. "
                "Only one BaseRunner instance can exist at a time. "
                "Call kill() on the existing runner before creating a new one."
            )
        logger.info("%s created and acquired runner quota", type(self).__name__)

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Execute the runner's main task.

        This method should start the subprocesses required for this runner.
        """

    def wait(self, callback: Callable[[int | None], None] | None = None) -> int | None:
        """Wait for all subprocesses to complete.

        This method blocks until all managed subprocesses have finished, then
        calls the optional callback with the return code. The callback runs
        even when there is no process to wait for, so a caller that reacts
        only to the callback never waits forever.

        Args:
            callback: Optional callback function to call with the return code.

        Returns:
            The subprocess return code (0 for success, non-zero for failure,
            negative when the process was killed by a signal), or None when
            there was no process to wait for or waiting failed. None keeps the
            signal code range free for genuine subprocess outcomes; the caller
            maps it to a protocol-level error code.
        """
        return_code: int | None = None

        if not self._processes:
            logger.warning("No processes to wait for")
        else:
            return_code = 0
            for process in self._processes:
                try:
                    process.wait()
                    if process.returncode != 0:
                        return_code = process.returncode
                        logger.error("Process exited with code %s", process.returncode)
                except Exception as e:
                    logger.error("Error waiting for process: %s", e)
                    return_code = None

        if callback:
            callback(return_code)

        return return_code

    def kill(self) -> None:
        """Force kill all managed subprocesses.

        This method sends SIGKILL (-9) to all subprocesses and releases
        the runner quota for this node.

        Note:
            This is the only method that releases the runner quota.
            The quota will be available for new runner instances after this call.
        """
        logger.info("Killing all runner processes...")
        for process in self._processes:
            try:
                if process.poll() is None:  # Process is still running
                    process.kill()
                    logger.info("Killed process with PID %s", process.pid)
            except Exception as e:
                logger.error("Error killing process: %s", e)

        self._processes.clear()

        # Release the runner quota
        self._quota_manager.release_quota(self)
        logger.info("All runner processes killed and quota released")

    def _add_process(self, process: subprocess.Popen[Any]) -> None:
        """Add a subprocess to the managed processes list.

        Args:
            process: The subprocess to add.
        """
        self._processes.append(process)
