# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Entry Point for Crystal Coordinator Client Management.

This module provides the command-line interface for managing CrystalLLM
Coordinator Client processes, including starting and stopping the client
in daemon or foreground mode.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from ..coordinator import CrystalClient

logger = logging.getLogger(__name__)

# Log file name template for daemon mode: {rank}.log
DAEMON_LOG_FILENAME = "{rank}.log"

# Subdirectory for client logs: workspace/client/{rank}.log
CLIENT_LOG_SUBDIR = "client"

# Subdirectory for PID files: workspace/client/pid/{rank}.pid
PID_SUBDIR = "pid"

# PID file name template: {rank}.pid
PID_FILENAME = "{rank}.pid"


def _parse_args() -> argparse.Namespace:
    r"""Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="crystalllm client",
        description="CrystalLLM Coordinator Client Management",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the CrystalLLM Coordinator Client",
    )
    start_parser.add_argument(
        "--crystal-workspace",
        type=str,
        default=".",
        help="Workspace directory for log files (default: current directory)",
    )
    start_parser.add_argument(
        "--server-addr",
        type=str,
        required=True,
        help="Server host address",
    )
    start_parser.add_argument(
        "--server-port",
        type=int,
        required=True,
        help="Server port",
    )
    start_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon process with output redirected to log file",
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the CrystalLLM Coordinator Client daemon",
    )
    stop_parser.add_argument(
        "--crystal-workspace",
        type=str,
        default=".",
        help="Workspace directory where the PID file is stored (default: current directory)",
    )

    return parser.parse_args()


def _daemonize(log_file: Path, working_dir: Path) -> None:
    r"""Daemonize the current process.

    This function performs the standard double-fork daemonization process:
    1. First fork to detach from parent process
    2. Create new session and set working directory to workspace
    3. Second fork to prevent reacquiring terminal
    4. Redirect stdout/stderr to the log file

    Args:
        log_file: Path to the log file for stdout/stderr redirection.
        working_dir: Working directory to switch to after daemonization.

    Raises:
        OSError: If fork fails.
    """
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        raise OSError(f"Fork #1 failed: {e}") from e

    # Decouple from parent environment
    os.chdir(working_dir)
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        raise OSError(f"Fork #2 failed: {e}") from e

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    with log_file.open("w+", encoding="utf-8") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    # Ignore SIGCHLD to prevent zombie processes
    # Comment out to prevent affact return code of subprocess
    # signal.signal(signal.SIGCHLD, signal.SIG_IGN)


def _get_rank_from_env() -> int:
    r"""Get client rank from RANK environment variable.

    Returns:
        int: Client rank identifier.

    Raises:
        ValueError: If RANK environment variable is not set or invalid.
    """
    rank_env = os.getenv("RANK")
    if rank_env is None:
        raise ValueError("RANK environment variable is not set.")
    try:
        return int(rank_env)
    except ValueError as e:
        raise ValueError(f"Invalid rank value in $RANK: {rank_env}") from e


def _validate_workspace(workspace_dir: str) -> Path:
    r"""Validate workspace directory exists and is a directory.

    Args:
        workspace_dir: Path to workspace directory.

    Returns:
        Validated workspace directory Path.

    Raises:
        ValueError: If workspace directory does not exist or is not a valid directory.
    """
    path = Path(workspace_dir).resolve()
    if not path.exists():
        raise ValueError(f"Workspace directory '{workspace_dir}' does not exist.")
    if not path.is_dir():
        raise ValueError(f"Workspace path '{workspace_dir}' is not a directory.")
    return path


def _setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    r"""Setup logging configuration.

    Args:
        log_file: Path to log file for file handler, or None for console only.
        level: Logging level.
    """
    handlers: list[logging.Handler] = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _handle_start_command(args: argparse.Namespace) -> None:
    r"""Handle the start command.

    Starts the CrystalClient in either daemon or foreground mode.
    In daemon mode, stdout/stderr are redirected to workspace/client/{rank}.log
    and the PID is written to workspace/client/pid/{rank}.pid.

    Args:
        args: Parsed command line arguments containing:
            - server_addr: Server host address
            - server_port: Server port number
            - crystal_workspace: Workspace directory path
            - daemon: Whether to run as daemon

    Raises:
        ValueError: If RANK environment variable is invalid or workspace is invalid.
        OSError: If daemonization fails.
    """
    # Get and validate parameters
    rank = _get_rank_from_env()
    workspace_dir = _validate_workspace(args.crystal_workspace)

    if args.daemon:
        # Get log file path
        log_dir = workspace_dir / CLIENT_LOG_SUBDIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / DAEMON_LOG_FILENAME.format(rank=rank)

        # Daemonize
        _daemonize(log_file, workspace_dir)

        # Setup logging to file
        _setup_logging(log_file=log_file)

        # Get and write PID file
        pid_dir = workspace_dir / CLIENT_LOG_SUBDIR / PID_SUBDIR
        pid_dir.mkdir(parents=True, exist_ok=True)
        pid_file = pid_dir / PID_FILENAME.format(rank=rank)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")

        logger.info("Daemon started with PID %s, logging to %s", os.getpid(), log_file)
    else:
        # Setup logging to console
        _setup_logging()

    # Print environment variables
    logger.info(os.environ)

    # Run the client
    try:
        client = CrystalClient(
            host=args.server_addr,
            port=args.server_port,
            rank=rank,
            crystal_workspace=str(workspace_dir),
        )
        client.start()
    except Exception as e:
        logger.error("Failed to start client: %s", e)
        raise


def _handle_stop_command(args: argparse.Namespace) -> None:
    r"""Handle the stop command.

    Stops the daemon process by reading the PID from workspace/client/pid/{rank}.pid
    and sending SIGTERM to that process.

    Args:
        args: Parsed command line arguments containing:
            - crystal_workspace: Workspace directory path where PID file is stored

    Raises:
        ValueError: If RANK environment variable is invalid, workspace is invalid,
            PID file is not found, or daemon process cannot be stopped.
    """
    workspace_dir = _validate_workspace(args.crystal_workspace)
    rank = _get_rank_from_env()

    # Setup logging
    _setup_logging()

    # Get PID file path
    pid_file = workspace_dir / CLIENT_LOG_SUBDIR / PID_SUBDIR / PID_FILENAME.format(rank=rank)

    if not pid_file.exists():
        raise ValueError(f"PID file not found: {pid_file}")

    # Read and remove PID file
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        pid_file.unlink(missing_ok=True)
    except (ValueError, OSError) as e:
        pid_file.unlink(missing_ok=True)
        raise ValueError(f"Failed to read PID from file: {e}") from e

    # Send SIGTERM to the daemon process
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to daemon process %s", pid)
    except ProcessLookupError as e:
        raise ValueError(f"Daemon process not found: {pid}") from e
    except OSError as e:
        raise ValueError(f"Failed to stop daemon: {e}") from e


def main() -> None:
    r"""Main entry point for crystalllm client command."""
    args = _parse_args()

    if args.command is None:
        print("Error: No command specified. Use 'start' or 'stop'.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "start":
            _handle_start_command(args)
        elif args.command == "stop":
            _handle_stop_command(args)
        else:
            raise ValueError(f"Unknown command '{args.command}'")
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
