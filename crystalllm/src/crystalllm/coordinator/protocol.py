# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Coordinator Protocol Definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class MessageType(str, Enum):
    """Message types for Coordinator communication."""

    CLIENT_HELLO = "CLIENT_HELLO"
    CLIENT_REPLY = "CLIENT_REPLY"
    SHUTDOWN_CLIENT = "SHUTDOWN_CLIENT"
    CLIENT_PRINT = "CLIENT_PRINT"

    # Training related messages
    TRAIN_START = "TRAIN_START"
    TRAIN_WAIT = "TRAIN_WAIT"
    TRAIN_STOP = "TRAIN_STOP"

    # Assistant related messages
    ASSISTANT_START = "ASSISTANT_START"
    ASSISTANT_STOP = "ASSISTANT_STOP"

    # Mock NCCL bootstrap related messages
    MOCK_BOOTSTRAP_PREPARE = "MOCK_BOOTSTRAP_PREPARE"


# POSIX exit codes are masked to 0..255 (WEXITSTATUS) and Python's subprocess
# reports a process killed by signal N as -N; Linux signal numbers go up to
# 64, so genuine subprocess outcomes live in 0..255 and -1..-64. Reply codes
# other than SUCCESS stay out of these reserved ranges.
MAX_EXIT_CODE = 255
MIN_SIGNAL_CODE = -64


class ReplyCode(IntEnum):
    """Reply codes carried by CLIENT_REPLY messages.

    The code space is shared with the subprocess exit codes that TRAIN_WAIT
    reports verbatim: 0..255 are exit codes of the training command (0 also
    marks success here) and -1..-64 are the signal codes Python's subprocess
    reports when a process is killed (returncode == -signum; Linux signal
    numbers go up to 64). Protocol-level codes stay out of those ranges and
    are grouped by origin: -1xx codes are reported by the client, -2xx codes
    by the server.
    """

    # Universal success; also the exit code of a clean subprocess exit.
    SUCCESS = 0
    # The client could not handle the request (a handler raised an exception
    # or the training process could not be waited on).
    INTERNAL_ERROR = -100
    # The node already runs another runner and refuses to start a second one.
    RUNNER_QUOTA_IN_USE = -101
    # The request needs an active runner, but none is active on the node.
    NO_ACTIVE_RUNNER = -102
    # The client went offline before replying (e.g., its machine was
    # preempted); reported by the server, which detects the closed connection.
    CLIENT_DISCONNECTED = -200
    # A controlled peer node went offline (e.g., its machine was preempted):
    # the run cannot complete without it, so the server abandons this rank's
    # pending reply instead of waiting for its doomed training to fail.
    PEER_DISCONNECTED = -201


def describe_reply_code(code: int) -> str:
    """Describe a reply code for logs and error messages.

    Protocol-level codes are described by their meaning, while raw subprocess
    exit codes (reported verbatim by TRAIN_WAIT) are described as exit codes
    or signal kills.

    Args:
        code: A reply code from a CLIENT_REPLY message.

    Returns:
        A short human-readable description of the code.
    """
    if code == ReplyCode.SUCCESS:
        return "success"
    if 0 < code <= MAX_EXIT_CODE:
        return f"process exited with code {code}"
    if MIN_SIGNAL_CODE <= code < 0:
        return f"process killed by signal {-code}"
    if code == ReplyCode.INTERNAL_ERROR:
        return "client internal error"
    if code == ReplyCode.RUNNER_QUOTA_IN_USE:
        return "runner quota already in use on the node"
    if code == ReplyCode.NO_ACTIVE_RUNNER:
        return "no active runner on the node"
    if code == ReplyCode.CLIENT_DISCONNECTED:
        return "client disconnected before replying (node offline or preempted)"
    if code == ReplyCode.PEER_DISCONNECTED:
        return "abandoned because a peer node went offline"
    return f"reply code {code}"


@dataclass
class Message:
    """Base message structure for Coordinator communication.

    Attributes:
        type: The message type
        data: The message payload
    """

    type: MessageType
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert message to JSON string.

        Returns:
            JSON string representation of the message
        """
        return json.dumps({"type": self.type.value, "data": self.data})

    @classmethod
    def from_json(cls, json_str: str) -> Message:
        """Parse message from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            Parsed Message object

        Raises:
            json.JSONDecodeError: If the string is not valid JSON
            KeyError: If required fields are missing
        """
        parsed = json.loads(json_str)
        return cls(type=MessageType(parsed["type"]), data=parsed.get("data", {}))


# Helper functions for creating specific message types


def create_client_hello(rank: int) -> Message:
    """Create a CLIENT_HELLO message.

    Args:
        rank: The client rank identifier

    Returns:
        CLIENT_HELLO message
    """
    return Message(type=MessageType.CLIENT_HELLO, data={"rank": rank})


def create_client_reply(
    code: int,
    rank: int,
    handle: str,
    msg: str = "",
    payload: dict[str, Any] | None = None,
) -> Message:
    """Create a CLIENT_REPLY message.

    Args:
        code: A ReplyCode member, or the subprocess exit code that TRAIN_WAIT
            reports verbatim.
        rank: The rank of the replying client.
        handle: The handle of the message being replied to.
        msg: Error message when code is non-zero (default: empty string).
        payload: Optional structured response payload.
    """
    return Message(
        type=MessageType.CLIENT_REPLY,
        data={"code": code, "_rank": rank, "_handle": handle, "msg": msg, "payload": payload},
    )


def create_shutdown_client() -> Message:
    """Create a SHUTDOWN_CLIENT message."""
    return Message(type=MessageType.SHUTDOWN_CLIENT, data={})


def create_client_print(message_str: str, handle: str) -> Message:
    """Create a CLIENT_PRINT message."""
    return Message(type=MessageType.CLIENT_PRINT, data={"str": message_str, "_handle": handle})


def create_train_start(
    master_addr: str,
    master_port: int,
    node_rank: int,
    log_dir_name: str,
    train_script_path: str,
    extra_args_str: str,
    extra_envs_str: str,
    handle: str,
) -> Message:
    """Create a TRAIN_START message."""
    return Message(
        type=MessageType.TRAIN_START,
        data={
            "master_addr": master_addr,
            "master_port": master_port,
            "node_rank": node_rank,
            "log_dir_name": log_dir_name,
            "train_script_path": train_script_path,
            "extra_args_str": extra_args_str,
            "extra_envs_str": extra_envs_str,
            "_handle": handle,
        },
    )


def create_train_wait(handle: str) -> Message:
    """Create a TRAIN_WAIT message."""
    return Message(type=MessageType.TRAIN_WAIT, data={"_handle": handle})


def create_train_stop() -> Message:
    """Create a TRAIN_STOP message."""
    return Message(type=MessageType.TRAIN_STOP, data={})


def create_assistant_start(
    master_addr: str,
    master_port: int,
    redis_host: str,
    log_dir_name: str,
    emulated_nodes_for_this_assistant: list[int],
    nnodes: int,
    sandbox_nodes: list[int],
    assistant_envs_str: str,
    trace_dir: str,
    graph_dir: str | None,
    graph_cache_path: str | None,
    profiler_dir: str | None,
    pp_rank_stride: int | None,
    enable_assistant_online_update: bool,
    tensor_provider: str | None,
    enable_mock_nccl_bootstrap: bool,
    assistant_nsight_system_ranks: list[int] | None,
    gpus_per_node: int,
    handle: str,
) -> Message:
    """Create an ASSISTANT_START message."""
    return Message(
        type=MessageType.ASSISTANT_START,
        data={
            "master_addr": master_addr,
            "master_port": master_port,
            "redis_host": redis_host,
            "log_dir_name": log_dir_name,
            "emulated_nodes_for_this_assistant": emulated_nodes_for_this_assistant,
            "nnodes": nnodes,
            "sandbox_nodes": sandbox_nodes,
            "assistant_envs_str": assistant_envs_str,
            "trace_dir": trace_dir,
            "graph_dir": graph_dir,
            "graph_cache_path": graph_cache_path,
            "profiler_dir": profiler_dir,
            "pp_rank_stride": pp_rank_stride,
            "enable_assistant_online_update": enable_assistant_online_update,
            "tensor_provider": tensor_provider,
            "enable_mock_nccl_bootstrap": enable_mock_nccl_bootstrap,
            "assistant_nsight_system_ranks": assistant_nsight_system_ranks,
            "gpus_per_node": gpus_per_node,
            "_handle": handle,
        },
    )


def create_mock_bootstrap_prepare(
    redis_host: str,
    trace_dir: str,
    nnodes: int,
    sandbox_nodes: list[int],
    gpus_per_node: int,
    handle: str,
) -> Message:
    """Create a MOCK_BOOTSTRAP_PREPARE message."""
    return Message(
        type=MessageType.MOCK_BOOTSTRAP_PREPARE,
        data={
            "redis_host": redis_host,
            "trace_dir": trace_dir,
            "nnodes": nnodes,
            "sandbox_nodes": sandbox_nodes,
            "gpus_per_node": gpus_per_node,
            "_handle": handle,
        },
    )


def create_assistant_stop() -> Message:
    """Create an ASSISTANT_STOP message."""
    return Message(type=MessageType.ASSISTANT_STOP, data={})


def encode_message(message: Message) -> bytes:
    """Encode a message to bytes with newline delimiter.

    Args:
        message: The message to encode

    Returns:
        Encoded bytes with newline suffix
    """
    return (message.to_json() + "\n").encode("utf-8")


def decode_message(data: bytes | str) -> Message:
    """Decode bytes to a Message.

    Args:
        data: The data to decode (bytes or string)

    Returns:
        Decoded Message object
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return Message.from_json(data.strip())
