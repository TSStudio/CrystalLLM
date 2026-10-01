# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Replay Dataclass Definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Any

import torch


@dataclass(frozen=True)
class ReplayerCommGroupInfo:
    """Communication group information for replay."""

    ranks: list[int]
    backend: str | None
    group_name: str
    group_desc: str
    timeout: int | None


@dataclass(frozen=True)
class ReplayerCommOpInfo:
    """Communication operation information for replay."""

    trace_comm_index: int
    c10d_func_name: str
    signature: str
    group_name: str
    group_desc: str
    output_shape: list[int] | None
    output_dtype: torch.dtype | None
    input_shape: list[int] | None
    input_dtype: torch.dtype | None
    async_op: bool
    meta: dict[str, Any]


@dataclass(frozen=True)
class ReplayerCoalescingInfo:
    """Coalescing operation information for replay."""

    trace_comm_index: int
    group_name: str
    group_desc: str
    device_set: bool
    async_ops: bool


@dataclass(frozen=True)
class ReplayerScheduleInfo:
    """Schedule information for replay iteration."""

    iteration: int


class ReplayerReplayOpType(IntEnum):
    """Enumeration of replay operation types."""

    COMM = 0
    COALESCING_START = 1
    COALESCING_END = 2
    ITERATION_BEGIN = 3


@dataclass(frozen=True)
class ReplayerReplayOpInfo:
    """Replay operation information container."""

    type: ReplayerReplayOpType
    info: ReplayerCommOpInfo | ReplayerCoalescingInfo | ReplayerScheduleInfo | None


class TensorMemoryLevel(Enum):
    """Memory management level assigned to a tensor demand.

    See ASSISTANT_MEMORY_MANAGER.md for the full semantics of each level.
    """

    ANYTHING = auto()
    SHARED = auto()
    PERSISTENT = auto()
    INSTANT = auto()


class TensorProvider(ABC):
    """Abstract base class for MemoryManager-driven tensor value providers.

    A TensorProvider is invoked by the MemoryManager, and only for INPUT tensor demands.
    """

    @abstractmethod
    def can_handle(
        self,
        rank: int,
        replay_op: ReplayerCommOpInfo,
        comm_group: ReplayerCommGroupInfo,
    ) -> bool:
        """Check if this provider can handle the given INPUT tensor demand.

        Args:
            rank: The rank that owns this tensor demand.
            replay_op: The communication operation info for the demand.
            comm_group: The communication group the operation belongs to.

        Returns:
            True if this provider can handle the demand, False otherwise.
        """

    @abstractmethod
    def tensor_memory_level(
        self,
        rank: int,
        replay_op: ReplayerCommOpInfo,
        comm_group: ReplayerCommGroupInfo,
    ) -> TensorMemoryLevel:
        """Return the memory level to assign to this INPUT tensor demand."""

    @abstractmethod
    def provide(
        self,
        rank: int,
        replay_op: ReplayerCommOpInfo,
        comm_group: ReplayerCommGroupInfo,
        use_index: int = -1,
    ) -> torch.Tensor:
        """Provide the CPU tensor value for this demand.

        Args:
            rank: The rank that owns this tensor demand.
            replay_op: The communication operation info for the demand.
            comm_group: The communication group the operation belongs to.
            use_index: For INSTANT-level demands, the 0-based index of the use
                being served. Ignored (-1) for non-INSTANT demands.

        Returns:
            A CPU torch.Tensor with the appropriate shape and dtype.
        """
