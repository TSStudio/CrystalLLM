# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Planning Graph Builder Examples.

This module provides some examples of building execution graphs for Megatron 1F1B
pipeline parallelism scheduling policies.
"""

from .megatron_1f1b import build_megatron_1f1b_graph, get_megatron_1f1b_channel_members
from .megatron_1f1b_interleaved import (
    build_megatron_1f1b_interleaved_graph,
    get_megatron_1f1b_interleaved_channel_members,
)

__all__ = [
    "build_megatron_1f1b_graph",
    "build_megatron_1f1b_interleaved_graph",
    "get_megatron_1f1b_channel_members",
    "get_megatron_1f1b_interleaved_channel_members",
]
