# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Rank layout and data parallel communication shared by the pipeline policy builders.

Ranks are laid out with PP outermost and DP innermost, i.e. rank = stage * dp_size +
dp_index, which is the layout described by a DP rank stride of 1 and a PP rank stride of
dp_size. Ranks sharing a dp_index therefore form one PP chain, and ranks sharing a stage
form the DP group of that stage.
"""

from __future__ import annotations

# Communication span type fields of the two collectives closing an iteration.
DP_ALL_REDUCE = "DPAR"  # Gradient all-reduce over the DP group of a stage
GLOBAL_ALL_REDUCE = "GAR"  # All-reduce over the whole world

DP_CHANNEL_BASE = "DP"
GLOBAL_CHANNEL = "GLOBAL"


def rank_of(stage: int, dp_index: int, data_parallel_size: int) -> int:
    """Return the global rank of a (stage, dp_index) position."""
    return stage * data_parallel_size + dp_index


def dp_channel_of(stage: int) -> str:
    """Return the DP group channel name of a PP stage."""
    return f"{DP_CHANNEL_BASE}_{stage}"


def dp_and_global_channel_members(pipeline_parallel_size: int, data_parallel_size: int) -> dict[str, list[int]]:
    """Return the member tables of every DP group channel plus the global channel."""
    members = {
        dp_channel_of(stage): [rank_of(stage, dp_index, data_parallel_size) for dp_index in range(data_parallel_size)]
        for stage in range(pipeline_parallel_size)
    }
    members[GLOBAL_CHANNEL] = list(range(pipeline_parallel_size * data_parallel_size))
    return members
