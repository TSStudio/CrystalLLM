# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Replay PyTorch Distributed C10d Trace Parser."""

from __future__ import annotations

import ast
import gzip
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .data import (
    ReplayerCoalescingInfo,
    ReplayerCommGroupInfo,
    ReplayerCommOpInfo,
    ReplayerReplayOpInfo,
    ReplayerReplayOpType,
    ReplayerScheduleInfo,
)


def parse_group_kwargs(kwargs_str: str) -> tuple[list[int], str | None, str, str, int | None]:
    """Parse group kwargs (ranks, backend, group_desc, timeout) from trace string."""
    ranks_search = re.search(r"'ranks':\s*(\[[^\]]*\])", kwargs_str)
    # backend can be None (nccl for default) and "gloo"
    backend_search = re.search(r"'backend':\s*([^\,\}]+)", kwargs_str)
    group_name_search = re.search(r"'group_name':\s*([^\,\}]+)", kwargs_str)
    group_desc_search = re.search(r"'group_desc':\s*([^\,\}]+)", kwargs_str)
    # timeout can be datetime.timedelta object or None
    timeout_search = re.search(r"'timeout':\s*datetime\.timedelta\(seconds=(\d+)\)", kwargs_str)
    if ranks_search and backend_search and group_desc_search:
        ranks = ast.literal_eval(ranks_search.group(1))
        backend = ast.literal_eval(backend_search.group(1))
        group_name = ast.literal_eval(group_name_search.group(1))
        group_desc = ast.literal_eval(group_desc_search.group(1))
        timeout = ast.literal_eval(timeout_search.group(1)) if timeout_search is not None else None
        return ranks, backend, group_name, group_desc, timeout
    raise ValueError(f"Unexpected new_group trace kwargs{kwargs_str}. Please check the trace format.")


def parse_tensor_shape_dtype(info_str: str, order=-1) -> tuple[list[int], torch.dtype]:
    """Parse tensor shape and dtype from trace string.

    Args:
        info_str: String containing tensor info in format 'torch.Size([...])-torch.dtype(...)'
        order: Index to select from multiple tensors. -1 means last tensor (input tensor).
    """
    import torch

    # Parse tensor shape from a string using regex.
    # In the trace, tensors were recored in format tensor-torch.Size(...)-torch.dtype(...)
    # When order==-1, input tensor infomation we be returned.
    #   1. For collectives with two tensors in their signature (like all_gather_into_tensor)
    #      tensors are passed in args with order [output, input], the input tensor is the last one
    #   2. For other collectives, there is only one tensor, so taking the last one is also safe
    shape_matches = re.findall(r"torch\.Size\((\[[^\]]*\])\)", info_str)
    dtype_matches = re.findall(r"torch\.dtype\(torch\.([^\)]+)\)", info_str)
    if shape_matches:
        shape_str = shape_matches[order]
        # Use ast.literal_eval to safely parse the shape string into a list of numbers
        shape = ast.literal_eval(shape_str)
    else:
        raise ValueError(f"Error parsing shape from {info_str}, Please check the trace format.")
    if dtype_matches:
        dtype_str = dtype_matches[order]
        # Use getattr to get the torch.dtype object from the dtype string
        dtype = getattr(torch, dtype_str)
    else:
        raise ValueError(f"Error parsing dtype from {info_str}, Please check the trace format.")
    return shape, dtype


def parse_async_op(kwargs_str: str) -> bool:
    """Parse async_op flag from kwargs string. Returns True if async_op is True."""
    async_op_search = re.search(r"'async_op':\s*True", kwargs_str)
    return async_op_search is not None


def parse_p2p_peer(args_str: str, kwargs_str: str, peer_name: str) -> int:
    """Parse a direct P2P peer from current or archived c10d traces."""
    peer_search = re.search(rf"'{peer_name}':\s*([^,\}}]+)", kwargs_str)
    if peer_search is None:
        # Current c10d traces record ``tensor`` and ``src``/``dst`` as the
        # first two positional arguments while keeping ``group`` in kwargs.
        peer_search = re.search(r"^\[\s*'tensor-[^']*'\s*,\s*([^,\]]+)", args_str)
    if peer_search is not None:
        return ast.literal_eval(peer_search.group(1))
    raise ValueError(
        f"Error parsing {peer_name} from args={args_str}, kwargs={kwargs_str}. "
        "Please check the trace format."
    )


def parse_reduce_op(kwargs_str: str) -> torch.distributed.ReduceOp.RedOpType:
    """Parse reduce operation from kwargs string. Defaults to SUM if not found."""
    import torch

    reduce_op_search = re.search(r"'op':\s*<RedOpType.([A-Z]+): \d+>", kwargs_str)
    if reduce_op_search is not None:
        return getattr(torch.distributed.ReduceOp, reduce_op_search.group(1))
    return torch.distributed.ReduceOp.SUM


def resolve_channel_members(channel: str, new_group_table: dict[str, list[int]]) -> list[int]:
    """Resolve a graph channel name into its member global ranks.

    A SYMM channel is named after its process group, so its members are the group
    members. A P2P channel is named f"{group_name}_{i}:{j}", where i and j are the two
    participating group ranks in ascending order, so its members are those two ranks
    mapped back to global ranks.

    Args:
        channel: The channel name recorded in the graph.
        new_group_table: Mapping from process group name to member global ranks.

    Returns:
        The member global ranks, ordered by group rank.

    Raises:
        ValueError: If the channel cannot be resolved from the table.
    """
    symm_members = new_group_table.get(channel)
    if symm_members is not None:
        return list(symm_members)

    group_name, separator, peers = channel.rpartition("_")
    if separator:
        group_members = new_group_table.get(group_name)
        first, colon, second = peers.partition(":")
        if group_members is not None and colon and first.isdigit() and second.isdigit():
            first_group_rank, second_group_rank = int(first), int(second)
            if first_group_rank < len(group_members) and second_group_rank < len(group_members):
                return [group_members[first_group_rank], group_members[second_group_rank]]

    raise ValueError(
        f"Cannot resolve members of channel '{channel}': it matches neither a process group name "
        f"nor a '{{group_name}}_{{i}}:{{j}}' P2P channel of a known group"
    )


class CrystalTraceParser:
    """Parser for CrystalLLM PyTorch Distributed C10d trace.

    Parses gzipped trace files containing NEW_GROUP, SCHEDULE, COALESCING,
    and communication operation lines into structured data.
    """

    def __init__(self, trace_file: str, include_replay_op: bool = True):
        r"""Initialize the CrystalTraceParser.

        Args:
            trace_file: Path to the gzipped trace file.
            include_replay_op: Whether to parse replay operation lines (SCHEDULE, COALESCING, COMM).
                When False, only NEW_GROUP lines are parsed. Defaults to True.
        """
        self.comm_groups_ordered_list: list[ReplayerCommGroupInfo] = []
        self.replay_ops_ordered_list: list[ReplayerReplayOpInfo] = []
        # { iteration -> trace_comm_index }
        self.trace_comm_index_dict: dict[int, int] = {}
        # { group_name -> ReplayerCommGroupInfo }
        self.comm_groups_dict: dict[str, ReplayerCommGroupInfo] = {}
        # { group_name -> { trace_comm_index -> [ReplayerReplayOpInfo] } }
        self.replay_comm_ops_dict: dict[str, dict[int, list[ReplayerReplayOpInfo]]] = {}

        self.comm_groups_dict["0"] = None
        self.replay_comm_ops_dict["0"] = defaultdict(list)

        self.now_iteration = 0
        self.now_trace_comm_index = 0
        with gzip.open(trace_file, "rt") as fp:
            # Iterate line by line to avoid loading entire file into memory
            for line in fp:
                if line.startswith("NEW_GROUP"):
                    self._parse_new_group_line(line)
                elif include_replay_op:
                    if line.startswith("SCHEDULE"):
                        self._parse_schedule_line(line)
                    elif line.startswith("COALESCING"):
                        self._parse_coalescing_line(line)
                    elif line and line[0].isdigit():
                        self._parse_comm_line(line)
        # Record the boundary of the last iteration
        self.trace_comm_index_dict[self.now_iteration + 1] = self.now_trace_comm_index

    def _parse_new_group_line(self, line: str) -> None:
        """Parse NEW_GROUP line and update comm_groups."""
        tokens = line.strip("\n").split("#")
        _, kwargs = tokens
        ranks, backend, group_name, group_desc, timeout = parse_group_kwargs(kwargs)
        comm_group_info = ReplayerCommGroupInfo(ranks, backend, group_name, group_desc, timeout)
        self.comm_groups_ordered_list.append(comm_group_info)
        self.comm_groups_dict[group_name] = comm_group_info
        self.replay_comm_ops_dict[group_name] = defaultdict(list)

    def _parse_schedule_line(self, line: str) -> None:
        """Parse SCHEDULE line and update iteration info."""
        self.now_iteration += 1
        replay_op_info = ReplayerReplayOpInfo(
            ReplayerReplayOpType.ITERATION_BEGIN, ReplayerScheduleInfo(self.now_iteration)
        )
        self.replay_ops_ordered_list.append(replay_op_info)
        self.trace_comm_index_dict[self.now_iteration] = self.now_trace_comm_index

    def _parse_coalescing_line(self, line: str) -> None:
        """Parse COALESCING line and update coalescing info."""
        tokens = line.strip("\n").split("#")
        _, stage, group_name, group_desc, device_set, async_ops, trace_comm_index = tokens
        trace_comm_index = int(trace_comm_index)
        self.now_trace_comm_index = trace_comm_index
        if group_desc.startswith("CONTEXT-"):
            return
        if stage == "START":
            replay_op_type = ReplayerReplayOpType.COALESCING_START
        else:  # stage == "END"
            replay_op_type = ReplayerReplayOpType.COALESCING_END
        replay_op_info = ReplayerReplayOpInfo(
            replay_op_type,
            ReplayerCoalescingInfo(
                trace_comm_index, group_name, group_desc, device_set == "device_set=True", async_ops == "async_ops=True"
            ),
        )
        self.replay_ops_ordered_list.append(replay_op_info)

    def _parse_comm_line(self, line: str) -> None:
        """Parse communication operation line and update comm ops info."""
        tokens = line.strip("\n").split("#")
        trace_comm_index, c10d_func_name, signature, group_name, group_desc, args, kwargs = tokens
        trace_comm_index = int(trace_comm_index)
        self.now_trace_comm_index = trace_comm_index
        # Keep raw CP traces/group metadata, but omit replay ops and memory demands.
        # The launch layout must keep each CP group on one side of the Sandbox boundary.
        if group_desc.startswith("CONTEXT-"):
            return
        output_shape, output_dtype, input_shape, input_dtype = None, None, None, None
        async_op = parse_async_op(kwargs)
        comm_op_meta = {}
        if c10d_func_name == "isend":
            async_op = True
            """isend(tensor: torch.Tensor, dst: Optional[int] = None,
                     group: Optional[ProcessGroup] = None, ...) -> Optional[Work]:"""
            if "batch_isend_irecv" in signature:
                (input_shape, input_dtype) = parse_tensor_shape_dtype(args, -1)
                group_dst_search = re.search(r"'group_dst':\s*([^\,\}]+)", kwargs)
                if group_dst_search is not None:
                    comm_op_meta["group_dst"] = ast.literal_eval(group_dst_search.group(1))
                else:
                    raise ValueError(f"Error parsing group_dst from {kwargs}, Please check the trace format.")
            else:
                tensor_info = args if "torch.Size" in args else kwargs
                (input_shape, input_dtype) = parse_tensor_shape_dtype(tensor_info, -1)
                comm_op_meta["dst"] = parse_p2p_peer(args, kwargs, "dst")
        elif c10d_func_name == "irecv":
            async_op = True
            """irecv(tensor: torch.Tensor, src: Optional[int] = None,
                     group: Optional[ProcessGroup] = None, ...) -> Optional[Work]:"""
            if "batch_isend_irecv" in signature:
                (output_shape, output_dtype) = parse_tensor_shape_dtype(args, -1)
                group_src_search = re.search(r"'group_src':\s*([^\,\}]+)", kwargs)
                if group_src_search is not None:
                    comm_op_meta["group_src"] = ast.literal_eval(group_src_search.group(1))
                else:
                    raise ValueError(f"Error parsing group_src from {kwargs}, Please check the trace format.")
            else:
                tensor_info = args if "torch.Size" in args else kwargs
                (output_shape, output_dtype) = parse_tensor_shape_dtype(tensor_info, 0)
                comm_op_meta["src"] = parse_p2p_peer(args, kwargs, "src")
        elif c10d_func_name == "broadcast":
            """broadcast(tensor: torch.Tensor, src: Optional[int] = None,
                         group: Optional[ProcessGroup] = None, async_op: bool = False, ...) -> Optional[Work]:"""
            (input_shape, input_dtype) = parse_tensor_shape_dtype(args, -1)
            # Current c10d records keyword arguments in the kwargs field. Keep
            # accepting the older positional trace form for archived artifacts.
            src_search = re.search(r"'src':\s*([^\,\}]+)", kwargs)
            if src_search is None:
                src_search = re.search(r"'\, (\d+)\]", args)
            if src_search is not None:
                comm_op_meta["src"] = ast.literal_eval(src_search.group(1))
            else:
                raise ValueError(f"Error parsing src from args={args}, kwargs={kwargs}. Please check the trace format.")
        elif c10d_func_name == "all_reduce":
            """all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)"""
            (input_shape, input_dtype) = parse_tensor_shape_dtype(args, -1)
            comm_op_meta["reduce_op"] = parse_reduce_op(kwargs)
        elif c10d_func_name in {"all_gather_into_tensor", "_all_gather_base"}:
            """all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)"""
            """_all_gather_base(output_tensor, input_tensor, group=None, async_op=False) (deprecated)"""
            (output_shape, output_dtype) = parse_tensor_shape_dtype(args, 0)
            (input_shape, input_dtype) = parse_tensor_shape_dtype(args, -1)
        elif c10d_func_name in {"reduce_scatter_tensor", "_reduce_scatter_base"}:
            """reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)"""
            """_reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None, async_op=False) (deprecated)"""
            (output_shape, output_dtype) = parse_tensor_shape_dtype(args, 0)
            (input_shape, input_dtype) = parse_tensor_shape_dtype(args, -1)
            comm_op_meta["reduce_op"] = parse_reduce_op(kwargs)
        elif c10d_func_name in {"all_to_all_single", "barrier"}:
            # All ranks in each all-to-all should not cross the boundary of sandbox
            """barrier(group: Optional[ProcessGroup] = GroupMember.WORLD, async_op=False, ...)"""
        else:
            raise ValueError(f"Unsupported collective operation: {c10d_func_name}")
        replay_op_info = ReplayerReplayOpInfo(
            ReplayerReplayOpType.COMM,
            ReplayerCommOpInfo(
                trace_comm_index,
                c10d_func_name,
                signature,
                group_name,
                group_desc,
                output_shape,
                output_dtype,
                input_shape,
                input_dtype,
                async_op,
                comm_op_meta,
            ),
        )
        self.replay_ops_ordered_list.append(replay_op_info)
        self.replay_comm_ops_dict[group_name][trace_comm_index].append(replay_op_info)
