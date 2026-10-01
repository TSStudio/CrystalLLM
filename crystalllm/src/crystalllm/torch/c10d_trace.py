# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM PyTorch Distributed C10d Trace Logger."""

from __future__ import annotations

import gzip
import inspect
import os
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

import torch
from typing_extensions import ParamSpec, Self

from ..util import get_from_args_kwargs

_T = TypeVar("_T")
_P = ParamSpec("_P")


_CRSYTAL_SIGNATURE_FRAME_DEPTH = 4
_CRSYTAL_TRACE_BUFFER_SIZE = 10000


@cache
def _group_argument_spec(func: Callable) -> tuple[int | None, Any]:
    """Resolve the group argument once, following functools.wraps metadata."""
    parameters = inspect.signature(func).parameters
    group = parameters.get("group")
    if group is None or group.kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }:
        raise ValueError(f"Cannot resolve group argument from {func!r}")
    positional = [
        name for name, parameter in parameters.items()
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    position = positional.index("group") if "group" in positional else None
    return position, group.default


def _get_collective_group(func: Callable, args: tuple, kwargs: dict) -> Any:
    """Read group without rebinding the entire call or changing its arguments."""
    position, default = _group_argument_spec(func)
    if "group" in kwargs:
        group = kwargs["group"]
    elif position is not None and position < len(args):
        group = args[position]
    elif default is inspect.Parameter.empty:
        raise TypeError(f"Missing required group argument for {func!r}")
    else:
        group = default
    if group is None:
        return torch.distributed.distributed_c10d._get_default_group()  # noqa: SLF001
    return group


class CrystalC10DTracer:
    """Singleton class for managing c10d collective communication tracing."""

    _instance: CrystalC10DTracer | None = None
    _initialized: bool = False

    def __new__(cls) -> Self:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the singleton instance."""
        if CrystalC10DTracer._initialized:
            return

        # Collective communication trace generation
        self._trace_file: gzip.GzipFile | None = None
        self._trace_buffer: list[str] = []
        self._trace_max_size: int = _CRSYTAL_TRACE_BUFFER_SIZE

        CrystalC10DTracer._initialized = True

    def init_trace_file(self) -> None:
        """Initialize the trace file for writing."""
        trace_dir = Path(os.getenv("CRYSTAL_LOG_DIR")) / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file_name = "trace_" + os.getenv("RANK") + ".log.gz"
        trace_file_path = trace_dir / trace_file_name
        self._trace_file = gzip.open(trace_file_path, "wt")  # noqa: SIM115
        print(f"Crystal c10d trace file initialized at {trace_file_path}")

    def close_trace_file(self) -> None:
        """Close the trace file, flushing any remaining buffer."""
        if self._trace_file is not None:
            if len(self._trace_buffer) > 0:
                self._flush_trace()
            self._trace_file.close()

    def _flush_trace(self) -> None:
        """Flush the trace buffer to the file."""
        assert self._trace_file is not None, "Crystal c10d trace file should be initialized first"
        self._trace_file.write("\n".join(self._trace_buffer))
        self._trace_file.write("\n")
        self._trace_file.flush()
        self._trace_buffer.clear()

    def _write_trace(self, log_type: str, log_detail: str) -> None:
        """Write a log string to the trace buffer."""
        assert self._trace_file is not None, "Crystal c10d trace file should be initialized first"
        trace_str = f"{log_type}#{log_detail}"
        self._trace_buffer.append(trace_str.replace("\n", ""))
        if len(self._trace_buffer) >= self._trace_max_size:
            self._flush_trace()

    def c10d_trace_writer(
        self, trace_index: int, func: Callable[_P, _T], args: list[Any], kwargs: dict[str, Any]
    ) -> None:
        """Write trace entry for a collective communication operation under ``trace_index``."""
        if func.__name__ not in {
            "isend",
            "irecv",
            "broadcast",
            "all_reduce",
            "all_gather_into_tensor",
            "_all_gather_base",
            "reduce_scatter_tensor",
            "_reduce_scatter_base",
            "all_to_all_single",
            "barrier",
        }:
            raise NotImplementedError(f"Unsupported collective operation: {func.__name__}")
        frame = sys._getframe(_CRSYTAL_SIGNATURE_FRAME_DEPTH)  # noqa: SLF001
        signature = f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
        group = _get_collective_group(func, args, kwargs)
        # We only log the tensor shapes to the trace file instead of the tensor values.
        filtered_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                filtered_args.append(f"tensor-{arg.shape}-torch.dtype({arg.dtype})")
            else:
                filtered_args.append(arg)
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                filtered_kwargs[k] = f"tensor-{v.shape}-torch.dtype({v.dtype})"
            else:
                filtered_kwargs[k] = v
        trace_str = (
            f"{func.__name__}#{signature}#{group.group_name}#{group.group_desc}#{filtered_args}#{filtered_kwargs}"
        )
        self._write_trace(trace_index, trace_str)

    def trace_start_coalescing(
        self, trace_index: int, group_name: str, group_desc: str, device_set: bool, async_ops: bool
    ) -> None:
        """Trace the start of a coalescing operation under ``trace_index``."""
        self._write_trace(
            "COALESCING",
            f"START#{group_name}#{group_desc}#device_set={device_set}#async_ops={async_ops}#{trace_index}",
        )

    def trace_end_coalescing(
        self, trace_index: int, group_name: str, group_desc: str, device_set: bool, async_ops: bool
    ) -> None:
        """Trace the end of a coalescing operation under ``trace_index``."""
        self._write_trace(
            "COALESCING",
            f"END#{group_name}#{group_desc}#device_set={device_set}#async_ops={async_ops}#{trace_index}",
        )


def print_operation(func: Callable[_P, _T], args: list[Any], kwargs: dict[str, Any]) -> None:
    """Print information about a collective communication operation."""
    rank = os.getenv("RANK")

    frame = sys._getframe(_CRSYTAL_SIGNATURE_FRAME_DEPTH - 1)  # noqa: SLF001
    signature = f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"

    def _get_group(
        position: int | None = None,
        keyword: str = "group",
    ) -> Any:
        """Get group from args or kwargs."""
        group = get_from_args_kwargs(args, kwargs, position=position, keyword=keyword)
        if group is not None:
            return group
        # When 'group' in kwargs is None, the default group is used.
        return torch.distributed.distributed_c10d._get_default_group()  # noqa: SLF001

    if func.__name__ == "isend":
        # isend(tensor, dst, group, tag, group_dst):
        tensor = get_from_args_kwargs(args, kwargs, position=0, keyword="tensor")
        dst = get_from_args_kwargs(args, kwargs, position=1, keyword="dst")
        group = _get_group(position=2)
        group_dst = get_from_args_kwargs(args, kwargs, position=4, keyword="group_dst")
        if sys._getframe(_CRSYTAL_SIGNATURE_FRAME_DEPTH).f_code.co_name == "batch_isend_irecv":  # noqa: SLF001
            print(
                (
                    f"[rank {rank}] isend() is called in batch_isend_irecv(): "
                    f"tensor shape: {tensor.shape}, group name:{group.group_name} "
                    f"group desc: {group.group_desc}, dst: {dst}, group_dst: {group_dst}"
                ),
                flush=True,
            )
        else:
            print(
                (
                    f"[rank {rank}] isend() is called at {signature}: "
                    f"tensor shape: {tensor.shape}, group name:{group.group_name} "
                    f"group desc: {group.group_desc}, dst: {dst}, group_dst: {group_dst}"
                ),
                flush=True,
            )
    elif func.__name__ == "irecv":
        # irecv(tensor, src, group, tag, group_src):
        tensor = get_from_args_kwargs(args, kwargs, position=0, keyword="tensor")
        src = get_from_args_kwargs(args, kwargs, position=1, keyword="src")
        group = _get_group(position=2)
        group_src = get_from_args_kwargs(args, kwargs, position=4, keyword="group_src")
        if sys._getframe(_CRSYTAL_SIGNATURE_FRAME_DEPTH).f_code.co_name == "batch_isend_irecv":  # noqa: SLF001
            print(
                (
                    f"[rank {rank}] irecv() is called in batch_isend_irecv(): "
                    f"tensor shape: {tensor.shape}, group name:{group.group_name} "
                    f"group desc: {group.group_desc}, src: {src}, group_src: {group_src}"
                ),
                flush=True,
            )
        else:
            print(
                (
                    f"[rank {rank}] irecv() is called at {signature}: "
                    f"tensor shape: {tensor.shape}, group name:{group.group_name} "
                    f"group desc: {group.group_desc}, src: {src}, group_src: {group_src}"
                ),
                flush=True,
            )
    elif func.__name__ == "broadcast":
        # broadcast(tensor, src, group, async_op, group_src):
        tensor = get_from_args_kwargs(args, kwargs, position=0, keyword="tensor")
        src = get_from_args_kwargs(args, kwargs, position=1, keyword="src")
        group = _get_group(position=2)
        group_src = get_from_args_kwargs(args, kwargs, position=4, keyword="group_src")
        print(
            (
                f"[rank {rank}] broadcast() is called at {signature}: "
                f"tensor shape: {tensor.shape}, group name:{group.group_name} "
                f"group desc: {group.group_desc}, src: {src}, group_src: {group_src}"
            ),
            flush=True,
        )
    elif func.__name__ == "all_reduce":
        # all_reduce(tensor, op, group, async_op):
        tensor = get_from_args_kwargs(args, kwargs, position=0, keyword="tensor")
        op = get_from_args_kwargs(args, kwargs, position=1, keyword="op")
        group = _get_group(position=2)
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        print(
            (
                f"[rank {rank}] all_reduce() is called at {signature}: tensor shape: {tensor.shape}, "
                f"group name:{group.group_name} group desc: {group.group_desc}, op: {op}"
            ),
            flush=True,
        )
    elif func.__name__ in {"all_gather_into_tensor", "_all_gather_base"}:
        # all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
        # _all_gather_base(output_tensor, input_tensor, group, async_op) (deprecated)
        output_tensor = get_from_args_kwargs(args, kwargs, position=0, keyword="output_tensor")
        input_tensor = get_from_args_kwargs(args, kwargs, position=1, keyword="input_tensor")
        group = _get_group(position=2)
        print(
            (
                f"[rank {rank}] {func.__name__}() is called at {signature}: "
                f"output_tensor shape: {output_tensor.shape}, "
                f"input_tensor shape: {input_tensor.shape}, "
                f"group name:{group.group_name} group desc: {group.group_desc}"
            ),
            flush=True,
        )
    elif func.__name__ in {"reduce_scatter_tensor", "_reduce_scatter_base"}:
        # reduce_scatter_tensor(output, input, op, group, async_op)
        # _reduce_scatter_base(output, input, op, group, async_op) (deprecated)
        output = get_from_args_kwargs(args, kwargs, position=0, keyword="output")
        input_tensor = get_from_args_kwargs(args, kwargs, position=1, keyword="input")
        op = get_from_args_kwargs(args, kwargs, position=2, keyword="op")
        group = _get_group(position=3)
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        print(
            (
                f"[rank {rank}] {func.__name__}() is called at {signature}: "
                f"output shape: {output.shape}, input shape: {input_tensor.shape}, "
                f"group name:{group.group_name} group desc: {group.group_desc}, op: {op}"
            ),
            flush=True,
        )
    elif func.__name__ == "all_to_all_single":
        # all_to_all_single(output, input, output_split_sizes, input_split_sizes, group, async_op):
        output = get_from_args_kwargs(args, kwargs, position=0, keyword="output")
        input_tensor = get_from_args_kwargs(args, kwargs, position=1, keyword="input")
        output_split_sizes = get_from_args_kwargs(args, kwargs, position=2, keyword="output_split_sizes")
        output_split_sizes = str(output_split_sizes).replace("\n", "") if output_split_sizes is not None else ""
        input_split_sizes = get_from_args_kwargs(args, kwargs, position=3, keyword="input_split_sizes")
        input_split_sizes = str(input_split_sizes).replace("\n", "") if input_split_sizes is not None else ""
        group = _get_group(position=4)
        print(
            (
                f"[rank {rank}] {func.__name__}() is called at {signature}: "
                f"output shape: {output.shape}, input shape: {input_tensor.shape}, "
                f"group name:{group.group_name} group desc: {group.group_desc} "
                f"output split sizes: {output_split_sizes} input split sizes: {input_split_sizes}"
            ),
            flush=True,
        )
    elif func.__name__ == "barrier":
        # barrier(group, async_op, device_ids):
        group = _get_group(position=0)
        print(
            (
                f"[rank {rank}] barrier() is called at {signature}: "
                f"group name:{group.group_name} group desc: {group.group_desc}"
            ),
            flush=True,
        )
    else:
        func_name = func.__name__
        print(
            f"[rank {rank}] Unsupported collective operation: {func_name}",
            flush=True,
        )
