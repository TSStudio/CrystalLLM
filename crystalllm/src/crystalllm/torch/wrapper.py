# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM PyTorch Distributed C10d Wrappers."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TypeVar

from typing_extensions import ParamSpec

from ..util import is_truthy_value
from .fake import CrystalFakeProcessGroup, CrystalFakeWork

_T = TypeVar("_T")
_P = ParamSpec("_P")


def cond_custom_write_crystal_c10d_trace(log_type: str, log_detail: str) -> None:
    """Conditionally write a custom trace entry if tracing is enabled."""
    from ..runtime import get_context

    get_context().cond_custom_write_c10d_trace(log_type, log_detail)


def cond_crystal_c10d_trace_writer(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorator to conditionally print and trace collective communication operations."""

    # !!! This must be the second wrapper of each communication function !!!
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_truthy_value(os.getenv("CRYSTAL_C10D_PRINTING")):
            from .c10d_trace import print_operation

            print_operation(func, args, kwargs)
        if is_truthy_value(os.getenv("CRYSTAL_C10D_CAPTURING")):
            from ..runtime import get_context

            get_context().cond_c10d_trace_writer(func, args, kwargs)

        return func(*args, **kwargs)

    return wrapper


def cond_crystal_c10d_start_coalescing(group: CrystalFakeProcessGroup, device_set: bool, async_ops: bool) -> None:
    """Conditionally trace the start of a capture-mode coalescing operation."""
    from ..runtime import get_context

    context = get_context()
    # Opening the manager scope reserves the comm trace index that the trace line below
    # logs, and ends the current computation span: the coalesced operations are launched
    # inside the manager scope.
    context.cond_start_comm_span_coalescing()
    context.cond_c10d_trace_start_coalescing(group.group_name, group.group_desc, device_set, async_ops)


def cond_crystal_c10d_end_coalescing(
    group: CrystalFakeProcessGroup, device_set: bool, async_ops: bool
) -> CrystalFakeWork:
    """Conditionally trace the end of a capture-mode coalescing operation."""
    from ..runtime import get_context
    from .collective_handler import crystal_c10d_coalescing_manager_handler

    # The trace line must be written before the handler closes the manager scope.
    get_context().cond_c10d_trace_end_coalescing(group.group_name, group.group_desc, device_set, async_ops)
    return crystal_c10d_coalescing_manager_handler(group, async_ops)


def cond_crystal_runtime_profiler_start_coalescing(group: CrystalFakeProcessGroup, async_ops: bool) -> None:
    """Conditionally start runtime-profiler coalescing collection."""
    from ..runtime import get_context

    # Same manager scope as capture mode: one shared comm trace index, and the real
    # coalesced kernels are enqueued inside the scope, so computation must stop here.
    get_context().cond_start_comm_span_coalescing()


def cond_crystal_runtime_profiler_end_coalescing(
    group: CrystalFakeProcessGroup,
    async_ops: bool,
    launch_func: Callable[[], object | None],
) -> object | None:
    """Conditionally end the coalescing span, launch the work, and wrap it for the profiler."""
    from .collective_handler import crystal_runtime_profiler_coalescing_manager_handler

    return crystal_runtime_profiler_coalescing_manager_handler(group, async_ops, launch_func)


def cond_crystal_collectives_handler(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorator to conditionally trace collective communication operations."""

    # !!! This must be the third wrapper of each communication function !!!
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_truthy_value(os.getenv("CRYSTAL_C10D_CAPTURING")):
            from .collective_handler import crystal_collectives_handler

            return crystal_collectives_handler(func, args, kwargs)
        if is_truthy_value(os.getenv("ENABLE_CRYSTAL_RUNTIME_PROFILER")):
            from .collective_handler import crystal_runtime_profiler_collectives_handler

            return crystal_runtime_profiler_collectives_handler(func, args, kwargs)

        return func(*args, **kwargs)

    return wrapper
