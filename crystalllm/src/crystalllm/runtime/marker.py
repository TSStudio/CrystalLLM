# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Runtime Computation Unit Marker."""

from contextlib import ContextDecorator

from .context import get_context


class cond_crystal_graph_mark(ContextDecorator):  # noqa: N801
    """Context manager for adding descriptions to captured computation spans.

    This context manager appends ``description`` to computation spans touched by
    the mark range. Nested entry is prohibited.
    """

    _entered = False

    def __init__(self, description: str) -> None:
        """Initialize the context manager with a span description."""
        self._description = description
        self._start_proc_index = None

    def __enter__(self) -> None:
        """Enter the mark range."""
        if cond_crystal_graph_mark._entered:
            msg = "cond_crystal_graph_mark is already entered"
            raise RuntimeError(msg)

        cond_crystal_graph_mark._entered = True
        self._start_proc_index = get_context().cond_graph_mark_start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the mark range and attach description to touched proc spans."""
        if not cond_crystal_graph_mark._entered:
            msg = "cond_crystal_graph_mark is not entered"
            raise RuntimeError(msg)

        get_context().cond_graph_mark_stop(self._start_proc_index, self._description)
        cond_crystal_graph_mark._entered = False
