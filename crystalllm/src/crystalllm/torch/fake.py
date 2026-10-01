# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM PyTorch Distributed C10d Fake Process Group and Fake Work."""

from ..planning import CrystalGSpanId


class CrystalFakeWork:
    """A fake work class for PyTorch collective operations.

    This class simulates a completed work object for distributed collective
    operations, always reporting completion and success.
    """

    def __init__(self, linked_comm_span_id: CrystalGSpanId | None = None):
        """Initialize the fake work object.

        Args:
            linked_comm_span_id: The linked communication span ID for context tracking.
        """
        self._linked_comm_span_id = linked_comm_span_id

    def wait(self):
        """Wait for the work to complete."""
        if self._linked_comm_span_id is not None:
            from ..runtime import get_context

            get_context().cond_comm_span_wait(self._linked_comm_span_id)

    @staticmethod
    def is_completed():
        """Check if the work has completed."""
        return True

    @staticmethod
    def is_success():
        """Check if the work completed successfully."""
        return True


class CrystalProfiledWork:
    """A thin wrapper around real c10d Work for profiling wait boundaries."""

    def __init__(self, work: object, linked_comm_span_id: CrystalGSpanId | None = None):
        """Initialize the wrapper with a real Work and linked span ID."""
        self._work = work
        self._linked_comm_span_id = linked_comm_span_id

    def wait(self, *args, **kwargs):
        """Wait for the real work and mark the runtime wait boundary."""
        if self._linked_comm_span_id is None:
            return self._work.wait(*args, **kwargs)
        from ..runtime import get_context

        context = get_context()
        proc_span_id = context.cond_start_comm_span_wait(self._linked_comm_span_id)
        result = self._work.wait(*args, **kwargs)
        context.cond_end_comm_span_wait(self._linked_comm_span_id, proc_span_id)
        return result

    def __getattr__(self, name: str):
        """Delegate all other attributes to the wrapped Work."""
        return getattr(self._work, name)


class CrystalFakeProcessGroup:
    """A fake process group for PyTorch distributed training.

    This class simulates a process group for distributed collective operations,
    providing basic group information and rank management without actual
    communication backend.

    Args:
        global_ranks_in_group: List of global ranks in this process group.
        group_rank: The rank of current process within this group.
        group_name: Name identifier for this process group.
        group_desc: Description of this process group.
    """

    def __init__(self, global_ranks_in_group: list[int], group_rank: int, group_name: str, group_desc: str):
        """Initialize the fake process group.

        Args:
            global_ranks_in_group: List of global ranks in this process group.
            group_rank: The rank of current process within this group.
            group_name: Name identifier for this process group.
            group_desc: Description of this process group.
        """
        self._global_ranks_in_group = global_ranks_in_group
        self._group_rank = group_rank
        self._group_name = group_name
        self._group_desc = group_desc

    def __str__(self):
        """Return a string representation with address."""
        return (
            f"CrystalFakeProcessGroup("
            f"global_ranks_in_group={self._global_ranks_in_group}, "
            f"group_rank={self._group_rank}, "
            f"group_name={self._group_name}, "
            f"group_desc={self._group_desc}"
            f") at 0x{id(self):x}"
        )

    def __repr__(self):
        """Return a string representation."""
        return (
            f"CrystalFakeProcessGroup("
            f"global_ranks_in_group={self._global_ranks_in_group}, "
            f"group_rank={self._group_rank}, "
            f"group_name={self._group_name}, "
            f"group_desc={self._group_desc}"
            f")"
        )

    def size(self):
        """Get the number of processes in this group."""
        return len(self._global_ranks_in_group)

    def rank(self):
        """Get the rank of current process within this group."""
        return self._group_rank

    @property
    def group_name(self):
        """The name of this process group."""
        return self._group_name

    @property
    def group_desc(self):
        """The description of this process group."""
        return self._group_desc
