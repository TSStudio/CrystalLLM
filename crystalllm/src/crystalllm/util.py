# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Library utility functions."""

import os
from typing import Any


def get_from_args_kwargs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    position: int | None = None,
    keyword: str | None = None,
) -> Any:
    """Get a value from args or kwargs.

    Args:
        args: The positional arguments tuple.
        kwargs: The keyword arguments dictionary.
        position: The index in args to look for the value.
        keyword: The key in kwargs to look for the value.

    Returns:
        The value found in args or kwargs, or None if not found.
    """
    if position is not None and position < len(args):
        return args[position]
    if keyword is not None and keyword in kwargs:
        return kwargs[keyword]
    return None


def is_truthy_value(value: str | int | None) -> bool:
    """Check if the given value is truthy.

    Args:
        value (str | int | None): The value to check.

    Returns:
        bool: True if the value is truthy, False otherwise.
    """
    if isinstance(value, str):
        return value.strip().lower() == "true" or value.strip() == "1"
    if isinstance(value, int):
        return value == 1
    return False


def print_rank_0(*message):
    """If distributed environment variables is set, print only on rank 0."""
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(*message, flush=True)


def print_rank_last(*message):
    """If distributed environment variables is set, print only on last rank."""
    if int(os.getenv("LOCAL_RANK", "0")) == (int(os.getenv("LOCAL_WOLRD_SIZE", "1")) - 1):
        print(*message, flush=True)
