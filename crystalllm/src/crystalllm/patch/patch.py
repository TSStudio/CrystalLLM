# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Patch."""

import functools
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from typing_extensions import ParamSpec


class PatchType(Enum):
    """Type of patch target."""

    ARGS = "args"
    KWARGS = "kwargs"
    RES = "res"


class BehaviorType(Enum):
    """Behavior type for patch."""

    ONLY_PATCH = "only_patch"
    ONLY_EXEC = "only_exec"
    EXEC_THEN_PATCH = "exec_then_patch"


# Short aliases for convenience
_ARGS = PatchType.ARGS
_KWARGS = PatchType.KWARGS
_RES = PatchType.RES
_ONLY_PATCH = BehaviorType.ONLY_PATCH
_ONLY_EXEC = BehaviorType.ONLY_EXEC
_EXEC_THEN_PATCH = BehaviorType.EXEC_THEN_PATCH


@dataclass(kw_only=True)
class CrystalPatchPlan:
    """Plan for patching a specific argument or return value."""

    patch_type: PatchType
    position: int | str
    action: Callable[[Any], Any]

    def __post_init__(self):
        """Validate position type based on patch type.

        Raises:
            TypeError: If position type is invalid for the patch type.
        """
        if self.patch_type in {_ARGS, _RES} and not isinstance(self.position, int):
            raise TypeError(f"position must be int for {self.patch_type.name}, got {type(self.position).__name__}")

        if self.patch_type == _KWARGS and not isinstance(self.position, str):
            raise TypeError(f"position must be str for KWARGS, got {type(self.position).__name__}")


@dataclass(kw_only=True)
class CrystalPatchRequest:
    """Request for patching a specific function."""

    patch_func: Callable
    capture_behavior: BehaviorType
    emulate_behavior: BehaviorType
    patch_plan: list[CrystalPatchPlan] | CrystalPatchPlan
    patch_res_length: int | None = None
    additional_condition: Callable[[], bool] | None = None
    additional_patch_actions: Callable[[], None] | None = None

    def __post_init__(self):
        """Validate the request after initialization.

        Raises:
            ValueError: If patch_plan is empty or if patch_res_length is not
                       specified when ONLY_PATCH is in behaviors.
        """
        # Normalize patch_plan to list for validation
        self.patch_plan = [self.patch_plan] if isinstance(self.patch_plan, CrystalPatchPlan) else self.patch_plan

        # Validate patch_plan is not empty
        if not self.patch_plan:
            raise ValueError("patch_plan cannot be empty")

        # Validate patch_res_length when ONLY_PATCH is in behaviors
        behaviors = [self.capture_behavior, self.emulate_behavior]
        if _ONLY_PATCH in behaviors and (self.patch_res_length is None or self.patch_res_length < 0):
            raise ValueError("patch_res_length must be a positive integer when ONLY_PATCH is used")


# Thread-local storage for active patch requests
_patch_storage = threading.local()


def _get_active_requests() -> dict:
    """Get the active requests dictionary for the current thread."""
    if not hasattr(_patch_storage, "requests"):
        _patch_storage.requests = {}
    return _patch_storage.requests


class cond_crystal_patch_request:  # noqa: N801
    """Context manager for conditional crystal patch requests.

    This context manager enables patching of functions decorated with
    @crystal_patch_response based on the CRYSTAL_MODE environment variable
    and the specified behaviors.
    """

    def __init__(
        self,
        request: list[CrystalPatchRequest] | CrystalPatchRequest,
    ):
        """Initialize the context manager.

        Args:
            request: A single CrystalPatchRequest or a list of requests.

        Raises:
            ValueError: If request is empty.
        """
        # Normalize request to list
        if isinstance(request, CrystalPatchRequest):
            self.requests = [request]
        else:
            self.requests = request

        # Validate request is not empty
        if not self.requests:
            raise ValueError("request cannot be empty")

        self._is_active = False
        self._saved_requests = {}

    def _should_activate(self) -> bool:
        """Determine if this context manager should activate patching.

        Returns:
            True if patching should be activated, False otherwise.
        """
        # If CRYSTAL_MODE is not set or is not CAPTURE/EMULATE, treat as nullcontext
        if os.getenv("CRYSTAL_MODE") not in {"CAPTURE", "EMULATE"}:
            return False

        # Check additional conditions for each request
        for req in self.requests:
            if req.additional_condition is not None and not req.additional_condition():
                return False

        return True

    def __enter__(self):
        """Enter the context manager."""
        if not self._should_activate():
            return self

        self._is_active = True
        active_requests = _get_active_requests()

        # Save current state for restoration on exit
        self._saved_requests = {}
        for req in self.requests:
            if req.patch_func in active_requests:
                self._saved_requests[req.patch_func] = active_requests[req.patch_func]

        # Update active requests
        for req in self.requests:
            active_requests[req.patch_func] = req

        return self

    def __exit__(self, *exc):
        """Exit the context manager."""
        if not self._is_active:
            return False

        active_requests = _get_active_requests()

        for req in self.requests:
            if req.patch_func in self._saved_requests:
                # Restore saved requests
                active_requests[req.patch_func] = self._saved_requests[req.patch_func]
            else:
                # Remove if it wasn't there before
                active_requests.pop(req.patch_func, None)

        self._is_active = False
        return False


_T = TypeVar("_T")
_P = ParamSpec("_P")


def crystal_patch_response(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorator for functions that can respond to crystal patches.

    Functions decorated with this can be patched when called within a
    cond_crystal_patch_request context manager.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Call the decorated function with potential patching."""
        if os.getenv("CRYSTAL_MODE") not in {"CAPTURE", "EMULATE"}:
            return func(*args, **kwargs)

        active_requests = _get_active_requests()
        # If no request for this wrapper function, execute normally
        # Note: we use wrapper itself as the key, not the original func
        if wrapper not in active_requests:
            return func(*args, **kwargs)

        req = active_requests[wrapper]
        crystal_mode = os.getenv("CRYSTAL_MODE")

        # Determine behavior based on CRYSTAL_MODE
        if crystal_mode == "CAPTURE":
            behavior = req.capture_behavior
        elif crystal_mode == "EMULATE":
            behavior = req.emulate_behavior

        # Execute based on behavior
        if behavior == _ONLY_EXEC:
            return func(*args, **kwargs)
        if behavior == _ONLY_PATCH:
            # Don't execute, return pre-generated None values
            result = None if req.patch_res_length == 0 else tuple(None for _ in range(req.patch_res_length))
        elif behavior == _EXEC_THEN_PATCH:
            # Execute function normally with args
            result = func(*args, **kwargs)

        # Apply patches to arguments and result
        patched_result = list(result) if isinstance(result, tuple) else [result]

        if behavior in {_ONLY_PATCH, _EXEC_THEN_PATCH}:
            for plan in req.patch_plan:
                if plan.patch_type == _ARGS:
                    plan.action(args[plan.position])
                elif plan.patch_type == _KWARGS:
                    plan.action(kwargs[plan.position])
                elif plan.patch_type == _RES:
                    patched_result[plan.position] = plan.action(patched_result[plan.position])

        # Call additional patch actions if patching was performed
        if req.additional_patch_actions is not None:
            req.additional_patch_actions()

        return patched_result[0] if len(patched_result) == 1 else tuple(patched_result)

    return wrapper
