# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Timer."""

from __future__ import annotations

import enum
import json
import os
import sys
import time
import typing
from collections import defaultdict
from contextlib import ContextDecorator
from pathlib import Path
from typing import Self

import torch

from ..util import is_truthy_value

_MS_TO_NS = 1_000_000


class CrystalTimer:
    """Base timer for tracing single operations."""

    @classmethod
    def default_name(cls) -> str:
        """Return the default name of the timer.

        Returns:
            str: Default timer name
        """
        return ""

    @classmethod
    def get_and_start(cls, name: str | None = None, **kwargs) -> CrystalTimer:
        """Fetch and start the timer for CrystalTimers.

        Args:
            name (str | None): Timer name
            **kwargs: Additional arguments

        Returns:
            CrystalTimer: Started timer instance
        """
        return CrystalTimers.get_timer(name=name, kind=cls, **kwargs).start()

    def __init__(self, name: str | None = None, **kwargs) -> None:
        """Initialize the timer.

        Args:
            name (str | None): Timer name
            **kwargs: Additional arguments
        """
        self._name = name or self.__class__.default_name()

    @property
    def name(self) -> str:
        """The timer name.

        Returns:
            str: Timer name
        """
        return self._name

    def start(self) -> CrystalTimer:
        """Start the timer."""
        timers = CrystalTimers.instance()
        if timers is None:
            raise RuntimeError("CrystalTimers is not initialized")
        timers.start_timer(self)
        return self

    def stop(self) -> None:
        """Stop the timer."""
        timers = CrystalTimers.instance()
        if timers is None:
            raise RuntimeError("CrystalTimers is not initialized")
        timers.stop_timer(self)


class CrystalTimers:
    """Manager class for handling multiple timers."""

    @enum.unique
    class Status(enum.Enum):
        """Enumeration of timer statuses."""

        NOT_STARTED = 0
        STARTED = 1
        STOPPED = 2

    class Handler:
        """Abstract base class for handling timer results."""

        def print_summary(self, step_timer_durs: dict[int, dict[CrystalTimer, list[list[int]]]]) -> None:
            """Print summary of timer results."""

    class LogHandler(Handler):
        """Log handler for printing timing results to stream."""

        def __init__(self, stream: typing.TextIO | None = sys.stdout):
            """Initialize the default log handler.

            Args:
                stream (typing.TextIO | None): Output stream for logging
            """
            self._stream = stream

        def print_summary(self, step_timer_durs: dict[int, dict[CrystalTimer, list[list[int]]]]) -> None:
            """Print summary of timer results."""
            if self._stream is None:
                return

            for step, timer_durs in step_timer_durs.items():
                self._stream.write(f"=== CrystalLLM Timer Results at Step {step} ===\n")
                for timer, durs in timer_durs.items():
                    for count, dur in enumerate(durs):
                        # Format {timer_name} #{execution_count} {elapsed_time_host} {elapsed_time_device}
                        self._stream.write(f"{timer.name} #{count} {dur[0]} {dur[1]}\n")
                self._stream.flush()

    class JSONHandler(Handler):
        """JSON handler for writing timing results to file."""

        def __init__(self, dir_name: str, worker_name: str):
            """Initialize the JSON handler.

            Args:
                dir_name (str): Directory for output files
                worker_name (str): Worker name for file naming
            """
            self._dir_name = dir_name
            Path(self._dir_name).mkdir(parents=True, exist_ok=True)
            self._worker_name = worker_name

        def print_summary(self, step_timer_durs: dict[int, dict[CrystalTimer, list[list[int]]]]) -> None:
            """Write timing results to JSON file."""
            # Convert timer data to JSON-serializable format
            json_data: dict[int, dict[str, list[list[int]]]] = {}
            for step, timer_durs in step_timer_durs.items():
                json_data[step] = {}
                for timer, durs in timer_durs.items():
                    json_data[step][timer.name] = []
                    for dur in durs:
                        json_data[step][timer.name].append([dur[0], dur[1]])

            # Write to JSON file with proper formatting
            json_filepath = Path(self._dir_name) / f"{self._worker_name}.json"
            with json_filepath.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"Crystal timer file stored at {json_filepath}")

    _instance: CrystalTimers | None = None
    _initialized: bool = False
    _timers: dict[str, CrystalTimer] = {}

    @classmethod
    def instance(cls) -> CrystalTimers | None:
        """Get the singleton instance of CrystalTimers.

        Returns:
            CrystalTimers | None: Singleton instance or None
        """
        return cls._instance

    @classmethod
    def get_timer(cls, name: str | None = None, kind: type[CrystalTimer] = CrystalTimer, **kwargs) -> CrystalTimer:
        """Get or create a timer instance.

        Args:
            name (str | None): Timer name
            kind (type[CrystalTimer]): Timer class type
            **kwargs: Additional arguments

        Returns:
            CrystalTimer: Timer instance

        Raises:
            ValueError: If name is not provided for default Timer
        """
        if name is None:
            name = kind.default_name()
            if not name:
                raise ValueError("name must be provided for default Timer")
        if name not in cls._timers:
            cls._timers[name] = kind(name, **kwargs)
        return cls._timers[name]

    def __new__(cls, *args, **kwargs) -> Self:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the singleton instance."""
        if not CrystalTimers._initialized:
            self._initialize(*args, **kwargs)
            CrystalTimers._initialized = True

    def _initialize(
        self,
        *,
        record_host_time=False,
        record_device_time=True,
        step_start: int | None = None,
        step_end: int | None = None,
        handlers: list[CrystalTimers.Handler] | None = None,
    ) -> None:
        """Initialize the timer manager.

        Args:
            record_host_time (bool): Whether to record host (CPU) time
            record_device_time (bool): Whether to record device (GPU) time
            step_start (int): Starting step for profiling
            step_end (int): Ending step for profiling
            handlers (list[CrystalTimers.Handler] | None): Output handlers
        """
        self._record_host_time = record_host_time
        self._record_device_time = record_device_time

        self._step = 0
        self._step_start = step_start
        self._step_end = step_end

        self._step_enable = (self._step_start is None or self._step_start <= self._step) and (
            self._step_end is None or self._step < self._step_end
        )

        self._handlers: list[CrystalTimers.Handler] = handlers or []

        self._timer_status: dict[str, CrystalTimers.Status] = {}

        def _default_dict_dict_list():
            return defaultdict(lambda: defaultdict(list))

        self._timer_start_ts: dict[int, dict[CrystalTimer, list[int]]] = _default_dict_dict_list()
        self._timer_stop_ts: dict[int, dict[CrystalTimer, list[int]]] = _default_dict_dict_list()
        self._timer_start_events: dict[int, dict[CrystalTimer, list[torch.cuda.Event]]] = _default_dict_dict_list()
        self._timer_stop_events: dict[int, dict[CrystalTimer, list[torch.cuda.Event]]] = _default_dict_dict_list()

    def __call__(self, name: str | None = None, kind: type[CrystalTimer] = CrystalTimer, **kwargs) -> CrystalTimer:
        """Get or create a timer (callable interface).

        Args:
            name (str | None): Timer name
            kind (type[CrystalTimer]): Timer class type
            **kwargs: Additional arguments

        Returns:
            CrystalTimer: Timer instance
        """
        return CrystalTimers.get_timer(name, kind, **kwargs)

    def start_timer(self, timer: CrystalTimer) -> None:
        """Start a timer by recording CUDA event.

        Args:
            timer (CrystalTimer): Timer to start
        """
        if not self._step_enable:
            return  # Not enabled iteration

        if timer.name in self._timer_status and self._timer_status[timer.name] == CrystalTimers.Status.STARTED:
            raise RuntimeError("Timer must could not be started twice")

        # Host side
        if self._record_host_time:
            ts = time.perf_counter_ns()
            self._timer_start_ts[self._step][timer].append(ts)
        # Device side
        if self._record_device_time:
            cuda_event = torch.cuda.Event(enable_timing=True)
            cuda_event.record()
            self._timer_start_events[self._step][timer].append(cuda_event)

        self._timer_status[timer.name] = CrystalTimers.Status.STARTED

    def stop_timer(self, timer: CrystalTimer):
        """Stop a timer by recording CUDA event.

        Args:
            timer (CrystalTimer): Timer to stop
        """
        if not self._step_enable:
            return  # Not enabled iteration

        if timer.name not in self._timer_status or self._timer_status[timer.name] != CrystalTimers.Status.STARTED:
            raise RuntimeError("Timer must be started before stopping")

        # Host side
        if self._record_host_time:
            ts = time.perf_counter_ns()
            self._timer_stop_ts[self._step][timer].append(ts)
        # Device side
        if self._record_device_time:
            cuda_event = torch.cuda.Event(enable_timing=True)
            cuda_event.record()
            self._timer_stop_events[self._step][timer].append(cuda_event)

        self._timer_status[timer.name] = CrystalTimers.Status.STOPPED

    def gather_result(self) -> dict[int, dict[CrystalTimer, list[list[int]]]]:
        """Calculate durations."""
        if not self._record_host_time and not self._record_device_time:
            return {}
        # Calculate durations from timestamp and CUDA events
        step_timer_durs: dict[int, dict[CrystalTimer, list[list[int]]]] = {
            step: {timer: [[-1, -1] for _ in range(len(data))] for timer, data in timer_data.items()}
            for step, timer_data in (
                self._timer_start_events.items() if self._record_device_time else self._timer_start_ts.items()
            )
        }
        if self._record_host_time:
            for step in self._timer_start_ts:
                for timer in self._timer_start_ts[step]:
                    start_ts = self._timer_start_ts[step][timer]
                    stop_ts = self._timer_stop_ts[step][timer]
                    for i, (start_t, end_t) in enumerate(zip(start_ts, stop_ts, strict=True)):
                        step_timer_durs[step][timer][i][0] = end_t - start_t
        if self._record_device_time:
            for step in self._timer_start_events:
                for timer in self._timer_start_events[step]:
                    start_events = self._timer_start_events[step][timer]
                    stop_events = self._timer_stop_events[step][timer]
                    for i, (start_event, end_event) in enumerate(zip(start_events, stop_events, strict=True)):
                        step_timer_durs[step][timer][i][1] = int(start_event.elapsed_time(end_event) * _MS_TO_NS)
        return step_timer_durs

    def step(self) -> None:
        """Advance to next step."""
        self._step += 1
        self._step_enable = (self._step_start is None or self._step_start <= self._step) and (
            self._step_end is None or self._step < self._step_end
        )

    def stop(self) -> None:
        """Stop and handle profiling."""
        torch.cuda.synchronize()
        step_timer_durs = self.gather_result()
        # Print results using handlers
        for handler in self._handlers:
            handler.print_summary(step_timer_durs)


def _is_crystal_timer_enabled() -> bool:
    """Check whether conditional timing is currently active.

    Conditional timing is active only when the ENABLE_CRYSTAL_TIMER environment variable
    is truthy and the CrystalTimers singleton has been initialized.

    Returns:
        bool: True if conditional timing is active, False otherwise
    """
    return is_truthy_value(os.getenv("ENABLE_CRYSTAL_TIMER")) and CrystalTimers.instance() is not None


def cond_crystal_timer_start(timer_name: str) -> None:
    """Start a conditional timer by name.

    The call is a no-op unless the ENABLE_CRYSTAL_TIMER environment variable is truthy and
    the CrystalTimers singleton has been initialized.

    Usage:
    ```
    cond_crystal_timer_start("timer_name")
    # Your code here
    cond_crystal_timer_stop("timer_name")
    ```

    Args:
        timer_name (str): Timer name
    """
    if _is_crystal_timer_enabled():
        CrystalTimers.get_timer(timer_name).start()


def cond_crystal_timer_stop(timer_name: str) -> None:
    """Stop a conditional timer previously started by name.

    The call is a no-op unless the ENABLE_CRYSTAL_TIMER environment variable is truthy and
    the CrystalTimers singleton has been initialized.

    Args:
        timer_name (str): Timer name
    """
    if _is_crystal_timer_enabled():
        CrystalTimers.get_timer(timer_name).stop()


class cond_crystal_timer_record(ContextDecorator):  # noqa: N801
    """Context manager for timer recording.

    Usage:
    ```
    with cond_crystal_timer_record("timer_name"):
        # Your code here
        pass


    @cond_crystal_timer_record("timer_name")
    def func():
        # Your code here
        pass
    ```
    """

    def __init__(self, timer_name: str) -> None:
        """Initialize the context manager with a timer name.

        Args:
            timer_name (str): Timer name
        """
        self._timer_name = timer_name

    def __enter__(self) -> None:
        """Start the timer when entering the context."""
        cond_crystal_timer_start(self._timer_name)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer when exiting the context."""
        cond_crystal_timer_stop(self._timer_name)
