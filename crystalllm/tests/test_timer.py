# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Comprehensive tests for CrystalLLM Timer."""

import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import torch

from crystalllm.timer import (
    CrystalTimer,
    CrystalTimers,
    cond_crystal_timer_record,
    cond_crystal_timer_start,
    cond_crystal_timer_stop,
)


def reset_timers_singleton() -> None:
    """Reset the CrystalTimers singleton so each test owns a fresh state."""
    CrystalTimers._instance = None
    CrystalTimers._initialized = False
    CrystalTimers._timers = {}


def initialize_timers(**kwargs) -> CrystalTimers:
    """Create the CrystalTimers singleton and enter its first step.

    Step counting starts at 0 and the driver calls step() once per iteration, so tests
    that expect their timing data at step 1 enter the first step here.
    """
    timers = CrystalTimers(**kwargs)
    timers.step()
    return timers


class TestCrystalTimerBasic(unittest.TestCase):
    """Test basic CrystalTimer functionality."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_default_name(self):
        """Test default name returns empty string."""
        self.assertEqual(CrystalTimer.default_name(), "")

    def test_timer_initialization(self):
        """Test timer initialization with name."""
        timer = CrystalTimer(name="test_timer")
        self.assertEqual(timer.name, "test_timer")

    def test_timer_initialization_default_name(self):
        """Test timer initialization with default name."""
        timer = CrystalTimer()
        self.assertEqual(timer.name, "")

    def test_get_timer_creates_new_timer(self):
        """Test get_timer creates new timer instance."""
        timer = CrystalTimers.get_timer("new_timer")
        self.assertIsInstance(timer, CrystalTimer)
        self.assertEqual(timer.name, "new_timer")

    def test_get_timer_returns_existing_timer(self):
        """Test get_timer returns existing timer instance."""
        timer1 = CrystalTimers.get_timer("existing_timer")
        timer2 = CrystalTimers.get_timer("existing_timer")
        self.assertIs(timer1, timer2)

    def test_get_timer_without_name_raises_error(self):
        """Test get_timer without name for default Timer raises ValueError."""
        with self.assertRaises(ValueError) as context:
            CrystalTimers.get_timer()
        self.assertIn("name must be provided", str(context.exception))


class TestCrystalTimersInitialization(unittest.TestCase):
    """Test CrystalTimers initialization."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_singleton_instance(self):
        """Test CrystalTimers is a singleton."""
        timers1 = CrystalTimers()
        timers2 = CrystalTimers.instance()
        self.assertIs(timers1, timers2)

    def test_second_construction_returns_existing_instance(self):
        """Test constructing again returns the existing instance without re-initializing."""
        timers1 = CrystalTimers()
        timers2 = CrystalTimers(record_host_time=True)  # Different params
        self.assertIs(timers1, timers2)
        self.assertFalse(timers2._record_host_time)

    def test_default_initialization(self):
        """Test default initialization parameters."""
        timers = CrystalTimers()
        self.assertFalse(timers._record_host_time)
        self.assertTrue(timers._record_device_time)
        self.assertIsNone(timers._step_start)
        self.assertIsNone(timers._step_end)
        # Without step bounds every step records, starting from step 0.
        self.assertEqual(timers._step, 0)
        self.assertTrue(timers._step_enable)

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        timers = CrystalTimers(
            record_host_time=True,
            record_device_time=False,
            step_start=5,
            step_end=10,
        )
        self.assertTrue(timers._record_host_time)
        self.assertFalse(timers._record_device_time)
        self.assertEqual(timers._step_start, 5)
        self.assertEqual(timers._step_end, 10)

    def test_step_enables_correctly(self):
        """Test step enable logic."""
        timers = CrystalTimers(step_start=2, step_end=5)
        # Step counting starts at 0 and is advanced by the driver.
        self.assertEqual(timers._step, 0)
        self.assertFalse(timers._step_enable)  # 0 < 2

        timers.step()
        self.assertEqual(timers._step, 1)
        self.assertFalse(timers._step_enable)  # 1 < 2

        timers.step()
        self.assertEqual(timers._step, 2)
        self.assertTrue(timers._step_enable)  # 2 >= 2 and 2 < 5

        timers.step()
        self.assertEqual(timers._step, 3)
        self.assertTrue(timers._step_enable)

        timers.step()
        timers.step()
        self.assertEqual(timers._step, 5)
        self.assertFalse(timers._step_enable)  # 5 >= 5


class TestCrystalTimersTimerOperations(unittest.TestCase):
    """Test timer start/stop operations."""

    def setUp(self):
        """Reset singleton and initialize timers before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.timers = initialize_timers(
            record_host_time=True,
            record_device_time=True,
            step_start=1,
            step_end=None,
        )

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_start_timer_without_initialization_raises(self):
        """Test starting timer without initialization raises RuntimeError."""
        CrystalTimers._instance = None
        timer = CrystalTimer("test")
        with self.assertRaises(RuntimeError) as context:
            timer.start()
        self.assertIn("CrystalTimers is not initialized", str(context.exception))

    def test_stop_timer_without_initialization_raises(self):
        """Test stopping timer without initialization raises RuntimeError."""
        CrystalTimers._instance = None
        timer = CrystalTimer("test")
        with self.assertRaises(RuntimeError) as context:
            timer.stop()
        self.assertIn("CrystalTimers is not initialized", str(context.exception))

    def test_stop_timer_without_start_raises(self):
        """Test stopping timer without starting raises RuntimeError."""
        timer = CrystalTimer("test")
        with self.assertRaises(RuntimeError) as context:
            timer.stop()
        self.assertIn("Timer must be started before stopping", str(context.exception))

    def test_start_stop_timer_records_data(self):
        """Test start/stop timer records timing data."""
        timer = CrystalTimer("test_timer")
        timer.start()
        # Simulate some work
        x = torch.randn(100, 100, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        timer.stop()

        # Check that data was recorded
        self.assertEqual(len(self.timers._timer_start_ts[1][timer]), 1)
        self.assertEqual(len(self.timers._timer_stop_ts[1][timer]), 1)
        self.assertEqual(len(self.timers._timer_start_events[1][timer]), 1)
        self.assertEqual(len(self.timers._timer_stop_events[1][timer]), 1)

    def test_timer_status_tracking(self):
        """Test timer status is tracked correctly."""
        timer = CrystalTimer("status_test")
        self.assertNotIn("status_test", self.timers._timer_status)

        timer.start()
        self.assertEqual(
            self.timers._timer_status["status_test"],
            CrystalTimers.Status.STARTED,
        )

        timer.stop()
        self.assertEqual(
            self.timers._timer_status["status_test"],
            CrystalTimers.Status.STOPPED,
        )

    def test_second_start_call_raises(self):
        """Test starting an already started timer raises RuntimeError."""
        timer = CrystalTimer("multi_start")
        timer.start()
        with self.assertRaises(RuntimeError) as context:
            timer.start()
        self.assertIn("could not be started twice", str(context.exception))

        self.assertEqual(len(self.timers._timer_start_ts[1][timer]), 1)
        self.assertEqual(len(self.timers._timer_start_events[1][timer]), 1)

    def test_multiple_timers_same_step(self):
        """Test multiple timers in same step."""
        timer1 = CrystalTimer("timer1")
        timer2 = CrystalTimer("timer2")

        timer1.start()
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        timer1.stop()

        timer2.start()
        z = torch.matmul(y, y)
        timer2.stop()

        self.assertEqual(len(self.timers._timer_start_ts[1][timer1]), 1)
        self.assertEqual(len(self.timers._timer_start_ts[1][timer2]), 1)

    def test_timer_across_multiple_steps(self):
        """Test timer operations across multiple steps."""
        timer = CrystalTimer("step_timer")

        # Step 1
        timer.start()
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        timer.stop()

        # Step 2
        self.timers.step()
        timer.start()
        z = torch.matmul(y, y)
        timer.stop()

        self.assertEqual(len(self.timers._timer_start_ts[1][timer]), 1)
        self.assertEqual(len(self.timers._timer_start_ts[2][timer]), 1)


class TestCrystalTimersGatherResult(unittest.TestCase):
    """Test gather_result functionality."""

    def setUp(self):
        """Reset singleton and initialize timers before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_gather_result_host_only(self):
        """Test gather_result with host time only."""
        timers = initialize_timers(
            record_host_time=True,
            record_device_time=False,
        )

        timer = CrystalTimer("host_only")
        timer.start()
        import time

        time.sleep(0.01)  # 10ms sleep
        timer.stop()

        result = timers.gather_result()
        self.assertIn(1, result)
        self.assertIn(timer, result[1])
        self.assertEqual(len(result[1][timer]), 1)
        # Host time should be recorded, device time should be -1
        self.assertGreater(result[1][timer][0][0], 0)  # host time in ns
        self.assertEqual(result[1][timer][0][1], -1)  # device time

    def test_gather_result_device_only(self):
        """Test gather_result with device time only."""
        timers = initialize_timers(
            record_host_time=False,
            record_device_time=True,
        )

        timer = CrystalTimer("device_only")
        timer.start()
        x = torch.randn(100, 100, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        timer.stop()

        result = timers.gather_result()
        self.assertIn(1, result)
        self.assertIn(timer, result[1])
        self.assertEqual(len(result[1][timer]), 1)
        # Host time should be -1, device time should be recorded
        self.assertEqual(result[1][timer][0][0], -1)  # host time
        self.assertGreater(result[1][timer][0][1], 0)  # device time in ns

    def test_gather_result_both_times(self):
        """Test gather_result with both host and device times."""
        timers = initialize_timers(
            record_host_time=True,
            record_device_time=True,
        )

        timer = CrystalTimer("both_times")
        timer.start()
        import time

        time.sleep(0.005)
        x = torch.randn(100, 100, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        time.sleep(0.005)
        timer.stop()

        result = timers.gather_result()
        self.assertIn(1, result)
        self.assertIn(timer, result[1])
        self.assertEqual(len(result[1][timer]), 1)
        # Both times should be recorded
        self.assertGreater(result[1][timer][0][0], 0)  # host time in ns
        self.assertGreater(result[1][timer][0][1], 0)  # device time in ns

    def test_gather_result_no_recording(self):
        """Test gather_result when no recording is enabled."""
        timers = initialize_timers(
            record_host_time=False,
            record_device_time=False,
        )

        timer = CrystalTimer("no_record")
        timer.start()
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        timer.stop()

        result = timers.gather_result()
        self.assertEqual(result, {})

    def test_gather_result_multiple_executions(self):
        """Test gather_result with multiple timer executions."""
        timers = initialize_timers(
            record_host_time=True,
            record_device_time=True,
        )

        timer = CrystalTimer("multi_exec")
        for _ in range(3):
            timer.start()
            x = torch.randn(50, 50, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            timer.stop()

        result = timers.gather_result()
        self.assertEqual(len(result[1][timer]), 3)
        for i in range(3):
            self.assertGreater(result[1][timer][i][0], 0)
            self.assertGreater(result[1][timer][i][1], 0)


class TestLogHandler(unittest.TestCase):
    """Test LogHandler output."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_log_handler_writes_timer_results(self):
        """Test LogHandler forwards the timer results to its stream.

        The human-readable layout is deliberately not pinned down: only the
        timer's name reaching the output is checked, so the wording stays free
        to change.
        """
        stream = io.StringIO()
        handler = CrystalTimers.LogHandler(stream=stream)

        timers = initialize_timers(
            record_host_time=False,
            record_device_time=True,
            handlers=[handler],
        )

        timer = CrystalTimer("TEST_TIMER")
        timer.start()
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        timer.stop()

        timers.stop()

        self.assertIn("TEST_TIMER", stream.getvalue())

    def test_log_handler_with_none_stream(self):
        """Test LogHandler with None stream does nothing."""
        handler = CrystalTimers.LogHandler(stream=None)
        handler.print_summary({1: {}})
        # Should not raise any exception


class TestJSONHandler(unittest.TestCase):
    """Test JSONHandler output."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_json_handler_output_format(self):
        """Test JSONHandler output format matches documentation."""
        handler = CrystalTimers.JSONHandler(
            dir_name=self.temp_dir,
            worker_name="test_worker",
        )

        timers = initialize_timers(
            record_host_time=False,
            record_device_time=True,
            handlers=[handler],
        )

        timer = CrystalTimer("JSON_TEST")
        timer.start()
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        timer.stop()

        timers.stop()

        # Read and verify JSON output
        json_path = pathlib.Path(self.temp_dir) / "test_worker.json"
        self.assertTrue(json_path.exists())

        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)

        # Check format: {step: {timer_name: [[host_time, device_time], ...]}}
        self.assertIn("1", data)
        self.assertIn("JSON_TEST", data["1"])
        self.assertIsInstance(data["1"]["JSON_TEST"], list)
        self.assertEqual(len(data["1"]["JSON_TEST"]), 1)
        self.assertIsInstance(data["1"]["JSON_TEST"][0], list)
        self.assertEqual(len(data["1"]["JSON_TEST"][0]), 2)


class TestCondCrystalTimerRecord(unittest.TestCase):
    """Test cond_crystal_timer_record decorator and context manager."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()
        # Clear environment variable
        if "ENABLE_CRYSTAL_TIMER" in os.environ:
            del os.environ["ENABLE_CRYSTAL_TIMER"]

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "true"})
    def test_context_manager_with_env_enabled(self):
        """Test context manager when environment variable is enabled."""
        initialize_timers(record_host_time=True, record_device_time=True)

        with cond_crystal_timer_record("CTX_TEST"):
            x = torch.randn(50, 50, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()

        # Verify timer was recorded
        timers = CrystalTimers.instance()
        self.assertEqual(len(timers._timer_start_ts[1]), 1)

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": ""})
    def test_context_manager_with_env_disabled(self):
        """Test context manager when environment variable is disabled."""
        with cond_crystal_timer_record("CTX_DISABLED"):
            x = torch.randn(50, 50, device="cuda")
            y = torch.matmul(x, x)

        # Timer should be None when env is not set
        # No exception should be raised

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "true"})
    def test_decorator_with_env_enabled(self):
        """Test decorator when environment variable is enabled."""
        initialize_timers(record_host_time=True, record_device_time=True)

        @cond_crystal_timer_record("DEC_TEST")
        def test_function():
            x = torch.randn(50, 50, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            return y

        result = test_function()
        self.assertIsNotNone(result)

        # Verify timer was recorded
        timers = CrystalTimers.instance()
        self.assertEqual(len(timers._timer_start_ts[1]), 1)

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "false"})
    def test_decorator_with_env_disabled(self):
        """Test decorator when environment variable is disabled."""

        @cond_crystal_timer_record("DEC_DISABLED")
        def test_function():
            return 42

        result = test_function()
        self.assertEqual(result, 42)

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "1"})
    def test_env_var_with_value_1(self):
        """Test environment variable with value '1'."""
        initialize_timers(record_host_time=True, record_device_time=True)

        with cond_crystal_timer_record("ENV_1_TEST"):
            pass

        timers = CrystalTimers.instance()
        self.assertEqual(len(timers._timer_start_ts[1]), 1)


class TestCondCrystalTimerStartStop(unittest.TestCase):
    """Test cond_crystal_timer_start and cond_crystal_timer_stop functions."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()
        # Clear environment variable
        if "ENABLE_CRYSTAL_TIMER" in os.environ:
            del os.environ["ENABLE_CRYSTAL_TIMER"]

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "true"})
    def test_start_stop_with_env_enabled(self):
        """Test start/stop pair when environment variable is enabled."""
        initialize_timers(record_host_time=True, record_device_time=True)

        cond_crystal_timer_start("START_STOP_TEST")
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        cond_crystal_timer_stop("START_STOP_TEST")

        # Verify timer was recorded and closed
        timers = CrystalTimers.instance()
        self.assertEqual(len(timers._timer_start_ts[1]), 1)
        self.assertEqual(timers._timer_status["START_STOP_TEST"], CrystalTimers.Status.STOPPED)

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "true"})
    def test_start_stop_measures_host_duration(self):
        """Test start/stop pair produces a valid host duration."""
        import time

        timers = initialize_timers(record_host_time=True, record_device_time=False)

        sleep_duration = 0.05  # 50ms
        cond_crystal_timer_start("DURATION_TEST")
        time.sleep(sleep_duration)
        cond_crystal_timer_stop("DURATION_TEST")

        result = timers.gather_result()
        timer = CrystalTimers.get_timer("DURATION_TEST")
        measured_ms = result[1][timer][0][0] / 1_000_000

        # Allow tolerance for host time measurement
        self.assertGreater(measured_ms, sleep_duration * 1000 - 10)
        self.assertLess(measured_ms, sleep_duration * 1000 + 50)

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "false"})
    def test_start_stop_with_env_disabled(self):
        """Test start/stop pair when environment variable is disabled."""
        initialize_timers(record_host_time=True, record_device_time=True)

        cond_crystal_timer_start("START_STOP_DISABLED")
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        cond_crystal_timer_stop("START_STOP_DISABLED")

        # Nothing should be recorded and no timer should be created
        timers = CrystalTimers.instance()
        self.assertEqual(len(timers._timer_start_ts), 0)
        self.assertEqual(CrystalTimers._timers, {})

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "true"})
    def test_start_stop_without_initialization(self):
        """Test start/stop pair is a no-op when timers are not initialized."""
        cond_crystal_timer_start("UNINITIALIZED_TEST")
        cond_crystal_timer_stop("UNINITIALIZED_TEST")

        # No exception should be raised and no timer should be created
        self.assertIsNone(CrystalTimers.instance())
        self.assertEqual(CrystalTimers._timers, {})

    @mock.patch.dict(os.environ, {"ENABLE_CRYSTAL_TIMER": "true"})
    def test_start_stop_multiple_cycles(self):
        """Test repeated start/stop cycles on the same timer name."""
        timers = initialize_timers(record_host_time=True, record_device_time=True)

        for _ in range(2):
            cond_crystal_timer_start("CYCLE_TEST")
            x = torch.randn(50, 50, device="cuda")
            y = torch.matmul(x, x)
            cond_crystal_timer_stop("CYCLE_TEST")

        timer = CrystalTimers.get_timer("CYCLE_TEST")
        self.assertEqual(len(timers._timer_start_ts[1][timer]), 2)
        self.assertEqual(len(timers._timer_stop_ts[1][timer]), 2)


class TestStepRangeControl(unittest.TestCase):
    """Test step range control functionality."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_timer_only_records_in_range(self):
        """Test timer only records within specified step range."""
        timers = initialize_timers(
            record_host_time=True,
            record_device_time=True,
            step_start=2,
            step_end=4,
        )

        timer = CrystalTimer("range_test")

        # Step 1 - should not record
        timer.start()
        x = torch.randn(50, 50, device="cuda")
        y = torch.matmul(x, x)
        timer.stop()
        self.assertEqual(len(timers._timer_start_ts.get(1, {})), 0)

        # Step 2 - should record
        timers.step()
        timer.start()
        y = torch.matmul(x, x)
        timer.stop()
        self.assertEqual(len(timers._timer_start_ts[2][timer]), 1)

        # Step 3 - should record
        timers.step()
        timer.start()
        y = torch.matmul(x, x)
        timer.stop()
        self.assertEqual(len(timers._timer_start_ts[3][timer]), 1)

        # Step 4 - should not record
        timers.step()
        timer.start()
        y = torch.matmul(x, x)
        timer.stop()
        self.assertEqual(len(timers._timer_start_ts.get(4, {})), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_timer_with_special_characters_in_name(self):
        """Test timer with special characters in name."""
        timers = initialize_timers(record_host_time=True, record_device_time=True)

        special_names = ["timer-with-dash", "timer.with.dots", "timer/with/slashes", "timer_with_underscores"]
        for name in special_names:
            timer = CrystalTimer(name)
            timer.start()
            x = torch.randn(10, 10, device="cuda")
            y = torch.matmul(x, x)
            timer.stop()
            self.assertIn(timer, timers._timer_start_ts[1])

    def test_nested_timer_usage(self):
        """Test nested timer usage."""
        timers = initialize_timers(record_host_time=True, record_device_time=True)

        outer_timer = CrystalTimer("outer")
        inner_timer = CrystalTimer("inner")

        outer_timer.start()
        x = torch.randn(50, 50, device="cuda")

        inner_timer.start()
        y = torch.matmul(x, x)
        inner_timer.stop()

        z = torch.matmul(y, y)
        outer_timer.stop()

        # Both timers should have recorded data
        self.assertEqual(len(timers._timer_start_ts[1][outer_timer]), 1)
        self.assertEqual(len(timers._timer_start_ts[1][inner_timer]), 1)

    def test_callable_interface(self):
        """Test CrystalTimers callable interface."""
        timers = initialize_timers()

        # Test __call__ method
        timer1 = timers("callable_timer")
        self.assertIsInstance(timer1, CrystalTimer)
        self.assertEqual(timer1.name, "callable_timer")

        # Test that it returns the same instance
        timer2 = timers("callable_timer")
        self.assertIs(timer1, timer2)

    def test_get_and_start_class_method(self):
        """Test CrystalTimer.get_and_start class method."""
        timers = initialize_timers(record_host_time=True, record_device_time=True)

        timer = CrystalTimer.get_and_start("auto_start")
        self.assertIsInstance(timer, CrystalTimer)
        self.assertEqual(timer.name, "auto_start")
        self.assertEqual(timers._timer_status["auto_start"], CrystalTimers.Status.STARTED)

        timer.stop()


class TestTimerAccuracy(unittest.TestCase):
    """Test timer accuracy and performance impact."""

    def setUp(self):
        """Reset singleton before each test."""
        reset_timers_singleton()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def tearDown(self):
        """Clean up after each test."""
        reset_timers_singleton()

    def test_host_time_accuracy(self):
        """Test host time measurement accuracy."""
        import time

        timers = initialize_timers(
            record_host_time=True,
            record_device_time=False,
        )

        sleep_duration = 0.05  # 50ms
        timer = CrystalTimer("accuracy_test")

        timer.start()
        time.sleep(sleep_duration)
        timer.stop()

        result = timers.gather_result()
        measured_ns = result[1][timer][0][0]
        measured_ms = measured_ns / 1_000_000

        # Allow 10ms tolerance for host time measurement
        self.assertGreater(measured_ms, sleep_duration * 1000 - 10)
        self.assertLess(measured_ms, sleep_duration * 1000 + 50)

    def test_device_time_reasonable(self):
        """Test device time is within reasonable range."""
        timers = initialize_timers(
            record_host_time=False,
            record_device_time=True,
        )

        timer = CrystalTimer("device_reasonable")
        timer.start()

        # Perform some GPU work
        for _ in range(10):
            x = torch.randn(100, 100, device="cuda")
            y = torch.matmul(x, x)
        torch.cuda.synchronize()

        timer.stop()

        result = timers.gather_result()
        measured_ns = result[1][timer][0][1]
        measured_ms = measured_ns / 1_000_000

        # Device time should be positive and reasonable (less than 1 second for this work)
        self.assertGreater(measured_ns, 0)
        self.assertLess(measured_ms, 1000)


if __name__ == "__main__":
    unittest.main()
