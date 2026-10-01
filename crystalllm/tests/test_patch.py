# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for CrystalLLM Patch and ParameterSTE."""

import os
import threading
import time
import unittest

import torch

from crystalllm.nn import ParameterCopySTE
from crystalllm.patch import (
    BehaviorType,
    CrystalPatchPlan,
    CrystalPatchRequest,
    PatchType,
    cond_crystal_patch_request,
    crystal_patch_response,
)
from crystalllm.patch.patch import _get_active_requests

# Enum values for convenience
ARGS = PatchType.ARGS
KWARGS = PatchType.KWARGS
RES = PatchType.RES
ONLY_PATCH = BehaviorType.ONLY_PATCH
ONLY_EXEC = BehaviorType.ONLY_EXEC
EXEC_THEN_PATCH = BehaviorType.EXEC_THEN_PATCH


class BasePatchTestCase(unittest.TestCase):
    """Base test case with common setup and teardown for patch tests."""

    def setUp(self):
        """Set up test environment."""
        _get_active_requests().clear()

    def tearDown(self):
        """Clean up after tests."""
        _get_active_requests().clear()
        if "CRYSTAL_MODE" in os.environ:
            del os.environ["CRYSTAL_MODE"]


class TestCrystalPatchPlan(unittest.TestCase):
    """Test CrystalPatchPlan dataclass validation."""

    def test_basic_creation_and_all_patch_types(self):
        """Test basic creation and all patch types."""
        plan_args = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x * 2)
        plan_kwargs = CrystalPatchPlan(patch_type=KWARGS, position="key", action=lambda x: x + "_patched")
        plan_res = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x - 1)

        self.assertEqual(plan_args.patch_type, ARGS)
        self.assertEqual(plan_kwargs.patch_type, KWARGS)
        self.assertEqual(plan_res.patch_type, RES)
        self.assertEqual(plan_args.action(5), 10)

    def test_position_type_validation(self):
        """Test position type validation for different patch types."""
        # ARGS requires int position
        with self.assertRaises(TypeError) as context:
            CrystalPatchPlan(patch_type=ARGS, position="invalid", action=lambda x: x)
        self.assertIn("position must be int for ARGS", str(context.exception))

        # RES requires int position
        with self.assertRaises(TypeError) as context:
            CrystalPatchPlan(patch_type=RES, position="invalid", action=lambda x: x)
        self.assertIn("position must be int for RES", str(context.exception))

        # KWARGS requires str position
        with self.assertRaises(TypeError) as context:
            CrystalPatchPlan(patch_type=KWARGS, position=0, action=lambda x: x)
        self.assertIn("position must be str for KWARGS", str(context.exception))


class TestCrystalPatchRequest(unittest.TestCase):
    """Test CrystalPatchRequest dataclass validation."""

    def test_basic_creation_and_normalization(self):
        """Test basic creation and single plan normalization to list."""

        def dummy_func():
            return "dummy"

        plan = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x)

        # Test with list
        req_list = CrystalPatchRequest(
            patch_func=dummy_func,
            capture_behavior=ONLY_EXEC,
            emulate_behavior=ONLY_EXEC,
            patch_res_length=1,
            patch_plan=[plan],
        )
        self.assertIsInstance(req_list.patch_plan, list)

        # Test with single plan (should be normalized to list)
        req_single = CrystalPatchRequest(
            patch_func=dummy_func,
            capture_behavior=ONLY_EXEC,
            emulate_behavior=ONLY_EXEC,
            patch_res_length=1,
            patch_plan=plan,
        )
        self.assertIsInstance(req_single.patch_plan, list)
        self.assertEqual(len(req_single.patch_plan), 1)

    def test_validation_errors(self):
        """Test validation errors for empty plan and missing patch_res_length."""

        def dummy_func():
            return "dummy"

        plan = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x)

        # Empty patch_plan should raise ValueError
        with self.assertRaises(ValueError) as context:
            CrystalPatchRequest(
                patch_func=dummy_func,
                capture_behavior=ONLY_EXEC,
                emulate_behavior=ONLY_EXEC,
                patch_res_length=1,
                patch_plan=[],
            )
        self.assertIn("patch_plan cannot be empty", str(context.exception))

        # ONLY_PATCH in capture_behavior requires patch_res_length
        with self.assertRaises(ValueError) as context:
            CrystalPatchRequest(
                patch_func=dummy_func,
                capture_behavior=ONLY_PATCH,
                emulate_behavior=ONLY_EXEC,
                patch_res_length=None,
                patch_plan=[plan],
            )
        self.assertIn("patch_res_length must be a positive integer when ONLY_PATCH is used", str(context.exception))

        # ONLY_PATCH in emulate_behavior also requires patch_res_length
        with self.assertRaises(ValueError) as context:
            CrystalPatchRequest(
                patch_func=dummy_func,
                capture_behavior=ONLY_EXEC,
                emulate_behavior=ONLY_PATCH,
                patch_res_length=None,
                patch_plan=[plan],
            )
        self.assertIn("patch_res_length must be a positive integer when ONLY_PATCH is used", str(context.exception))

    def test_optional_callbacks(self):
        """Test optional callback functions."""

        def dummy_func():
            return "dummy"

        plan = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x)

        def condition():
            return True

        def action_callback():
            return None

        req = CrystalPatchRequest(
            patch_func=dummy_func,
            capture_behavior=ONLY_EXEC,
            emulate_behavior=ONLY_EXEC,
            patch_res_length=1,
            patch_plan=[plan],
            additional_condition=condition,
            additional_patch_actions=action_callback,
        )
        self.assertEqual(req.additional_condition, condition)
        self.assertEqual(req.additional_patch_actions, action_callback)


class TestCrystalPatchResponse(BasePatchTestCase):
    """Test crystal_patch_response decorator functionality."""

    def test_normal_execution_without_context(self):
        """Test that decorated function executes normally without context."""

        @crystal_patch_response
        def test_func(a, b):
            return a + b

        result = test_func(1, 2)
        self.assertEqual(result, 3)

    def test_behavior_types(self):
        """Test all behavior types: ONLY_EXEC, ONLY_PATCH, EXEC_THEN_PATCH."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func(a, b):
            return a + b

        # Test ONLY_EXEC - execute without patching
        plan_exec = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x * 10)
        req_exec = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_EXEC,
            emulate_behavior=ONLY_EXEC,
            patch_res_length=1,
            patch_plan=[plan_exec],
        )

        with cond_crystal_patch_request(request=[req_exec]):
            result = test_func(1, 2)
        self.assertEqual(result, 3)  # Should execute normally

        # Test ONLY_PATCH - return pre-generated None values
        plan_patch = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req_patch_single = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan_patch],
        )

        with cond_crystal_patch_request(request=[req_patch_single]):
            result = test_func(5, 3)
        self.assertIsNone(result)

        # Test ONLY_PATCH with multiple return values
        @crystal_patch_response
        def test_func_multi(a, b):
            return a, b

        req_patch_multi = CrystalPatchRequest(
            patch_func=test_func_multi,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=2,
            patch_plan=[plan_patch],
        )

        with cond_crystal_patch_request(request=[req_patch_multi]):
            result = test_func_multi(1, 2)
        self.assertEqual(result, (None, None))

        # Test EXEC_THEN_PATCH - execute then patch result
        plan_exec_patch = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x * 2)
        req_exec_patch = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan_exec_patch],
        )

        with cond_crystal_patch_request(request=[req_exec_patch]):
            result = test_func(3, 4)
        self.assertEqual(result, 14)  # Execute (3+4=7), then patch (7*2=14)

    def test_patch_args_and_kwargs(self):
        """Test patching positional and keyword arguments."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        # Test patching mutable ARGS
        @crystal_patch_response
        def test_func_list(data):
            return sum(data)

        plan_args = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x.append(100))
        req_args = CrystalPatchRequest(
            patch_func=test_func_list,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan_args],
        )

        data = [1, 2, 3]
        with cond_crystal_patch_request(request=[req_args]):
            result = test_func_list(data)

        self.assertEqual(result, 6)  # sum([1, 2, 3])
        self.assertEqual(data, [1, 2, 3, 100])  # Modified by patch action

        # Test patching mutable KWARGS
        @crystal_patch_response
        def test_func_kwargs(data=None):
            return sum(data) if data else 0

        plan_kwargs = CrystalPatchPlan(patch_type=KWARGS, position="data", action=lambda x: x.append(100))
        req_kwargs = CrystalPatchRequest(
            patch_func=test_func_kwargs,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan_kwargs],
        )

        data = [1, 2, 3]
        with cond_crystal_patch_request(request=[req_kwargs]):
            result = test_func_kwargs(data=data)

        self.assertEqual(result, 6)
        self.assertEqual(data, [1, 2, 3, 100])

    def test_patch_result_and_multiple_returns(self):
        """Test patching return values including multiple returns."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func(a, b, c=0):
            return a + b + c

        plan_res = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x + 100)
        req_res = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan_res],
        )

        with cond_crystal_patch_request(request=[req_res]):
            result = test_func(1, 2, c=3)
        self.assertEqual(result, 106)  # (1+2+3) + 100

        # Test multiple return values
        @crystal_patch_response
        def test_func_multi(a, b):
            return a, b, a + b

        plans_multi = [
            CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x * 10),
            CrystalPatchPlan(patch_type=RES, position=2, action=lambda x: x + 1),
        ]
        req_multi = CrystalPatchRequest(
            patch_func=test_func_multi,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=3,
            patch_plan=plans_multi,
        )

        with cond_crystal_patch_request(request=[req_multi]):
            result = test_func_multi(1, 2)
        self.assertEqual(result, (10, 2, 4))  # (1*10, 2, 3+1)


class TestCondCrystalPatchRequest(BasePatchTestCase):
    """Test cond_crystal_patch_request context manager."""

    def test_empty_request_raises(self):
        """Test that empty request list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            cond_crystal_patch_request(request=[])
        self.assertIn("request cannot be empty", str(context.exception))

    def test_nullcontext_behavior(self):
        """Test nullcontext behavior without or with invalid CRYSTAL_MODE."""

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        # Without CRYSTAL_MODE
        if "CRYSTAL_MODE" in os.environ:
            del os.environ["CRYSTAL_MODE"]

        with cond_crystal_patch_request(request=[req]):
            result = test_func()
        self.assertEqual(result, "executed")

        # With invalid CRYSTAL_MODE
        os.environ["CRYSTAL_MODE"] = "INVALID"
        with cond_crystal_patch_request(request=[req]):
            result = test_func()
        self.assertEqual(result, "executed")

    def test_crystal_modes(self):
        """Test CAPTURE and EMULATE modes use correct behaviors."""

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,  # Used in CAPTURE mode
            emulate_behavior=ONLY_EXEC,  # Used in EMULATE mode
            patch_res_length=1,
            patch_plan=[plan],
        )

        # CAPTURE mode should use ONLY_PATCH
        os.environ["CRYSTAL_MODE"] = "CAPTURE"
        with cond_crystal_patch_request(request=[req]):
            result = test_func()
        self.assertIsNone(result)

        # EMULATE mode should use ONLY_EXEC
        os.environ["CRYSTAL_MODE"] = "EMULATE"
        with cond_crystal_patch_request(request=[req]):
            result = test_func()
        self.assertEqual(result, "executed")

    def test_additional_condition(self):
        """Test additional_condition callback."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)

        # Condition returns False - should act as nullcontext
        req_false = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
            additional_condition=lambda: False,
        )

        with cond_crystal_patch_request(request=[req_false]):
            result = test_func()
        self.assertEqual(result, "executed")

        # Condition returns True - should enable patching
        req_true = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
            additional_condition=lambda: True,
        )

        with cond_crystal_patch_request(request=[req_true]):
            result = test_func()
        self.assertIsNone(result)

    def test_additional_patch_actions(self):
        """Test additional_patch_actions callback."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"
        action_called = []

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)

        def patch_action():
            action_called.append(True)

        # Should be called for ONLY_PATCH
        req_patch = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
            additional_patch_actions=patch_action,
        )

        with cond_crystal_patch_request(request=[req_patch]):
            test_func()
        self.assertEqual(len(action_called), 1)

        # Should not be called for ONLY_EXEC
        action_called.clear()
        req_exec = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_EXEC,
            emulate_behavior=ONLY_EXEC,
            patch_res_length=1,
            patch_plan=[plan],
            additional_patch_actions=patch_action,
        )

        with cond_crystal_patch_request(request=[req_exec]):
            test_func()
        self.assertEqual(len(action_called), 0)


class TestNestedContextManagers(BasePatchTestCase):
    """Test nested context managers."""

    def test_nested_context_managers(self):
        """Test that nested context managers work correctly."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func():
            return "executed"

        plan1 = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req1 = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan1],
        )

        plan2 = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req2 = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_EXEC,
            emulate_behavior=ONLY_EXEC,
            patch_res_length=1,
            patch_plan=[plan2],
        )

        with cond_crystal_patch_request(request=[req1]):
            result1 = test_func()
            self.assertIsNone(result1)

            with cond_crystal_patch_request(request=[req2]):
                result2 = test_func()
                self.assertEqual(result2, "executed")

            result3 = test_func()
            self.assertIsNone(result3)  # Should restore to outer context

    def test_nested_same_function_different_requests(self):
        """Test nested contexts with different requests for same function."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func(a):
            return a * 2

        req_outer = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)],
        )

        req_inner = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x + 100)],
        )

        with cond_crystal_patch_request(request=[req_outer]):
            result1 = test_func(5)
            self.assertIsNone(result1)

            with cond_crystal_patch_request(request=[req_inner]):
                result2 = test_func(5)
                self.assertEqual(result2, 110)  # (5*2) + 100

            result3 = test_func(5)
            self.assertIsNone(result3)


class TestThreadSafety(BasePatchTestCase):
    """Test thread safety of patch system."""

    def test_thread_isolation(self):
        """Test that patch requests are isolated between threads."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"
        results = []

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        def thread_with_patch():
            with cond_crystal_patch_request(request=[req]):
                time.sleep(0.01)
                results.append(("with_patch", test_func()))

        def thread_without_patch():
            time.sleep(0.005)
            results.append(("without_patch", test_func()))

        t1 = threading.Thread(target=thread_with_patch)
        t2 = threading.Thread(target=thread_without_patch)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        with_patch_results = [r for name, r in results if name == "with_patch"]
        without_patch_results = [r for name, r in results if name == "without_patch"]

        self.assertEqual(with_patch_results[0], None)
        self.assertEqual(without_patch_results[0], "executed")

    def test_concurrent_nested_contexts(self):
        """Test concurrent nested context managers in different threads."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"
        results = []

        @crystal_patch_response
        def test_func(x):
            return x * 2

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x + 10)
        req = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        def worker(thread_id, value):
            with cond_crystal_patch_request(request=[req]):
                time.sleep(0.01)
                result = test_func(value)
                results.append((thread_id, result))

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i, i + 1))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = {i: ((i + 1) * 2) + 10 for i in range(5)}
        actual = dict(results)
        self.assertEqual(actual, expected)


class TestEdgeCases(BasePatchTestCase):
    """Test edge cases and error conditions."""

    def test_function_not_in_request_executes_normally(self):
        """Test that functions not in request execute normally."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def patched_func():
            return "patched"

        @crystal_patch_response
        def unpatched_func():
            return "unpatched"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=patched_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        with cond_crystal_patch_request(request=[req]):
            result1 = patched_func()
            result2 = unpatched_func()

        self.assertIsNone(result1)
        self.assertEqual(result2, "unpatched")

    def test_single_request_not_list(self):
        """Test that single request can be passed without list."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        with cond_crystal_patch_request(request=req):  # Single request, not list
            result = test_func()

        self.assertIsNone(result)

    def test_index_and_key_errors(self):
        """Test that IndexError and KeyError are raised for invalid positions."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func_no_args():
            return "executed"

        # ARGS position out of range
        plan_args = CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x)
        req_args = CrystalPatchRequest(
            patch_func=test_func_no_args,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan_args],
        )

        with cond_crystal_patch_request(request=[req_args]), self.assertRaises(IndexError):
            test_func_no_args()

        @crystal_patch_response
        def test_func_single_return():
            return "executed"

        # RES position out of range
        plan_res = CrystalPatchPlan(patch_type=RES, position=1, action=lambda x: x)
        req_res = CrystalPatchRequest(
            patch_func=test_func_single_return,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan_res],
        )

        with cond_crystal_patch_request(request=[req_res]), self.assertRaises(IndexError):
            test_func_single_return()

        @crystal_patch_response
        def test_func_no_kwargs():
            return "executed"

        # KWARGS key doesn't exist
        plan_kwargs = CrystalPatchPlan(patch_type=KWARGS, position="nonexistent", action=lambda x: x)
        req_kwargs = CrystalPatchRequest(
            patch_func=test_func_no_kwargs,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan_kwargs],
        )

        with cond_crystal_patch_request(request=[req_kwargs]), self.assertRaises(KeyError):
            test_func_no_kwargs()

    def test_context_exit_restores_state(self):
        """Test that context exit properly restores state."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def test_func():
            return "executed"

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=test_func,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        # Before context
        result1 = test_func()
        self.assertEqual(result1, "executed")

        # Inside context
        with cond_crystal_patch_request(request=[req]):
            result2 = test_func()
            self.assertIsNone(result2)

        # After context
        result3 = test_func()
        self.assertEqual(result3, "executed")


class TestParameterSTE(unittest.TestCase):
    """Test ParameterSTE class functionality."""

    def test_forward_without_functions(self):
        """Test forward pass without any transformation functions."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ParameterCopySTE.apply(x, None, None)

        self.assertTrue(torch.equal(y, x))
        self.assertIsNotNone(y.grad_fn)

        # Test backward
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_forward_with_output_func(self):
        """Test forward pass with output_tensor_func."""

        def output_func(t):
            return t * 2

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ParameterCopySTE.apply(x, output_func, None)

        expected = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.equal(y, expected))

        # Test backward (should pass gradient through unchanged)
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_backward_with_grad_func(self):
        """Test backward pass with grad_input_func."""

        def grad_func(g):
            return g * 0.5

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ParameterCopySTE.apply(x, None, grad_func)

        # Use explicit gradient to avoid broadcasted tensor memory issue with copy_
        y.backward(torch.ones_like(y))

        # Gradient should be scaled by 0.5
        expected_grad = torch.tensor([0.5, 0.5, 0.5])
        self.assertTrue(torch.equal(x.grad, expected_grad))

    def test_forward_backward_with_both_funcs(self):
        """Test forward and backward with both transformation functions."""

        def output_func(t):
            return t + 10

        def grad_func(g):
            return g * 2

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ParameterCopySTE.apply(x, output_func, grad_func)

        # Forward: [1, 2, 3] + 10 = [11, 12, 13]
        expected = torch.tensor([11.0, 12.0, 13.0])
        self.assertTrue(torch.equal(y, expected))

        # Use explicit gradient to avoid broadcasted tensor memory issue with copy_
        y.backward(torch.ones_like(y))

        # Backward: gradient [1, 1, 1] * 2 = [2, 2, 2]
        expected_grad = torch.tensor([2.0, 2.0, 2.0])
        self.assertTrue(torch.equal(x.grad, expected_grad))

    def test_shape_validation_forward(self):
        """Test shape validation in forward pass."""

        def bad_output_func(t):
            return t.view(-1)  # Changes shape

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        with self.assertRaises(AssertionError) as context:
            ParameterCopySTE.apply(x, bad_output_func, None)

        self.assertIn("shape mismatch", str(context.exception))

    def test_dtype_validation_forward(self):
        """Test dtype validation in forward pass."""

        def bad_output_func(t):
            return t.to(torch.float64)  # Changes dtype

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)

        with self.assertRaises(AssertionError) as context:
            ParameterCopySTE.apply(x, bad_output_func, None)

        self.assertIn("dtype mismatch", str(context.exception))

    def test_shape_validation_backward(self):
        """Test shape validation in backward pass."""

        def bad_grad_func(g):
            return g.view(-1)  # Changes shape

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = ParameterCopySTE.apply(x, None, bad_grad_func)

        with self.assertRaises(AssertionError) as context:
            y.sum().backward()

        self.assertIn("shape mismatch", str(context.exception))

    def test_dtype_validation_backward(self):
        """Test dtype validation in backward pass."""

        def bad_grad_func(g):
            return g.to(torch.float64)  # Changes dtype

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        y = ParameterCopySTE.apply(x, None, bad_grad_func)

        with self.assertRaises(AssertionError) as context:
            y.sum().backward()

        self.assertIn("dtype mismatch", str(context.exception))

    def test_device_consistency(self):
        """Test that output tensor is on the same device as input."""
        # This test assumes CPU. For CUDA, additional checks would be needed.
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        def output_func(t):
            # Return tensor that might be on different device in real scenarios
            return t.clone()

        y = ParameterCopySTE.apply(x, output_func, None)

        self.assertEqual(y.device, x.device)

    def test_complex_gradient_flow(self):
        """Test complex gradient flow through multiple operations."""

        def output_func(t):
            return t * 2

        def grad_func(g):
            return g * 0.5

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Chain of operations
        y = ParameterCopySTE.apply(x, output_func, grad_func)
        z = y * 3
        w = z.sum()

        w.backward()

        # Forward: x * 2 * 3 = x * 6
        # Backward: grad_output = 1 (from sum)
        #           grad_z = 3 (from multiplication by 3)
        #           grad_y = grad_func(grad_z) = 3 * 0.5 = 1.5
        # Note: STE is a straight-through estimator, so grad_func directly
        # transforms the gradient. The gradient flow is:
        #   dL/dx = grad_func(dL/dz * dz/dy) = grad_func(3 * 1) = 1.5
        expected_grad = torch.tensor([1.5, 1.5, 1.5])
        self.assertTrue(torch.allclose(x.grad, expected_grad))


class TestParameterSTEWithPatch(BasePatchTestCase):
    """Integration tests between ParameterSTE and patch system."""

    def test_ste_inside_patched_function(self):
        """Test ParameterSTE used inside a patched function."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        def output_func(t):
            return t * 2

        @crystal_patch_response
        def compute_with_ste(x):
            # Use ParameterSTE inside the function
            y = ParameterCopySTE.apply(x, output_func, None)
            return y.sum()

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda r: r + 100)
        req = CrystalPatchRequest(
            patch_func=compute_with_ste,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        with cond_crystal_patch_request(request=[req]):
            result = compute_with_ste(x)

        # Forward: x * 2 -> sum = 12, then + 100 = 112
        self.assertEqual(result.item(), 112.0)

        # Test backward
        result.backward()

        # Gradient flows through STE and patch
        # grad_output = 1 (from scalar)
        # Through patch: unchanged (just adds constant)
        # Through STE: grad_input_func is None, so gradient passes through unchanged
        # STE forward applies output_func, but backward just passes gradient through
        # when grad_input_func is None
        # Final gradient should be 1.0 (from sum) for each element
        expected_grad = torch.tensor([1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_ste_with_patch_result(self):
        """Test ParameterSTE with patched result transformation."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def ste_operation(x):
            return ParameterCopySTE.apply(x, lambda t: t + 10, None)

        # Patch the result to multiply by 2
        plan = CrystalPatchPlan(
            patch_type=RES, position=0, action=lambda t: t * 2 if isinstance(t, torch.Tensor) else t
        )
        req = CrystalPatchRequest(
            patch_func=ste_operation,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        with cond_crystal_patch_request(request=[req]):
            result = ste_operation(x)

        # Forward: x + 10 = [11, 12, 13], then * 2 = [22, 24, 26]
        expected = torch.tensor([22.0, 24.0, 26.0])
        self.assertTrue(torch.equal(result, expected))

    def test_ste_with_only_patch_behavior(self):
        """Test ParameterSTE when function uses ONLY_PATCH behavior."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        @crystal_patch_response
        def ste_operation(x):
            # This should not execute when ONLY_PATCH is used
            return ParameterCopySTE.apply(x, lambda t: t * 100, None)

        plan = CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x)
        req = CrystalPatchRequest(
            patch_func=ste_operation,
            capture_behavior=ONLY_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        x = torch.tensor([1.0, 2.0, 3.0])

        with cond_crystal_patch_request(request=[req]):
            result = ste_operation(x)

        # ONLY_PATCH returns None, function body doesn't execute
        self.assertIsNone(result)

    def test_ste_args_patching(self):
        """Test patching tensor arguments with ParameterSTE."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        patched_tensors = []

        def patch_action(t):
            if isinstance(t, torch.Tensor):
                patched_tensors.append(t.clone())

        @crystal_patch_response
        def ste_with_tensor(x):
            y = ParameterCopySTE.apply(x, lambda t: t * 2, None)
            return y.sum()

        # Patch the tensor argument
        plan = CrystalPatchPlan(patch_type=ARGS, position=0, action=patch_action)
        req = CrystalPatchRequest(
            patch_func=ste_with_tensor,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=EXEC_THEN_PATCH,
            patch_res_length=1,
            patch_plan=[plan],
        )

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        with cond_crystal_patch_request(request=[req]):
            result = ste_with_tensor(x)

        # Function executes with original tensor
        self.assertEqual(result.item(), 12.0)  # sum([1,2,3] * 2)

        # Patch action should have recorded the tensor
        self.assertEqual(len(patched_tensors), 1)
        self.assertTrue(torch.equal(patched_tensors[0], x))

    def test_ste_integration_full_workflow(self):
        """Test full workflow combining STE, patching, and gradient computation."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"

        execution_log = []

        def on_patch():
            execution_log.append("patch_applied")

        @crystal_patch_response
        def complex_operation(x, scale=1.0):
            execution_log.append("function_executed")

            # STE with forward and backward transforms
            # Note: torch.autograd.Function.apply() does not accept keyword arguments
            # Return y directly instead of y.sum() to avoid broadcasted gradient issue
            return ParameterCopySTE.apply(x, lambda t: t * scale, lambda g: g * 0.5)

        plans = [
            CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda t: execution_log.append("arg_patched")),
            CrystalPatchPlan(
                patch_type=KWARGS, position="scale", action=lambda s: execution_log.append("kwarg_patched")
            ),
            CrystalPatchPlan(patch_type=RES, position=0, action=lambda r: r + 10 if isinstance(r, torch.Tensor) else r),
        ]

        req = CrystalPatchRequest(
            patch_func=complex_operation,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=1,
            patch_plan=plans,
            additional_patch_actions=on_patch,
        )

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Test in CAPTURE mode (EXEC_THEN_PATCH)
        with cond_crystal_patch_request(request=[req]):
            result = complex_operation(x, scale=2.0)

        # Verify execution order
        self.assertIn("function_executed", execution_log)
        self.assertIn("patch_applied", execution_log)

        # Forward: x * 2.0 = [2, 4, 6], + 10 = [12, 14, 16]
        expected_result = torch.tensor([12.0, 14.0, 16.0])
        self.assertTrue(torch.equal(result, expected_result))

        # Backward
        result.backward(torch.ones_like(result))

        # Gradient: grad_output = 1
        # Through result patch: unchanged (adds constant)
        # Through STE backward: grad_func scales by 0.5
        # Note: STE is a straight-through estimator, grad_func directly transforms
        # the incoming gradient without considering the forward transformation
        # grad_x = 1 * 0.5 = 0.5 for each element
        expected_grad = torch.tensor([0.5, 0.5, 0.5])
        self.assertTrue(torch.allclose(x.grad, expected_grad))

        # Test in EMULATE mode (ONLY_PATCH)
        execution_log.clear()
        os.environ["CRYSTAL_MODE"] = "EMULATE"

        with cond_crystal_patch_request(request=[req]):
            result = complex_operation(x, scale=2.0)

        # ONLY_PATCH: function doesn't execute, returns None
        self.assertIsNone(result)
        self.assertNotIn("function_executed", execution_log)
        self.assertIn("patch_applied", execution_log)


class TestIntegration(BasePatchTestCase):
    """Integration tests combining multiple patch features."""

    def test_full_workflow(self):
        """Test a complete workflow with all features."""
        os.environ["CRYSTAL_MODE"] = "CAPTURE"
        patch_actions_log = []

        def on_patch():
            patch_actions_log.append("patch_applied")

        @crystal_patch_response
        def compute(a, b, multiplier=1):
            return (a + b) * multiplier, a * b

        plans = [
            CrystalPatchPlan(patch_type=ARGS, position=0, action=lambda x: x + 10),
            CrystalPatchPlan(patch_type=ARGS, position=1, action=lambda x: x + 20),
            CrystalPatchPlan(patch_type=KWARGS, position="multiplier", action=lambda x: x * 2),
            CrystalPatchPlan(patch_type=RES, position=0, action=lambda x: x + 1000 if x is not None else None),
            CrystalPatchPlan(patch_type=RES, position=1, action=lambda x: x + 100 if x is not None else None),
        ]

        req = CrystalPatchRequest(
            patch_func=compute,
            capture_behavior=EXEC_THEN_PATCH,
            emulate_behavior=ONLY_PATCH,
            patch_res_length=2,
            patch_plan=plans,
            additional_condition=lambda: True,
            additional_patch_actions=on_patch,
        )

        # Test in CAPTURE mode (EXEC_THEN_PATCH)
        with cond_crystal_patch_request(request=[req]):
            sum_result, product_result = compute(1, 2, multiplier=3)

        # Function executes: (1+2)*3=9, 1*2=2
        # Results patched: 9+1000=1009, 2+100=102
        self.assertEqual(sum_result, 1009)
        self.assertEqual(product_result, 102)
        self.assertEqual(len(patch_actions_log), 1)

        # Test in EMULATE mode (ONLY_PATCH)
        patch_actions_log.clear()
        os.environ["CRYSTAL_MODE"] = "EMULATE"

        with cond_crystal_patch_request(request=[req]):
            sum_result, product_result = compute(100, 200, multiplier=50)

        # Should return Nones
        self.assertIsNone(sum_result)
        self.assertIsNone(product_result)
        self.assertEqual(len(patch_actions_log), 1)


if __name__ == "__main__":
    unittest.main()
