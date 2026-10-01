# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Parameterized Straight-Through Estimators (STE)."""

from collections.abc import Callable

import torch


class ParameterSTE(torch.autograd.Function):
    """Generic Parameterized Straight-Through Estimator with standard tensor operations.

    This class implements a custom autograd function that allows flexible
    transformations of input tensors during forward pass and gradient tensors
    during backward pass through user-provided functions.

    The forward pass can optionally transform the input tensor using a provided
    function, while the backward pass can optionally transform the gradient
    using another provided function. This enables controlled gradient flow
    and custom differentiation behavior.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor | None,
        output_tensor_func: Callable[[torch.Tensor | None], torch.Tensor | None] = None,
        grad_input_func: Callable[[torch.Tensor | None, torch.Tensor | None], torch.Tensor | None] = None,
    ):
        """Forward pass with optional tensor transformation using standard operations.

        Args:
            ctx: Autograd context for saving information for backward pass.
            input_tensor: The input tensor to be processed. Can be None.
            output_tensor_func: Optional function to transform input_tensor.
                If provided, the function is applied to input_tensor and the
                result is returned after shape/dtype/device validation.
                If None, input_tensor is returned unchanged.
            grad_input_func: Optional function to transform gradients during
                backward pass. This is saved in ctx for later use.

        Returns:
            torch.Tensor | None: Either input_tensor (if output_tensor_func is None)
                or the transformed tensor (with same shape/dtype/device as input_tensor).
                Returns None if input_tensor is None and output_tensor_func is None.

        Raises:
            AssertionError: If output_tensor shape, dtype, or device does not match input_tensor.
        """
        # Save grad_input_func for backward pass
        ctx.input_tensor = input_tensor
        ctx.grad_input_func = grad_input_func

        if output_tensor_func is not None:
            func_output_tensor = output_tensor_func(input_tensor)
            if func_output_tensor is not None and input_tensor is not None:
                assert func_output_tensor.device == input_tensor.device, (
                    f"Output tensor device mismatch: {func_output_tensor.device} vs {input_tensor.device}"
                )
            output_tensor = func_output_tensor
        else:
            output_tensor = input_tensor

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None):
        """Backward pass with optional gradient transformation using standard operations.

        The gradient flow is controlled entirely by grad_input_func. If provided,
        the gradient is transformed; otherwise, grad_output is passed through unchanged.

        Args:
            ctx: Autograd context containing saved grad_input_func and input_tensor.
            grad_output: The gradient tensor from the output. Can be None.

        Returns:
            tuple: A tuple containing:
                - grad_input: The gradient for input_tensor (transformed if
                  grad_input_func is provided, otherwise grad_output unchanged)
                - None: Gradient for output_tensor_func (always None)
                - None: Gradient for grad_input_func (always None)

        Raises:
            AssertionError: If grad_input shape, dtype, or device does not match grad_output.
        """
        input_tensor: torch.Tensor | None = ctx.input_tensor
        grad_input_func: Callable[[torch.Tensor | None, torch.Tensor | None], torch.Tensor | None] = ctx.grad_input_func

        if grad_input_func is not None:
            func_grad_input = grad_input_func(input_tensor, grad_output)
            if func_grad_input is not None and grad_output is not None:
                assert func_grad_input.device == grad_output.device, (
                    f"Gradient input device mismatch: {func_grad_input.device} vs {grad_output.device}"
                )
            grad_input = func_grad_input
        else:
            grad_input = grad_output

        # Return gradients: grad_input for input_tensor, None for both function parameters
        return grad_input, None, None


class ParameterCopySTE(torch.autograd.Function):
    """Generic Parameterized Straight-Through Estimator using in-place copy_() operations.

    This class implements a custom autograd function that allows flexible
    transformations of input tensors during forward pass and gradient tensors
    during backward pass through user-provided functions.

    The forward pass can optionally transform the input tensor using a provided
    function, while the backward pass can optionally transform the gradient
    using another provided function. This enables controlled gradient flow
    and custom differentiation behavior.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        output_tensor_func: Callable[[torch.Tensor], torch.Tensor] = None,
        grad_input_func: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """Forward pass with optional in-place tensor transformation.

        Uses copy_() to modify the input tensor in-place when output_tensor_func
        is provided. The original input_tensor is modified and returned.

        Args:
            ctx: Autograd context for saving information for backward pass.
            input_tensor: The input tensor to be processed. Must NOT be None.
            output_tensor_func: Optional function to transform input_tensor.
                If provided, the function is applied to input_tensor and the
                result is copied back to input_tensor using copy_().
                If None, input_tensor is returned unchanged.
            grad_input_func: Optional function to transform gradients during
                backward pass. This is saved in ctx for later use.

        Returns:
            torch.Tensor: The input_tensor (modified in-place if output_tensor_func
                is provided, otherwise unchanged).

        Raises:
            AssertionError: If input_tensor is None, or if output_tensor shape or dtype
                does not match input_tensor.
        """
        # Save grad_input_func for backward pass
        ctx.grad_input_func = grad_input_func

        assert input_tensor is not None

        if output_tensor_func is not None:
            func_output_tensor = output_tensor_func(input_tensor)
            assert func_output_tensor is not None
            assert func_output_tensor.shape == input_tensor.shape, (
                f"Output tensor shape mismatch: {func_output_tensor.shape} vs {input_tensor.shape}"
            )
            assert func_output_tensor.dtype == input_tensor.dtype, (
                f"Output tensor dtype mismatch: {func_output_tensor.dtype} vs {input_tensor.dtype}"
            )
            output_tensor = input_tensor.copy_(func_output_tensor, non_blocking=True)
        else:
            output_tensor = input_tensor

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass with optional in-place gradient transformation.

        Uses copy_() to modify the gradient tensor in-place when grad_input_func
        is provided. The original grad_output is modified and returned.

        The gradient flow is controlled entirely by grad_input_func. If provided,
        the gradient is transformed in-place; otherwise, grad_output is passed
        through unchanged.

        Args:
            ctx: Autograd context containing saved grad_input_func.
            grad_output: The gradient tensor from the output. Must NOT be None.

        Returns:
            tuple: A tuple containing:
                - grad_input: The gradient for input_tensor (transformed in-place if
                  grad_input_func is provided, otherwise grad_output unchanged)
                - None: Gradient for output_tensor_func (always None)
                - None: Gradient for grad_input_func (always None)

        Raises:
            AssertionError: If grad_output is None, or if grad_input shape or dtype
                does not match grad_output.
        """
        grad_input_func: Callable[[torch.Tensor], torch.Tensor] = ctx.grad_input_func

        assert grad_output is not None

        if grad_input_func is not None:
            func_grad_input = grad_input_func(grad_output)
            assert func_grad_input is not None
            assert func_grad_input.shape == grad_output.shape, (
                f"Gradient input shape mismatch: {func_grad_input.shape} vs {grad_output.shape}"
            )
            assert func_grad_input.dtype == grad_output.dtype, (
                f"Gradient input dtype mismatch: {func_grad_input.dtype} vs {grad_output.dtype}"
            )
            grad_input = grad_output.copy_(func_grad_input, non_blocking=True)
        else:
            grad_input = grad_output

        # Return gradients: grad_input for input_tensor, None for both function parameters
        return grad_input, None, None
