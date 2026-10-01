# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Mock Router Module for MoE (Mixture of Experts).

This module provides functionality to simulate specific routing distributions
using ParameterCopySTE for MoE router gating.
It allows controlling the exact token distribution across devices and experts
for benchmarking and analysis purposes.
"""

import functools
import itertools

import torch

from ..nn import ParameterCopySTE
from ..util import print_rank_0


class MockRouter:
    """Mock router for MoE with controlled routing distributions.

    This class encapsulates the state and logic for simulating specific routing
    patterns in MoE models. It pre-computes routing logits based on desired
    balance ratios and expert distributions, then uses ParameterCopySTE to inject
    these logits during the forward pass.

    Attributes:
        _logits_dict: Dictionary mapping gating_index to pre-computed logits.
    """

    def __init__(self) -> None:
        """Initialize a new MockRouter instance.

        Creates empty logits dictionary.
        """
        self._logits_dict: dict[int, torch.Tensor] | None = None

    def is_initialized(self) -> bool:
        """Check if the mock router has been initialized.

        Returns:
            bool: True if logits_dict is set, False otherwise.
        """
        return self._logits_dict is not None

    def clear(self) -> None:
        """Clear the mock router state.

        Resets the logits dictionary to None, effectively uninitializing the router.
        This can be used to free memory or reset the state for re-initialization.
        """
        self._logits_dict = None

    def initialize(
        self,
        num_experts: int,
        ep_size: int,
        br_list_part: list[list[float]],
        f_avg_list_part: list[list[float]],
        routed_experts_per_token: int,
        local_tokens_per_device: int,
        gating_count: int,
        micro_batch_size: int,
        logits_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the mock router with pre-computed routing logits.

        This method builds the balance ratio list and expert distribution list,
        then computes the routing logits for each gating operation.

        Args:
            num_experts: Total number of experts in the model.
            ep_size: Expert model parallelism world size.
            br_list_part: Partial balance ratio list for specified devices.
            f_avg_list_part: Expert distribution percentages (f10, f20, f50) per device.
            routed_experts_per_token: Number of experts to route each token to (top-k).
            local_tokens_per_device: Number of tokens processed locally per device.
            gating_count: Total number of gating operations.
            micro_batch_size: Micro batch size.
            logits_dtype: Logtis data type (default: bfloat16).

        Raises:
            AssertionError: If input parameters are invalid.
        """
        # Build br_list
        d = len(br_list_part) if br_list_part else 0
        assert ep_size >= d, f"The size of br_list_part {d} is greater than ep_size"
        br_len = gating_count
        assert all(len(br_list_part[i]) == br_len for i in range(d)), (
            f"The shape of br_list_part is invalid {br_list_part}"
        )

        # Convert br_list_part to tensor and pad with zeros if needed
        if d > 0:
            br_tensor = torch.tensor(br_list_part, dtype=torch.float)
            # Pad with zeros if needed
            if d < ep_size:
                # Calculate remaining tokens, for each position j, the sum should be ep_size
                # Sum across devices for each position
                sum_existing = torch.sum(br_tensor, dim=0)
                # Remaining balance ratio for unspecified devices
                remaining_br = ep_size - sum_existing
                remaining_br_per_device = remaining_br / (ep_size - d)
                # Create the full tensor with the right size
                br_list = torch.cat([br_tensor, remaining_br_per_device.expand(ep_size - d, br_len)], dim=0)
            else:
                br_list = br_tensor
        else:
            # If br_list_part is empty, create uniform distribution
            br_list = torch.full((ep_size, br_len), 1.0, dtype=torch.float)

        br_list = br_list.reshape((ep_size, gating_count)).permute(1, 0)
        print_rank_0("[MockRouter] Balance ratio list")
        print_rank_0(br_list)

        # Build f_list
        num_local_experts = num_experts // ep_size
        assert num_local_experts > 0, "The amount of local experts should be greater than 0."

        d = len(f_avg_list_part) if f_avg_list_part else 0
        assert ep_size >= d, f"The size of f_avg_list_part {d} is greater than ep_size"
        assert all(len(f_avg_list_part[i]) == 3 for i in range(d)), (  # noqa: PLR2004
            f"The shape of f_avg_list_part is invalid {f_avg_list_part}"
        )

        f_list = torch.zeros((ep_size, num_local_experts), dtype=torch.float)

        for k in range(ep_size):
            if k < d:
                # Use f_avg to construct step-shaped CDF
                f_10_avg, f_20_avg, f_50_avg = f_avg_list_part[k]
                assert 0.0 <= f_10_avg <= f_20_avg <= f_50_avg <= 100.0, (  # noqa: PLR2004
                    f"f_avg_list[{k}] = ({f_10_avg}, {f_20_avg}, {f_50_avg}) is invalid"
                )

                experts_10_percent = num_local_experts // 10
                experts_20_percent = num_local_experts // 5
                experts_50_percent = num_local_experts // 2

                remaining_ratio = 100.0

                # First 10% experts
                if experts_10_percent > 0:
                    ratio_per_expert = f_10_avg / experts_10_percent
                    f_list[k][:experts_10_percent] = ratio_per_expert
                    remaining_ratio -= f_10_avg

                # 10%-20% experts
                experts_20_only = experts_20_percent - experts_10_percent
                ratio_20_only = f_20_avg - f_10_avg
                if experts_20_only > 0:
                    ratio_per_expert = ratio_20_only / experts_20_only
                    f_list[k][experts_10_percent:experts_20_percent] = ratio_per_expert
                    remaining_ratio -= ratio_20_only

                # 20%-50% experts
                experts_50_only = experts_50_percent - experts_20_percent
                ratio_50_only = f_50_avg - f_20_avg
                if experts_50_only > 0:
                    ratio_per_expert = ratio_50_only / experts_50_only
                    f_list[k][experts_20_percent:experts_50_percent] = ratio_per_expert
                    remaining_ratio -= ratio_50_only

                # Remaining experts
                experts_remaining = num_local_experts - experts_50_percent
                ratio_remaining = remaining_ratio / experts_remaining
                f_list[k][experts_50_percent:] = ratio_remaining
            else:
                avg_ratio = 100.0 / num_local_experts
                f_list[k] = torch.full((num_local_experts,), avg_ratio, dtype=torch.float)

        print_rank_0("[MockRouter] Percentage of tokens received by local experts sorted by index")
        print_rank_0(f_list)
        f_list /= 100.0

        all_tokens_to_dispatch = local_tokens_per_device * routed_experts_per_token

        def floor_divide_and_guarantee_sum(all_val: int, ratio: torch.Tensor) -> torch.Tensor:
            """Floor divide while guaranteeing the sum equals all_val."""
            assert ratio.nelement() > 0
            result = torch.floor(ratio * all_val).view(-1)
            for i in range(int(all_val - torch.sum(result))):
                result[i] += 1
            assert torch.sum(result) == all_val
            return result.view(ratio.shape)

        def get_logits(gating_idx: int) -> torch.Tensor:
            """Compute routing logits for a specific gating operation."""
            br_per_device = br_list[gating_idx] / ep_size
            tokens_per_device = floor_divide_and_guarantee_sum(all_tokens_to_dispatch, br_per_device)
            assert torch.sum(tokens_per_device) == all_tokens_to_dispatch, tokens_per_device

            print_rank_0(f"[MockRouter] gating_idx: {gating_idx}", tokens_per_device)

            tokens_per_experts = (
                torch
                .stack([floor_divide_and_guarantee_sum(tokens_per_device[d], f_list[d]) for d in range(ep_size)])
                .view(-1)
                .int()
            )
            assert torch.sum(tokens_per_experts) == all_tokens_to_dispatch, tokens_per_experts

            t_expert = [[] for _ in range(local_tokens_per_device)]
            token_iter = itertools.cycle(range(local_tokens_per_device))
            for e in range(num_experts):
                for _ in range(tokens_per_experts[e]):
                    t_expert[next(token_iter)].append(e)
            assert all(len(set(t_expert[i])) == routed_experts_per_token for i in range(local_tokens_per_device)), (
                t_expert
            )

            local_sequence_length = local_tokens_per_device // micro_batch_size
            t_expert_seq_batch = torch.tensor(t_expert).reshape(
                local_sequence_length, micro_batch_size, routed_experts_per_token
            )
            parameter_logits = torch.empty(
                (local_sequence_length, micro_batch_size, num_experts),
                device="cpu",
                dtype=logits_dtype,
                pin_memory=True,
            ).fill_(-1.0)
            for s in range(local_sequence_length):
                for b in range(micro_batch_size):
                    for j in range(routed_experts_per_token):
                        parameter_logits[s][b][t_expert_seq_batch[s][b][j]] = 1.0
            return parameter_logits

        # Compute logits for all gating operations using single-dimensional index
        logits_dict: dict[int, torch.Tensor] = {}
        for gating_idx in range(gating_count):
            logits_dict[gating_idx] = get_logits(gating_idx)

        self._logits_dict = logits_dict

    def apply_mock_logits(
        self,
        logits: torch.Tensor,
        gating_index: int,
    ) -> torch.Tensor:
        """Apply pre-computed parameter logits to the input logits.

        This method uses ParameterCopySTE to replace the input logits with
        pre-computed values from the internal dictionary.

        Args:
            logits: Original router logits from the model.
            gating_index: Single-dimensional index for gating operation.

        Returns:
            torch.Tensor: Modified logits from the pre-computed dictionary.

        Raises:
            RuntimeError: If the mock router has not been initialized.
        """
        if not self.is_initialized():
            raise RuntimeError("MockRouter has not been initialized. Call initialize() first.")

        output_func = functools.partial(lambda gating_idx, in_tensor: self._logits_dict[gating_idx], gating_index)
        return ParameterCopySTE.apply(logits, output_func, None)


def calculate_br_list_from_statistics(
    gating_count: int,
    objective: str,
    br_min: float,
    br_max: float,
    br_avg: float | None = None,
    br_std: float | None = None,
    br_med: float | None = None,
    br_skew: float | None = None,
    num_of_local_moe_layers: int | None = None,
    pp_size: int | None = None,
) -> list[float]:
    """Calculate balance ratio list from statistical parameters.

    This function uses optimization to generate a sequence of balance ratios
    that match the specified statistical properties (mean, std, median, skewness).

    Args:
        gating_count: Number of gating operations in one iteration.
        objective: Optimization objective ('smooth', 'smooth_shuffle', or 'max_1f1b').
        br_min: Minimum balance ratio.
        br_max: Maximum balance ratio.
        br_avg: Target average balance ratio (optional).
        br_std: Target standard deviation (optional).
        br_med: Target median (optional).
        br_skew: Target skewness (optional).
        num_of_local_moe_layers: Number of local MoE layers (for max_1f1b objective).
        pp_size: Pipeline model parallelism world size (for max_1f1b objective).

    Returns:
        List of balance ratios satisfying the statistical constraints.

    Raises:
        AssertionError: If the optimization fails to converge.
    """
    import numpy as np
    from scipy.optimize import minimize

    assert objective in {"smooth_shuffle", "max_1f1b", "smooth"}
    if objective == "max_1f1b":
        assert num_of_local_moe_layers is not None
        assert pp_size is not None

    def solve_sequence(
        objective_name: str,
        n: int,
        v_min: float,
        v_max: float,
        v_mean: float | None,
        v_std: float | None,
        v_median: float | None,
        v_skew: float | None,
    ):
        """Internal solver for the sequence optimization problem."""

        def get_x(d: np.ndarray) -> np.ndarray:
            """Reconstruct sequence x from increments d."""
            increments = np.concatenate(([0], d))
            return v_min + np.cumsum(increments)

        def objective_smooth(d: np.ndarray) -> float:
            """Minimize curvature for smooth sequences."""
            diffs = np.diff(d)
            return np.sum(diffs**2)

        def objective_max_1f1b(d: np.ndarray) -> float:
            """Maximize sum of last pp_size * num_of_local_moe_layers elements."""
            x = get_x(d)
            return -np.sum(x[-(pp_size * num_of_local_moe_layers) :])

        def constr_max(d: np.ndarray) -> float:
            """Constraint: x_n = v_max."""
            return np.sum(d) - (v_max - v_min)

        def constr_mean(d: np.ndarray) -> float:
            """Constraint: mean(x) = v_mean."""
            x = get_x(d)
            return np.mean(x) - v_mean

        def constr_median(d: np.ndarray) -> float:
            """Constraint: median(x) = v_median."""
            x = get_x(d)
            return np.median(x) - v_median

        def constr_std(d: np.ndarray) -> float:
            """Constraint: std(x) = v_std."""
            x = get_x(d)
            return np.std(x, ddof=0) - v_std

        def constr_skew(d: np.ndarray) -> float:
            """Constraint: skewness(x) = v_skew."""
            x = get_x(d)
            mu = np.mean(x)
            return (np.mean((x - mu) ** 3) / (np.std(x, ddof=0) ** 3)) - v_skew

        # Build constraints list
        cons = [{"type": "eq", "fun": constr_max}]

        if v_mean is not None:
            cons.append({"type": "eq", "fun": constr_mean})
        if v_std is not None:
            cons.append({"type": "eq", "fun": constr_std})
        if v_median is not None:
            cons.append({"type": "eq", "fun": constr_median})
        if v_skew is not None:
            cons.append({"type": "eq", "fun": constr_skew})

        # Bounds: d_i >= 0 (monotonicity)
        bounds = [(0, None) for _ in range(n - 1)]

        # Initial guess: linear interpolation
        d0_val = (v_max - v_min) / (n - 1)
        d0 = np.full(n - 1, d0_val)

        # Optimization
        obj_func = objective_max_1f1b if objective_name == "max_1f1b" else objective_smooth
        result = minimize(
            obj_func,
            d0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 10000, "ftol": 1e-8, "disp": False},
        )

        return result, get_x(result.x)

    status, result = solve_sequence(objective, gating_count, br_min, br_max, br_avg, br_std, br_med, br_skew)

    print_rank_0(f"[MockRouter] {status.message}")
    assert status.success, "[MockRouter] Calculate balance ratio from statistics: failed to converge"
    print_rank_0("[MockRouter] Calculate balance ratio from statistics result")

    if objective == "smooth_shuffle":
        np.random.shuffle(result)  # noqa: NPY002

    print_rank_0(result)
    return result.tolist()
