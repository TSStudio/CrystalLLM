# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystallLLM Balance Ratio Recorder Module for MoE (Mixture of Experts).

This module provides functionality to record and analyze the balance ratio
of token distribution across devices and experts in MoE dispatch operations.

The ideal_tokens_per_device is calculated as:
    ideal_tokens_per_device = local_tokens_per_device * routed_experts_per_token
where:
    - local_tokens_per_device = micro_batch_size * seq_length / (context_parallel_size * sequence_parallel_size)
    - routed_experts_per_token = experts_per_token - num_shared_experts
"""

import torch


class BalanceRecorder:
    """Balance ratio recorder for MoE token distribution analysis.

    This class provides functionality to record and analyze the balance ratio
    of token distribution across devices and experts in MoE dispatch operations.
    It encapsulates all tracking state and provides methods for data collection
    and statistical analysis.

    Attributes:
        _device_counts: List tracking token counts per device.
        _expert_counts: List tracking token counts per expert (as tensors).
        _evict_mask10: Tensor mask for top 10% experts by evict criteria.
        _evict_mask20: Tensor mask for top 20% experts by evict criteria.
        _evict_mask50: Tensor mask for top 50% experts by evict criteria.
    """

    def __init__(self) -> None:
        """Initialize a new BalanceRecorder instance.

        Creates empty tracking lists and resets evict masks to None.
        """
        self._device_counts: list[int] = []
        self._expert_counts: list[torch.Tensor] = []
        self._evict_mask10: torch.Tensor | None = None
        self._evict_mask20: torch.Tensor | None = None
        self._evict_mask50: torch.Tensor | None = None

    def save(
        self,
        recv_count_per_device: int,
        recv_count_per_expert: torch.Tensor,
    ) -> None:
        """Save the receive counts to the balance tracker.

        This method records the number of tokens received per device and per expert
        during MoE dispatch operations for later balance ratio analysis.

        Args:
            recv_count_per_device: Total number of tokens received by this device.
            recv_count_per_expert: Tensor containing token counts per expert.
        """
        self._device_counts.append(recv_count_per_device)
        self._expert_counts.append(recv_count_per_expert)

    def clear(self) -> None:
        """Clear the balance tracker."""
        self._device_counts.clear()
        self._expert_counts.clear()
        self._evict_mask10 = None
        self._evict_mask20 = None
        self._evict_mask50 = None

    def _calculate_expert_distribution(
        self, income_lists: list[torch.Tensor], ideal_tokens_per_device: float
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """Calculate expert receive distribution statistics."""
        income_tensor_raw = torch.stack(income_lists)
        income_sum_by_row_mask = income_tensor_raw.sum(1) > ideal_tokens_per_device
        num10 = income_tensor_raw.size(1) // 10
        num20 = income_tensor_raw.size(1) // 5
        num50 = income_tensor_raw.size(1) // 2

        if not income_sum_by_row_mask.any():
            total_counts_by_expert = income_tensor_raw.sum(0)
            sorted_index = torch.argsort(total_counts_by_expert, descending=True)
            self._evict_mask10 = sorted_index[:num10]
            self._evict_mask20 = sorted_index[:num20]
            self._evict_mask50 = sorted_index[:num50]
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        income_tensor = income_tensor_raw[income_sum_by_row_mask]
        sorted_income, _ = torch.sort(income_tensor, dim=-1, descending=True)
        income_sum_by_row = income_tensor.sum(1)

        t10 = (sorted_income[:, :num10].sum(1) / income_sum_by_row).mean().item()
        t20 = (sorted_income[:, :num20].sum(1) / income_sum_by_row).mean().item()
        t50 = (sorted_income[:, :num50].sum(1) / income_sum_by_row).mean().item()

        f10 = (income_tensor[:, :num10].sum(1) / income_sum_by_row).mean().item()
        f20 = (income_tensor[:, :num20].sum(1) / income_sum_by_row).mean().item()
        f50 = (income_tensor[:, :num50].sum(1) / income_sum_by_row).mean().item()

        if self._evict_mask10 is not None and self._evict_mask20 is not None and self._evict_mask50 is not None:
            e10 = (income_tensor[:, self._evict_mask10].sum(1) / income_sum_by_row).mean().item()
            e20 = (income_tensor[:, self._evict_mask20].sum(1) / income_sum_by_row).mean().item()
            e50 = (income_tensor[:, self._evict_mask50].sum(1) / income_sum_by_row).mean().item()
        else:
            e10 = f10
            e20 = f20
            e50 = f50

        total_counts_by_expert = income_tensor_raw.sum(0)
        sorted_index = torch.argsort(total_counts_by_expert, descending=True)
        self._evict_mask10 = sorted_index[:num10]
        self._evict_mask20 = sorted_index[:num20]
        self._evict_mask50 = sorted_index[:num50]

        return t10, t20, t50, f10, f20, f50, e10, e20, e50

    def _calculate_device_distribution(  # noqa: PLR6301
        self, income_lists: list[int], ideal_tokens_per_device: float
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate device receive distribution statistics."""
        income_tensor = torch.tensor(income_lists) / ideal_tokens_per_device
        br_avg = income_tensor.mean()
        br_max = income_tensor.max()
        br_min = income_tensor.min()
        br_std = income_tensor.std(unbiased=False)
        br_med = income_tensor.median()
        third_moment = torch.mean((income_tensor - br_avg) ** 3)
        std_cubed = br_std**3
        br_skew = third_moment / std_cubed
        return (br_avg.item(), br_std.item(), br_med.item(), br_max.item(), br_min.item(), br_skew.item())

    def calculate_balance_ratio(self, ideal_tokens_per_device: float) -> str | None:
        """Calculate the balance ratio metrics from recorded data.

        This method computes comprehensive balance ratio statistics from the
        tracked device and expert token distributions and return statistics string.

        Args:
            ideal_tokens_per_device: Ideal number of tokens per device for calculating ratios.

        Returns:
            Formatted balance ratio string if data exists, "N/A" if no data or ideal_tokens_per_device is invalid.
        """
        if ideal_tokens_per_device is not None and ideal_tokens_per_device > 0 and len(self._device_counts) > 0:
            br_avg, br_std, br_med, br_max, br_min, br_skew = self._calculate_device_distribution(
                self._device_counts, ideal_tokens_per_device
            )
            t10, t20, t50, f10, f20, f50, e10, e20, e50 = self._calculate_expert_distribution(
                self._expert_counts, ideal_tokens_per_device
            )

            return (
                "["
                f"{br_avg:.2f}(+/-{br_std:.2f}),"
                f"{br_min:.2f}~{br_max:.2f},"
                f"{br_med:.2f},"
                f"{br_skew:.2f}"
                "],"
                f"[{t10 * 100:.2f},"
                f"{t20 * 100:.2f},"
                f"{t50 * 100:.2f}],"
                f"[{f10 * 100:.2f},"
                f"{f20 * 100:.2f},"
                f"{f50 * 100:.2f}],"
                f"[{e10 * 100:.2f},"
                f"{e20 * 100:.2f},"
                f"{e50 * 100:.2f}]"
            )

        return "N/A"
