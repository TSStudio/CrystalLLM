# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Monitor utility functions."""


def get_dual_network_interfaces_for_gpu(gpu_index):
    """Get the two network interface names for a given GPU index.

    Each GPU is mapped to two network interfaces (reth*).
    GPU 0 -> reth0, reth1
    GPU 1 -> reth2, reth3
    GPU N -> reth(2*N), reth(2*N+1)

    Args:
        gpu_index: The index of the GPU.

    Returns:
        A list of two network interface names.
    """
    base = gpu_index * 2
    return [f"reth{base}", f"reth{base + 1}"]
