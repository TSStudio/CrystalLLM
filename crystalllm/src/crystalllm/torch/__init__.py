# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM PyTorch Distributed C10d Modifications."""

from .fake import CrystalFakeProcessGroup, CrystalFakeWork, CrystalProfiledWork
from .wrapper import (
    cond_crystal_c10d_end_coalescing,
    cond_crystal_c10d_start_coalescing,
    cond_crystal_c10d_trace_writer,
    cond_crystal_collectives_handler,
    cond_crystal_runtime_profiler_end_coalescing,
    cond_crystal_runtime_profiler_start_coalescing,
    cond_custom_write_crystal_c10d_trace,
)

__all__ = [
    "CrystalFakeProcessGroup",
    "CrystalFakeWork",
    "CrystalProfiledWork",
    "cond_crystal_c10d_end_coalescing",
    "cond_crystal_c10d_start_coalescing",
    "cond_crystal_c10d_trace_writer",
    "cond_crystal_collectives_handler",
    "cond_crystal_runtime_profiler_end_coalescing",
    "cond_crystal_runtime_profiler_start_coalescing",
    "cond_custom_write_crystal_c10d_trace",
]
