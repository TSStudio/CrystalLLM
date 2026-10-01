# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM NCCL interface."""

from .util import (
    get_base_lib_path,
    get_fake_lib_path,
    get_mock_lib_path,
)

__all__ = ["get_base_lib_path", "get_fake_lib_path", "get_mock_lib_path"]
