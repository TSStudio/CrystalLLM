# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM NCCL utility functions."""

import importlib.util
from pathlib import Path


def _get_nccl_module_dir():
    r"""Get the directory of the crystalllm.nccl module."""
    nccl_module_spec = importlib.util.find_spec("crystalllm.nccl")
    if nccl_module_spec is None or nccl_module_spec.origin is None:
        raise RuntimeError("Failed to locate crystalllm.nccl module")
    return Path(nccl_module_spec.origin).parent.absolute()


def get_base_lib_path():
    r"""Get the absolute path of the baseBuild version of libnccl.so."""
    nccl_module_dir = _get_nccl_module_dir()
    lib_path = nccl_module_dir / "lib" / "baseBuild" / "libnccl.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"NCCL base library not found: {lib_path}")
    return lib_path


def get_fake_lib_path():
    r"""Get the absolute path of the fakeBuild version of libnccl.so."""
    nccl_module_dir = _get_nccl_module_dir()
    lib_path = nccl_module_dir / "lib" / "fakeBuild" / "libnccl.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"NCCL emu library not found: {lib_path}")
    return lib_path


def get_mock_lib_path():
    r"""Get the absolute path of the mockBuild version of libnccl.so."""
    nccl_module_dir = _get_nccl_module_dir()
    lib_path = nccl_module_dir / "lib" / "mockBuild" / "libnccl.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"NCCL mock library not found: {lib_path}")
    return lib_path
