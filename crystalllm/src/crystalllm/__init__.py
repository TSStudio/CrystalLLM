# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Library Interface."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
