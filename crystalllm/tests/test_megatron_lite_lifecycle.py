# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Lite owns mode gating and exit finalization, without changing runtime APIs."""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch  # noqa: F401  # Load PyTorch before the optional Crystal hooks.

from crystalllm.integration import megatron_lite as lite


@pytest.mark.parametrize("mode", [None, "", "PLANNING", "capture"])
def test_inactive_adapter_does_not_touch_runtime_or_te(monkeypatch, mode):
    """Leave runtime and Transformer Engine untouched outside Crystal modes."""
    if mode is None:
        monkeypatch.delenv("CRYSTAL_MODE", raising=False)
    else:
        monkeypatch.setenv("CRYSTAL_MODE", mode)
    initialize, step, te, register = (mock.Mock() for _ in range(4))
    monkeypatch.setattr(lite, "initialize_crystal_runtime", initialize)
    monkeypatch.setattr(lite, "crystal_runtime_step_start", step)
    monkeypatch.setattr(lite, "_enable_transformer_engine_fake_process_groups", te)
    monkeypatch.setattr(lite.atexit, "register", register)

    lite.initialize()
    lite.step_start()
    for function in (initialize, step, te, register):
        function.assert_not_called()


@pytest.mark.parametrize("mode", [None, "CAPTURE", "EMULATE"])
def test_step_stop_uses_new_runtime_boundary(monkeypatch, mode):
    """Close profiler windows only when the Lite adapter is active."""
    if mode is None:
        monkeypatch.delenv("CRYSTAL_MODE", raising=False)
    else:
        monkeypatch.setenv("CRYSTAL_MODE", mode)
    stop = mock.Mock()
    monkeypatch.setattr(lite, "crystal_runtime_step_stop", stop)
    lite.step_stop()
    assert stop.call_count == (0 if mode is None else 1)


@pytest.mark.parametrize("mode", ["CAPTURE", "EMULATE"])
def test_adapter_registers_runtime_finalization_once(monkeypatch, mode):
    """Register one adapter finalizer without adding generic runtime idempotency."""
    monkeypatch.setenv("CRYSTAL_MODE", mode)
    monkeypatch.setattr(lite, "_ATEXIT_REGISTERED", False)
    initialize, step, destroy, te, register = (mock.Mock() for _ in range(5))
    monkeypatch.setattr(lite, "initialize_crystal_runtime", initialize)
    monkeypatch.setattr(lite, "crystal_runtime_step_start", step)
    monkeypatch.setattr(lite, "destroy_crystal_runtime", destroy)
    monkeypatch.setattr(lite, "_enable_transformer_engine_fake_process_groups", te)
    monkeypatch.setattr(lite.atexit, "register", register)

    lite.initialize()
    lite.initialize()
    lite.step_start()

    assert initialize.call_args_list == [mock.call(), mock.call()]
    assert te.call_args_list == [mock.call(), mock.call()]
    step.assert_called_once_with()
    register.assert_called_once_with(destroy)
    destroy.assert_not_called()
    register.call_args.args[0]()
    destroy.assert_called_once_with()


def test_failed_initialization_does_not_register_finalizer(monkeypatch):
    """Avoid registering teardown when adapter initialization fails."""
    monkeypatch.setenv("CRYSTAL_MODE", "CAPTURE")
    monkeypatch.setattr(lite, "_ATEXIT_REGISTERED", False)
    monkeypatch.setattr(lite, "initialize_crystal_runtime", mock.Mock(side_effect=RuntimeError("init failed")))
    register = mock.Mock()
    monkeypatch.setattr(lite.atexit, "register", register)

    with pytest.raises(RuntimeError, match="init failed"):
        lite.initialize()

    register.assert_not_called()
    assert not lite._ATEXIT_REGISTERED


def test_normal_exit_flushes_real_trace_and_graph_without_explicit_destroy(tmp_path):
    """Exercise real CPU-only runtime finalization in a separate interpreter."""
    env = {key: value for key, value in os.environ.items() if "CRYSTAL" not in key}
    env.update({
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "CRYSTAL_MODE": "CAPTURE",
        "RANK": "0",
        "CRYSTAL_LOG_DIR": str(tmp_path),
        "ENABLE_CRYSTAL_C10D_TRACE": "true",
        "CRYSTAL_GRAPH_CAPTURING": "true",
        "CRYSTAL_ANALYSIS_STEP_START": "1",
        "CRYSTAL_ANALYSIS_STEP_END": "2",
    })
    code = """
import torch
from crystalllm.integration import megatron_lite as lite
from crystalllm.runtime import get_context
lite._enable_transformer_engine_fake_process_groups = lambda: None
lite.initialize()
lite.initialize()
lite.step_start()
get_context().cond_custom_write_c10d_trace("TEST", "pending buffer")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "Exception ignored in atexit callback" not in result.stderr
    with gzip.open(tmp_path / "trace/trace_0.log.gz", "rt") as stream:
        trace = stream.read()
    assert "SCHEDULE#iteration 1 begin" in trace
    assert "TEST#pending buffer" in trace
    with gzip.open(tmp_path / "graph/graph_0.json.gz", "rt") as stream:
        graph = json.load(stream)
    assert "1" in graph
    assert not graph["1"]["span_desc"]
    assert graph["1"]["spans"]["_proc#0"]["span_type"] == "proc"
