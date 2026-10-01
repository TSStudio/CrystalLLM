# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Resolve trace group metadata from wrapped c10d signatures."""

import functools
import inspect
import io
from unittest import mock

import pytest
import torch

from crystalllm.torch import c10d_trace
from crystalllm.torch.fake import CrystalFakeProcessGroup


@pytest.fixture
def tracer(monkeypatch):
    """Isolate the trace buffer and counters without opening artifact files."""
    tracer = c10d_trace.CrystalC10DTracer()
    for name, value in {
        "_trace_buffer": [], "_trace_file": io.StringIO(),
    }.items():
        monkeypatch.setattr(tracer, name, value)
    return tracer


@pytest.mark.parametrize("op", [
    "isend", "irecv", "broadcast", "all_reduce", "all_gather_into_tensor",
    "_all_gather_base", "reduce_scatter_tensor", "_reduce_scatter_base",
    "all_to_all_single", "barrier",
])
@pytest.mark.parametrize("style", ["positional", "keyword", "omitted", "explicit_none"])
def test_group_metadata_from_real_c10d_signatures(tracer, monkeypatch, op, style):
    """Resolve group metadata without changing positional or keyword call styles."""
    func = getattr(torch.distributed, op)
    parameters = inspect.signature(func).parameters
    positional = [name for name, parameter in parameters.items()
                  if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}]
    position = positional.index("group")
    cp = CrystalFakeProcessGroup([0, 8], 0, "17", "CONTEXT-0-17-nccl")
    world = CrystalFakeProcessGroup([0, 8], 0, "0", "default_pg")
    monkeypatch.setattr(torch.distributed.distributed_c10d, "_get_default_group", lambda: world)
    args, kwargs = [None] * position, {}
    if style == "positional":
        args.append(cp)
    elif style == "keyword":
        kwargs["group"] = cp
    elif style == "explicit_none":
        kwargs["group"] = None

    tracer.c10d_trace_writer(7, func, args, kwargs)

    expected = ["17", "CONTEXT-0-17-nccl"] if style in {"positional", "keyword"} else ["0", "default_pg"]
    assert tracer._trace_buffer[0].split("#")[3:5] == expected


def test_signature_is_unwrapped_and_inspected_once():
    """Follow wrapper metadata once and reuse the cached argument specification."""
    def isend(tensor, dst, group=None):
        pass

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    wrapped = decorate(decorate(decorate(isend)))
    group = object()
    with mock.patch.object(c10d_trace.inspect, "signature", wraps=inspect.signature) as signature:
        assert c10d_trace._get_collective_group(wrapped, (None, 8, group), {}) is group
        assert c10d_trace._get_collective_group(wrapped, (None, 8), {"group": group}) is group
        assert c10d_trace._get_collective_group(wrapped, (None, 8, group), {}) is group
        signature.assert_called_once_with(wrapped)


def test_keyword_only_group_and_declared_default(monkeypatch):
    """Honor the declared group default and explicit None as the WORLD group."""
    group, world = object(), object()

    def collective(tensor, *, group=group):
        pass

    monkeypatch.setattr(torch.distributed.distributed_c10d, "_get_default_group", lambda: world)
    assert c10d_trace._get_collective_group(collective, (None,), {}) is group
    assert c10d_trace._get_collective_group(collective, (None,), {"group": None}) is world


def test_positional_only_group():
    """Resolve group from a positional-only parameter without a hardcoded offset."""
    def collective(tensor, group, /):
        pass

    group = object()
    assert c10d_trace._get_collective_group(collective, (None, group), {}) is group


def test_missing_required_group_is_not_silently_world():
    """Reject a missing required argument instead of mislabeling its group."""
    def collective(tensor, group):
        pass

    with pytest.raises(TypeError, match="Missing required group"):
        c10d_trace._get_collective_group(collective, (None,), {})


def test_erased_signature_is_not_silently_world():
    """Reject wrappers whose real group parameter cannot be recovered."""
    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(ValueError, match="Cannot resolve group"):
        c10d_trace._get_collective_group(wrapper, (None, 8, object()), {})
