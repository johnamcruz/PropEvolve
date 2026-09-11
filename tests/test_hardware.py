"""Explicit accelerator requests never silently fall back."""

import pytest

from propevolve import hardware


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.torch.backends.mps, "is_available", lambda: True)
    assert str(hardware.resolve_device("auto")) == "cuda"

    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    assert str(hardware.resolve_device("auto")) == "mps"

    monkeypatch.setattr(hardware.torch.backends.mps, "is_available", lambda: False)
    assert str(hardware.resolve_device("auto")) == "cpu"


def test_explicit_unavailable_or_unknown_device_fails_closed(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS training was requested"):
        hardware.resolve_device("mps")
    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        hardware.resolve_device("cuda")
    with pytest.raises(ValueError, match="device must be"):
        hardware.resolve_device("tpu")
    assert str(hardware.resolve_device("cpu")) == "cpu"
