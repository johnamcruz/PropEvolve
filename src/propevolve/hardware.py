"""Shared PyTorch device selection for frozen specialist teachers."""

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve an accelerator without silently changing explicit requests."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS training was requested but this PyTorch runtime cannot use MPS")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but this PyTorch runtime cannot use CUDA")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    return torch.device(device)


__all__ = ["resolve_device"]
