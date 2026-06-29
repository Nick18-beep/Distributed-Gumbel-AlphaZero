"""Detect the active PyTorch runtime backend."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeBackend:
    name: str
    torch_available: bool
    device: str
    device_count: int
    reason: str

def _can_import(module_name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if spec is None:
        return False, "module not found"
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "available"


def detect_torch_runtime() -> RuntimeBackend:
    torch_available, torch_reason = _can_import("torch")
    if not torch_available:
        return RuntimeBackend(
            name="none",
            torch_available=False,
            device="none",
            device_count=0,
            reason=f"PyTorch unavailable ({torch_reason})",
        )

    import torch

    torch_cuda = getattr(torch, "cuda", None)
    if torch_cuda is None:
        return RuntimeBackend(
            name="none",
            torch_available=False,
            device="none",
            device_count=0,
            reason="PyTorch installation appears incomplete (torch.cuda is missing)",
        )

    if torch_cuda.is_available():
        return RuntimeBackend(
            name="torch",
            torch_available=True,
            device="cuda",
            device_count=torch_cuda.device_count(),
            reason="PyTorch available; using CUDA",
        )
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return RuntimeBackend(
            name="torch",
            torch_available=True,
            device="mps",
            device_count=1,
            reason="PyTorch available; using Apple MPS",
        )
    return RuntimeBackend(
        name="torch",
        torch_available=True,
        device="cpu",
        device_count=1,
        reason="PyTorch available; using CPU",
    )


def detect_runtime_backend() -> RuntimeBackend:
    return detect_torch_runtime()
