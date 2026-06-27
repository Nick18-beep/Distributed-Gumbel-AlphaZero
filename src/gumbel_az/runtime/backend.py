"""Detect the active ML runtime backend."""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeBackend:
    name: str
    jax_available: bool
    torch_available: bool
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


def detect_runtime_backend() -> RuntimeBackend:
    jax_available, jax_reason = _can_import("jax")
    torch_available, torch_reason = _can_import("torch")
    forced = os.environ.get("GAZ_FORCE_RUNTIME_BACKEND", "").strip().lower()
    if forced and forced not in {"jax", "torch"}:
        return RuntimeBackend(
            name="none",
            jax_available=jax_available,
            torch_available=torch_available,
            reason=f"unsupported GAZ_FORCE_RUNTIME_BACKEND={forced!r}; expected 'jax' or 'torch'",
        )
    if forced == "torch":
        if torch_available:
            return RuntimeBackend(
                name="torch",
                jax_available=jax_available,
                torch_available=True,
                reason="GAZ_FORCE_RUNTIME_BACKEND=torch; using PyTorch fallback",
            )
        return RuntimeBackend(
            name="none",
            jax_available=jax_available,
            torch_available=False,
            reason=f"GAZ_FORCE_RUNTIME_BACKEND=torch but PyTorch unavailable ({torch_reason})",
        )
    if forced == "jax" and not jax_available:
        return RuntimeBackend(
            name="none",
            jax_available=False,
            torch_available=torch_available,
            reason=f"GAZ_FORCE_RUNTIME_BACKEND=jax but JAX unavailable ({jax_reason})",
        )
    if jax_available:
        return RuntimeBackend(
            name="jax",
            jax_available=True,
            torch_available=torch_available,
            reason="JAX available; using primary backend",
        )
    if torch_available:
        return RuntimeBackend(
            name="torch",
            jax_available=False,
            torch_available=True,
            reason=f"JAX unavailable ({jax_reason}); using PyTorch fallback",
        )
    return RuntimeBackend(
        name="none",
        jax_available=False,
        torch_available=False,
        reason=f"JAX unavailable ({jax_reason}); PyTorch unavailable ({torch_reason})",
    )
