"""Execution backends."""

from gumbel_az.execution.base import ExecutionBackend, ExecutionResult
from gumbel_az.execution.single_process import SingleProcessExecutionBackend

__all__ = [
    "ExecutionBackend",
    "ExecutionResult",
    "LanRayExecutionBackend",
    "LocalMultiprocessExecutionBackend",
    "SingleProcessExecutionBackend",
]


def __getattr__(name: str):
    if name == "LocalMultiprocessExecutionBackend":
        from gumbel_az.execution.local_multiprocess import LocalMultiprocessExecutionBackend

        return LocalMultiprocessExecutionBackend
    if name == "LanRayExecutionBackend":
        from gumbel_az.execution.lan_ray import LanRayExecutionBackend

        return LanRayExecutionBackend
    raise AttributeError(name)
