"""Self-play generation."""

from typing import Any

__all__ = ["SelfPlayResult", "SelfPlayWorker"]


def __getattr__(name: str) -> Any:
    if name in {"SelfPlayResult", "SelfPlayWorker"}:
        from gumbel_az.selfplay.worker import SelfPlayResult, SelfPlayWorker

        return {"SelfPlayResult": SelfPlayResult, "SelfPlayWorker": SelfPlayWorker}[name]
    raise AttributeError(name)
