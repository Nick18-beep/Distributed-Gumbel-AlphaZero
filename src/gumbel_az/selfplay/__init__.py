"""Self-play generation."""

__all__ = ["SelfPlayResult", "SelfPlayWorker"]


def __getattr__(name: str):
    if name in {"SelfPlayResult", "SelfPlayWorker"}:
        from gumbel_az.selfplay.worker import SelfPlayResult, SelfPlayWorker

        return {"SelfPlayResult": SelfPlayResult, "SelfPlayWorker": SelfPlayWorker}[name]
    raise AttributeError(name)
