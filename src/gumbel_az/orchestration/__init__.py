"""Run orchestration."""

from gumbel_az.orchestration.resume import (
    ResumeContext,
    load_resume_context,
    load_run_state,
    rebuild_replay_index,
)
from gumbel_az.orchestration.scheduler import LocalScheduler, SchedulerDecision, SchedulerSignals

__all__ = [
    "LocalScheduler",
    "RunOrchestrator",
    "SchedulerDecision",
    "SchedulerSignals",
    "ResumeContext",
    "load_resume_context",
    "load_run_state",
    "rebuild_replay_index",
]


def __getattr__(name: str):
    if name == "RunOrchestrator":
        from gumbel_az.orchestration.run import RunOrchestrator

        return RunOrchestrator
    raise AttributeError(name)
