"""Single-process execution backend."""

from __future__ import annotations

from gumbel_az.config.loader import save_resolved_config
from gumbel_az.config.schema import AppConfig
from gumbel_az.execution.base import ExecutionResult
from gumbel_az.logging import JsonlWriter, MetricWriter
from gumbel_az.orchestration.run import RunOrchestrator
from gumbel_az.runtime import detect_runtime_backend
from gumbel_az.storage import create_run_directory


class SingleProcessExecutionBackend:
    name = "single_process"

    def run(self, config: AppConfig) -> ExecutionResult:
        if config.execution.backend != self.name:
            raise ValueError(
                f"SingleProcessExecutionBackend cannot run backend {config.execution.backend!r}"
            )

        paths = create_run_directory(config)
        save_resolved_config(config, paths.run_dir)
        event_writer = JsonlWriter(paths.events_path)
        metric_writer = MetricWriter(paths.metrics_path)

        event_writer.write({"event": "run_initialized", "run_id": paths.run_id})
        metric_writer.write_metrics(0, {"run_initialized": True})
        runtime_backend = detect_runtime_backend()
        return RunOrchestrator(
            config,
            paths=paths,
            runtime_backend=runtime_backend,
            event_writer=event_writer,
            metric_writer=metric_writer,
        ).run()
