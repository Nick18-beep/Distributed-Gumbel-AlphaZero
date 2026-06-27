"""Single-process run orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from gumbel_az.config.schema import AppConfig
from gumbel_az.execution.base import ExecutionResult
from gumbel_az.logging import JsonlWriter, MetricWriter
from gumbel_az.replay import ReplayWriter
from gumbel_az.runtime.backend import RuntimeBackend
from gumbel_az.storage.atomic import atomic_write_json
from gumbel_az.storage.filesystem import RunPaths


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RunOrchestrator:
    """Wire self-play, replay, training, checkpointing and evaluation."""

    def __init__(
        self,
        config: AppConfig,
        *,
        paths: RunPaths,
        runtime_backend: RuntimeBackend,
        event_writer: JsonlWriter,
        metric_writer: MetricWriter,
    ) -> None:
        self.config = config
        self.paths = paths
        self.runtime_backend = runtime_backend
        self.event_writer = event_writer
        self.metric_writer = metric_writer

    def _write_state(self, **updates: Any) -> dict[str, Any]:
        previous: dict[str, Any] = {}
        if self.paths.run_state_path.exists():
            try:
                previous = json.loads(self.paths.run_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        state = {
            **previous,
            "run_id": self.paths.run_id,
            "backend": self.config.execution.backend,
            "runtime_backend": self.runtime_backend.name,
            "runtime_backend_reason": self.runtime_backend.reason,
            "updated_at": _utc_now(),
            **updates,
        }
        atomic_write_json(self.paths.run_state_path, state)
        return state

    def _run_selfplay(
        self,
        *,
        iteration: int,
        games_to_generate: int,
        worker: Any | None = None,
    ) -> Any:
        if games_to_generate <= 0:
            raise ValueError("single-process run requires at least one self-play game")

        if worker is None and self.runtime_backend.name == "torch":
            replay_writer = ReplayWriter(self.paths.run_dir / "replay")
            from gumbel_az.selfplay.torch_fallback import TorchFallbackSelfPlayWorker

            worker = TorchFallbackSelfPlayWorker(self.config, replay_writer=replay_writer)
        elif worker is None:
            raise RuntimeError(self.runtime_backend.reason)
        seed = self.config.run.seed + iteration * self.config.selfplay.games_per_iteration
        _, result = worker.play_batch(games_to_generate, seed)
        self.event_writer.write(
            {
                "event": "selfplay_completed",
                "iteration": iteration,
                "runtime_backend": self.runtime_backend.name,
                "games": result.games,
                "positions": result.positions,
                "replay_shard": result.replay_shard,
            }
        )
        self.metric_writer.write_metrics(
            iteration,
            {
                "games_per_sec": result.games_per_sec,
                "positions_per_sec": result.positions_per_sec,
                "illegal_action_rate": result.illegal_action_rate,
                "policy_entropy_mean": result.policy_entropy_mean,
                "root_value_mean": result.root_value_mean,
                "runtime_backend_is_torch_fallback": self.runtime_backend.name == "torch",
            },
        )
        return result

    def run(self) -> ExecutionResult:
        started = perf_counter()
        self._write_state(
            status="running",
            created_at=_utc_now(),
            config_path=str(self.paths.resolved_config_path),
            train_step=0,
            games_seen=0,
            samples_seen=0,
            iterations_completed=0,
        )

        try:
            self.event_writer.write(
                {
                    "event": "runtime_backend_selected",
                    "runtime_backend": self.runtime_backend.name,
                    "reason": self.runtime_backend.reason,
                    "jax_available": self.runtime_backend.jax_available,
                    "torch_available": self.runtime_backend.torch_available,
                }
            )
            max_iterations = self.config.stop.max_iterations or 1
            total_games = 0
            total_positions = 0
            latest_replay_shard: str | None = None
            latest_train_result: Any | None = None
            latest_eval_payload: dict[str, Any] | None = None
            iterations_completed = 0

            if self.runtime_backend.name != "jax":
                games_to_generate = min(
                    self.config.stop.max_games or self.config.selfplay.games_per_iteration,
                    self.config.selfplay.games_per_iteration,
                )
                selfplay_result = self._run_selfplay(
                    iteration=0,
                    games_to_generate=games_to_generate,
                )
                self._write_state(
                    status="completed_torch_fallback",
                    train_step=0,
                    games_seen=selfplay_result.games,
                    samples_seen=selfplay_result.positions,
                    replay_shard=selfplay_result.replay_shard,
                    note="JAX unavailable; PyTorch fallback generated replay only.",
                )
                return ExecutionResult(
                    run_id=self.paths.run_id,
                    run_dir=self.paths.run_dir,
                    status="completed_torch_fallback",
                )

            from gumbel_az.eval import Arena, should_promote
            from gumbel_az.model.checkpoint import CheckpointManager
            from gumbel_az.orchestration.scheduler import LocalScheduler, SchedulerSignals
            from gumbel_az.replay import ReplayReader
            from gumbel_az.selfplay.worker import SelfPlayWorker
            from gumbel_az.training.trainer import Trainer

            checkpoint_manager = CheckpointManager(self.paths.run_dir / "checkpoints")
            replay_reader = ReplayReader(self.paths.run_dir / "replay")
            trainer = Trainer(
                self.config,
                replay_reader=replay_reader,
                checkpoint_manager=checkpoint_manager,
                metric_writer=self.metric_writer,
            )
            scheduler = LocalScheduler(self.config)
            selfplay_worker = SelfPlayWorker(
                self.config,
                replay_writer=ReplayWriter(self.paths.run_dir / "replay"),
                params=trainer.state.params,
            )

            for iteration in range(max_iterations):
                if self.config.stop.max_wall_time_sec is not None:
                    elapsed = perf_counter() - started
                    if iteration > 0 and elapsed >= self.config.stop.max_wall_time_sec:
                        break
                if self.config.stop.max_train_steps is not None:
                    if int(trainer.state.step) >= self.config.stop.max_train_steps:
                        break
                replay_samples_available = sum(
                    int(entry.get("samples", 0)) for entry in replay_reader.shard_paths_metadata()
                )
                decision = scheduler.decide(
                    SchedulerSignals(
                        replay_samples_available=replay_samples_available,
                        checkpoint_pending=iteration > 0,
                        evaluation_pending=self.config.eval.enabled,
                        model_staleness=iteration,
                    )
                )
                self.event_writer.write(
                    {
                        "event": "scheduler_decision",
                        "iteration": iteration,
                        "stage": "before_selfplay",
                        "decision": decision.to_event(),
                    }
                )
                remaining_games = None
                if self.config.stop.max_games is not None:
                    remaining_games = self.config.stop.max_games - total_games
                    if remaining_games <= 0:
                        break
                if decision.allow_selfplay:
                    games_to_generate = min(
                        remaining_games or self.config.selfplay.games_per_iteration,
                        self.config.selfplay.games_per_iteration,
                    )
                    selfplay_worker.params = trainer.state.params
                    selfplay_result = self._run_selfplay(
                        iteration=iteration,
                        games_to_generate=games_to_generate,
                        worker=selfplay_worker,
                    )
                    total_games += selfplay_result.games
                    total_positions += selfplay_result.positions
                    latest_replay_shard = selfplay_result.replay_shard

                replay_samples_available = sum(
                    int(entry.get("samples", 0)) for entry in replay_reader.shard_paths_metadata()
                )
                training_decision = scheduler.decide(
                    SchedulerSignals(
                        replay_samples_available=replay_samples_available,
                        checkpoint_pending=iteration > 0,
                        evaluation_pending=self.config.eval.enabled,
                        model_staleness=iteration,
                    )
                )
                self.event_writer.write(
                    {
                        "event": "scheduler_decision",
                        "iteration": iteration,
                        "stage": "before_training",
                        "decision": training_decision.to_event(),
                    }
                )
                if not training_decision.allow_training:
                    continue

                remaining_steps = self.config.training.steps_per_iteration
                if self.config.stop.max_train_steps is not None:
                    remaining_steps = min(
                        remaining_steps,
                        self.config.stop.max_train_steps - int(trainer.state.step),
                    )
                    if remaining_steps <= 0:
                        break
                latest_train_result = trainer.run(max_steps=remaining_steps)
                self.event_writer.write(
                    {
                        "event": "training_completed",
                        "iteration": iteration,
                        "train_step": latest_train_result.checkpoint_version,
                        "checkpoint_version": latest_train_result.checkpoint_version,
                    }
                )
                self.metric_writer.write_metrics(
                    latest_train_result.checkpoint_version,
                    {
                        "train_samples_per_sec": latest_train_result.samples_per_sec,
                        "replay_sample_age_mean": latest_train_result.replay_sample_age_mean,
                        "checkpoint_version": latest_train_result.checkpoint_version,
                    },
                )

                iterations_completed += 1
                promoted = False
                if self.config.eval.enabled:
                    arena = Arena(
                        self.config,
                        eval_dir=self.paths.run_dir / "eval",
                        event_writer=self.event_writer,
                    )
                    eval_result = arena.evaluate_vs_random(
                        params=latest_train_result.state.params,
                        checkpoint_version=latest_train_result.checkpoint_version,
                    )
                    is_first_best = not checkpoint_manager.best_path.exists()
                    promoted = is_first_best or should_promote(
                        eval_result,
                        min_games=self.config.eval.games,
                        promotion_win_rate=self.config.eval.promotion_win_rate,
                    )
                    if promoted:
                        checkpoint_manager.promote(latest_train_result.checkpoint_version)
                    latest_eval_payload = {
                        "checkpoint_version": eval_result.checkpoint_version,
                        "games": eval_result.games,
                        "wins": eval_result.wins,
                        "losses": eval_result.losses,
                        "draws": eval_result.draws,
                        "win_rate": eval_result.win_rate,
                        "games_per_sec": eval_result.games_per_sec,
                        "promoted": promoted,
                        "promotion_reason": (
                            "initial_best"
                            if is_first_best
                            else "threshold"
                            if promoted
                            else "not_promoted"
                        ),
                    }
                    self.metric_writer.write_metrics(
                        latest_train_result.checkpoint_version,
                        {
                            "eval_win_rate": eval_result.win_rate,
                            "eval_games_per_sec": eval_result.games_per_sec,
                            "checkpoint_promoted": promoted,
                        },
                    )

                self._write_state(
                    status="running",
                    train_step=latest_train_result.checkpoint_version,
                    games_seen=total_games,
                    samples_seen=total_positions,
                    replay_shard=latest_replay_shard,
                    checkpoint_version=latest_train_result.checkpoint_version,
                    iterations_completed=iterations_completed,
                    eval=latest_eval_payload,
                )

            if latest_train_result is None:
                raise RuntimeError("run stopped before completing a training iteration")

            self._write_state(
                status="completed",
                train_step=latest_train_result.checkpoint_version,
                games_seen=total_games,
                samples_seen=total_positions,
                replay_shard=latest_replay_shard,
                checkpoint_version=latest_train_result.checkpoint_version,
                iterations_completed=iterations_completed,
                eval=latest_eval_payload,
            )
            return ExecutionResult(
                run_id=self.paths.run_id,
                run_dir=self.paths.run_dir,
                status="completed",
            )
        except KeyboardInterrupt:
            self._write_state(status="interrupted")
            self.event_writer.write({"event": "run_interrupted"})
            return ExecutionResult(
                run_id=self.paths.run_id,
                run_dir=self.paths.run_dir,
                status="interrupted",
            )
