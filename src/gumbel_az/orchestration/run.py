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
        skip_initial_selfplay_if_replay_available: bool = False,
        resume: bool = False,
        started_at: float | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.runtime_backend = runtime_backend
        self.event_writer = event_writer
        self.metric_writer = metric_writer
        self.skip_initial_selfplay_if_replay_available = skip_initial_selfplay_if_replay_available
        self.resume = resume
        self.started_at = started_at

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
            from gumbel_az.selfplay.worker import SelfPlayWorker

            worker = SelfPlayWorker(
                self.config,
                replay_writer=replay_writer,
                device=self.runtime_backend.device,
            )
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
                "runtime_backend_is_torch": self.runtime_backend.name == "torch",
            },
        )
        return result

    def run(self) -> ExecutionResult:
        started = self.started_at if self.started_at is not None else perf_counter()
        previous_state: dict[str, Any] = {}
        if self.paths.run_state_path.exists():
            try:
                previous_state = json.loads(self.paths.run_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous_state = {}
        self._write_state(
            status="running",
            error=None,
            created_at=previous_state.get("created_at", _utc_now()),
            config_path=str(self.paths.resolved_config_path),
            resumed_from_checkpoint=bool(self.resume),
            train_step=int(previous_state.get("train_step", 0)) if self.resume else 0,
            games_seen=int(previous_state.get("games_seen", 0)),
            samples_seen=int(previous_state.get("samples_seen", 0)),
            iterations_completed=int(previous_state.get("iterations_completed", 0))
            if self.resume
            else 0,
        )

        try:
            self.event_writer.write(
                {
                    "event": "runtime_backend_selected",
                    "runtime_backend": self.runtime_backend.name,
                    "reason": self.runtime_backend.reason,
                    "torch_available": self.runtime_backend.torch_available,
                    "device": self.runtime_backend.device,
                }
            )
            max_iterations = self.config.stop.max_iterations or 1
            total_games = 0
            total_positions = 0
            latest_replay_shard: str | None = None
            latest_train_result: Any | None = None
            latest_eval_payload: dict[str, Any] | None = None
            iterations_completed = 0

            if self.runtime_backend.name != "torch":
                raise RuntimeError(self.runtime_backend.reason)

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
                device=self.runtime_backend.device,
            )
            if self.resume:
                try:
                    checkpoint = checkpoint_manager.load(map_location=trainer.device)
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"cannot resume {self.paths.run_dir}: latest checkpoint not found"
                    ) from exc
                trainer.load_checkpoint(checkpoint)
                metadata = checkpoint.get("metadata", {})
                self.event_writer.write(
                    {
                        "event": "checkpoint_loaded_for_resume",
                        "checkpoint_version": metadata.get("version", trainer.state.step),
                        "train_step": int(trainer.state.step),
                    }
                )
            scheduler = LocalScheduler(self.config)
            selfplay_worker = SelfPlayWorker(
                self.config,
                replay_writer=ReplayWriter(self.paths.run_dir / "replay"),
                model=trainer.state.model,
                device=self.runtime_backend.device,
            )
            selfplay_worker.model_version = int(trainer.state.step)

            total_games = int(previous_state.get("games_seen", 0)) + int(
                previous_state.get("remote_games", 0)
            )
            total_positions = int(previous_state.get("samples_seen", 0)) + int(
                previous_state.get("remote_positions", 0)
            )
            iterations_completed = (
                int(previous_state.get("iterations_completed", 0)) if self.resume else 0
            )
            iteration_offset = iterations_completed if self.resume else 0

            for local_iteration in range(max_iterations):
                iteration = iteration_offset + local_iteration
                if self.config.stop.max_wall_time_sec is not None:
                    elapsed = perf_counter() - started
                    if local_iteration > 0 and elapsed >= self.config.stop.max_wall_time_sec:
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
                skip_selfplay_for_game_limit = False
                if self.config.stop.max_games is not None:
                    remaining_games = self.config.stop.max_games - total_games
                    if remaining_games <= 0:
                        if replay_samples_available > 0:
                            skip_selfplay_for_game_limit = True
                        else:
                            break
                skip_selfplay = (
                    self.skip_initial_selfplay_if_replay_available
                    and iteration == 0
                    and replay_samples_available > 0
                )
                if decision.allow_selfplay and skip_selfplay_for_game_limit:
                    self.event_writer.write(
                        {
                            "event": "selfplay_skipped",
                            "iteration": iteration,
                            "reason": "max_games_reached_existing_replay_available",
                            "replay_samples_available": replay_samples_available,
                        }
                    )
                elif decision.allow_selfplay and skip_selfplay:
                    self.event_writer.write(
                        {
                            "event": "selfplay_skipped",
                            "iteration": iteration,
                            "reason": "remote_replay_available",
                            "replay_samples_available": replay_samples_available,
                        }
                    )
                elif decision.allow_selfplay:
                    games_to_generate = min(
                        remaining_games or self.config.selfplay.games_per_iteration,
                        self.config.selfplay.games_per_iteration,
                    )
                    selfplay_worker.model = trainer.state.model
                    selfplay_worker.model_version = int(trainer.state.step)
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
                        device=self.runtime_backend.device,
                    )
                    eval_result = arena.evaluate_vs_random(
                        model=latest_train_result.state.model,
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
                if self.resume:
                    self.event_writer.write(
                        {
                            "event": "resume_no_training_needed",
                            "train_step": int(trainer.state.step),
                        }
                    )
                    self._write_state(
                        status="completed",
                        train_step=int(trainer.state.step),
                        games_seen=total_games,
                        samples_seen=total_positions,
                        replay_shard=latest_replay_shard,
                        checkpoint_version=int(trainer.state.step),
                        iterations_completed=iterations_completed,
                        eval=latest_eval_payload,
                    )
                    return ExecutionResult(
                        run_id=self.paths.run_id,
                        run_dir=self.paths.run_dir,
                        status="completed",
                    )
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
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            self._write_state(status="failed", error=error)
            self.event_writer.write({"event": "run_failed", "error": error})
            raise
