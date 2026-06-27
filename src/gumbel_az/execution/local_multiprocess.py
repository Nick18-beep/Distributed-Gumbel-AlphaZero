"""Local multiprocessing execution backend."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gumbel_az.config.loader import save_resolved_config
from gumbel_az.config.schema import AppConfig
from gumbel_az.eval import Arena, should_promote
from gumbel_az.execution.base import ExecutionResult
from gumbel_az.logging import JsonlWriter, MetricWriter
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.orchestration.scheduler import LocalScheduler, SchedulerSignals
from gumbel_az.replay import ReplayReader, ReplayWriter
from gumbel_az.runtime import detect_runtime_backend
from gumbel_az.storage import create_run_directory
from gumbel_az.storage.atomic import atomic_write_json
from gumbel_az.training.trainer import Trainer


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _selfplay_process(
    config_data: dict[str, Any],
    run_dir: str,
    games: int,
    seed: int,
    out,
) -> None:
    try:
        config = AppConfig.model_validate(config_data)
        runtime = detect_runtime_backend()
        replay_writer = ReplayWriter(Path(run_dir) / "replay")
        if runtime.name == "jax":
            from gumbel_az.selfplay.worker import SelfPlayWorker

            worker = SelfPlayWorker(config, replay_writer=replay_writer)
        elif runtime.name == "torch":
            from gumbel_az.selfplay.torch_fallback import TorchFallbackSelfPlayWorker

            worker = TorchFallbackSelfPlayWorker(config, replay_writer=replay_writer)
        else:
            raise RuntimeError(runtime.reason)
        _, result = worker.play_batch(games, seed)
        out.put(
            {
                "ok": True,
                "runtime_backend": runtime.name,
                "games": result.games,
                "positions": result.positions,
                "replay_shard": result.replay_shard,
                "games_per_sec": result.games_per_sec,
                "positions_per_sec": result.positions_per_sec,
                "illegal_action_rate": result.illegal_action_rate,
                "policy_entropy_mean": result.policy_entropy_mean,
                "root_value_mean": result.root_value_mean,
            }
        )
    except BaseException as exc:
        out.put({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})


class LocalMultiprocessExecutionBackend:
    """Run self-play in a child process and training/evaluation in the parent."""

    name = "local_multiprocess"

    def run(self, config: AppConfig) -> ExecutionResult:
        if config.execution.backend != self.name:
            raise ValueError(
                f"LocalMultiprocessExecutionBackend cannot run backend {config.execution.backend!r}"
            )
        paths = create_run_directory(config)
        save_resolved_config(config, paths.run_dir)
        event_writer = JsonlWriter(paths.events_path)
        metric_writer = MetricWriter(paths.metrics_path)
        runtime = detect_runtime_backend()
        state = {
            "run_id": paths.run_id,
            "backend": self.name,
            "runtime_backend": runtime.name,
            "runtime_backend_reason": runtime.reason,
            "status": "running",
            "created_at": _utc_now(),
            "config_path": str(paths.resolved_config_path),
            "train_step": 0,
            "games_seen": 0,
            "samples_seen": 0,
            "worker_processes_started": 0,
        }
        atomic_write_json(paths.run_state_path, state)
        event_writer.write({"event": "run_initialized", "run_id": paths.run_id})

        games = min(
            config.stop.max_games or config.selfplay.games_per_iteration,
            config.selfplay.games_per_iteration,
        )
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_selfplay_process,
            args=(
                config.model_dump(mode="json"),
                str(paths.run_dir),
                games,
                config.run.seed,
                result_queue,
            ),
            name="gaz-selfplay-worker-0",
        )
        event_writer.write({"event": "worker_process_starting", "worker": process.name})
        process.start()
        state["worker_processes_started"] = 1
        atomic_write_json(paths.run_state_path, state)
        process.join(timeout=config.stop.max_wall_time_sec or None)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            state["status"] = "interrupted"
            state["worker_exitcode"] = process.exitcode
            atomic_write_json(paths.run_state_path, state)
            event_writer.write(
                {
                    "event": "worker_process_timeout",
                    "worker": process.name,
                    "exitcode": process.exitcode,
                }
            )
            raise TimeoutError("local_multiprocess self-play worker timed out")
        try:
            message = result_queue.get(timeout=5.0)
        except queue.Empty as exc:
            state["status"] = "failed"
            state["worker_exitcode"] = process.exitcode
            atomic_write_json(paths.run_state_path, state)
            event_writer.write(
                {
                    "event": "worker_process_missing_result",
                    "worker": process.name,
                    "exitcode": process.exitcode,
                }
            )
            raise RuntimeError(
                f"self-play worker exited without result: {process.exitcode}"
            ) from exc
        if not message.get("ok"):
            state["status"] = "failed"
            state["worker_exitcode"] = process.exitcode
            state["worker_error"] = message.get("error", "unknown worker error")
            state["worker_traceback"] = message.get("traceback", "")
            atomic_write_json(paths.run_state_path, state)
            event_writer.write(
                {
                    "event": "worker_process_failed",
                    "worker": process.name,
                    "exitcode": process.exitcode,
                    "error": message.get("error"),
                    "traceback": message.get("traceback"),
                }
            )
            raise RuntimeError(
                f"self-play worker failed: {message.get('error', 'unknown worker error')}"
            )
        event_writer.write(
            {
                "event": "worker_process_completed",
                "worker": process.name,
                "exitcode": process.exitcode,
                **message,
            }
        )
        metric_writer.write_metrics(
            0,
            {
                "games_per_sec": message["games_per_sec"],
                "positions_per_sec": message["positions_per_sec"],
                "illegal_action_rate": message["illegal_action_rate"],
                "policy_entropy_mean": message["policy_entropy_mean"],
                "root_value_mean": message["root_value_mean"],
                "selfplay_queue_depth": 0,
                "replay_write_queue_depth": 0,
            },
        )

        if runtime.name != "jax":
            state.update(
                {
                    "status": "completed_torch_fallback",
                    "runtime_backend": runtime.name,
                    "worker_runtime_backend": message["runtime_backend"],
                    "worker_exitcode": process.exitcode,
                    "games_seen": message["games"],
                    "samples_seen": message["positions"],
                    "replay_shard": message["replay_shard"],
                }
            )
            atomic_write_json(paths.run_state_path, state)
            return ExecutionResult(paths.run_id, paths.run_dir, state["status"])

        replay_reader = ReplayReader(paths.run_dir / "replay")
        replay_samples_available = sum(
            int(entry.get("samples", 0)) for entry in replay_reader.shard_paths_metadata()
        )
        decision = LocalScheduler(config).decide(
            SchedulerSignals(
                replay_samples_available=replay_samples_available,
                selfplay_queue_depth=0,
                replay_write_queue_depth=0,
                evaluation_pending=config.eval.enabled,
            )
        )
        event_writer.write(
            {
                "event": "scheduler_decision",
                "stage": "local_multiprocess_training",
                "decision": decision.to_event(),
            }
        )
        checkpoint_manager = CheckpointManager(paths.run_dir / "checkpoints")
        trainer = Trainer(
            config,
            replay_reader=replay_reader,
            checkpoint_manager=checkpoint_manager,
            metric_writer=metric_writer,
        )
        max_steps = min(
            config.training.steps_per_iteration,
            config.stop.max_train_steps or config.training.steps_per_iteration,
        )
        try:
            train_result = trainer.run(max_steps=max_steps)
            eval_payload = None
            if config.eval.enabled:
                arena = Arena(config, eval_dir=paths.run_dir / "eval", event_writer=event_writer)
                eval_result = arena.evaluate_vs_random(
                    params=train_result.state.params,
                    checkpoint_version=train_result.checkpoint_version,
                )
                promoted = (
                    should_promote(
                        eval_result,
                        min_games=config.eval.games,
                        promotion_win_rate=config.eval.promotion_win_rate,
                    )
                    or not checkpoint_manager.best_path.exists()
                )
                if promoted:
                    checkpoint_manager.promote(train_result.checkpoint_version)
                eval_payload = {
                    "checkpoint_version": eval_result.checkpoint_version,
                    "games": eval_result.games,
                    "wins": eval_result.wins,
                    "losses": eval_result.losses,
                    "draws": eval_result.draws,
                    "win_rate": eval_result.win_rate,
                    "promoted": promoted,
                }
        except Exception as exc:
            state.update(
                {
                    "status": "failed",
                    "failure_stage": "local_multiprocess_training",
                    "error": repr(exc),
                    "worker_runtime_backend": message["runtime_backend"],
                    "worker_exitcode": process.exitcode,
                    "games_seen": message["games"],
                    "samples_seen": message["positions"],
                    "replay_shard": message["replay_shard"],
                }
            )
            atomic_write_json(paths.run_state_path, state)
            event_writer.write(
                {
                    "event": "local_multiprocess_training_failed",
                    "error": repr(exc),
                    "worker": process.name,
                    "exitcode": process.exitcode,
                }
            )
            raise
        state.update(
            {
                "status": "completed",
                "train_step": train_result.checkpoint_version,
                "games_seen": message["games"],
                "samples_seen": message["positions"],
                "replay_shard": message["replay_shard"],
                "worker_runtime_backend": message["runtime_backend"],
                "worker_exitcode": process.exitcode,
                "checkpoint_version": train_result.checkpoint_version,
                "eval": eval_payload,
            }
        )
        atomic_write_json(paths.run_state_path, state)
        return ExecutionResult(paths.run_id, paths.run_dir, "completed")
