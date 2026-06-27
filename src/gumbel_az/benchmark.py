"""Project smoke benchmarks."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.eval import Arena
from gumbel_az.logging import JsonlWriter
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.replay import ReplayReader, ReplayWriter
from gumbel_az.selfplay.worker import SelfPlayWorker
from gumbel_az.training.train_state import create_train_state, train_step


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%fZ")


def _config_snapshot(config: AppConfig) -> dict[str, Any]:
    return {
        "game": config.game.model_dump(mode="json"),
        "algorithm": config.algorithm.model_dump(mode="json"),
        "search": config.search.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
    }


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_benchmark(config: AppConfig, *, output_dir: Path | None = None) -> Path:
    benchmark_dir = output_dir or Path("artifacts") / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_path = benchmark_dir / f"benchmark_{_utc_stamp()}.jsonl"
    writer = JsonlWriter(output_path)
    run_dir = benchmark_dir / f"workspace_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(
        jax.random.PRNGKey(config.run.seed),
        game.observation_shape,
        game.num_actions,
    )
    writer.write(
        {
            "benchmark": "metadata",
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "git_commit": _git_commit(),
            "config": _config_snapshot(config),
        }
    )

    tx, schedule = create_optimizer(config.training)
    state = create_train_state(params=params, apply_fn=network.apply, tx=tx)
    batch = {
        "observation": jnp.zeros((config.training.batch_size, *game.observation_shape)),
        "policy_target": jnp.full(
            (config.training.batch_size, game.num_actions),
            1.0 / game.num_actions,
        ),
        "value_target": jnp.zeros((config.training.batch_size,)),
    }
    compile_start = perf_counter()
    state, metrics = train_step(state, batch, schedule(state.step))
    jax.block_until_ready(metrics["total_loss"])
    compile_seconds = perf_counter() - compile_start
    steady_start = perf_counter()
    iterations = 10
    for _ in range(iterations):
        state, metrics = train_step(state, batch, schedule(state.step))
    jax.block_until_ready(metrics["total_loss"])
    steady_seconds = perf_counter() - steady_start
    writer.write(
        {
            "benchmark": "train_step",
            "compile_seconds": compile_seconds,
            "warmup_iterations": 1,
            "steady_iterations": iterations,
            "samples_per_sec": iterations
            * config.training.batch_size
            / max(steady_seconds, 1.0e-9),
        }
    )

    replay_writer = ReplayWriter(run_dir / "replay")
    selfplay_worker = SelfPlayWorker(config, replay_writer=replay_writer, params=params)
    warmup_games = 1
    selfplay_worker.play_batch(warmup_games, config.run.seed)
    measured_games = min(config.selfplay.games_per_iteration, config.selfplay.batch_size)
    selfplay_start = perf_counter()
    _, selfplay_result = selfplay_worker.play_batch(measured_games, config.run.seed + warmup_games)
    selfplay_seconds = perf_counter() - selfplay_start
    writer.write(
        {
            "benchmark": "selfplay",
            "warmup_games": warmup_games,
            "measured_games": measured_games,
            "games_per_sec": selfplay_result.games_per_sec,
            "positions_per_sec": selfplay_result.positions_per_sec,
            "search_simulations_per_sec": selfplay_result.positions
            * config.search.simulations_per_move
            / max(selfplay_seconds, 1.0e-9),
            "replay_write_throughput_samples_per_sec": selfplay_result.positions
            / max(selfplay_seconds, 1.0e-9),
        }
    )
    read_start = perf_counter()
    samples = ReplayReader(run_dir / "replay").read_all()
    read_seconds = perf_counter() - read_start
    writer.write(
        {
            "benchmark": "replay_read",
            "samples": len(samples),
            "samples_per_sec": len(samples) / max(read_seconds, 1.0e-9),
        }
    )

    manager = CheckpointManager(run_dir / "checkpoints")
    save_start = perf_counter()
    manager.save(
        version=1,
        state={"params": state.params, "opt_state": state.opt_state, "step": state.step},
        metadata={"training_step": int(state.step), "game": game.name},
        best=True,
    )
    save_seconds = perf_counter() - save_start
    load_start = perf_counter()
    manager.load()
    load_seconds = perf_counter() - load_start
    writer.write(
        {
            "benchmark": "checkpoint",
            "save_seconds": save_seconds,
            "load_seconds": load_seconds,
        }
    )

    eval_config = config.model_copy(update={"eval": config.eval.model_copy(update={"games": 2})})
    eval_start = perf_counter()
    eval_result = Arena(eval_config, eval_dir=run_dir / "eval").evaluate_vs_random(
        params=state.params,
        checkpoint_version=1,
    )
    eval_seconds = perf_counter() - eval_start
    writer.write(
        {
            "benchmark": "evaluation",
            "games": eval_result.games,
            "games_per_sec": eval_result.games / max(eval_seconds, 1.0e-9),
        }
    )
    return output_path
