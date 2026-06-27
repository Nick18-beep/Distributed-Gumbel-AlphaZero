"""Benchmark helpers for PyTorch training and self-play."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.eval import Arena
from gumbel_az.logging import JsonlWriter
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.replay import ReplayReader, ReplayWriter
from gumbel_az.runtime import detect_torch_runtime
from gumbel_az.selfplay.worker import SelfPlayWorker
from gumbel_az.storage import create_run_directory
from gumbel_az.training.train_state import TorchTrainState, train_step
from gumbel_az.training.trainer import Trainer


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def benchmark_training(config_path: Path, *, output: Path) -> dict:
    config = load_config(config_path)
    runtime = detect_torch_runtime()
    device = torch.device(runtime.device)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    model = network.init(config.run.seed, game.observation_shape, game.num_actions, device=device)
    optimizer, schedule = create_optimizer(model, config.training)
    state = TorchTrainState(
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        step=0,
        device=device,
        scaler=torch.amp.GradScaler("cuda") if device.type == "cuda" else None,
    )
    batch = {
        "observation": torch.zeros(
            (config.training.batch_size, *game.observation_shape),
            dtype=torch.float32,
            device=device,
        ),
        "policy_target": torch.full(
            (config.training.batch_size, game.num_actions),
            1.0 / game.num_actions,
            dtype=torch.float32,
            device=device,
        ),
        "value_target": torch.zeros(
            (config.training.batch_size,),
            dtype=torch.float32,
            device=device,
        ),
    }
    train_step(state, batch, config.training)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(config.training.steps_per_iteration):
        state, metrics = train_step(state, batch, config.training)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    payload = {
        "benchmark": "training",
        "runtime": runtime.name,
        "device": runtime.device,
        "steps": config.training.steps_per_iteration,
        "batch_size": config.training.batch_size,
        "samples_per_sec": (
            config.training.steps_per_iteration * config.training.batch_size
        )
        / max(elapsed, 1.0e-9),
        "seconds": elapsed,
        "total_loss": metrics["total_loss"],
    }
    _write_jsonl(output, payload)
    return payload


def benchmark_selfplay(config_path: Path, *, output: Path) -> dict:
    config = load_config(config_path)
    paths = create_run_directory(config)
    worker = SelfPlayWorker(config, replay_writer=ReplayWriter(paths.run_dir / "replay"))
    _, result = worker.play_batch(config.selfplay.games_per_iteration, config.run.seed)
    payload = {
        "benchmark": "selfplay",
        "games": result.games,
        "positions": result.positions,
        "games_per_sec": result.games_per_sec,
        "positions_per_sec": result.positions_per_sec,
        "replay_shard": result.replay_shard,
    }
    _write_jsonl(output, payload)
    return payload


def benchmark_end_to_end(config_path: Path, *, output: Path) -> dict:
    config = load_config(config_path)
    paths = create_run_directory(config)
    replay_writer = ReplayWriter(paths.run_dir / "replay")
    worker = SelfPlayWorker(config, replay_writer=replay_writer)
    _, selfplay = worker.play_batch(config.selfplay.games_per_iteration, config.run.seed)
    checkpoint_manager = CheckpointManager(paths.run_dir / "checkpoints")
    trainer = Trainer(
        config,
        replay_reader=ReplayReader(paths.run_dir / "replay"),
        checkpoint_manager=checkpoint_manager,
    )
    train = trainer.run(max_steps=config.training.steps_per_iteration)
    eval_result = Arena(config, eval_dir=paths.run_dir / "eval").evaluate_vs_random(
        model=train.state.model,
        checkpoint_version=train.checkpoint_version,
    )
    JsonlWriter(paths.logs_dir / "benchmark.jsonl").write(
        {
            "selfplay_games": selfplay.games,
            "train_step": train.checkpoint_version,
            "eval_win_rate": eval_result.win_rate,
        }
    )
    payload = {
        "benchmark": "end_to_end",
        "run_dir": str(paths.run_dir),
        "selfplay_games_per_sec": selfplay.games_per_sec,
        "train_samples_per_sec": train.samples_per_sec,
        "eval_games_per_sec": eval_result.games_per_sec,
    }
    _write_jsonl(output, payload)
    return payload


def run_benchmark(config, *, output_dir: Path | None = None) -> Path:
    output_root = output_dir or Path("artifacts/benchmarks")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%fZ")
    output = output_root / f"benchmark_{stamp}.jsonl"
    runtime = detect_torch_runtime()
    paths = create_run_directory(config)
    _write_jsonl(
        output,
        {
            "benchmark": "metadata",
            "runtime": runtime.name,
            "device": runtime.device,
            "run_dir": str(paths.run_dir),
        },
    )

    replay_writer = ReplayWriter(paths.run_dir / "replay")
    worker = SelfPlayWorker(config, replay_writer=replay_writer)
    worker.play_batch(1, config.run.seed)
    measured_games = min(
        config.selfplay.games_per_iteration,
        config.stop.max_games or config.selfplay.games_per_iteration,
    )
    _, selfplay = worker.play_batch(
        measured_games,
        config.run.seed + 10_000,
    )
    _write_jsonl(
        output,
        {
            "benchmark": "selfplay",
            "warmup_games": 1,
            "measured_games": selfplay.games,
            "games_per_sec": selfplay.games_per_sec,
            "positions_per_sec": selfplay.positions_per_sec,
        },
    )

    replay_reader = ReplayReader(paths.run_dir / "replay")
    started = perf_counter()
    samples = replay_reader.read_all()
    _write_jsonl(
        output,
        {
            "benchmark": "replay_read",
            "samples": len(samples),
            "seconds": perf_counter() - started,
        },
    )

    checkpoint_manager = CheckpointManager(paths.run_dir / "checkpoints")
    trainer = Trainer(
        config,
        replay_reader=replay_reader,
        checkpoint_manager=checkpoint_manager,
    )
    measured_steps = min(
        config.training.steps_per_iteration,
        config.stop.max_train_steps or config.training.steps_per_iteration,
    )
    train = trainer.run(max_steps=measured_steps)
    _write_jsonl(
        output,
        {
            "benchmark": "train_step",
            "steps": train.steps,
            "samples_per_sec": train.samples_per_sec,
            "checkpoint_version": train.checkpoint_version,
        },
    )
    _write_jsonl(
        output,
        {
            "benchmark": "checkpoint",
            "version": train.checkpoint_version,
            "latest_exists": checkpoint_manager.latest_path.exists(),
        },
    )

    eval_result = Arena(config, eval_dir=paths.run_dir / "eval").evaluate_vs_random(
        model=train.state.model,
        checkpoint_version=train.checkpoint_version,
    )
    _write_jsonl(
        output,
        {
            "benchmark": "evaluation",
            "games": eval_result.games,
            "win_rate": eval_result.win_rate,
            "games_per_sec": eval_result.games_per_sec,
        },
    )
    return output
