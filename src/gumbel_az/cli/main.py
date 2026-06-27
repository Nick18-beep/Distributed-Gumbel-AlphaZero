"""Top-level CLI entrypoint for the ``gaz`` command."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep
from typing import Annotated, Literal

import typer
from pydantic import ValidationError

from gumbel_az import __version__
from gumbel_az.cli.doctor import run_doctor
from gumbel_az.config import load_config
from gumbel_az.config.loader import save_resolved_config
from gumbel_az.config.schema import AppConfig
from gumbel_az.execution import SingleProcessExecutionBackend
from gumbel_az.logging import JsonlWriter, MetricWriter
from gumbel_az.orchestration import load_resume_context
from gumbel_az.storage import RunPaths, create_run_directory
from gumbel_az.storage.atomic import atomic_write_json

app = typer.Typer(
    name="gaz",
    help="Distributed Gumbel AlphaZero command line interface.",
    no_args_is_help=True,
)
cluster_app = typer.Typer(
    name="cluster",
    help="LAN cluster commands.",
    no_args_is_help=True,
)
app.add_typer(cluster_app, name="cluster")

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a YAML config file.",
    ),
]
OverridesOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help="Override config values with dotted keys, e.g. --set run.seed=123.",
    ),
]
ExecutionOption = Annotated[
    Literal["single_process", "local_multiprocess", "lan_ray"] | None,
    typer.Option("--execution", help="Override execution backend for this run."),
]


def _not_implemented(feature: str, roadmap_context: str) -> None:
    typer.echo(f"{feature} is registered but not implemented yet.", err=True)
    typer.echo(f"Roadmap context: {roadmap_context}.", err=True)
    raise typer.Exit(code=1)


def _validate_config(config: Path, overrides: list[str] | None) -> AppConfig:
    try:
        return load_config(config, overrides)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Invalid config {config}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _append_override(overrides: list[str] | None, override: str | None) -> list[str] | None:
    if override is None:
        return overrides
    merged = list(overrides or [])
    merged.append(override)
    return merged


def _create_runtime_dirs(root: Path) -> None:
    for relative in ("artifacts", "artifacts/runs", "artifacts/cache"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _create_dev_run(config: AppConfig) -> tuple[RunPaths, JsonlWriter, MetricWriter]:
    paths = create_run_directory(config)
    save_resolved_config(config, paths.run_dir)
    event_writer = JsonlWriter(paths.events_path)
    metric_writer = MetricWriter(paths.metrics_path)
    event_writer.write({"event": "run_initialized", "run_id": paths.run_id, "mode": "dev"})
    metric_writer.write_metrics(0, {"run_initialized": True})
    return paths, event_writer, metric_writer


def _reject_checkpoint_shape_overrides(overrides: list[str] | None) -> None:
    blocked_prefixes = ("game.", "model.", "algorithm.")
    for override in overrides or []:
        key = override.split("=", 1)[0]
        if key.startswith(blocked_prefixes):
            typer.echo(
                f"when --run-dir is used, checkpoint-shape overrides are not allowed: {key}",
                err=True,
            )
            raise typer.Exit(code=2)


def _ray_cli_path() -> str:
    ray_cli = shutil.which("ray")
    if ray_cli is None:
        raise RuntimeError("Ray CLI not found; run `uv sync --extra distributed`.")
    return ray_cli


def _detect_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def _ray_node_ip(host: str) -> str:
    if host in {"", "0.0.0.0", "::"}:
        return _detect_lan_ip()
    return host


def _ray_cluster_env() -> dict[str, str]:
    env = os.environ.copy()
    if sys.platform in {"win32", "darwin"}:
        env.setdefault("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", "1")
    return env


def _ray_cluster_platform_warning() -> str | None:
    if sys.platform == "win32":
        return (
            "Ray multi-node on Windows is experimental; "
            "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 was set for this command."
        )
    if sys.platform == "darwin":
        return (
            "Ray multi-node on macOS is experimental; "
            "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 was set for this command."
        )
    return None


def _apply_ray_cluster_env() -> None:
    os.environ.update(_ray_cluster_env())


def _ray_node_label(node: dict) -> str:
    address = node.get("NodeManagerAddress") or node.get("NodeManagerHostname") or "unknown"
    resources = node.get("Resources", {})
    cpu = resources.get("CPU", 0)
    gpu = resources.get("GPU", 0)
    return f"{address} cpu={cpu:g} gpu={gpu:g}"


def _wait_for_ray_workers(
    *,
    head: str,
    min_workers: int,
    timeout_sec: float,
    poll_sec: float,
) -> None:
    _apply_ray_cluster_env()
    try:
        import ray  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        typer.echo("Ray is not installed; run `uv sync --extra distributed`.", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        sys.modules.pop("ray", None)
        typer.echo(f"Ray could not be imported: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not ray.is_initialized():
        ray.init(address=head, ignore_reinit_error=True)

    typer.echo(f"waiting for {min_workers} Ray worker(s) at {head}...")
    seen_node_ids: set[str] = set()
    last_worker_count = -1
    deadline = monotonic() + timeout_sec
    while True:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        for node in alive_nodes:
            node_id = str(node.get("NodeID", _ray_node_label(node)))
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                typer.echo(f"ray node connected: {_ray_node_label(node)}")

        worker_count = max(0, len(alive_nodes) - 1)
        if worker_count != last_worker_count:
            last_worker_count = worker_count
            typer.echo(f"ray workers ready: {worker_count}/{min_workers}")
        if worker_count >= min_workers:
            typer.echo(f"required Ray workers connected: {worker_count}/{min_workers}")
            return
        if monotonic() >= deadline:
            typer.echo(
                f"Timed out waiting for Ray workers: {worker_count}/{min_workers} connected.",
                err=True,
            )
            raise typer.Exit(code=1)
        sleep(poll_sec)


def _prompt_int(label: str) -> int | None:
    typer.echo(f"{label}: ", nl=False)
    line = sys.stdin.readline()
    if line == "":
        return None
    try:
        return int(line.strip())
    except ValueError:
        typer.echo("input non valido: inserire un numero intero", err=True)
        return _prompt_int(label)


@app.callback()
def root() -> None:
    """Distributed Gumbel AlphaZero command line interface."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def init() -> None:
    """Create local runtime directories."""
    project_root = Path.cwd()
    _create_runtime_dirs(project_root)
    typer.echo(f"initialized: {project_root}")


@app.command()
def doctor(
    fix: Annotated[bool, typer.Option("--fix", help="Apply safe local fixes.")] = False,
    distributed: Annotated[
        bool,
        typer.Option("--distributed", help="Check optional Ray/distributed dependencies."),
    ] = False,
    cuda: Annotated[
        bool,
        typer.Option("--cuda", help="Diagnose PyTorch CUDA/GPU availability."),
    ] = False,
) -> None:
    """Run environment checks."""
    run_doctor(fix=fix, distributed=distributed, cuda=cuda)


@app.command()
def run(
    config: ConfigOption,
    overrides: OverridesOption = None,
    execution: ExecutionOption = None,
) -> None:
    """Run a training orchestration from a config file."""
    execution_override = None if execution is None else f"execution.backend={execution}"
    app_config = _validate_config(config, _append_override(overrides, execution_override))
    try:
        if app_config.execution.backend == "single_process":
            result = SingleProcessExecutionBackend().run(app_config)
        elif app_config.execution.backend == "local_multiprocess":
            from gumbel_az.execution.local_multiprocess import LocalMultiprocessExecutionBackend

            result = LocalMultiprocessExecutionBackend().run(app_config)
        elif app_config.execution.backend == "lan_ray":
            from gumbel_az.execution.lan_ray import LanRayExecutionBackend

            result = LanRayExecutionBackend().run(app_config)
        else:
            _not_implemented(
                f"gaz run with execution backend {app_config.execution.backend}",
                "Fase 25 - LAN Ray base",
            )
    except (KeyError, ValueError, RuntimeError, NotImplementedError, TimeoutError) as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"run {result.status}: {result.run_dir}")


@app.command()
def resume(
    run_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Run directory to resume.",
        ),
    ],
    rebuild_replay_index: Annotated[
        bool,
        typer.Option(
            "--rebuild-replay-index",
            help="Rebuild replay/index.json from valid local shards before reporting resume state.",
        ),
    ] = False,
) -> None:
    """Resume an existing run."""
    try:
        context = load_resume_context(run_dir, rebuild_replay=rebuild_replay_index)
    except (OSError, ValueError) as exc:
        typer.echo(f"Cannot resume {run_dir}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    state = context.run_state
    latest_version = (
        None if context.latest_checkpoint is None else context.latest_checkpoint.get("version")
    )
    typer.echo(
        "resume state: "
        f"status={state.get('status', 'unknown')} "
        f"train_step={state.get('train_step', 0)} "
        f"games_seen={state.get('games_seen', 0)} "
        f"replay_samples={context.replay_index.get('total_samples', 0)} "
        f"latest_checkpoint={latest_version}"
    )


@app.command()
def selfplay(
    config: ConfigOption,
    overrides: OverridesOption = None,
    games: Annotated[
        int | None,
        typer.Option("--games", min=1, help="Number of self-play games to generate."),
    ] = None,
) -> None:
    """Developer command for self-play only."""
    app_config = _validate_config(config, overrides)
    try:
        from gumbel_az.replay import ReplayWriter
        from gumbel_az.selfplay.worker import SelfPlayWorker

        paths, event_writer, metric_writer = _create_dev_run(app_config)
        games_to_play = games
        if games_to_play is None:
            games_to_play = min(
                app_config.stop.max_games or app_config.selfplay.games_per_iteration,
                app_config.selfplay.games_per_iteration,
            )
        worker = SelfPlayWorker(
            app_config,
            replay_writer=ReplayWriter(paths.run_dir / "replay"),
        )
        _, result = worker.play_batch(games_to_play, app_config.run.seed)
        event_writer.write(
            {
                "event": "selfplay_completed",
                "games": result.games,
                "positions": result.positions,
                "replay_shard": result.replay_shard,
            }
        )
        metric_writer.write_metrics(
            0,
            {
                "games_per_sec": result.games_per_sec,
                "positions_per_sec": result.positions_per_sec,
                "illegal_action_rate": result.illegal_action_rate,
                "policy_entropy_mean": result.policy_entropy_mean,
                "root_value_mean": result.root_value_mean,
            },
        )
        atomic_write_json(
            paths.run_state_path,
            {
                "run_id": paths.run_id,
                "status": "completed_selfplay",
                "games_seen": result.games,
                "samples_seen": result.positions,
                "replay_shard": result.replay_shard,
            },
        )
    except (KeyError, ValueError, RuntimeError, NotImplementedError) as exc:
        typer.echo(f"Self-play failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"selfplay completed: {paths.run_dir}")


@app.command()
def train(config: ConfigOption, overrides: OverridesOption = None) -> None:
    """Developer command for training only."""
    app_config = _validate_config(config, overrides)
    try:
        from gumbel_az.model.checkpoint import CheckpointManager
        from gumbel_az.replay import ReplayReader, ReplayWriter
        from gumbel_az.selfplay.worker import SelfPlayWorker
        from gumbel_az.training.trainer import Trainer

        paths, event_writer, metric_writer = _create_dev_run(app_config)
        replay_dir = paths.run_dir / "replay"
        replay_writer = ReplayWriter(replay_dir)
        warmup_games = min(
            app_config.stop.max_games or app_config.selfplay.games_per_iteration,
            app_config.selfplay.games_per_iteration,
        )
        worker = SelfPlayWorker(app_config, replay_writer=replay_writer)
        _, selfplay_result = worker.play_batch(warmup_games, app_config.run.seed)
        trainer = Trainer(
            app_config,
            replay_reader=ReplayReader(replay_dir),
            checkpoint_manager=CheckpointManager(paths.run_dir / "checkpoints"),
            metric_writer=metric_writer,
        )
        steps = app_config.training.steps_per_iteration
        if app_config.stop.max_train_steps is not None:
            steps = min(steps, app_config.stop.max_train_steps)
        train_result = trainer.run(max_steps=steps)
        event_writer.write(
            {
                "event": "training_completed",
                "train_step": train_result.checkpoint_version,
                "checkpoint_version": train_result.checkpoint_version,
            }
        )
        metric_writer.write_metrics(
            train_result.checkpoint_version,
            {
                "train_samples_per_sec": train_result.samples_per_sec,
                "replay_sample_age_mean": train_result.replay_sample_age_mean,
                "checkpoint_version": train_result.checkpoint_version,
            },
        )
        atomic_write_json(
            paths.run_state_path,
            {
                "run_id": paths.run_id,
                "status": "completed_train",
                "train_step": train_result.checkpoint_version,
                "games_seen": selfplay_result.games,
                "samples_seen": selfplay_result.positions,
                "replay_shard": selfplay_result.replay_shard,
                "checkpoint_version": train_result.checkpoint_version,
            },
        )
    except (KeyError, ValueError, RuntimeError, NotImplementedError) as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"train completed: {paths.run_dir}")


@app.command("eval")
def eval_command(config: ConfigOption, overrides: OverridesOption = None) -> None:
    """Developer command for evaluation only."""
    app_config = _validate_config(config, overrides)
    try:
        from gumbel_az.envs import create_game
        from gumbel_az.eval import Arena
        from gumbel_az.model import create_network
        from gumbel_az.runtime import detect_torch_runtime

        paths, event_writer, metric_writer = _create_dev_run(app_config)
        game = create_game(app_config.game.name)
        network = create_network(app_config.model, num_actions=game.num_actions)
        runtime = detect_torch_runtime()
        model = network.init(
            app_config.run.seed,
            game.observation_shape,
            game.num_actions,
            device=runtime.device,
        )
        result = Arena(
            app_config,
            eval_dir=paths.run_dir / "eval",
            event_writer=event_writer,
            device=runtime.device,
        ).evaluate_vs_random(model=model, checkpoint_version=0)
        metric_writer.write_metrics(
            0,
            {
                "eval_win_rate": result.win_rate,
                "eval_games_per_sec": result.games_per_sec,
            },
        )
        atomic_write_json(
            paths.run_state_path,
            {
                "run_id": paths.run_id,
                "status": "completed_eval",
                "eval": {
                    "checkpoint_version": result.checkpoint_version,
                    "games": result.games,
                    "wins": result.wins,
                    "losses": result.losses,
                    "draws": result.draws,
                    "win_rate": result.win_rate,
                    "games_per_sec": result.games_per_sec,
                },
            },
        )
    except (KeyError, ValueError, RuntimeError, NotImplementedError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"eval completed: {paths.run_dir}")


@app.command()
def play(
    config: ConfigOption,
    overrides: OverridesOption = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Run directory containing checkpoints to play against.",
        ),
    ] = None,
    checkpoint: Annotated[
        Literal["latest", "best"],
        typer.Option("--checkpoint", help="Checkpoint pointer to load from --run-dir."),
    ] = "best",
    human_player: Annotated[
        int,
        typer.Option("--human-player", min=0, max=1, help="Human player index."),
    ] = 0,
    moves: Annotated[
        list[int] | None,
        typer.Option("--move", help="Scripted human move; repeat for smoke tests."),
    ] = None,
) -> None:
    """Play against an agent checkpoint."""
    if run_dir is not None and (run_dir / "config.resolved.yaml").exists():
        _reject_checkpoint_shape_overrides(overrides)
        app_config = _validate_config(run_dir / "config.resolved.yaml", overrides)
    else:
        app_config = _validate_config(config, overrides)
    from gumbel_az.envs import create_game
    from gumbel_az.play import play_scripted_game, result_message
    from gumbel_az.play.session import AgentPlayer, apply_human_action, load_play_model

    if moves is not None:
        result = play_scripted_game(
            app_config,
            human_actions=moves,
            run_dir=run_dir,
            checkpoint=checkpoint,
            human_player=human_player,
        )
        typer.echo(result.board_text)
        typer.echo(result.message)
        raise typer.Exit(code=0)

    game = create_game(app_config.game.name)
    model = load_play_model(app_config, run_dir=run_dir, checkpoint=checkpoint)
    agent = AgentPlayer(app_config, model=model)
    state = game.init(app_config.run.seed)
    while not bool(game.is_terminal(state)):
        typer.echo(game.render_text(state))
        current_player = int(game.current_player(state))
        if current_player == human_player:
            action = _prompt_int("colonna")
            if action is None:
                typer.echo("partita interrotta", err=True)
                raise typer.Exit(code=1)
            try:
                state = apply_human_action(game, state, action)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
        else:
            action = agent.select_action(
                state,
                seed=app_config.run.seed + 50_000 + int(state.move_count),
            )
            typer.echo(f"agente: {action}")
            state = game.step(state, action)
    typer.echo(game.render_text(state))
    typer.echo(result_message(game, state, human_player=human_player))


@app.command()
def benchmark(
    config: ConfigOption,
    overrides: OverridesOption = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Directory where benchmark JSONL output is written.",
        ),
    ] = None,
) -> None:
    """Run project benchmarks."""
    app_config = _validate_config(config, overrides)
    from gumbel_az.benchmark import run_benchmark

    output_path = run_benchmark(app_config, output_dir=output_dir)
    typer.echo(f"benchmark written: {output_path}")


@app.command("inspect")
def inspect_command(
    subject: Annotated[str, typer.Argument(help="Object type: run, replay, or checkpoint.")],
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Path to inspect.",
        ),
    ],
) -> None:
    """Inspect a run, replay directory, or checkpoint."""
    allowed_subjects = {"run", "replay", "checkpoint"}
    if subject not in allowed_subjects:
        allowed = ", ".join(sorted(allowed_subjects))
        raise typer.BadParameter(f"subject must be one of: {allowed}")
    if subject == "run":
        from gumbel_az.analysis.inspect import inspect_run

        report = inspect_run(path)
    elif subject == "replay":
        from gumbel_az.analysis.inspect import inspect_replay

        report = inspect_replay(path)
    else:
        from gumbel_az.analysis.inspect import inspect_checkpoint

        report = inspect_checkpoint(path)
    typer.echo(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


@cluster_app.command("head")
def cluster_head(
    config: ConfigOption,
    overrides: OverridesOption = None,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help=(
                "Ray node IP. Use 0.0.0.0 to auto-detect and print the LAN IP workers should use."
            ),
        ),
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Ray head port.")] = 6379,
    wait_workers: Annotated[
        bool,
        typer.Option("--wait-workers", help="Keep this terminal open until workers connect."),
    ] = False,
    min_workers: Annotated[
        int,
        typer.Option("--min-workers", min=0, help="Minimum Ray worker nodes to wait for."),
    ] = 1,
    timeout_sec: Annotated[
        float,
        typer.Option("--timeout-sec", min=1.0, help="Seconds to wait for workers."),
    ] = 300.0,
    poll_sec: Annotated[
        float,
        typer.Option("--poll-sec", min=0.1, help="Worker polling interval in seconds."),
    ] = 2.0,
) -> None:
    """Start a Ray head node."""
    _validate_config(config, overrides)
    node_ip = _ray_node_ip(host)
    ray_env = _ray_cluster_env()
    platform_warning = _ray_cluster_platform_warning()
    if platform_warning is not None:
        typer.echo(platform_warning)
    try:
        import ray  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        typer.echo("Ray is not installed; run `uv sync --extra distributed`.", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        sys.modules.pop("ray", None)
        typer.echo(f"Ray could not be imported: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    del ray
    try:
        subprocess.run(
            [
                _ray_cli_path(),
                "start",
                "--head",
                f"--node-ip-address={node_ip}",
                f"--port={port}",
                "--include-dashboard=false",
                "--disable-usage-stats",
            ],
            check=True,
            env=ray_env,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        typer.echo(f"Cannot start Ray head: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ray head ready: {node_ip}:{port}")
    if node_ip != host:
        typer.echo(f"bind host {host!r} resolved to Ray node IP {node_ip!r}")
    if wait_workers:
        _wait_for_ray_workers(
            head=f"{node_ip}:{port}",
            min_workers=min_workers,
            timeout_sec=timeout_sec,
            poll_sec=poll_sec,
        )


@cluster_app.command("worker")
def cluster_worker(
    head: Annotated[str, typer.Option("--head", help="Ray head address host:port.")],
    config: ConfigOption,
    overrides: OverridesOption = None,
    auto: Annotated[bool, typer.Option("--auto", help="Register worker capabilities.")] = False,
) -> None:
    """Connect a worker to a Ray head."""
    _validate_config(config, overrides)
    ray_env = _ray_cluster_env()
    platform_warning = _ray_cluster_platform_warning()
    if platform_warning is not None:
        typer.echo(platform_warning)
    try:
        import ray  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        typer.echo("Ray is not installed; run `uv sync --extra distributed`.", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        sys.modules.pop("ray", None)
        typer.echo(f"Ray could not be imported: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    del ray
    from gumbel_az.execution.lan_ray import detect_worker_capabilities

    try:
        subprocess.run(
            [_ray_cli_path(), "start", f"--address={head}"],
            check=True,
            env=ray_env,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        typer.echo(f"Cannot start Ray worker: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if auto:
        capabilities = detect_worker_capabilities()
        typer.echo(json.dumps(capabilities.__dict__, sort_keys=True))
    else:
        typer.echo(f"ray worker connected: {head}")


@cluster_app.command("wait")
def cluster_wait(
    head: Annotated[str, typer.Option("--head", help="Ray head address host:port.")],
    min_workers: Annotated[
        int,
        typer.Option("--min-workers", min=0, help="Minimum Ray worker nodes to wait for."),
    ] = 1,
    timeout_sec: Annotated[
        float,
        typer.Option("--timeout-sec", min=1.0, help="Seconds to wait for workers."),
    ] = 300.0,
    poll_sec: Annotated[
        float,
        typer.Option("--poll-sec", min=0.1, help="Worker polling interval in seconds."),
    ] = 2.0,
) -> None:
    """Wait for Ray worker nodes and log them as they connect."""
    _wait_for_ray_workers(
        head=head,
        min_workers=min_workers,
        timeout_sec=timeout_sec,
        poll_sec=poll_sec,
    )


@cluster_app.command("status")
def cluster_status(
    head: Annotated[
        str | None,
        typer.Option("--head", help="Optional Ray head address host:port."),
    ] = None,
) -> None:
    """Print local Ray cluster status."""
    ray_env = _ray_cluster_env()
    try:
        import ray  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        typer.echo("Ray is not installed; run `uv sync --extra distributed`.", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        sys.modules.pop("ray", None)
        typer.echo(f"Ray could not be imported: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    del ray
    command = [_ray_cli_path(), "status"]
    if head is not None:
        command.append(f"--address={head}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=ray_env,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        typer.echo(f"Cannot query Ray status: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.stdout.strip())


def main() -> None:
    """Run the CLI application."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    main()
