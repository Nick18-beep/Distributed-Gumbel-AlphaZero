from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gumbel_az.cli.main import app
from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.execution import SingleProcessExecutionBackend
from gumbel_az.play import play_scripted_game
from gumbel_az.play.session import apply_human_action

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"
REAL_CONFIG = PROJECT_ROOT / "configs" / "connect_four.yaml"


def test_play_scripted_game_makes_human_and_agent_moves() -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            "search.simulations_per_move=2",
        ],
    )

    result = play_scripted_game(config, human_actions=[3], human_player=0)

    assert result.moves[0] == 3
    assert len(result.moves) >= 2
    assert "0 1 2 3 4 5 6" in result.board_text


def test_play_rejects_illegal_human_move() -> None:
    game = create_game("connect_four")
    state = game.init(None)

    with pytest.raises(ValueError, match="mossa illegale"):
        apply_human_action(game, state, 99)


def test_cli_play_scripted_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "play",
            "--config",
            str(DEBUG_CONFIG),
            "--set",
            "search.simulations_per_move=2",
            "--move",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "0 1 2 3 4 5 6" in result.output


def test_cli_play_eof_exits_cleanly() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "play",
            "--config",
            str(DEBUG_CONFIG),
            "--set",
            "search.simulations_per_move=2",
        ],
        input="",
    )

    assert result.exit_code == 1
    assert "partita interrotta" in result.output
    assert "Aborted" not in result.output


def test_cli_play_uses_run_resolved_config_for_checkpoint(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )
    run = SingleProcessExecutionBackend().run(config)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "play",
            "--config",
            str(REAL_CONFIG),
            "--run-dir",
            str(run.run_dir),
            "--checkpoint",
            "best",
            "--move",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "0 1 2 3 4 5 6" in result.output


def test_cli_play_rejects_structural_override_with_checkpoint(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )
    run = SingleProcessExecutionBackend().run(config)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "play",
            "--config",
            str(DEBUG_CONFIG),
            "--run-dir",
            str(run.run_dir),
            "--set",
            "model.hidden_size=128",
            "--move",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "checkpoint-shape overrides are not allowed" in result.output
