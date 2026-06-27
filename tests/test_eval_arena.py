from __future__ import annotations

from pathlib import Path

import jax

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.eval import Arena, EvalResult, should_promote
from gumbel_az.model import create_network

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_arena_evaluates_against_random_and_writes_matches(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "eval.games=2",
        ],
    )
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(
        jax.random.PRNGKey(config.run.seed),
        game.observation_shape,
        game.num_actions,
    )

    arena = Arena(config, eval_dir=tmp_path / "eval")
    result = arena.evaluate_vs_random(params=params, checkpoint_version=1)

    assert result.games == 2
    assert result.wins + result.losses + result.draws == 2
    assert 0.0 <= result.win_rate <= 1.0
    assert (tmp_path / "eval" / "matches.jsonl").exists()


def test_arena_evaluates_checkpoint_against_checkpoint(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "eval.games=2",
        ],
    )
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(
        jax.random.PRNGKey(config.run.seed),
        game.observation_shape,
        game.num_actions,
    )
    opponent_params = network.init(
        jax.random.PRNGKey(config.run.seed + 1),
        game.observation_shape,
        game.num_actions,
    )

    result = Arena(config, eval_dir=tmp_path / "eval").evaluate_vs_params(
        candidate_params=params,
        opponent_params=opponent_params,
        checkpoint_version=2,
        opponent_version=1,
    )

    assert result.games == 2
    assert result.wins + result.losses + result.draws == 2


def test_promotion_requires_enough_games_and_threshold() -> None:
    result = EvalResult(
        checkpoint_version=1,
        games=4,
        wins=3,
        losses=1,
        draws=0,
        win_rate=0.75,
        games_per_sec=100.0,
    )

    assert should_promote(result, min_games=4, promotion_win_rate=0.55)
    assert not should_promote(result, min_games=8, promotion_win_rate=0.55)
    assert not should_promote(result, min_games=4, promotion_win_rate=0.80)
