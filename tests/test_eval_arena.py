from __future__ import annotations

from pathlib import Path

import torch

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.eval import Arena, EvalResult, should_promote
from gumbel_az.model import create_network
from gumbel_az.model.common import NetworkOutput

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
    model = network.init(
        config.run.seed,
        game.observation_shape,
        game.num_actions,
    )

    arena = Arena(config, eval_dir=tmp_path / "eval")
    result = arena.evaluate_vs_random(model=model, checkpoint_version=1)

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
    model = network.init(
        config.run.seed,
        game.observation_shape,
        game.num_actions,
    )
    opponent_model = network.init(
        config.run.seed + 1,
        game.observation_shape,
        game.num_actions,
    )

    result = Arena(config, eval_dir=tmp_path / "eval").evaluate_vs_models(
        candidate_model=model,
        opponent_model=opponent_model,
        checkpoint_version=2,
        opponent_version=1,
    )

    assert result.games == 2
    assert result.wins + result.losses + result.draws == 2


def test_arena_network_action_moves_model_to_device_and_eval(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    game = create_game(config.game.name)

    class TrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls = 0
            self.eval_called = False

        def to(self, *args, **kwargs):
            self.to_calls += 1
            return super().to(*args, **kwargs)

        def eval(self):
            self.eval_called = True
            return super().eval()

        def forward(self, observations):
            batch_size = observations.shape[0]
            return NetworkOutput(
                policy_logits=torch.zeros(
                    (batch_size, game.num_actions),
                    device=observations.device,
                ),
                value=torch.zeros((batch_size,), device=observations.device),
            )

    model = TrackingModel()
    action = Arena(config, eval_dir=tmp_path / "eval", device="cpu")._network_action(
        model,
        game.init(),
    )

    assert action == 0
    assert model.to_calls == 1
    assert model.eval_called


def test_arena_prepares_model_once_for_repeated_network_actions(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    game = create_game(config.game.name)

    class TrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls += 1
            return super().to(*args, **kwargs)

        def forward(self, observations):
            batch_size = observations.shape[0]
            return NetworkOutput(
                policy_logits=torch.zeros(
                    (batch_size, game.num_actions),
                    device=observations.device,
                ),
                value=torch.zeros((batch_size,), device=observations.device),
            )

    model = TrackingModel()
    arena = Arena(config, eval_dir=tmp_path / "eval", device="cpu")
    state = game.init()

    arena._network_action(model, state)
    arena._network_action(model, state)

    assert model.to_calls == 1


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
