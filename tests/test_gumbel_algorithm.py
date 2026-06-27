from __future__ import annotations

import numpy as np
import pytest
import torch

from gumbel_az.algorithms import create_algorithm, registered_algorithms
from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search import TorchGumbelSearchBackend
from gumbel_az.selfplay.trajectory import Trajectory, TrajectoryStep

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _network_apply(observation):
    return NetworkOutput(
        policy_logits=torch.zeros((observation.shape[0], 7), dtype=torch.float32),
        value=torch.zeros((observation.shape[0],), dtype=torch.float32),
    )


def test_algorithm_registry_only_lists_implemented_algorithms() -> None:
    assert registered_algorithms() == ("gumbel_alphazero",)


def test_unknown_algorithm_fails_before_claiming_registration() -> None:
    config = load_config(CONFIG, ["algorithm.name=random_baseline"])
    game = create_game(config.game.name)

    with pytest.raises(KeyError, match="unknown algorithm"):
        create_algorithm(config, game=game, search_backend=TorchGumbelSearchBackend(game=game))


def test_gumbel_algorithm_select_action_is_deterministic_and_masks_illegal() -> None:
    config = load_config(CONFIG, ["search.simulations_per_move=4"])
    game = create_game(config.game.name)
    algorithm = create_algorithm(
        config,
        game=game,
        search_backend=TorchGumbelSearchBackend(game=game),
    )
    state = game.init()
    for _ in range(6):
        state = game.step(state, 0)

    output_a = algorithm.select_action(
        game_state=state,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(0),
        temperature=0.0,
    )
    output_b = algorithm.select_action(
        game_state=state,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(0),
        temperature=0.0,
    )

    assert int(output_a.selected_action) != 0
    assert float(output_a.policy_target[0]) == 0.0
    assert torch.isclose(output_a.policy_target.sum(), torch.tensor(1.0))
    assert int(output_a.selected_action) == int(output_b.selected_action)
    assert torch.allclose(output_a.policy_target, output_b.policy_target)


def test_gumbel_algorithm_temperature_sampling_is_deterministic_for_seed() -> None:
    config = load_config(CONFIG, ["search.simulations_per_move=4"])
    game = create_game(config.game.name)
    algorithm = create_algorithm(
        config,
        game=game,
        search_backend=TorchGumbelSearchBackend(game=game),
    )
    state = game.init()

    output_a = algorithm.select_action(
        game_state=state,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(42),
        temperature=1.0,
    )
    output_b = algorithm.select_action(
        game_state=state,
        network_apply=_network_apply,
        rng=torch.Generator().manual_seed(42),
        temperature=1.0,
    )

    assert int(output_a.selected_action) == int(output_b.selected_action)
    assert 0 <= int(output_a.selected_action) < game.num_actions


def test_temperature_sampling_never_selects_zero_probability_action() -> None:
    config = load_config(CONFIG, ["search.simulations_per_move=4"])
    game = create_game(config.game.name)
    algorithm = create_algorithm(
        config,
        game=game,
        search_backend=TorchGumbelSearchBackend(game=game),
    )
    state = game.init()
    for _ in range(6):
        state = game.step(state, 0)

    for seed in range(20):
        output = algorithm.select_action(
            game_state=state,
            network_apply=_network_apply,
            rng=torch.Generator().manual_seed(seed),
            temperature=1.0,
        )
        assert int(output.selected_action) != 0
        assert float(output.policy_target[0]) == 0.0


def test_gumbel_algorithm_generates_value_target_from_to_play_perspective() -> None:
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    algorithm = create_algorithm(
        config,
        game=game,
        search_backend=TorchGumbelSearchBackend(game=game),
    )
    observation = np.zeros(game.observation_shape, dtype=np.float32)
    legal = np.ones((game.num_actions,), dtype=bool)
    policy = np.ones((game.num_actions,), dtype=np.float32) / game.num_actions
    trajectory = Trajectory(
        game_id="g",
        game_name=game.name,
        algorithm_name=algorithm.name,
        model_version=0,
        final_rewards=np.asarray([1.0, -1.0], dtype=np.float32),
        steps=[
            TrajectoryStep(observation, legal, policy, 0, 0.0, 0, 0, {}),
            TrajectoryStep(observation, legal, policy, 1, 0.0, 1, 1, {}),
        ],
    )

    samples = algorithm.generate_targets(trajectory, trajectory.final_rewards)

    assert samples[0]["value_target"] == 1.0
    assert samples[1]["value_target"] == -1.0
    assert samples[0]["policy_target"].shape == (game.num_actions,)
