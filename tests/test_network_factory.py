from __future__ import annotations

import torch

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_mlp_init_is_deterministic_and_uses_num_actions() -> None:
    config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions + 2)

    model_a = network.init(0, game.observation_shape, game.num_actions + 2)
    model_b = network.init(0, game.observation_shape, game.num_actions + 2)
    output = model_a(torch.zeros((3, *game.observation_shape), dtype=torch.float32))

    assert all(
        torch.equal(model_a.state_dict()[key], model_b.state_dict()[key])
        for key in model_a.state_dict()
    )
    assert output.policy_logits.shape == (3, game.num_actions + 2)
    assert output.value.shape == (3,)
    assert torch.all(output.value <= 1.0)
    assert torch.all(output.value >= -1.0)


def test_resnet_forward_shapes() -> None:
    config = load_config(CONFIG_DIR / "connect_four.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)

    model = network.init(1, game.observation_shape, game.num_actions)
    output = model(torch.zeros((2, *game.observation_shape), dtype=torch.float32))

    assert output.policy_logits.shape == (2, game.num_actions)
    assert output.value.shape == (2,)


def test_resnet_long_preset_uses_configurable_head_channels() -> None:
    config = load_config(CONFIG_DIR / "connect_four_lan_long.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)

    model = network.init(1, game.observation_shape, game.num_actions)
    output = model(torch.zeros((2, *game.observation_shape), dtype=torch.float32))

    assert output.policy_logits.shape == (2, game.num_actions)
    assert output.value.shape == (2,)
    assert model.policy_head[0].out_channels == 32
    assert model.value_head[0].out_channels == 32


def test_model_init_rejects_num_actions_mismatch() -> None:
    config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)

    try:
        network.init(0, game.observation_shape, game.num_actions + 1)
    except ValueError as exc:
        assert "num_actions" in str(exc)
    else:
        raise AssertionError("expected num_actions mismatch to fail")
