from __future__ import annotations

import numpy as np

from gumbel_az.domain.game import GameAdapter
from gumbel_az.envs import create_game, registered_games


def _assert_contract(game: GameAdapter) -> None:
    state = game.init(0)
    legal = game.legal_action_mask(state)
    observation = game.canonical_observation(state)
    rewards = game.rewards(state)

    assert game.num_players == 2
    assert legal.shape == (game.num_actions,)
    assert legal.dtype == np.bool_
    assert observation.shape == game.observation_shape
    assert rewards.shape == (game.num_players,)
    assert int(game.current_player(state)) == 0
    assert not bool(game.is_terminal(state))
    assert isinstance(game.render_text(state), str)


def test_connect_four_is_registered_and_satisfies_contract() -> None:
    assert "connect_four" in registered_games()
    _assert_contract(create_game("connect_four"))


def test_unknown_game_fails_with_available_games() -> None:
    try:
        create_game("missing")
    except KeyError as exc:
        assert "connect_four" in str(exc)
    else:
        raise AssertionError("expected unknown game to fail")
