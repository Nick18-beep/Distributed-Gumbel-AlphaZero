from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gumbel_az.envs.custom.connect_four import COLUMNS, ROWS, ConnectFourGame


def _play(game: ConnectFourGame, actions: list[int]):
    state = game.init()
    for action in actions:
        state = game.step(state, action)
    return state


def test_legal_moves_empty_board() -> None:
    game = ConnectFourGame()
    state = game.init()

    assert game.legal_action_mask(state).tolist() == [True] * COLUMNS


def test_legal_moves_full_column() -> None:
    game = ConnectFourGame()
    state = _play(game, [0, 0, 0, 0, 0, 0])

    assert game.legal_action_mask(state).tolist()[0] is False
    assert game.legal_action_mask(state).tolist()[1:] == [True] * (COLUMNS - 1)


def test_out_of_range_actions_are_illegal_noops() -> None:
    game = ConnectFourGame()
    state = game.init()

    negative = game.step(state, -1)
    too_large = game.step(state, COLUMNS)

    np.testing.assert_array_equal(negative.board, state.board)
    np.testing.assert_array_equal(too_large.board, state.board)
    assert int(negative.current_player) == 0
    assert int(too_large.current_player) == 0
    assert int(negative.move_count) == 0
    assert int(too_large.move_count) == 0


def test_horizontal_win() -> None:
    game = ConnectFourGame()
    state = _play(game, [0, 0, 1, 1, 2, 2, 3])

    assert bool(game.is_terminal(state))
    assert int(state.winner) == 0
    assert game.rewards(state).tolist() == [1.0, -1.0]
    assert float(game.terminal_value(state)) == 1.0


def test_vertical_win() -> None:
    game = ConnectFourGame()
    state = _play(game, [0, 1, 0, 1, 0, 1, 0])

    assert bool(game.is_terminal(state))
    assert int(state.winner) == 0


def test_diagonal_up_win() -> None:
    game = ConnectFourGame()
    state = _play(game, [0, 1, 1, 2, 6, 2, 2, 3, 6, 3, 6, 3, 3])

    assert bool(game.is_terminal(state))
    assert int(state.winner) == 0


def test_diagonal_down_win() -> None:
    game = ConnectFourGame()
    state = _play(game, [3, 2, 2, 1, 6, 1, 1, 0, 6, 0, 6, 0, 0])

    assert bool(game.is_terminal(state))
    assert int(state.winner) == 0


def test_draw_detection_with_full_board_without_four() -> None:
    game = ConnectFourGame()
    rows = [
        [1, 1, -1, -1, 1, -1, -1],
        [-1, -1, 1, -1, 1, 1, -1],
        [1, -1, 1, -1, 1, 1, 1],
        [1, -1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, 1, -1, 1],
    ]
    state = game.init()._replace(
        board=jnp.asarray(rows, dtype=jnp.int8),
        move_count=jnp.asarray(42, dtype=jnp.int16),
        terminated=jnp.asarray(True),
    )

    assert bool(game.is_terminal(state))
    assert int(state.winner) == -1
    assert game.legal_action_mask(state).tolist() == [False] * COLUMNS
    assert game.rewards(state).tolist() == [0.0, -0.0]


def test_draw_detection_on_final_step() -> None:
    game = ConnectFourGame()
    actions = [
        4,
        2,
        0,
        2,
        6,
        0,
        6,
        2,
        2,
        1,
        3,
        4,
        2,
        6,
        6,
        2,
        1,
        4,
        0,
        1,
        3,
        6,
        4,
        5,
        3,
        5,
        5,
        6,
        4,
        3,
        4,
        3,
        5,
        1,
        0,
        1,
        5,
        0,
        1,
        5,
        0,
        3,
    ]
    state = _play(game, actions[:-1])

    assert not bool(game.is_terminal(state))
    next_state = game.step(state, actions[-1])

    assert bool(game.is_terminal(next_state))
    assert int(next_state.winner) == -1
    assert int(next_state.move_count) == 42
    assert game.rewards(next_state).tolist() == [0.0, -0.0]


def test_current_player_switches_after_legal_move() -> None:
    game = ConnectFourGame()
    state = game.step(game.init(), 0)

    assert int(game.current_player(state)) == 1


def test_canonical_observation_tracks_current_player_perspective() -> None:
    game = ConnectFourGame()
    state = _play(game, [0])
    observation = game.canonical_observation(state)

    assert observation.shape == (ROWS, COLUMNS, 2)
    assert float(observation[ROWS - 1, 0, 1]) == 1.0
    assert float(observation[ROWS - 1, 0, 0]) == 0.0


def test_horizontal_symmetry_flips_observation_and_actions() -> None:
    game = ConnectFourGame()
    observation = jnp.arange(ROWS * COLUMNS * 2).reshape((ROWS, COLUMNS, 2))
    sample = {
        "observation": observation,
        "legal_action_mask": jnp.asarray([True, False, True, False, True, False, True]),
        "policy_target": jnp.arange(COLUMNS),
    }

    original, flipped = game.symmetries(sample)

    assert original is sample
    np.testing.assert_array_equal(flipped["observation"], jnp.flip(observation, axis=1))
    assert flipped["legal_action_mask"].tolist() == [True, False, True, False, True, False, True]
    assert flipped["policy_target"].tolist() == [6, 5, 4, 3, 2, 1, 0]


def test_step_supports_jit_and_vmap() -> None:
    game = ConnectFourGame()
    state = game.init()

    jitted_state = jax.jit(game.step)(state, 0)
    assert int(jitted_state.current_player) == 1

    states = jax.tree.map(lambda value: jnp.stack([value, value]), state)
    actions = jnp.asarray([0, 1])
    next_states = jax.vmap(game.step)(states, actions)
    assert next_states.board.shape == (2, ROWS, COLUMNS)
