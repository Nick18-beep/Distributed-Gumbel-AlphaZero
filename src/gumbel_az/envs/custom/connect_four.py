"""Connect Four environment with JAX-friendly state transitions."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER0 = 1
PLAYER1 = -1
NO_WINNER = -1


class ConnectFourState(NamedTuple):
    board: jnp.ndarray
    current_player: jnp.ndarray
    move_count: jnp.ndarray
    terminated: jnp.ndarray
    winner: jnp.ndarray


def _piece_for_player(player: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(player == 0, PLAYER0, PLAYER1)


def _has_four(board: jnp.ndarray, piece: jnp.ndarray) -> jnp.ndarray:
    occupied = board == piece
    horizontal = jnp.any(occupied[:, :-3] & occupied[:, 1:-2] & occupied[:, 2:-1] & occupied[:, 3:])
    vertical = jnp.any(occupied[:-3, :] & occupied[1:-2, :] & occupied[2:-1, :] & occupied[3:, :])
    diagonal_down = jnp.any(
        occupied[:-3, :-3] & occupied[1:-2, 1:-2] & occupied[2:-1, 2:-1] & occupied[3:, 3:]
    )
    diagonal_up = jnp.any(
        occupied[3:, :-3] & occupied[2:-1, 1:-2] & occupied[1:-2, 2:-1] & occupied[:-3, 3:]
    )
    return horizontal | vertical | diagonal_up | diagonal_down


class ConnectFourGame:
    name = "connect_four"
    num_players = 2
    num_actions = COLUMNS
    observation_shape = (ROWS, COLUMNS, 2)
    max_moves = ROWS * COLUMNS
    supports_jit = True
    supports_vmap = True

    def init(self, rng_key: Any = None) -> ConnectFourState:
        del rng_key
        return ConnectFourState(
            board=jnp.zeros((ROWS, COLUMNS), dtype=jnp.int8),
            current_player=jnp.asarray(0, dtype=jnp.int8),
            move_count=jnp.asarray(0, dtype=jnp.int16),
            terminated=jnp.asarray(False),
            winner=jnp.asarray(NO_WINNER, dtype=jnp.int8),
        )

    def legal_action_mask(self, state: ConnectFourState) -> jnp.ndarray:
        return (state.board[0, :] == EMPTY) & ~state.terminated

    def step(self, state: ConnectFourState, action: int) -> ConnectFourState:
        action_array = jnp.asarray(action, dtype=jnp.int32)
        action_in_bounds = (action_array >= 0) & (action_array < COLUMNS)
        safe_action = jnp.clip(action_array, 0, COLUMNS - 1)
        legal_mask = self.legal_action_mask(state)
        is_legal = action_in_bounds & legal_mask[safe_action]
        piece = _piece_for_player(state.current_player)
        column = state.board[:, safe_action]
        row = jnp.sum(column == EMPTY, dtype=jnp.int32) - 1
        safe_row = jnp.maximum(row, 0)

        candidate_board = state.board.at[safe_row, safe_action].set(piece)
        next_board = jnp.where(is_legal, candidate_board, state.board)
        next_move_count = jnp.where(is_legal, state.move_count + 1, state.move_count)
        won = is_legal & _has_four(next_board, piece)
        draw = is_legal & (next_move_count >= self.max_moves) & ~won
        terminated = state.terminated | won | draw
        winner = jnp.where(won, state.current_player, state.winner)
        next_player = jnp.where(
            is_legal & ~terminated,
            1 - state.current_player,
            state.current_player,
        )

        return ConnectFourState(
            board=next_board,
            current_player=next_player.astype(jnp.int8),
            move_count=next_move_count.astype(jnp.int16),
            terminated=terminated,
            winner=winner.astype(jnp.int8),
        )

    def is_terminal(self, state: ConnectFourState) -> jnp.ndarray:
        return state.terminated

    def current_player(self, state: ConnectFourState) -> jnp.ndarray:
        return state.current_player

    def canonical_observation(self, state: ConnectFourState) -> jnp.ndarray:
        own_piece = _piece_for_player(state.current_player)
        opponent_piece = -own_piece
        own = state.board == own_piece
        opponent = state.board == opponent_piece
        return jnp.stack([own, opponent], axis=-1).astype(jnp.float32)

    def terminal_value(self, state: ConnectFourState) -> jnp.ndarray:
        rewards = self.rewards(state)
        return rewards[state.current_player.astype(jnp.int32)]

    def rewards(self, state: ConnectFourState) -> jnp.ndarray:
        player0_reward = jnp.where(
            state.winner == 0,
            1.0,
            jnp.where(state.winner == 1, -1.0, 0.0),
        )
        return jnp.asarray([player0_reward, -player0_reward], dtype=jnp.float32)

    def symmetries(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        flipped = dict(sample)
        if "observation" in flipped:
            flipped["observation"] = jnp.flip(flipped["observation"], axis=1)
        if "state_or_observation" in flipped:
            flipped["state_or_observation"] = jnp.flip(flipped["state_or_observation"], axis=1)
        for key in ("legal_action_mask", "policy_target"):
            if key in flipped:
                flipped[key] = jnp.flip(flipped[key], axis=0)
        return [sample, flipped]

    def render_text(self, state: ConnectFourState) -> str:
        board = np.asarray(state.board)
        symbols = {EMPTY: ".", PLAYER0: "X", PLAYER1: "O"}
        rows = [" ".join(symbols[int(value)] for value in row) for row in board]
        rows.append("0 1 2 3 4 5 6")
        return "\n".join(rows)
