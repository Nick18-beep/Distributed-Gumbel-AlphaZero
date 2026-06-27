"""Connect Four environment implemented with NumPy state transitions."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER0 = 1
PLAYER1 = -1
NO_WINNER = -1


class ConnectFourState(NamedTuple):
    board: np.ndarray
    current_player: np.int8
    move_count: np.int16
    terminated: np.bool_
    winner: np.int8


def _piece_for_player(player: int | np.integer) -> int:
    return PLAYER0 if int(player) == 0 else PLAYER1


def _has_four(board: np.ndarray, piece: int) -> bool:
    occupied = board == piece
    return bool(
        np.any(occupied[:, :-3] & occupied[:, 1:-2] & occupied[:, 2:-1] & occupied[:, 3:])
        or np.any(occupied[:-3, :] & occupied[1:-2, :] & occupied[2:-1, :] & occupied[3:, :])
        or np.any(
            occupied[:-3, :-3] & occupied[1:-2, 1:-2] & occupied[2:-1, 2:-1] & occupied[3:, 3:]
        )
        or np.any(
            occupied[3:, :-3] & occupied[2:-1, 1:-2] & occupied[1:-2, 2:-1] & occupied[:-3, 3:]
        )
    )


class ConnectFourGame:
    name = "connect_four"
    num_players = 2
    num_actions = COLUMNS
    observation_shape = (ROWS, COLUMNS, 2)
    max_moves = ROWS * COLUMNS
    supports_jit = False
    supports_vmap = False

    def init(self, rng_key: Any = None) -> ConnectFourState:
        del rng_key
        return ConnectFourState(
            board=np.zeros((ROWS, COLUMNS), dtype=np.int8),
            current_player=np.int8(0),
            move_count=np.int16(0),
            terminated=np.bool_(False),
            winner=np.int8(NO_WINNER),
        )

    def legal_action_mask(self, state: ConnectFourState) -> np.ndarray:
        return ((state.board[0, :] == EMPTY) & (not bool(state.terminated))).astype(bool)

    def step(self, state: ConnectFourState, action: int) -> ConnectFourState:
        action = int(action)
        if bool(state.terminated) or action < 0 or action >= COLUMNS:
            return state
        legal = self.legal_action_mask(state)
        if not bool(legal[action]):
            return state

        board = state.board.copy()
        empty_rows = np.flatnonzero(board[:, action] == EMPTY)
        row = int(empty_rows[-1])
        piece = _piece_for_player(state.current_player)
        board[row, action] = piece
        move_count = np.int16(int(state.move_count) + 1)
        won = _has_four(board, piece)
        draw = not won and int(move_count) >= self.max_moves
        terminated = np.bool_(won or draw)
        winner = np.int8(int(state.current_player) if won else int(state.winner))
        next_player = np.int8(
            int(state.current_player) if terminated else 1 - int(state.current_player)
        )
        return ConnectFourState(
            board=board,
            current_player=next_player,
            move_count=move_count,
            terminated=terminated,
            winner=winner,
        )

    def is_terminal(self, state: ConnectFourState) -> np.bool_:
        return state.terminated

    def current_player(self, state: ConnectFourState) -> np.int8:
        return state.current_player

    def canonical_observation(self, state: ConnectFourState) -> np.ndarray:
        own_piece = _piece_for_player(state.current_player)
        own = state.board == own_piece
        opponent = state.board == -own_piece
        return np.stack([own, opponent], axis=-1).astype(np.float32)

    def terminal_value(self, state: ConnectFourState) -> np.float32:
        rewards = self.rewards(state)
        return np.float32(rewards[int(state.current_player)])

    def rewards(self, state: ConnectFourState) -> np.ndarray:
        if int(state.winner) == 0:
            return np.asarray([1.0, -1.0], dtype=np.float32)
        if int(state.winner) == 1:
            return np.asarray([-1.0, 1.0], dtype=np.float32)
        return np.asarray([0.0, 0.0], dtype=np.float32)

    def symmetries(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        flipped = dict(sample)
        if "observation" in flipped:
            flipped["observation"] = np.flip(np.asarray(flipped["observation"]), axis=1).copy()
        if "state_or_observation" in flipped:
            flipped["state_or_observation"] = np.flip(
                np.asarray(flipped["state_or_observation"]),
                axis=1,
            ).copy()
        for key in ("legal_action_mask", "policy_target"):
            if key in flipped:
                flipped[key] = np.flip(np.asarray(flipped[key]), axis=0).copy()
        return [sample, flipped]

    def render_text(self, state: ConnectFourState) -> str:
        symbols = {EMPTY: ".", PLAYER0: "X", PLAYER1: "O"}
        rows = [" ".join(symbols[int(value)] for value in row) for row in state.board]
        rows.append("0 1 2 3 4 5 6")
        return "\n".join(rows)
