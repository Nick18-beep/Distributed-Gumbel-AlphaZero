"""PyTorch fallback self-play when JAX is unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from gumbel_az.config.schema import AppConfig
from gumbel_az.replay import ReplayWriter

ROWS = 6
COLUMNS = 7
PLAYER0 = 1
PLAYER1 = -1
NO_WINNER = -1


@dataclass(frozen=True)
class TorchFallbackResult:
    games: int
    positions: int
    illegal_action_rate: float
    policy_entropy_mean: float
    root_value_mean: float
    replay_shard: str
    seconds: float

    @property
    def games_per_sec(self) -> float:
        return self.games / max(self.seconds, 1.0e-9)

    @property
    def positions_per_sec(self) -> float:
        return self.positions / max(self.seconds, 1.0e-9)


class TorchPolicyValue(nn.Module):
    def __init__(self, hidden_size: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ROWS * COLUMNS * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(observation)
        return self.policy_head(hidden), torch.tanh(self.value_head(hidden)).squeeze(-1)


def _legal_mask(board: np.ndarray, terminal: bool) -> np.ndarray:
    return (board[0] == 0) & (not terminal)


def _piece(player: int) -> int:
    return PLAYER0 if player == 0 else PLAYER1


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


def _observation(board: np.ndarray, player: int) -> np.ndarray:
    own = board == _piece(player)
    opponent = board == -_piece(player)
    return np.stack([own, opponent], axis=-1).astype(np.float32)


def _step(board: np.ndarray, player: int, action: int) -> tuple[np.ndarray, int, bool, int]:
    next_board = board.copy()
    rows = np.where(next_board[:, action] == 0)[0]
    row = int(rows[-1])
    next_board[row, action] = _piece(player)
    won = _has_four(next_board, _piece(player))
    draw = not won and not np.any(next_board[0] == 0)
    terminal = won or draw
    winner = player if won else NO_WINNER
    next_player = player if terminal else 1 - player
    return next_board, next_player, terminal, winner


def _rewards(winner: int) -> np.ndarray:
    if winner == 0:
        return np.asarray([1.0, -1.0], dtype=np.float32)
    if winner == 1:
        return np.asarray([-1.0, 1.0], dtype=np.float32)
    return np.asarray([0.0, 0.0], dtype=np.float32)


def _entropy(policy: np.ndarray) -> float:
    positive = policy[policy > 0.0]
    return float(-np.sum(positive * np.log(positive))) if positive.size else 0.0


class TorchFallbackSelfPlayWorker:
    """Fallback generator that keeps replay/logging alive without JAX.

    This path is intentionally logged as ``torch_fallback``. It does not pretend to
    be the primary MCTX Gumbel search path; it uses PyTorch policy/value inference
    and legal masking so local runs can still produce valid replay when JAX cannot
    import on the machine.
    """

    def __init__(self, config: AppConfig, *, replay_writer: ReplayWriter) -> None:
        self.config = config
        torch.manual_seed(config.run.seed)
        hidden_size = config.model.hidden_size or 64
        self.model = TorchPolicyValue(hidden_size=hidden_size, num_actions=COLUMNS)
        self.model.eval()
        self.replay_writer = replay_writer

    def _policy_value(self, observation: np.ndarray, legal: np.ndarray) -> tuple[np.ndarray, float]:
        with torch.inference_mode():
            obs = torch.from_numpy(observation[None, ...])
            logits, value = self.model(obs)
            logits_np = logits[0].detach().cpu().numpy()
        masked = np.where(legal, logits_np, -np.inf)
        finite = masked[np.isfinite(masked)]
        if finite.size == 0:
            raise ValueError("no legal actions available for PyTorch fallback")
        shifted = masked - np.max(finite)
        probs = np.where(legal, np.exp(shifted), 0.0)
        probs = probs / np.sum(probs)
        return probs.astype(np.float32), float(value[0])

    def play_batch(self, num_games: int, seed: int) -> tuple[list[list[dict]], TorchFallbackResult]:
        rng = np.random.default_rng(seed)
        start = perf_counter()
        all_games: list[list[dict]] = []
        samples: list[dict] = []
        entropies: list[float] = []
        root_values: list[float] = []

        for _game_index in range(num_games):
            board = np.zeros((ROWS, COLUMNS), dtype=np.int8)
            player = 0
            terminal = False
            winner = NO_WINNER
            game_id = uuid4().hex
            game_steps: list[dict] = []
            pending: list[dict] = []
            for move_index in range(ROWS * COLUMNS):
                if terminal:
                    break
                legal = _legal_mask(board, terminal)
                obs = _observation(board, player)
                policy, root_value = self._policy_value(obs, legal)
                action = int(rng.choice(COLUMNS, p=policy))
                game_steps.append({"action": action, "to_play": player})
                pending.append(
                    {
                        "game_name": "connect_four",
                        "algorithm_name": "gumbel_alphazero",
                        "state_or_observation": obs,
                        "legal_action_mask": legal,
                        "policy_target": policy,
                        "to_play": player,
                        "move_index": move_index,
                        "game_id": game_id,
                        "model_version": 0,
                        "search_stats": {
                            "backend": "torch_fallback",
                            "root_value": root_value,
                            "policy_entropy": _entropy(policy),
                            "selected_action": action,
                        },
                    }
                )
                entropies.append(_entropy(policy))
                root_values.append(root_value)
                board, player, terminal, winner = _step(board, player, action)

            rewards = _rewards(winner)
            for sample in pending:
                sample["value_target"] = float(rewards[int(sample["to_play"])])
                samples.append(sample)
            all_games.append(game_steps)

        shard = self.replay_writer.write_shard(samples)
        elapsed = perf_counter() - start
        positions = len(samples)
        return all_games, TorchFallbackResult(
            games=num_games,
            positions=positions,
            illegal_action_rate=0.0,
            policy_entropy_mean=float(np.mean(entropies)) if entropies else 0.0,
            root_value_mean=float(np.mean(root_values)) if root_values else 0.0,
            replay_shard=str(shard),
            seconds=elapsed,
        )
