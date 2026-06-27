"""Non-interactive and CLI helpers for playing against an agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax

from gumbel_az.algorithms import create_algorithm
from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.search import MctxSearchBackend

CheckpointSelector = Literal["latest", "best"]


@dataclass(frozen=True)
class PlayResult:
    moves: list[int]
    message: str
    board_text: str
    terminal: bool


class AgentPlayer:
    def __init__(self, config: AppConfig, *, params: Any) -> None:
        self.config = config
        self.game = create_game(config.game.name)
        self.network = create_network(config.model, num_actions=self.game.num_actions)
        self.algorithm = create_algorithm(
            config,
            game=self.game,
            search_backend=MctxSearchBackend(),
        )
        self.params = params

    def select_action(self, state: Any, *, seed: int) -> int:
        def network_apply(observations):
            return self.network.apply(self.params, observations, train=False)

        output = self.algorithm.select_action(
            game_state=state,
            network_apply=network_apply,
            rng_key=jax.random.PRNGKey(seed),
            temperature=0.0,
        )
        return int(output.selected_action)


def load_play_params(
    config: AppConfig,
    *,
    run_dir: Path | None = None,
    checkpoint: CheckpointSelector = "best",
) -> Any:
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    if run_dir is None:
        return network.init(
            jax.random.PRNGKey(config.run.seed),
            game.observation_shape,
            game.num_actions,
        )
    payload = CheckpointManager(run_dir / "checkpoints").load(best=checkpoint == "best")
    return payload["state"]["params"]


def select_agent_action(
    config: AppConfig,
    *,
    params: Any,
    state: Any,
    seed: int,
) -> int:
    return AgentPlayer(config, params=params).select_action(state, seed=seed)


def result_message(game: Any, state: Any, *, human_player: int) -> str:
    if not bool(game.is_terminal(state)):
        return "partita in corso"
    reward = float(game.rewards(state)[human_player])
    if reward > 0:
        return "hai vinto"
    if reward < 0:
        return "hai perso"
    return "pareggio"


def apply_human_action(game: Any, state: Any, action: int) -> Any:
    legal = game.legal_action_mask(state)
    if action < 0 or action >= game.num_actions or not bool(legal[action]):
        raise ValueError(f"mossa illegale: {action}")
    return game.step(state, action)


def play_scripted_game(
    config: AppConfig,
    *,
    human_actions: Sequence[int],
    run_dir: Path | None = None,
    checkpoint: CheckpointSelector = "best",
    human_player: int = 0,
) -> PlayResult:
    game = create_game(config.game.name)
    params = load_play_params(config, run_dir=run_dir, checkpoint=checkpoint)
    agent = AgentPlayer(config, params=params)
    state = game.init(jax.random.PRNGKey(config.run.seed))
    moves: list[int] = []
    human_index = 0

    for move_index in range(game.max_moves):
        if bool(game.is_terminal(state)):
            break
        current_player = int(game.current_player(state))
        if current_player == human_player:
            if human_index >= len(human_actions):
                break
            action = int(human_actions[human_index])
            human_index += 1
            state = apply_human_action(game, state, action)
        else:
            action = agent.select_action(state, seed=config.run.seed + 50_000 + move_index)
            state = game.step(state, action)
        moves.append(action)

    return PlayResult(
        moves=moves,
        message=result_message(game, state, human_player=human_player),
        board_text=game.render_text(state),
        terminal=bool(game.is_terminal(state)),
    )
