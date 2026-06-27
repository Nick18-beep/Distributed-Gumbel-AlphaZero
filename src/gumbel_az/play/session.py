"""Non-interactive and CLI helpers for playing against an agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from gumbel_az.algorithms import create_algorithm
from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.runtime import detect_torch_runtime
from gumbel_az.search import TorchGumbelSearchBackend

CheckpointSelector = Literal["latest", "best"]


@dataclass(frozen=True)
class PlayResult:
    moves: list[int]
    message: str
    board_text: str
    terminal: bool


class AgentPlayer:
    def __init__(self, config: AppConfig, *, model: torch.nn.Module) -> None:
        self.config = config
        self.runtime = detect_torch_runtime()
        self.device = torch.device(self.runtime.device)
        self.game = create_game(config.game.name)
        self.model = model.to(self.device)
        self.model.eval()
        self.algorithm = create_algorithm(
            config,
            game=self.game,
            search_backend=TorchGumbelSearchBackend(game=self.game, device=self.device),
        )

    def select_action(self, state: Any, *, seed: int) -> int:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        def network_apply(observations):
            self.model.eval()
            return self.model(observations.to(self.device, non_blocking=True))

        output = self.algorithm.select_action(
            game_state=state,
            network_apply=network_apply,
            rng=generator,
            temperature=0.0,
        )
        return int(output.selected_action.detach().cpu().item())


def load_play_model(
    config: AppConfig,
    *,
    run_dir: Path | None = None,
    checkpoint: CheckpointSelector = "best",
) -> torch.nn.Module:
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    runtime = detect_torch_runtime()
    model = network.init(
        config.run.seed,
        game.observation_shape,
        game.num_actions,
        device=runtime.device,
    )
    if run_dir is None:
        return model
    payload = CheckpointManager(run_dir / "checkpoints").load(
        best=checkpoint == "best",
        map_location=runtime.device,
    )
    model.load_state_dict(payload["state"]["model_state_dict"])
    model.eval()
    return model


def load_play_params(
    config: AppConfig,
    *,
    run_dir: Path | None = None,
    checkpoint: CheckpointSelector = "best",
) -> torch.nn.Module:
    return load_play_model(config, run_dir=run_dir, checkpoint=checkpoint)


def select_agent_action(
    config: AppConfig,
    *,
    params: Any,
    state: Any,
    seed: int,
) -> int:
    return AgentPlayer(config, model=params).select_action(state, seed=seed)


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
    model = load_play_model(config, run_dir=run_dir, checkpoint=checkpoint)
    agent = AgentPlayer(config, model=model)
    state = game.init(config.run.seed)
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
