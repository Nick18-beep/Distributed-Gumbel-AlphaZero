from __future__ import annotations

import json
from pathlib import Path

import torch

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import create_optimizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _state():
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    model = network.init(0, game.observation_shape, game.num_actions)
    optimizer, _ = create_optimizer(model, config.training)
    return game, model, optimizer, {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 0,
        "scaler_state_dict": None,
    }


def test_checkpoint_save_load_and_registry(tmp_path: Path) -> None:
    game, model, _optimizer, state = _state()
    manager = CheckpointManager(tmp_path / "checkpoints")

    checkpoint_dir = manager.save(
        version=1,
        state=state,
        metadata={"training_step": int(state["step"]), "game": game.name},
        best=True,
    )
    restored = manager.load()
    restored_best = manager.load(best=True)
    index = json.loads((tmp_path / "checkpoints" / "index.json").read_text(encoding="utf-8"))

    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "checkpoint.pt").exists()
    assert index["checkpoints"][0]["version"] == 1
    assert (tmp_path / "checkpoints" / "latest.json").exists()
    assert (tmp_path / "checkpoints" / "best.json").exists()
    assert int(restored["state"]["step"]) == 0
    assert int(restored_best["state"]["step"]) == 0
    assert all(
        torch.equal(restored["state"]["model_state_dict"][key], value)
        for key, value in model.state_dict().items()
    )


def test_checkpoint_registry_uses_paths_stable_across_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    game, _model, _optimizer, state = _state()
    root = tmp_path / "checkpoints"
    manager = CheckpointManager(root)
    manager.save(
        version=2,
        state=state,
        metadata={"training_step": int(state["step"]), "game": game.name},
    )

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    restored = CheckpointManager(root).load()

    assert int(restored["state"]["step"]) == 0


def test_checkpoint_manager_resolves_relative_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    game, _model, _optimizer, state = _state()
    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(Path("relative_checkpoints"))

    manager.save(
        version=3,
        state=state,
        metadata={"training_step": int(state["step"]), "game": game.name},
    )
    restored = manager.load()

    assert manager.root.is_absolute()
    assert int(restored["state"]["step"]) == 0


def test_incomplete_checkpoint_is_ignored_by_latest(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")
    incomplete = manager.checkpoint_dir(99)
    incomplete.mkdir(parents=True)

    try:
        manager.load()
    except FileNotFoundError as exc:
        assert "latest.json" in str(exc)
    else:
        raise AssertionError("expected missing latest pointer")


def test_incomplete_checkpoint_is_not_loaded_by_version(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")
    incomplete = manager.checkpoint_dir(99)
    incomplete.mkdir(parents=True)

    try:
        manager.load(version=99)
    except FileNotFoundError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("expected unregistered incomplete checkpoint to fail")
