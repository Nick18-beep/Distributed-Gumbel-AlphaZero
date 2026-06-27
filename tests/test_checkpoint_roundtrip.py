from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.training import create_train_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_checkpoint_save_load_and_registry(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(jax.random.PRNGKey(0), game.observation_shape, game.num_actions)
    tx, _ = create_optimizer(config.training)
    state = create_train_state(params=params, apply_fn=network.apply, tx=tx)
    manager = CheckpointManager(tmp_path / "checkpoints")

    checkpoint_dir = manager.save(
        version=1,
        state={"params": state.params, "opt_state": state.opt_state, "step": state.step},
        metadata={"training_step": int(state.step), "game": game.name},
        best=True,
    )
    restored = manager.load()
    restored_best = manager.load(best=True)
    index = json.loads((tmp_path / "checkpoints" / "index.json").read_text(encoding="utf-8"))

    assert checkpoint_dir.exists()
    assert index["checkpoints"][0]["version"] == 1
    assert (tmp_path / "checkpoints" / "latest.json").exists()
    assert (tmp_path / "checkpoints" / "best.json").exists()
    assert int(restored["state"]["step"]) == 0
    assert int(restored_best["state"]["step"]) == 0
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.array_equal(a, b),
            restored["state"]["params"],
            state.params,
        )
    )


def test_checkpoint_registry_uses_paths_stable_across_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(jax.random.PRNGKey(0), game.observation_shape, game.num_actions)
    tx, _ = create_optimizer(config.training)
    state = create_train_state(params=params, apply_fn=network.apply, tx=tx)
    root = tmp_path / "checkpoints"
    manager = CheckpointManager(root)
    manager.save(
        version=2,
        state={"params": state.params, "opt_state": state.opt_state, "step": state.step},
        metadata={"training_step": int(state.step), "game": game.name},
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
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(jax.random.PRNGKey(0), game.observation_shape, game.num_actions)
    tx, _ = create_optimizer(config.training)
    state = create_train_state(params=params, apply_fn=network.apply, tx=tx)
    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(Path("relative_checkpoints"))

    manager.save(
        version=3,
        state={"params": state.params, "opt_state": state.opt_state, "step": state.step},
        metadata={"training_step": int(state.step), "game": game.name},
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
