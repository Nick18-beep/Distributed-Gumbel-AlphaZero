from __future__ import annotations

import numpy as np
import torch

import gumbel_az.training.trainer as trainer_module
from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import WarmupCosineSchedule, create_optimizer
from gumbel_az.replay import ReplayReader, ReplayWriter
from gumbel_az.training import TorchTrainState, train_step
from gumbel_az.training.trainer import (
    Trainer,
    _checkpoint_model_state_dict,
    _maybe_compile_model,
)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _replay_sample(index: int = 0) -> dict:
    return {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": np.zeros((6, 7, 2), dtype=np.float32),
        "legal_action_mask": np.asarray([True, True, True, True, True, True, True]),
        "policy_target": np.asarray([1 / 7] * 7, dtype=np.float32),
        "value_target": 1.0 if index % 2 == 0 else -1.0,
        "to_play": index % 2,
        "move_index": index,
        "game_id": f"game-{index}",
        "model_version": 0,
        "search_stats": {"root_value": 0.0},
    }


def test_train_step_is_finite_and_updates_params() -> None:
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    model = network.init(0, game.observation_shape, game.num_actions)
    optimizer, schedule = create_optimizer(model, config.training)
    state = TorchTrainState(
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        step=0,
        device=torch.device("cpu"),
    )
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    batch = {
        "observation": torch.ones((config.training.batch_size, *game.observation_shape)),
        "policy_target": torch.ones((config.training.batch_size, game.num_actions))
        / game.num_actions,
        "value_target": torch.ones((config.training.batch_size,)) * 0.5,
    }

    new_state, metrics = train_step(state, batch, config.training)

    assert int(new_state.step) == 1
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
    assert torch.isfinite(torch.tensor(metrics["grad_norm"]))
    assert metrics["grad_norm"] > 0.0
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.state_dict().items()
        if value.dtype.is_floating_point
    )


def test_warmup_cosine_schedule_uses_global_total_steps() -> None:
    short_config = load_config(CONFIG, ["training.steps_per_iteration=10"])
    long_config = load_config(
        CONFIG,
        [
            "training.steps_per_iteration=10",
            "training.total_steps=100",
        ],
    )

    short_schedule = WarmupCosineSchedule(short_config.training)
    long_schedule = WarmupCosineSchedule(long_config.training)

    assert short_schedule(10) == 0.0
    assert long_schedule(10) > 0.0
    assert long_schedule(100) == 0.0


def test_compile_auto_keeps_cpu_debug_eager() -> None:
    model = torch.nn.Linear(2, 2)

    compiled, mode = _maybe_compile_model(
        model,
        compile_policy="auto",
        device=torch.device("cpu"),
        run_name="connect-four-cpu-debug",
    )

    assert compiled is model
    assert mode == "eager"


def test_compile_on_reports_fallback_when_compile_fails(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile unavailable")

    monkeypatch.setattr(torch, "compile", fail_compile)

    compiled, mode = _maybe_compile_model(
        model,
        compile_policy="on",
        device=torch.device("cpu"),
        run_name="train",
    )

    assert compiled is model
    assert mode == "fallback"


def test_compile_on_reports_compiled_when_compile_succeeds(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    calls = []

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    compiled, mode = _maybe_compile_model(
        model,
        compile_policy="on",
        device=torch.device("cpu"),
        run_name="train",
    )

    assert compiled is model
    assert mode == "compiled"
    assert calls == [{"mode": "reduce-overhead"}]


def test_checkpoint_state_dict_unwraps_compiled_module_prefix() -> None:
    model = torch.nn.Linear(2, 2)

    class CompiledWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = wrapped

        def forward(self, inputs):
            return self._orig_mod(inputs)

    state_dict = _checkpoint_model_state_dict(CompiledWrapper(model))

    assert set(state_dict) == set(model.state_dict())
    assert not any(key.startswith("_orig_mod.") for key in state_dict)


def test_trainer_falls_back_to_eager_when_compiled_step_fails(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_config(
        CONFIG,
        [
            "training.compile=on",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "training.checkpoint_every_steps=1",
            "replay.min_samples_to_train=1",
        ],
    )
    replay_dir = tmp_path / "replay"
    ReplayWriter(replay_dir).write_shard([_replay_sample(0), _replay_sample(1)])

    class CompiledWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = wrapped

        def forward(self, inputs):
            return self._orig_mod(inputs)

    def fake_compile_model(model, *, compile_policy, device, run_name):
        del compile_policy, device, run_name
        return CompiledWrapper(model), "compiled"

    real_train_step = trainer_module.train_step
    calls = {"count": 0}

    def flaky_train_step(state, batch, training_config):
        calls["count"] += 1
        if state.compile_enabled:
            raise RuntimeError("compiled graph failed")
        return real_train_step(state, batch, training_config)

    monkeypatch.setattr(trainer_module, "_maybe_compile_model", fake_compile_model)
    monkeypatch.setattr(trainer_module, "train_step", flaky_train_step)

    trainer = Trainer(
        config,
        replay_reader=ReplayReader(replay_dir),
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        device="cpu",
    )

    result = trainer.run(max_steps=1)

    assert calls["count"] == 2
    assert result.state.compile_mode == "fallback"
    assert not result.state.compile_enabled
    assert result.steps == 1


def test_compile_fallback_preserves_optimizer_state(tmp_path, monkeypatch) -> None:
    config = load_config(CONFIG, ["training.compile=on"])
    replay_dir = tmp_path / "replay"
    ReplayWriter(replay_dir).write_shard([_replay_sample(0), _replay_sample(1)])

    class CompiledWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = wrapped

        def forward(self, inputs):
            return self._orig_mod(inputs)

    def fake_compile_model(model, *, compile_policy, device, run_name):
        del compile_policy, device, run_name
        return CompiledWrapper(model), "compiled"

    monkeypatch.setattr(trainer_module, "_maybe_compile_model", fake_compile_model)
    trainer = Trainer(
        config,
        replay_reader=ReplayReader(replay_dir),
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        device="cpu",
    )
    trainer.state.model.train()
    trainer.state.optimizer.zero_grad(set_to_none=True)
    batch = {
        "observation": torch.ones((config.training.batch_size, 6, 7, 2)),
        "policy_target": torch.ones((config.training.batch_size, 7)) / 7,
        "value_target": torch.ones((config.training.batch_size,)),
    }
    output = trainer.state.model(batch["observation"])
    loss = output.value.square().mean() + output.policy_logits.square().mean()
    loss.backward()
    trainer.state.optimizer.step()
    before_state = trainer.state.optimizer.state_dict()

    fallback = trainer._fallback_state_to_eager(trainer.state)

    assert fallback.optimizer.state_dict()["param_groups"] == before_state["param_groups"]
    assert fallback.optimizer.state_dict()["state"]
    assert fallback.compile_mode == "fallback"
