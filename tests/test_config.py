from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gumbel_az.config import AppConfig, load_config, save_resolved_config
from gumbel_az.config.overrides import apply_overrides, parse_override

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.mark.parametrize(
    "config_name",
    [
        "connect_four.yaml",
        "connect_four_lan_fast.yaml",
        "connect_four_lan.yaml",
        "connect_four_lan_long.yaml",
        "connect_four_cpu_debug.yaml",
        "connect_four_gpu.yaml",
    ],
)
def test_project_configs_are_valid(config_name: str) -> None:
    config = load_config(CONFIG_DIR / config_name)

    assert isinstance(config, AppConfig)
    assert config.game.name == "connect_four"
    assert config.algorithm.name == "gumbel_alphazero"
    assert config.search.max_num_considered_actions == 7


def test_cpu_debug_config_has_finite_stop_conditions() -> None:
    config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")

    assert config.stop.max_iterations == 1
    assert config.stop.max_train_steps == 4
    assert config.stop.max_games == 8
    assert config.stop.max_wall_time_sec == 60


def test_real_config_is_not_debug_preset() -> None:
    real_config = load_config(CONFIG_DIR / "connect_four.yaml")
    debug_config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")

    assert real_config.model.name == "resnet_board"
    assert debug_config.model.name == "mlp_small"
    assert real_config.search.simulations_per_move > debug_config.search.simulations_per_move
    assert real_config.selfplay.games_per_iteration > debug_config.selfplay.games_per_iteration


def test_invalid_config_schema_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
run:
  name: broken
  seed: 1
  output_dir: artifacts/runs
execution:
  backend: invalid_backend
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_yaml_root_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_config(config_path)


def test_overrides_are_applied() -> None:
    config = load_config(
        CONFIG_DIR / "connect_four_cpu_debug.yaml",
        ["run.seed=123", "training.batch_size=16", "eval.enabled=false"],
    )

    assert config.run.seed == 123
    assert config.training.batch_size == 16
    assert config.eval.enabled is False


def test_training_compile_defaults_to_auto_and_can_be_overridden() -> None:
    default_config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")
    override_config = load_config(
        CONFIG_DIR / "connect_four_cpu_debug.yaml",
        ["training.compile=off"],
    )

    assert default_config.training.compile == "auto"
    assert override_config.training.compile == "off"


def test_lan_actor_resources_default_to_auto_and_accept_manual_caps() -> None:
    automatic = load_config(CONFIG_DIR / "connect_four_lan.yaml")
    manual = load_config(
        CONFIG_DIR / "connect_four_lan.yaml",
        [
            "cluster.max_selfplay_actors_per_node=4",
            "cluster.head_selfplay_actors=2",
        ],
    )

    assert automatic.cluster.max_selfplay_actors_per_node == "auto"
    assert automatic.cluster.head_selfplay_actors == "auto"
    assert manual.cluster.max_selfplay_actors_per_node == 4
    assert manual.cluster.head_selfplay_actors == 2


@pytest.mark.parametrize(
    "override",
    [
        "cluster.max_selfplay_actors_per_node=0",
        "cluster.max_selfplay_actors_per_node=invalid",
        "cluster.head_selfplay_actors=-1",
        "cluster.head_selfplay_actors=invalid",
    ],
)
def test_lan_actor_resource_policy_rejects_invalid_values(override: str) -> None:
    with pytest.raises(ValidationError):
        load_config(CONFIG_DIR / "connect_four_lan.yaml", [override])


def test_parse_override_requires_equals() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_override("run.seed")


def test_apply_overrides_does_not_mutate_original() -> None:
    original = {"run": {"seed": 1}}
    updated = apply_overrides(original, ["run.seed=2"])

    assert original["run"]["seed"] == 1
    assert updated["run"]["seed"] == 2


def test_save_resolved_config(tmp_path: Path) -> None:
    config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")
    output_path = save_resolved_config(config, tmp_path / "run")

    assert output_path == tmp_path / "run" / "config.resolved.yaml"
    assert output_path.exists()
    loaded = load_config(output_path)
    assert loaded == config


def test_relative_paths_are_preserved_cross_platform() -> None:
    config = load_config(CONFIG_DIR / "connect_four.yaml")

    assert config.run.output_dir == Path("artifacts/runs")
    assert config.storage.root == Path("artifacts/runs")
