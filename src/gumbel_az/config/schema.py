"""Pydantic schema for project YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    output_dir: Path


class ExecutionConfig(StrictBaseModel):
    backend: Literal["single_process", "local_multiprocess", "lan_ray"]


class ClusterConfig(StrictBaseModel):
    enabled: bool
    head_address: str | None = None

    @model_validator(mode="after")
    def validate_head_address(self) -> ClusterConfig:
        if self.enabled and not self.head_address:
            raise ValueError("cluster.head_address is required when cluster.enabled is true")
        return self


class InstallConfig(StrictBaseModel):
    profile: Literal["cpu", "cuda", "cuda12", "distributed", "analysis", "dev"]


class StorageConfig(StrictBaseModel):
    backend: Literal["filesystem"]
    root: Path


class LoggingConfig(StrictBaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: Literal["jsonl", "text"]


class GameConfig(StrictBaseModel):
    name: str = Field(min_length=1)
    implementation: str = Field(min_length=1)


class AlgorithmConfig(StrictBaseModel):
    name: str = Field(min_length=1)


class SearchConfig(StrictBaseModel):
    backend: str = Field(min_length=1)
    simulations_per_move: PositiveInt
    max_num_considered_actions: PositiveInt
    gumbel_scale: float = Field(ge=0.0)
    q_transform: str = Field(min_length=1)


class ModelConfig(StrictBaseModel):
    name: str = Field(min_length=1)
    channels: PositiveInt | None = None
    blocks: PositiveInt | None = None
    hidden_size: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_model_parameters(self) -> ModelConfig:
        if self.name == "resnet_board" and (self.channels is None or self.blocks is None):
            raise ValueError("resnet_board requires model.channels and model.blocks")
        if self.name == "mlp_small" and self.hidden_size is None:
            raise ValueError("mlp_small requires model.hidden_size")
        return self


class SelfPlayConfig(StrictBaseModel):
    games_per_iteration: PositiveInt
    batch_size: PositiveInt
    temperature_moves: int = Field(ge=0)
    shard_max_samples: PositiveInt


class ReplayConfig(StrictBaseModel):
    window_samples: PositiveInt
    min_samples_to_train: PositiveInt
    low_watermark: PositiveInt
    high_watermark: PositiveInt

    @model_validator(mode="after")
    def validate_watermarks(self) -> ReplayConfig:
        if self.min_samples_to_train > self.window_samples:
            raise ValueError("replay.min_samples_to_train cannot exceed replay.window_samples")
        if self.low_watermark >= self.high_watermark:
            raise ValueError("replay.low_watermark must be lower than replay.high_watermark")
        if self.high_watermark > self.window_samples:
            raise ValueError("replay.high_watermark cannot exceed replay.window_samples")
        return self


class TrainingConfig(StrictBaseModel):
    batch_size: PositiveInt
    steps_per_iteration: PositiveInt
    optimizer: Literal["adamw"]
    learning_rate: PositiveFloat
    weight_decay: float = Field(ge=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    gradient_clip_norm: PositiveFloat | None = 1.0
    checkpoint_every_steps: PositiveInt


class EvalConfig(StrictBaseModel):
    enabled: bool
    games: PositiveInt
    promotion_win_rate: float = Field(gt=0.0, lt=1.0)


class StopConfig(StrictBaseModel):
    max_iterations: PositiveInt | None = None
    max_train_steps: PositiveInt | None = None
    max_games: PositiveInt | None = None
    max_wall_time_sec: PositiveFloat | None = None


class AppConfig(StrictBaseModel):
    run: RunConfig
    execution: ExecutionConfig
    cluster: ClusterConfig
    install: InstallConfig
    storage: StorageConfig
    logging: LoggingConfig
    game: GameConfig
    algorithm: AlgorithmConfig
    search: SearchConfig
    model: ModelConfig
    selfplay: SelfPlayConfig
    replay: ReplayConfig
    training: TrainingConfig
    eval: EvalConfig
    stop: StopConfig

    @model_validator(mode="after")
    def validate_execution_cluster(self) -> AppConfig:
        if self.execution.backend == "lan_ray" and not self.cluster.enabled:
            raise ValueError("execution.backend lan_ray requires cluster.enabled true")
        if self.game.name == "connect_four" and self.search.max_num_considered_actions > 7:
            raise ValueError("Connect Four has at most 7 actions")
        return self
