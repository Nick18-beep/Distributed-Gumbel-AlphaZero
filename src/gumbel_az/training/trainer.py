"""Single-process trainer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.logging import MetricWriter
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.replay.reader import ReplayReader
from gumbel_az.replay.sampler import ReplaySampler
from gumbel_az.training.train_state import GAZTrainState, create_train_state, train_step


@dataclass(frozen=True)
class TrainLoopResult:
    state: GAZTrainState
    checkpoint_version: int
    steps: int
    samples_seen: int
    samples_per_sec: float
    replay_sample_age_mean: float
    latest_metrics: dict[str, float]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sample_age_seconds(samples: list[dict]) -> float:
    ages = []
    now = _utc_now()
    for sample in samples:
        timestamp = sample.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        try:
            created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        ages.append((now - created).total_seconds())
    return float(np.mean(ages)) if ages else 0.0


class Trainer:
    def __init__(
        self,
        config: AppConfig,
        *,
        replay_reader: ReplayReader,
        checkpoint_manager: CheckpointManager,
        metric_writer: MetricWriter | None = None,
    ) -> None:
        self.config = config
        self.game = create_game(config.game.name)
        self.network = create_network(config.model, num_actions=self.game.num_actions)
        self.replay_reader = replay_reader
        self.replay_sampler = ReplaySampler(replay_reader, config.replay.window_samples)
        self.checkpoint_manager = checkpoint_manager
        self.metric_writer = metric_writer
        self.tx, self.schedule = create_optimizer(config.training)
        params = self.network.init(
            jax.random.PRNGKey(config.run.seed),
            self.game.observation_shape,
            self.game.num_actions,
        )
        self.state = create_train_state(params=params, apply_fn=self.network.apply, tx=self.tx)

    def _save_checkpoint(self, state: GAZTrainState) -> int:
        checkpoint_version = int(state.step)
        self.checkpoint_manager.save(
            version=checkpoint_version,
            state={"params": state.params, "opt_state": state.opt_state, "step": state.step},
            metadata={
                "training_step": checkpoint_version,
                "game": self.game.name,
                "algorithm": self.config.algorithm.name,
                "model": self.config.model.name,
            },
        )
        return checkpoint_version

    def _augmented_samples(self) -> list[dict]:
        samples = self.replay_sampler.samples()
        augmented: list[dict] = []
        for sample in samples:
            augmented.extend(self.game.symmetries(sample))
        return augmented

    def _sample_batch(self, arrays: dict[str, np.ndarray], step: int) -> dict[str, jax.Array]:
        batch_size = self.config.training.batch_size
        return self.replay_sampler.sample_arrays(
            arrays,
            batch_size=batch_size,
            seed=self.config.run.seed + step,
            replace_if_needed=True,
        )

    def run(self, *, max_steps: int | None = None) -> TrainLoopResult:
        steps = max_steps or self.config.training.steps_per_iteration
        samples = self._augmented_samples()
        sample_arrays = self.replay_sampler.arrays_from(samples)
        replay_age = _sample_age_seconds(samples)
        start = perf_counter()
        latest_metrics: dict[str, float] = {}
        state = self.state
        for _ in range(steps):
            batch = self._sample_batch(sample_arrays, int(state.step))
            learning_rate = self.schedule(state.step)
            state, metrics = train_step(state, batch, learning_rate)
            latest_metrics = {
                key: float(value) if hasattr(value, "item") else float(value)
                for key, value in metrics.items()
            }
            if self.metric_writer is not None:
                self.metric_writer.write_metrics(
                    int(state.step),
                    {
                        "policy_loss": latest_metrics["policy_loss"],
                        "value_loss": latest_metrics["value_loss"],
                        "total_loss": latest_metrics["total_loss"],
                        "learning_rate": latest_metrics["learning_rate"],
                        "grad_norm": latest_metrics["grad_norm"],
                        "replay_sample_age_mean": replay_age,
                    },
                )
            if int(state.step) % self.config.training.checkpoint_every_steps == 0:
                self._save_checkpoint(state)
        self.state = state
        elapsed = perf_counter() - start
        checkpoint_version = int(state.step)
        if checkpoint_version % self.config.training.checkpoint_every_steps != 0:
            checkpoint_version = self._save_checkpoint(state)
        return TrainLoopResult(
            state=state,
            checkpoint_version=checkpoint_version,
            steps=steps,
            samples_seen=steps * self.config.training.batch_size,
            samples_per_sec=(steps * self.config.training.batch_size) / max(elapsed, 1.0e-9),
            replay_sample_age_mean=replay_age,
            latest_metrics=latest_metrics,
        )


def greedy_action_from_params(
    network,
    params: dict,
    observation: jax.Array,
    legal: jax.Array,
) -> int:
    output = network.apply(params, observation[None, ...], train=False)
    logits = jnp.where(legal, output.policy_logits[0], -jnp.inf)
    return int(jnp.argmax(logits))
