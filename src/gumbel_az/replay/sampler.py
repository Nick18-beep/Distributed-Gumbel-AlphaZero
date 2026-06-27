"""Uniform replay sampler."""

from __future__ import annotations

import numpy as np

try:
    import jax
except ImportError:  # pragma: no cover - exercised only without ML extras.
    jax = None  # type: ignore[assignment]

from gumbel_az.replay.reader import ReplayReader


class ReplaySampler:
    """Uniform sampler over the latest replay window."""

    def __init__(self, reader: ReplayReader, window_samples: int) -> None:
        self.reader = reader
        self.window_samples = window_samples

    def samples(self) -> list[dict]:
        samples = self.reader.read_all()
        if self.window_samples > 0:
            return samples[-self.window_samples :]
        return samples

    def arrays_from(self, samples: list[dict]) -> dict[str, np.ndarray]:
        if not samples:
            raise ValueError("cannot sample from empty replay")
        return {
            "observation": np.stack(
                [np.asarray(sample["state_or_observation"], dtype=np.float32) for sample in samples]
            ),
            "policy_target": np.stack(
                [np.asarray(sample["policy_target"], dtype=np.float32) for sample in samples]
            ),
            "value_target": np.asarray(
                [sample["value_target"] for sample in samples],
                dtype=np.float32,
            ),
        }

    def sample_arrays(
        self,
        arrays: dict[str, np.ndarray],
        *,
        batch_size: int,
        seed: int,
        replace_if_needed: bool = False,
    ):
        sample_count = int(arrays["observation"].shape[0])
        if sample_count == 0:
            raise ValueError("cannot sample from empty replay")
        replace = replace_if_needed and sample_count < batch_size
        if not replace and sample_count < batch_size:
            raise ValueError("not enough replay samples for batch")
        rng = np.random.default_rng(seed)
        indices = rng.choice(sample_count, size=batch_size, replace=replace)
        batch = {key: value[indices] for key, value in arrays.items()}
        if jax is None:
            return batch
        return jax.tree.map(jax.numpy.asarray, batch)

    def sample(self, *, batch_size: int, seed: int):
        arrays = self.arrays_from(self.samples())
        return self.sample_arrays(
            arrays,
            batch_size=batch_size,
            seed=seed,
            replace_if_needed=False,
        )
