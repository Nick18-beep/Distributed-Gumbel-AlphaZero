"""Uniform replay sampler."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gumbel_az.replay.reader import ReplayReader


class ReplaySampler:
    """Uniform sampler over the latest replay window."""

    def __init__(self, reader: ReplayReader, window_samples: int) -> None:
        self.reader = reader
        self.window_samples = window_samples

    def samples(self) -> list[dict[str, Any]]:
        samples = self.reader.read_all()
        if self.window_samples > 0:
            return samples[-self.window_samples :]
        return samples

    def arrays_from(self, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
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

    def tensors_from_arrays(
        self,
        arrays: dict[str, np.ndarray],
        *,
        pin_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32) for key, value in arrays.items()
        }
        if pin_memory:
            try:
                tensors = {key: value.pin_memory() for key, value in tensors.items()}
            except RuntimeError:
                pass
        return tensors

    def sample_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        batch_size: int,
        seed: int,
        replace_if_needed: bool = False,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        sample_count = int(tensors["observation"].shape[0])
        if sample_count == 0:
            raise ValueError("cannot sample from empty replay")
        replace = replace_if_needed and sample_count < batch_size
        if not replace and sample_count < batch_size:
            raise ValueError("not enough replay samples for batch")
        rng = np.random.default_rng(seed)
        indices_np = rng.choice(sample_count, size=batch_size, replace=replace)
        source_device = tensors["observation"].device
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=source_device)
        device = torch.device(device)
        return {
            key: value.index_select(0, indices).to(device, non_blocking=True)
            for key, value in tensors.items()
        }

    def sample_arrays(
        self,
        arrays: dict[str, np.ndarray],
        *,
        batch_size: int,
        seed: int,
        replace_if_needed: bool = False,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        tensors = self.tensors_from_arrays(arrays, pin_memory=pin_memory and device.type == "cuda")
        return self.sample_tensors(
            tensors,
            batch_size=batch_size,
            seed=seed,
            replace_if_needed=replace_if_needed,
            device=device,
        )

    def sample(
        self,
        *,
        batch_size: int,
        seed: int,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        arrays = self.arrays_from(self.samples())
        return self.sample_arrays(
            arrays,
            batch_size=batch_size,
            seed=seed,
            replace_if_needed=False,
            device=device,
            pin_memory=pin_memory,
        )
