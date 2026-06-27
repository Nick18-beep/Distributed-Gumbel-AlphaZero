"""Replay shard msgpack/zstd codec."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
import zstandard as zstd


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(contiguous.dtype),
            "shape": contiguous.shape,
            "data": contiguous.tobytes(),
        }
    if hasattr(value, "__array__") and not isinstance(value, (str, bytes, bytearray)):
        return _to_serializable(np.asarray(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _from_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            dtype = np.dtype(value["dtype"])
            shape = tuple(value["shape"])
            data = value["data"]
            if isinstance(data, list):
                return np.asarray(data, dtype=dtype).reshape(shape)
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {key: _from_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_serializable(item) for item in value]
    return value


def encode_samples(samples: list[dict[str, Any]]) -> bytes:
    payload = [_to_serializable(sample) for sample in samples]
    packed = msgpack.packb(payload, use_bin_type=True)
    return zstd.ZstdCompressor(level=3).compress(packed)


def decode_samples(data: bytes) -> list[dict[str, Any]]:
    unpacked = msgpack.unpackb(
        zstd.ZstdDecompressor().decompress(data),
        raw=False,
        strict_map_key=False,
    )
    if not isinstance(unpacked, list):
        raise ValueError("replay shard payload must be a list")
    samples = [_from_serializable(sample) for sample in unpacked]
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("replay shard contains non-object samples")
    return samples
