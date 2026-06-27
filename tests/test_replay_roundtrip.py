from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import zstandard as zstd

from gumbel_az.replay import ReplayReader, ReplaySampler, ReplayWriter
from gumbel_az.replay.codec import decode_samples, encode_samples


def _sample(index: int = 0) -> dict:
    return {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": jnp.zeros((6, 7, 2), dtype=jnp.float32),
        "legal_action_mask": jnp.asarray([True, True, False, True, True, True, True]),
        "policy_target": jnp.asarray([0.2, 0.2, 0.0, 0.2, 0.2, 0.1, 0.1]),
        "value_target": 1.0 if index % 2 == 0 else -1.0,
        "to_play": index % 2,
        "move_index": index,
        "game_id": f"game-{index}",
        "model_version": 0,
        "search_stats": {"root_value": np.float32(0.0)},
    }


def test_replay_roundtrip_and_sampler(tmp_path: Path) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    writer.write_shard([_sample(0), _sample(1), _sample(2)])
    reader = ReplayReader(tmp_path / "replay")

    samples = reader.read_all()
    batch = ReplaySampler(reader, window_samples=2).sample(batch_size=2, seed=0)

    assert len(samples) == 3
    assert batch["observation"].shape == (2, 6, 7, 2)
    assert batch["policy_target"].shape == (2, 7)
    assert batch["value_target"].shape == (2,)
    assert isinstance(batch["observation"], jax.Array)


def test_replay_codec_reads_legacy_list_encoded_arrays() -> None:
    sample = {"schema_version": 1, "timestamp": "now", **_sample(0)}
    sample["search_stats"] = {"root_value": 0.0}
    legacy_payload = [
        {
            **sample,
            "state_or_observation": {
                "__ndarray__": True,
                "dtype": "float32",
                "shape": (6, 7, 2),
                "data": np.zeros((6, 7, 2), dtype=np.float32).tolist(),
            },
            "legal_action_mask": {
                "__ndarray__": True,
                "dtype": "bool",
                "shape": (7,),
                "data": [True, True, False, True, True, True, True],
            },
            "policy_target": {
                "__ndarray__": True,
                "dtype": "float32",
                "shape": (7,),
                "data": [0.2, 0.2, 0.0, 0.2, 0.2, 0.1, 0.1],
            },
        }
    ]
    encoded = zstd.ZstdCompressor(level=3).compress(
        msgpack.packb(legacy_payload, use_bin_type=True)
    )

    decoded = decode_samples(encoded)

    assert decoded[0]["state_or_observation"].shape == (6, 7, 2)
    assert decoded[0]["legal_action_mask"].dtype == np.bool_
    assert np.isclose(np.sum(decoded[0]["policy_target"]), 1.0)


def test_replay_index_paths_survive_cwd_change(tmp_path: Path, monkeypatch) -> None:
    replay_dir = tmp_path / "replay"
    writer = ReplayWriter(replay_dir)
    writer.write_shard([_sample(0)])
    other = tmp_path / "other"
    other.mkdir()

    monkeypatch.chdir(other)
    samples = ReplayReader(replay_dir).read_all()

    assert len(samples) == 1


def test_replay_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    bad = {"schema_version": 999, "timestamp": "now", **_sample(0)}

    try:
        writer.write_shard([bad])
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("expected schema mismatch to fail")


def test_replay_writer_uses_authoritative_schema_and_timestamp(tmp_path: Path) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    sample = {"schema_version": 1, "timestamp": "caller-timestamp", **_sample(0)}
    writer.write_shard([sample])

    stored = ReplayReader(tmp_path / "replay").read_all()[0]

    assert stored["schema_version"] == 1
    assert stored["timestamp"] != "caller-timestamp"


def test_replay_writer_does_not_overwrite_existing_orphan_shard(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    shards_dir = replay_dir / "shards"
    shards_dir.mkdir(parents=True)
    orphan = shards_dir / "shard_000000001.msgpack.zst"
    orphan.write_bytes(b"orphan")

    written = ReplayWriter(replay_dir).write_shard([_sample(0)])

    assert written.name == "shard_000000002.msgpack.zst"
    assert orphan.read_bytes() == b"orphan"


def test_corrupted_shard_is_quarantined(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    shards_dir = replay_dir / "shards"
    shards_dir.mkdir(parents=True)
    (replay_dir / "quarantine").mkdir()
    shard = shards_dir / "shard_000000001.msgpack.zst"
    shard.write_bytes(b"not-zstd")
    (replay_dir / "index.json").write_text(
        json.dumps({"schema_version": 1, "shards": [{"path": str(shard), "samples": 1}]}),
        encoding="utf-8",
    )
    reader = ReplayReader(replay_dir)

    try:
        reader.read_all()
    except Exception:
        pass
    else:
        raise AssertionError("expected corrupted shard to fail")

    assert not shard.exists()
    assert list((replay_dir / "quarantine").iterdir())


def test_replay_codec_schema_version_mismatch_on_read(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    shards_dir = replay_dir / "shards"
    shards_dir.mkdir(parents=True)
    (replay_dir / "quarantine").mkdir()
    shard = shards_dir / "shard_000000001.msgpack.zst"
    shard.write_bytes(encode_samples([{"schema_version": 999, "timestamp": "now", **_sample(0)}]))
    (replay_dir / "index.json").write_text(
        json.dumps({"schema_version": 1, "shards": [{"path": str(shard), "samples": 1}]}),
        encoding="utf-8",
    )

    try:
        ReplayReader(replay_dir).read_all()
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("expected schema mismatch to fail")
