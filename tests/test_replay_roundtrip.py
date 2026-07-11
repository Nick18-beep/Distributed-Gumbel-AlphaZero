from __future__ import annotations

import json
from pathlib import Path

import msgpack
import numpy as np
import torch
import zstandard as zstd

from gumbel_az.replay import ReplayReader, ReplaySampler, ReplayWriter
from gumbel_az.replay.codec import decode_samples, encode_samples


def _sample(index: int = 0) -> dict:
    return {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": np.zeros((6, 7, 2), dtype=np.float32),
        "legal_action_mask": np.asarray([True, True, False, True, True, True, True]),
        "policy_target": np.asarray([0.2, 0.2, 0.0, 0.2, 0.2, 0.1, 0.1]),
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
    assert isinstance(batch["observation"], torch.Tensor)


def test_replay_sampler_samples_prebuilt_tensors_deterministically(tmp_path: Path) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    writer.write_shard([_sample(0), _sample(1), _sample(2)])
    reader = ReplayReader(tmp_path / "replay")
    sampler = ReplaySampler(reader, window_samples=3)
    tensors = sampler.tensors_from_arrays(sampler.arrays_from(reader.read_all()))

    batch_a = sampler.sample_tensors(tensors, batch_size=4, seed=11, replace_if_needed=True)
    batch_b = sampler.sample_tensors(tensors, batch_size=4, seed=11, replace_if_needed=True)

    assert torch.equal(batch_a["observation"], batch_b["observation"])
    assert torch.equal(batch_a["policy_target"], batch_b["policy_target"])
    assert torch.equal(batch_a["value_target"], batch_b["value_target"])


def test_replay_sampler_arrays_and_tensors_match_for_same_seed(tmp_path: Path) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    writer.write_shard([_sample(0), _sample(1), _sample(2)])
    reader = ReplayReader(tmp_path / "replay")
    sampler = ReplaySampler(reader, window_samples=3)
    arrays = sampler.arrays_from(reader.read_all())
    tensors = sampler.tensors_from_arrays(arrays)

    from_arrays = sampler.sample_arrays(
        arrays,
        batch_size=2,
        seed=99,
        replace_if_needed=False,
    )
    from_tensors = sampler.sample_tensors(
        tensors,
        batch_size=2,
        seed=99,
        replace_if_needed=False,
    )

    assert torch.equal(from_arrays["observation"], from_tensors["observation"])
    assert torch.equal(from_arrays["policy_target"], from_tensors["policy_target"])
    assert torch.equal(from_arrays["value_target"], from_tensors["value_target"])


def test_replay_sampler_rejects_large_batch_without_replacement(tmp_path: Path) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    writer.write_shard([_sample(0)])
    reader = ReplayReader(tmp_path / "replay")
    sampler = ReplaySampler(reader, window_samples=1)
    tensors = sampler.tensors_from_arrays(sampler.arrays_from(reader.read_all()))

    try:
        sampler.sample_tensors(tensors, batch_size=2, seed=0, replace_if_needed=False)
    except ValueError as exc:
        assert "not enough replay samples" in str(exc)
    else:
        raise AssertionError("expected insufficient replay samples to fail")


def test_replay_sampler_pin_memory_guard_keeps_tensors_when_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = ReplayWriter(tmp_path / "replay")
    writer.write_shard([_sample(0)])
    reader = ReplayReader(tmp_path / "replay")
    sampler = ReplaySampler(reader, window_samples=1)
    arrays = sampler.arrays_from(reader.read_all())

    def fail_pin_memory(self):
        raise RuntimeError("pinned memory unavailable")

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_pin_memory)

    tensors = sampler.tensors_from_arrays(arrays, pin_memory=True)

    assert tensors["observation"].shape == (1, 6, 7, 2)
    assert not tensors["observation"].is_pinned()


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


def test_replay_index_is_portable_when_directory_moves(tmp_path: Path) -> None:
    source = tmp_path / "source" / "replay"
    ReplayWriter(source).write_shard([_sample(0)])
    moved = tmp_path / "moved" / "replay"
    moved.parent.mkdir()
    source.rename(moved)

    index = json.loads((moved / "index.json").read_text(encoding="utf-8"))

    assert index["shards"][0]["path"] == "shards/shard_000000001.msgpack.zst"
    assert len(ReplayReader(moved).read_all()) == 1


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
    index = json.loads((replay_dir / "index.json").read_text(encoding="utf-8"))
    assert index["shards"] == []
    assert index["total_samples"] == 0


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
