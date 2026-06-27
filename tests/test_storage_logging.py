from __future__ import annotations

import json
from pathlib import Path

import yaml

from gumbel_az.config import load_config, save_resolved_config
from gumbel_az.logging import JsonlWriter, MetricWriter
from gumbel_az.storage import create_run_directory
from gumbel_az.storage.atomic import atomic_write_json, atomic_write_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_atomic_write_json_and_yaml(tmp_path: Path) -> None:
    json_path = atomic_write_json(tmp_path / "nested" / "data.json", {"value": 1})
    yaml_path = atomic_write_yaml(tmp_path / "nested" / "data.yaml", {"value": 2})

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 1}
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(tmp_path.rglob("*.tmp"))


def test_run_directory_layout_and_latest_pointer(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG, [f"run.output_dir={tmp_path.as_posix()}"])

    paths = create_run_directory(config)
    save_resolved_config(config, paths.run_dir)

    assert paths.run_dir.is_dir()
    assert paths.logs_dir.is_dir()
    assert paths.resolved_config_path.exists()
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == paths.run_id
    assert Path(latest["run_dir"]) == paths.run_dir


def test_jsonl_event_and_metric_writers(tmp_path: Path) -> None:
    events_path = tmp_path / "logs" / "events.jsonl"
    metrics_path = tmp_path / "logs" / "metrics.jsonl"

    JsonlWriter(events_path).write({"event": "started"})
    MetricWriter(metrics_path).write_metrics(3, {"loss": 1.5})

    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    metric = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["event"] == "started"
    assert "timestamp" in event
    assert metric["step"] == 3
    assert metric["metrics"]["loss"] == 1.5
