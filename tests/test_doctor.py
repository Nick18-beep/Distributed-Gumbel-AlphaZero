from __future__ import annotations

import builtins
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gumbel_az.cli import doctor as doctor_module
from gumbel_az.cli.main import app

runner = CliRunner()


def _hide_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            raise ModuleNotFoundError("No module named 'ray'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_doctor_reports_core_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "[OK] python:" in result.output
    assert "[OK] os:" in result.output
    assert "[OK] uv:" in result.output
    assert "[OK] import pydantic:" in result.output
    assert "[OK] artifacts writable:" in result.output
    assert "[WARN] connect_four config:" in result.output


def test_doctor_fix_creates_runtime_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor", "--fix"])

    assert result.exit_code == 0
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "artifacts" / "runs").is_dir()
    assert (tmp_path / "artifacts" / "cache").is_dir()
    assert "[SKIP] uv sync: pyproject.toml not found" in result.output


def test_doctor_fix_preserves_installed_optional_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"pytest", "ruff", "ray"}
    monkeypatch.setattr(
        doctor_module,
        "_safe_find_spec",
        lambda module_name: object() if module_name in installed else None,
    )

    command = doctor_module._uv_sync_command("uv")

    assert command == [
        "uv",
        "sync",
        "--extra",
        "cpu",
        "--extra",
        "dev",
        "--extra",
        "distributed",
    ]


def test_doctor_distributed_requires_ray(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    _hide_ray(monkeypatch)

    result = runner.invoke(app, ["doctor", "--distributed"])

    assert result.exit_code == 1
    assert "[ERROR] ray:" in result.output


def test_doctor_cuda_reports_clear_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()

    result = runner.invoke(app, ["doctor", "--cuda"])

    assert "torch cuda available" in result.output or "[ERROR] torch:" in result.output
    if result.exit_code != 0:
        assert "cuda" in result.output.lower() or "torch" in result.output.lower()
