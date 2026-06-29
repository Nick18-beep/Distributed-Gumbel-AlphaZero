"""Environment diagnostics for the ``gaz doctor`` command."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Protocol

import typer


class Reporter(Protocol):
    def __call__(self, message: str, *, err: bool = False) -> None: ...


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str


BASE_IMPORTS = ("pydantic", "typer", "yaml", "numpy", "msgpack", "zstandard")
ML_IMPORTS = ("torch",)
OPTIONAL_EXTRA_MARKERS = {
    "dev": ("pytest", "ruff", "mypy"),
    "distributed": ("ray",),
    "analysis": ("duckdb", "pandas", "matplotlib"),
}


def _local_uv_path() -> Path:
    executable = "uv.exe" if sys.platform == "win32" else "uv"
    return Path.home() / ".local" / "bin" / executable


def _safe_find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except Exception:
        return None


def find_uv() -> str | None:
    found = which("uv")
    if found:
        return found

    local_uv = _local_uv_path()
    if local_uv.exists():
        return str(local_uv)

    return None


def _is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    proc_version = Path("/proc/version")
    if proc_version.exists():
        try:
            return "microsoft" in proc_version.read_text(encoding="utf-8").lower()
        except OSError:
            return False
    return False


def _os_name() -> str:
    system = platform.system()
    if _is_wsl():
        return "WSL"
    if system == "Windows":
        return "Windows"
    if system == "Linux":
        return "Linux"
    return system or "unknown"


def _in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def _import_status(
    module_name: str,
    *,
    missing_status: str = "ERROR",
    error_status: str = "ERROR",
) -> CheckResult:
    if _safe_find_spec(module_name) is None:
        return CheckResult(missing_status, f"import {module_name}", "module not found")

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive diagnostic path.
        return CheckResult(error_status, f"import {module_name}", f"{type(exc).__name__}: {exc}")

    version = getattr(module, "__version__", "installed")
    return CheckResult("OK", f"import {module_name}", str(version))


def _write_artifacts_check(project_root: Path, *, fix: bool) -> CheckResult:
    artifacts = project_root / "artifacts"
    if not artifacts.exists():
        if fix:
            artifacts.mkdir(parents=True, exist_ok=True)
        else:
            return CheckResult("WARN", "artifacts writable", "artifacts/ does not exist")

    probe = artifacts / f".doctor_write_test_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult("ERROR", "artifacts writable", str(exc))

    return CheckResult("OK", "artifacts writable", str(artifacts))


def _config_check(project_root: Path) -> CheckResult:
    config_path = project_root / "configs" / "connect_four.yaml"
    if not config_path.exists():
        return CheckResult("WARN", "connect_four config", "configs/connect_four.yaml not found yet")

    try:
        from gumbel_az.config import load_config

        load_config(config_path)
    except Exception as exc:  # pragma: no cover - defensive diagnostic path.
        return CheckResult("ERROR", "connect_four config", f"{type(exc).__name__}: {exc}")

    return CheckResult("OK", "connect_four config", str(config_path))


def _installed_optional_extras() -> list[str]:
    extras = ["cpu"]
    for extra, modules in OPTIONAL_EXTRA_MARKERS.items():
        if any(_safe_find_spec(module_name) is not None for module_name in modules):
            extras.append(extra)
    return extras


def _uv_sync_command(uv_path: str) -> list[str]:
    command = [uv_path, "sync"]
    for extra in _installed_optional_extras():
        command.extend(("--extra", extra))
    return command


def _run_uv_sync(uv_path: str, project_root: Path, reporter: Reporter) -> CheckResult:
    command = _uv_sync_command(uv_path)
    reporter(f"+ {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=project_root,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        retry = [*command, "--system-certs"]
        reporter("uv sync failed; retrying with --system-certs")
        reporter(f"+ {' '.join(retry)}")
        result = subprocess.run(
            retry,
            cwd=project_root,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
        )

    if result.stdout:
        reporter(result.stdout.rstrip())

    if result.returncode == 0:
        return CheckResult("OK", "uv sync", "completed")
    return CheckResult("ERROR", "uv sync", f"exit code {result.returncode}")


def _torch_checks(cuda_requested: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    if _safe_find_spec("torch") is None:
        results.append(CheckResult("ERROR", "torch", "not installed"))
        return results

    try:
        import torch
    except Exception as exc:
        results.append(CheckResult("ERROR", "torch", f"{type(exc).__name__}: {exc}"))
        return results

    torch_cuda = getattr(torch, "cuda", None)
    if torch_cuda is None:
        results.append(CheckResult("ERROR", "torch", getattr(torch, "__version__", "installed")))
        results.append(
            CheckResult(
                "ERROR",
                "torch cuda available",
                "torch.cuda is missing; PyTorch installation appears incomplete",
            )
        )
        return results

    cuda_available = torch_cuda.is_available()
    mps_available = (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    results.append(CheckResult("OK", "torch", getattr(torch, "__version__", "installed")))
    results.append(CheckResult("OK", "torch cuda available", str(cuda_available)))
    if cuda_available:
        devices = [
            f"cuda:{index} {torch_cuda.get_device_name(index)}"
            for index in range(torch_cuda.device_count())
        ]
        results.append(CheckResult("OK", "torch cuda devices", ", ".join(devices)))
    results.append(CheckResult("OK", "torch mps available", str(mps_available)))

    if cuda_requested:
        if not cuda_available:
            results.append(CheckResult("ERROR", "cuda devices", "no PyTorch CUDA device found"))
        else:
            results.append(CheckResult("OK", "cuda devices", "PyTorch CUDA detected"))

    return results


def _runtime_backend_check() -> CheckResult:
    from gumbel_az.runtime import detect_runtime_backend

    backend = detect_runtime_backend()
    status = "OK" if backend.name == "torch" else "ERROR"
    return CheckResult(status, "runtime backend", f"{backend.name}: {backend.reason}")


def _distributed_checks() -> list[CheckResult]:
    if _safe_find_spec("ray") is None:
        return [CheckResult("ERROR", "ray", "not installed; install extra 'distributed'")]

    try:
        ray = importlib.import_module("ray")
    except Exception as exc:  # pragma: no cover - defensive diagnostic path.
        return [CheckResult("ERROR", "ray", f"{type(exc).__name__}: {exc}")]

    return [CheckResult("OK", "ray", getattr(ray, "__version__", "installed"))]


def _emit_results(results: list[CheckResult], reporter: Reporter) -> tuple[int, int]:
    warnings = 0
    errors = 0
    for result in results:
        if result.status == "WARN":
            warnings += 1
        elif result.status == "ERROR":
            errors += 1
        reporter(f"[{result.status}] {result.name}: {result.detail}")
    return warnings, errors


def run_doctor(
    *,
    fix: bool,
    distributed: bool,
    cuda: bool,
    reporter: Reporter = typer.echo,
) -> None:
    project_root = Path.cwd()
    results: list[CheckResult] = []

    if fix:
        for relative in ("artifacts", "artifacts/runs", "artifacts/cache"):
            (project_root / relative).mkdir(parents=True, exist_ok=True)

    uv_path = find_uv()

    results.append(CheckResult("OK", "python", platform.python_version()))
    results.append(CheckResult("OK", "os", _os_name()))
    results.append(
        CheckResult(
            "OK" if _in_virtualenv() else "WARN",
            "virtualenv",
            sys.prefix if _in_virtualenv() else "not active",
        )
    )
    results.append(CheckResult("OK" if uv_path else "ERROR", "uv", uv_path or "not found"))

    if fix and uv_path and (project_root / "pyproject.toml").exists():
        results.append(_run_uv_sync(uv_path, project_root, reporter))
    elif fix:
        results.append(CheckResult("SKIP", "uv sync", "pyproject.toml not found"))

    for module_name in BASE_IMPORTS:
        results.append(_import_status(module_name))

    for module_name in ML_IMPORTS:
        if _safe_find_spec(module_name) is not None:
            results.append(
                _import_status(
                    module_name,
                    missing_status="SKIP",
                    error_status="WARN",
                )
            )

    results.extend(_torch_checks(cuda))
    results.append(_runtime_backend_check())

    results.append(_write_artifacts_check(project_root, fix=fix))
    results.append(_config_check(project_root))

    if distributed:
        results.extend(_distributed_checks())

    warnings, errors = _emit_results(results, reporter)
    reporter(f"doctor summary: {errors} error(s), {warnings} warning(s)")

    if errors:
        raise typer.Exit(code=1)
