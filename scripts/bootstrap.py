"""Bootstrap the local development environment."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_PROFILES = ("cpu", "cuda", "dev", "distributed", "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Distributed Gumbel AlphaZero.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=VALID_PROFILES,
        default=[],
        help="Dependency profile to install. Can be passed multiple times.",
    )
    return parser.parse_args()


def local_uv_path() -> Path:
    if os.name == "nt":
        return Path.home() / ".local" / "bin" / "uv.exe"
    return Path.home() / ".local" / "bin" / "uv"


def ensure_local_bin_on_path() -> None:
    uv_dir = str(local_uv_path().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if uv_dir not in path_parts:
        os.environ["PATH"] = uv_dir + os.pathsep + os.environ.get("PATH", "")


def find_uv() -> str | None:
    ensure_local_bin_on_path()
    found = shutil.which("uv")
    if found:
        return found

    candidate = local_uv_path()
    if candidate.exists():
        return str(candidate)

    return None


def run(command: list[str], *, retry_system_certs: bool = False) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode == 0:
        return

    if retry_system_certs and "--system-certs" not in command:
        retry = [*command, "--system-certs"]
        print("Command failed; retrying with --system-certs.", flush=True)
        print(f"+ {' '.join(retry)}", flush=True)
        retry_result = subprocess.run(
            retry,
            cwd=PROJECT_ROOT,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
        )
        if retry_result.stdout:
            print(retry_result.stdout, end="", flush=True)
        if retry_result.returncode == 0:
            return
        raise SystemExit(retry_result.returncode)

    raise SystemExit(result.returncode)


def install_uv() -> None:
    print("uv not found; installing uv with the official Astral installer.", flush=True)
    if os.name == "nt":
        command = [
            "powershell",
            "-ExecutionPolicy",
            "ByPass",
            "-NoProfile",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    else:
        command = ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
    run(command)
    ensure_local_bin_on_path()


def ensure_python_version() -> None:
    if sys.version_info < (3, 11):  # noqa: UP036 - bootstrap can run before packaging metadata.
        raise SystemExit("Python 3.11+ is required.")
    if sys.version_info >= (3, 14):
        raise SystemExit("Python 3.14 is not supported yet; use Python 3.11, 3.12, or 3.13.")


def ensure_artifact_dirs() -> None:
    for relative in ("artifacts", "artifacts/runs", "artifacts/cache"):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def sync_profiles(uv: str, profiles: list[str]) -> None:
    command = [uv, "sync"]
    for profile in profiles:
        command.extend(["--extra", profile])
    run(command, retry_system_certs=True)


def run_doctor(uv: str) -> None:
    run([uv, "run", "gaz", "doctor"])


def main() -> None:
    args = parse_args()
    profiles = args.profile or ["cpu"]

    ensure_python_version()
    print(f"Platform: {platform.system()} {platform.release()}", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"Profiles: {', '.join(profiles)}", flush=True)

    uv = find_uv()
    if uv is None:
        install_uv()
        uv = find_uv()
    if uv is None:
        raise SystemExit("uv installation finished, but uv was not found on PATH.")

    print(f"uv: {uv}", flush=True)
    ensure_artifact_dirs()
    sync_profiles(uv, profiles)
    run_doctor(uv)

    print(flush=True)
    print("Bootstrap complete.", flush=True)
    print("Next command:", flush=True)
    print("  uv run gaz run --config configs/connect_four.yaml", flush=True)


if __name__ == "__main__":
    main()
