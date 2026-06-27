"""Run the PyTorch self-play/search benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from gumbel_az.benchmark import benchmark_selfplay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/connect_four_cpu_debug.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/search.jsonl"))
    args = parser.parse_args()
    print(benchmark_selfplay(args.config, output=args.output))


if __name__ == "__main__":
    main()
