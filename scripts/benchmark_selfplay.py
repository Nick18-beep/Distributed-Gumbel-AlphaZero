"""Benchmark self-play generation after worker warmup."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from gumbel_az.config import load_config
from gumbel_az.replay import ReplayWriter
from gumbel_az.selfplay import SelfPlayWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/connect_four_cpu_debug.yaml"))
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    with tempfile.TemporaryDirectory(prefix="gaz_selfplay_bench_") as tmp:
        worker = SelfPlayWorker(config, replay_writer=ReplayWriter(Path(tmp) / "replay"))
        worker.play_batch(1, config.run.seed)
        _, result = worker.play_batch(args.games, config.run.seed + 10_000)
        print(
            f"games={result.games} positions={result.positions} "
            f"games_per_sec={result.games_per_sec:.4f} "
            f"positions_per_sec={result.positions_per_sec:.4f}"
        )


if __name__ == "__main__":
    main()
