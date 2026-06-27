# Distributed Gumbel AlphaZero

Framework local-first per addestrare agenti self-play su giochi a informazione perfetta.
Il target corrente e' Connect Four con Gumbel AlphaZero PyTorch-native, replay locale,
checkpoint PyTorch, evaluation, CLI e distribuzione opzionale con Ray.

## Quickstart CPU

```bash
python scripts/bootstrap.py --profile cpu
uv run gaz doctor
uv run gaz run --config configs/connect_four.yaml
```

Smoke test rapido:

```bash
uv run gaz run --config configs/connect_four_cpu_debug.yaml
```

## Quickstart GPU

CUDA NVIDIA e' supportata tramite PyTorch su Windows nativo e Linux quando driver e
runtime NVIDIA sono gia' installati correttamente.

```bash
python scripts/bootstrap.py --profile cuda
uv run gaz doctor --cuda
uv run gaz run --config configs/connect_four_gpu.yaml
```

Python supportato: 3.11, 3.12, 3.13. Python 3.14 non e' supportato per ora.

## Quickstart LAN Ray

Ray e' opzionale e sta dietro `ExecutionBackend`.

Master/head:

```bash
uv run --extra cpu --extra distributed gaz cluster head --config configs/connect_four_lan.yaml --host 0.0.0.0 --port 6379 --wait-workers --min-workers 1
```

Worker:

```bash
uv run --extra cpu --extra distributed gaz cluster worker --head 192.168.1.50:6379 --config configs/connect_four_lan.yaml --auto
```

Training LAN dal master:

```bash
uv run --extra cpu --extra distributed gaz run --config configs/connect_four_lan.yaml --execution lan_ray --set cluster.head_address=192.168.1.50:6379
```

Su Windows/macOS la CLI imposta automaticamente
`RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` per i comandi Ray multi-node.

## Comandi principali

```bash
uv run gaz doctor
uv run gaz doctor --fix
uv run gaz doctor --cuda
uv run gaz doctor --distributed
uv run gaz run --config configs/connect_four.yaml
uv run gaz run --config configs/connect_four.yaml --execution local_multiprocess
uv run gaz run --config configs/connect_four_lan.yaml --execution lan_ray --set cluster.head_address=HEAD_IP:6379
uv run gaz selfplay --config configs/connect_four_cpu_debug.yaml --games 2
uv run gaz train --config configs/connect_four_cpu_debug.yaml
uv run gaz eval --config configs/connect_four_cpu_debug.yaml
uv run gaz resume artifacts/runs/<run_id>
uv run gaz play --config configs/connect_four.yaml --run-dir artifacts/runs/<run_id>
uv run gaz benchmark --config configs/connect_four_cpu_debug.yaml --output-dir artifacts/benchmarks
uv run gaz inspect run artifacts/runs/<run_id>
uv run gaz inspect replay artifacts/runs/<run_id>/replay
uv run gaz inspect checkpoint artifacts/runs/<run_id>/checkpoints
```

`gaz run --execution ...` e' un alias pratico per `--set execution.backend=...`.
`--set dotted.key=value` resta disponibile per ogni override puntuale.

## Artifact runtime

```text
artifacts/runs/<run_id>/
  config.resolved.yaml
  run_state.json
  logs/events.jsonl
  logs/metrics.jsonl
  replay/shards/
  replay/index.json
  replay/quarantine/
  checkpoints/index.json
  checkpoints/latest.json
  checkpoints/best.json
  checkpoints/ckpt_000001/checkpoint.pt
  eval/matches.jsonl
```

Replay usa msgpack + zstd con shard append-only. Checkpoint usa `torch.save`
atomico con registry JSON.

## Gumbel AlphaZero

Il path primario usa:

- Connect Four NumPy/PyTorch-friendly;
- modelli PyTorch (`mlp_small`, `resnet_board`);
- backend search `torch_gumbel`;
- legal masking obbligatorio;
- Gumbel root noise, candidate set, sequential halving e Q transform;
- policy target migliorata e value target dalla prospettiva di `to_play`;
- training PyTorch con AdamW, cosine schedule, clipping, AMP CUDA, `training.compile`
  in modalita' `auto|on|off` e checkpoint.

JAX, Flax, Optax, Orbax, MCTX, Chex e PGX non sono runtime attivi del progetto.

## Troubleshooting

- `Ray is not installed`: eseguire `uv sync --extra distributed`.
- `torch cuda available: False`: controllare driver NVIDIA e installazione PyTorch CUDA.
- Worker macOS/Windows Ray: usare i comandi `gaz cluster ...`; la CLI imposta la variabile Ray richiesta.
- Replay corrotto: `gaz inspect replay ...` mostra `read_errors`; gli shard corrotti vanno in `quarantine/`.
- Checkpoint mancante: controllare `checkpoints/latest.json`, `checkpoints/best.json` e `checkpoint.pt`.

## Limiti noti

- LAN Ray e' opzionale e richiede extra `distributed`.
- Docker e database server non sono richiesti.
- Il bootstrap remoto automatico multi-PC non fa parte della roadmap corrente.
