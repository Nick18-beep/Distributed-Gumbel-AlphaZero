# Distributed Gumbel AlphaZero

Framework local-first per addestrare agenti self-play su giochi a informazione perfetta.
Il target corrente e' Connect Four con una implementazione reale di Gumbel AlphaZero
basata su JAX/Flax/Optax, MCTX, replay locale, checkpoint Orbax, evaluation e CLI.

## Quickstart CPU

```bash
python scripts/bootstrap.py --profile cpu
uv run gaz doctor
uv run gaz run --config configs/connect_four.yaml
```

Per smoke test rapidi su laptop:

```bash
uv run gaz run --config configs/connect_four_cpu_debug.yaml
```

## Quickstart GPU

JAX GPU NVIDIA non e' supportato su Windows nativo. Per CUDA usare Linux o WSL2.

```bash
python scripts/bootstrap.py --profile cuda
uv run gaz doctor --cuda
uv run gaz run --config configs/connect_four_gpu.yaml
```

Se serve CUDA 12:

```bash
python scripts/bootstrap.py --profile cuda12
```

## Quickstart Windows

Windows nativo e' supportato come piattaforma CPU:

```powershell
python scripts/bootstrap.py --profile cpu --profile dev
uv run gaz doctor
uv run pytest
uv run gaz run --config configs/connect_four_cpu_debug.yaml
```

CUDA su Windows nativo non viene promessa: usare WSL2/Linux.

## Quickstart Linux/WSL

```bash
python scripts/bootstrap.py --profile cpu --profile dev
uv run gaz doctor
uv run pytest
uv run gaz run --config configs/connect_four_cpu_debug.yaml
```

## Quickstart LAN Ray

Ray e' opzionale e sta dietro `ExecutionBackend`. Il progetto deve funzionare senza
Ray installato. Per LAN:

```bash
python scripts/bootstrap.py --profile cpu --profile distributed
uv run gaz doctor --distributed
```

Head:

```bash
uv run gaz cluster head --config configs/connect_four_lan.yaml --host 0.0.0.0 --port 6379
```

Il comando stampa un indirizzo del tipo `ray head ready: 192.168.1.50:6379`.
Usare quell'IP LAN reale nei comandi worker e run. Non usare letteralmente
`HEAD_IP` o `MASTER_IP`, e non usare `0.0.0.0` come indirizzo di connessione.

Worker:

```bash
uv run gaz cluster worker --head 192.168.1.50:6379 --config configs/connect_four_lan.yaml --auto
```

Run LAN:

```bash
uv run gaz run --config configs/connect_four_lan.yaml --set cluster.head_address=192.168.1.50:6379
```

Se Ray non e' installato, i comandi LAN falliscono con un errore esplicito e il resto
del progetto resta utilizzabile.

## Comandi principali

```bash
uv run gaz doctor
uv run gaz doctor --fix
uv run gaz doctor --cuda
uv run gaz doctor --distributed
uv run gaz run --config configs/connect_four.yaml
uv run gaz resume artifacts/runs/<run_id>
uv run gaz play --config configs/connect_four.yaml --run-dir artifacts/runs/<run_id>
uv run gaz benchmark --config configs/connect_four_cpu_debug.yaml
uv run gaz inspect run artifacts/runs/<run_id>
uv run gaz inspect replay artifacts/runs/<run_id>/replay
uv run gaz inspect checkpoint artifacts/runs/<run_id>/checkpoints
```

## Artifact runtime

Ogni run scrive:

```text
artifacts/runs/<run_id>/
  config.resolved.yaml
  run_state.json
  logs/
    events.jsonl
    metrics.jsonl
  replay/
    shards/
    index.json
    quarantine/
  checkpoints/
    index.json
    latest.json
    best.json
  eval/
    matches.jsonl
```

Replay usa msgpack + zstd, con shard append-only e import atomico. Checkpoint usa
Orbax con registry JSON atomico.

## Gumbel AlphaZero

Il path primario usa:

- Connect Four custom JAX-friendly;
- network Flax (`mlp_small` per debug, `resnet_board` per preset reale);
- MCTX `gumbel_muzero_policy`;
- legal masking obbligatorio;
- recurrent function con dinamica reale del gioco;
- replay target con policy migliorata e value dalla prospettiva di `to_play`;
- training JIT con AdamW, cosine schedule, clipping e checkpoint.

Se JAX non e' disponibile, il runtime puo' usare fallback PyTorch per mantenere
diagnostica e replay operativo. Il fallback e' loggato chiaramente come
`runtime_backend=torch` e non viene mascherato come path Gumbel/MCTX primario.

## Aggiungere un gioco

1. Implementare il contratto `GameAdapter` in `src/gumbel_az/envs/custom/<game>.py`
   oppure dietro un adapter a libreria matura.
2. Registrare il gioco in `src/gumbel_az/envs/registry.py`.
3. Aggiungere una config dedicata in `configs/`.
4. Aggiungere test di contratto, legal mask, terminal rewards e self-play smoke.

Il training loop non deve cambiare quando si aggiunge un gioco.

## Aggiungere un algoritmo

Il registry algoritmi deve contenere solo algoritmi realmente implementati e testati.
Oggi il solo algoritmo registrato e' `gumbel_alphazero`. Un nuovo algoritmo deve
passare da `TrainingAlgorithm`, usare config dedicata e non modificare il
`GameAdapter`.

## Aggiungere un modello

1. Implementare la factory in `src/gumbel_az/model/`.
2. Registrarla nel model registry.
3. Accettare `observation_shape` e `num_actions` dal gioco.
4. Aggiungere test init/forward/shape.

## Troubleshooting

- `Ray is not installed`: eseguire `uv sync --extra distributed`.
- `jax backend: cpu` su Windows: e' atteso su Windows nativo.
- CUDA NVIDIA: usare Linux/WSL2 e profilo `cuda` o `cuda12`.
- Replay corrotto: `gaz inspect replay ...` mostra `read_errors`; gli shard corrotti
  vengono messi in `quarantine/` quando rilevati.
- Checkpoint mancante: controllare `checkpoints/latest.json` e `checkpoints/best.json`.

## Limiti noti

- LAN Ray e' opzionale e richiede extra `distributed`.
- Test Linux/WSL richiedono una distro WSL o CI Linux disponibile.
- Docker e database server non sono richiesti.
- Il bootstrap remoto automatico multi-PC non fa parte della roadmap corrente.
