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

Ray e' opzionale e sta dietro `ExecutionBackend`. Il trainer resta sul master/head;
i worker generano self-play replay shard e li inviano al master.

Comandi gia' compilati per il setup corrente:

```text
master Windows/GPU: 192.168.1.12
worker macOS CPU:   192.168.1.161
Ray head port:      6379
Ray fixed ports:    6380-6386, 10002-10101
```

Prima di partire, assicurarsi che firewall e rete permettano TCP tra worker e
master su `6379`, `6380-6386` e `10002-10101`.
Se un avvio Ray fallito lascia porte occupate, il comando segnala quali porte
sono in uso: fermare Ray con `gaz cluster stop` e ripartire.
Dopo il bootstrap/sync iniziale, usare `uv run --no-sync` nei terminali cluster:
evita sync concorrenti della stessa `.venv` mentre Ray e' attivo.
Sequenza obbligatoria:

```text
1. fermare Ray su master e worker
2. avviare Ray head sul master con porte fisse
3. avviare Ray worker sul Mac/worker con le stesse porte fisse
4. aspettare sul master: "required Ray workers connected: 1/1"
5. solo dopo avviare il training dal secondo terminale master
```

Non avviare `gaz run --execution lan_ray` prima che il worker sia connesso: il
training partirebbe solo sul master e finirebbe senza replay remoto.

### 0. Setup ambiente

Master Windows/GPU:

```bash
cd "D:\nicol\Distributed Gumbel AlphaZero"
python scripts/bootstrap.py --profile cuda --profile distributed
uv run --extra cuda --extra distributed gaz doctor --cuda --distributed
```

Worker macOS:

```bash
cd /Users/nicolo/Desktop/Distributed-Gumbel-AlphaZero
uv sync --extra cpu --extra distributed
uv run --extra cpu --extra distributed gaz doctor --distributed
```

### 1. Master terminale 1: Ray head

Prima ferma eventuali processi Ray vecchi sul master:

```powershell
cd "D:\nicol\Distributed Gumbel AlphaZero"
uv run --no-sync --extra cuda --extra distributed gaz cluster stop
```

Poi avvia il nodo head. Lascia questo terminale aperto:

```powershell
uv run --no-sync --extra cuda --extra distributed gaz cluster head `
  --config configs/connect_four_lan.yaml `
  --host 0.0.0.0 `
  --port 6379 `
  --node-manager-port 6380 `
  --object-manager-port 6381 `
  --runtime-env-agent-port 6382 `
  --dashboard-agent-listen-port 6384 `
  --dashboard-agent-grpc-port 6385 `
  --metrics-export-port 6386 `
  --min-worker-port 10002 `
  --max-worker-port 10101 `
  --wait-workers `
  --min-workers 1
```

Il comando resta in attesa. Quando il worker si collega deve stampare una riga
simile a:

```text
required Ray workers connected: 1/1
```

### 2. Worker macOS

Prima ferma eventuali processi Ray vecchi sul worker:

```bash
cd /Users/nicolo/Desktop/Distributed-Gumbel-AlphaZero
uv run --no-sync --extra cpu --extra distributed gaz cluster stop
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
```

Poi collega il worker al master usando porte fisse. Non omettere questi
argomenti sulle porte: senza porte fisse Ray puo' usare porte random e andare in
timeout su LAN/firewall.
Su macOS il comando `gaz cluster worker` applica automaticamente path temporanei
corti (`/tmp/ray-gaz`, `/tmp`, `/tmp/ray-gaz-spill`) quando non sono indicati.
Sono lasciati espliciti nel comando sotto per rendere chiara la configurazione:
Ray/Plasma puo' crashare con `Invalid argument` se usa un path temporaneo non
adatto, per esempio il `TMPDIR` lungo sotto `/var/folders/...` oppure un path
Windows rimasto in cache come `C:\Users\nicol\AppData\Local\Temp\ray`.

```bash
mkdir -p /tmp/ray-gaz /tmp/ray-gaz-spill

uv run --no-sync --extra cpu --extra distributed gaz cluster worker \
  --head 192.168.1.12:6379 \
  --node-ip 192.168.1.161 \
  --config configs/connect_four_lan.yaml \
  --node-manager-port 6380 \
  --object-manager-port 6381 \
  --runtime-env-agent-port 6382 \
  --dashboard-agent-listen-port 6384 \
  --dashboard-agent-grpc-port 6385 \
  --metrics-export-port 6386 \
  --min-worker-port 10002 \
  --max-worker-port 10101 \
  --temp-dir /tmp/ray-gaz \
  --plasma-directory /tmp \
  --object-spilling-directory /tmp/ray-gaz-spill \
  --keep-alive \
  --keep-alive-poll-sec 10 \
  --auto
```

`--keep-alive` lascia il terminale del worker attivo dopo l'avvio di Ray. La CLI
controlla periodicamente lo stato del cluster ogni `--keep-alive-poll-sec`
secondi e, con `Ctrl+C`, ferma il worker Ray locale.

Se dopo un pull compare:

```text
No such option: --keep-alive
```

la virtualenv sta ancora eseguendo una versione vecchia della CLI. Aggiornare
l'ambiente e verificare che l'help mostri i flag keep-alive:

```bash
uv sync --extra cpu --extra distributed
uv run --no-sync --extra cpu --extra distributed gaz cluster worker --help | grep keep-alive
```

Se il worker va in timeout con `RPC error: Deadline Exceeded`, controllare prima
la connettivita' base dal Mac. Se queste porte rispondono, il master e'
raggiungibile e il problema e' quasi certamente nello startup locale del worker
Ray, non nella rete.

```bash
nc -vz 192.168.1.12 6379
nc -vz 192.168.1.12 6380
nc -vz 192.168.1.12 6381
nc -vz 192.168.1.12 6382
nc -vz 192.168.1.12 6384
nc -vz 192.168.1.12 6385
nc -vz 192.168.1.12 6386
```

Per diagnosticare il caso Ray/Plasma su macOS, cercare l'ultimo `raylet.out` e
`raylet.err`:

```bash
find /tmp/ray-gaz -maxdepth 4 -type f \
  \( -name 'raylet.out' -o -name 'raylet.err' \) -print | tail -20
```

Il sintomo tipico e':

```text
Starting object store with directory C:\Users\nicol\AppData\Local\Temp\ray
Unhandled exception ... Invalid argument [system:22]
```

In quel caso fermare Ray e rilanciare il worker con i tre argomenti
`--temp-dir`, `--plasma-directory` e `--object-spilling-directory` mostrati
sopra:

```bash
uv run --no-sync --extra cpu --extra distributed gaz cluster stop
mkdir -p /tmp/ray-gaz /tmp/ray-gaz-spill
```

Il training non va avviato finche' il master non vede il worker connesso.
Una verifica positiva da master deve mostrare due nodi attivi:

```powershell
uv run --no-sync --extra cuda --extra distributed gaz cluster status --head 192.168.1.12:6379
```

Con master `192.168.1.12` e worker `192.168.1.161`, l'output atteso contiene:

```text
Active:
  ...
  ...
Resources:
  0.0/20.0 CPU
```

### 3. Master terminale 2: training distribuito

Eseguire questo comando solo dopo che il terminale 1 mostra:

```text
required Ray workers connected: 1/1
```

```powershell
cd "D:\nicol\Distributed Gumbel AlphaZero"

uv run --no-sync --extra cuda --extra distributed gaz run `
  --config configs/connect_four_lan.yaml `
  --execution lan_ray `
  --set cluster.head_address=192.168.1.12:6379
```

Durante il run il master stampa eventi di avanzamento come:

```text
[lan_ray] connected to Ray cluster: 192.168.1.12:6379
[lan_ray] remote worker completed: worker=... imported_samples=...
[run] scheduler: iteration=0 stage=before_training selfplay=True training=True ...
[run] training checkpoint: iteration=0 step=...
```

A fine run, controllare che `run_state.json` contenga replay remoto importato:

```powershell
$latest = Get-Content artifacts\runs\latest.json | ConvertFrom-Json
Get-Content (Join-Path $latest.run_dir "run_state.json") |
  ConvertFrom-Json |
  Select-Object status,backend,remote_workers_available,remote_workers_completed,remote_workers_failed,remote_replay_samples_imported
```

Valori attesi con un worker:

```text
remote_workers_available              1
remote_workers_completed              1
remote_workers_failed                 0
remote_replay_samples_imported        > 0
```

### 4. Status e stop

Status dal master:

```powershell
uv run --no-sync --extra cuda --extra distributed gaz cluster status --head 192.168.1.12:6379
```

Fermare Ray quando il training e' concluso.

Master Windows:

```powershell
uv run --no-sync --extra cuda --extra distributed gaz cluster stop
```

Worker Linux/macOS:

```bash
uv run --no-sync --extra cpu --extra distributed gaz cluster stop
```

Per worker Linux/WSL2 usare lo stesso comando worker macOS, cambiando solo
`--node-ip` con l'IP LAN del worker Linux oppure omettendo `--node-ip` se
l'auto-detect rileva l'interfaccia corretta.

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
