# AGENTS.md - Istruzioni permanenti per AI Coding Agent

Questo repository implementa un framework local-first per addestrare agenti
self-play su giochi a informazione perfetta. Il target corrente e':

```text
Connect Four + Gumbel AlphaZero + PyTorch + replay locale + checkpoint + evaluation + CLI + Ray opzionale
```

Il runtime ML attivo e' solo PyTorch. Non reintrodurre JAX, Flax, Optax,
Orbax, MCTX, Chex, PGX, `custom_jax`, `mctx_backend`, `jax_distributed` o
fallback legacy.

Python supportato: `>=3.11,<3.14`.

---

## 0. Esperienza prioritaria

Il quickstart locale deve restare sempre funzionante:

```bash
python scripts/bootstrap.py --profile cpu
uv run --extra cpu gaz doctor
uv run --extra cpu gaz run --config configs/connect_four.yaml
```

Il quickstart GPU deve usare PyTorch CUDA, non JAX:

```bash
python scripts/bootstrap.py --profile cuda
uv run --extra cuda gaz doctor --cuda
uv run --extra cuda gaz run --config configs/connect_four_gpu.yaml
```

Non installare driver GPU, CUDA di sistema, firewall rules, Docker o pacchetti
admin senza conferma esplicita.

---

## 1. Decisioni non negoziabili

1. Connect Four e' il primo gioco ufficiale.
   Il codice non deve conoscere Connect Four fuori da `GameAdapter`, config,
   test specifici e rendering.

2. Gumbel AlphaZero e' il primo algoritmo ufficiale.
   Algoritmi e search devono stare dietro `TrainingAlgorithm` e
   `SearchBackend`.

3. PyTorch e' l'unico runtime ML attivo.
   Training, self-play, search, checkpoint, evaluation, play, benchmark e Ray
   LAN devono usare PyTorch.

4. `uv` e' il package manager principale.
   Usare `pyproject.toml` e `uv.lock`. Non usare `requirements.txt` come fonte
   primaria.

5. Preferire librerie mature quando sono compatibili.
   LightZero puo' essere riferimento/benchmark futuro, ma non e' dipendenza
   primaria. MCTX non va usato perche' e' JAX-native.

6. Windows, Linux/WSL2 e macOS CPU sono piattaforme di prima classe.
   CUDA NVIDIA e' supportata tramite PyTorch su Windows e Linux. macOS puo'
   usare CPU o MPS se disponibile.

7. Non vendere prototipi come implementazione completa.
   Una demo senza replay robusto, checkpoint, evaluation, promotion, metriche,
   resume e test non soddisfa il progetto.

8. Le performance sono requisito ingegneristico.
   Il path caldo deve usare batch, device placement esplicito, trasferimenti
   host/device minimizzati, AMP CUDA quando utile e benchmark riproducibili.

9. Ogni nuova estensione deve passare da registry e contratti.
   Nuovi giochi, algoritmi, modelli e search backend devono essere registrati e
   testati senza rompere Connect Four.

---

## 2. Architettura

Architettura target:

```text
local-first modular monolith + backend di esecuzione pluggabile
```

Principi:

- un solo package Python principale `gumbel_az`;
- una CLI principale `gaz`;
- filesystem locale come storage base;
- nessun Docker obbligatorio;
- nessun database server obbligatorio;
- Ray opzionale dietro `ExecutionBackend`;
- astrazioni esplicite per gioco, algoritmo, modello, search, replay,
  scheduler, storage ed execution.

Il ciclo base deve restare:

```text
self-play -> replay -> training -> checkpoint -> evaluation -> promotion
```

---

## 3. Runtime e dipendenze

Extra supportati:

```text
cpu          -> PyTorch CPU
cuda         -> PyTorch CUDA
dev          -> pytest, pytest-cov, ruff, mypy
distributed  -> Ray
analysis     -> DuckDB, pandas, matplotlib
```

Regole:

- non aggiungere extra JAX o `cuda12` legacy;
- `torch` deve essere dichiarato in `pyproject.toml`;
- il profilo CUDA deve usare gli index ufficiali PyTorch configurati per `uv`;
- import pesanti devono restare lazy dove possibile, cosi' `gaz doctor` puo'
  spiegare cosa manca;
- `detect_torch_runtime()` e' la fonte runtime per device `cuda`, `mps`, `cpu`;
- `gaz doctor` deve riportare versione PyTorch, CUDA availability, device count,
  device name, MPS availability, Ray opzionale e artifact writability.

---

## 4. Modalita' di esecuzione

Tutte le modalita' passano da `ExecutionBackend`.

### 4.1 `single_process`

```bash
uv run --extra cpu gaz run --config configs/connect_four.yaml
```

Caratteristiche:

- self-play, replay, training, checkpoint ed evaluation nello stesso processo;
- modalita' primaria per smoke test e debug;
- deve restare funzionante quando si aggiungono backend paralleli.

### 4.2 `local_multiprocess`

```bash
uv run --extra cpu --extra distributed gaz run --config configs/connect_four.yaml --execution local_multiprocess
```

Caratteristiche:

- processi self-play locali;
- trainer centrale nello stesso PC;
- replay importato atomicamente;
- shutdown pulito e backpressure;
- evitare inizializzazioni concorrenti inutili del runtime PyTorch.

### 4.3 `lan_ray`

Master/head:

```bash
uv run --extra cuda --extra distributed gaz cluster head --config configs/connect_four_lan.yaml --host 0.0.0.0 --port 6379 --wait-workers --min-workers 1
```

Worker:

```bash
uv run --extra cpu --extra distributed gaz cluster worker --head HEAD_IP:6379 --config configs/connect_four_lan.yaml --auto
```

Training LAN dal master:

```bash
uv run --extra cuda --extra distributed gaz run --config configs/connect_four_lan.yaml --execution lan_ray --set cluster.head_address=HEAD_IP:6379
```

Regole:

- Ray deve stare solo in execution/cluster CLI, non in dominio, gioco o algoritmo;
- trainer principale sul nodo head;
- worker Ray generano replay shard con PyTorch e li importano atomicamente sul
  master;
- nessun worker deve scrivere nello storage del trainer salvo filesystem
  condiviso esplicito;
- su Windows/macOS la CLI imposta `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1`.

---

## 5. GameAdapter

Ogni gioco deve implementare:

```python
class GameAdapter(Protocol):
    name: str
    num_players: int
    num_actions: int
    observation_shape: tuple[int, ...]
    max_moves: int
    supports_jit: bool
    supports_vmap: bool

    def init(self, seed): ...
    def legal_action_mask(self, state): ...
    def step(self, state, action): ...
    def is_terminal(self, state): ...
    def current_player(self, state): ...
    def canonical_observation(self, state): ...
    def terminal_value(self, state): ...
    def rewards(self, state): ...
    def symmetries(self, sample): ...
    def render_text(self, state): ...
```

Regole:

- nessuna logica training deve conoscere Connect Four direttamente;
- `legal_action_mask` e' obbligatoria;
- `rewards(state)` e' la fonte autorevole per risultati terminali per-player;
- `value_target` e' sempre dalla prospettiva di `to_play`;
- usare NumPy/PyTorch-friendly state e array;
- test di contratto per ogni gioco.

Connect Four deve coprire board 6x7, gravita', legal mask, win orizzontale,
verticale, diagonali, draw, osservazione canonica, simmetria orizzontale e
render testuale.

---

## 6. Gumbel AlphaZero

Backend operativo: `torch_gumbel`.

Componenti obbligatori:

- policy logits e value dalla rete PyTorch;
- legal masking;
- Gumbel root noise;
- candidate action set;
- sequential halving;
- Q transform documentata;
- improved policy target;
- selected action dalla policy/search migliorata;
- value target dal risultato finale;
- temperature schedule;
- metriche search;
- fallimento/test se azioni illegali ricevono probabilita' positiva.

Il backend search deve lavorare su batch dove possibile e usare inferenza rete
batchata sul device PyTorch selezionato.

---

## 7. Modelli e training

Modelli registrati:

```text
mlp_small
resnet_board
```

Output standard:

```text
policy_logits: [batch, num_actions]
value: [batch]
```

Training:

- `TorchTrainState` con `model`, optimizer, scheduler, scaler AMP opzionale;
- `training.compile` supporta `auto`, `on`, `off`; default `auto`;
- in `auto`, non forzare compile sui preset debug CPU se peggiora il warmup;
- AdamW;
- cosine schedule con warmup opzionale;
- gradient clipping;
- policy loss cross entropy su improved policy target;
- value loss;
- weight decay;
- checkpoint periodici;
- evaluation periodica;
- promotion;
- resume;
- finite loss e gradienti non NaN testati.

Checkpoint:

- usare `torch.save` atomico;
- salvare `model_state_dict`, `optimizer_state_dict`, scheduler, scaler, step,
  metadata e config hash;
- registry locale con `index.json`, `latest.json`, `best.json`;
- non sovrascrivere checkpoint validi.

---

## 8. Replay e storage

Replay locale append-only:

```text
artifacts/runs/<run_id>/
  replay/shards/
  replay/index.json
  replay/quarantine/
  checkpoints/
  eval/matches.jsonl
  logs/events.jsonl
  logs/metrics.jsonl
  config.resolved.yaml
  run_state.json
```

Schema sample:

```text
schema_version
game_name
algorithm_name
state_or_observation
legal_action_mask
policy_target
value_target
to_play
move_index
game_id
model_version
search_stats
timestamp
```

Regole:

- msgpack per serializzazione;
- zstd per compressione;
- temp file -> fsync dove opportuno -> rename atomico;
- shard incompleti mai indicizzati;
- shard corrotti in `quarantine/`;
- sampler restituisce `torch.Tensor`;
- `pin_memory=True` quando il training device e' CUDA.

---

## 9. CLI

Comandi principali:

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
uv run gaz play --config configs/connect_four.yaml --run-dir artifacts/runs/<run_id>
uv run gaz benchmark --config configs/connect_four_cpu_debug.yaml --output-dir artifacts/benchmarks
uv run gaz inspect run artifacts/runs/<run_id>
uv run gaz inspect replay artifacts/runs/<run_id>/replay
uv run gaz inspect checkpoint artifacts/runs/<run_id>/checkpoints
```

`--set dotted.key=value` deve restare supportato per override puntuali.

---

## 10. Test e verifica

Prima di dichiarare completa una modifica rilevante:

```bash
uv run --extra cpu --extra dev --extra distributed ruff check src/gumbel_az scripts tests
uv run --extra cpu --extra dev --extra distributed pytest -q
uv run --extra cpu --extra distributed gaz doctor --distributed
uv run --extra cpu --extra distributed gaz run --config configs/connect_four_cpu_debug.yaml --set stop.max_iterations=1
```

Quando si toccano CLI/execution, testare anche:

```bash
uv run --extra cpu --extra distributed gaz selfplay --config configs/connect_four_cpu_debug.yaml --games 2
uv run --extra cpu --extra distributed gaz benchmark --config configs/connect_four_cpu_debug.yaml --output-dir artifacts/benchmarks
uv run --extra cpu --extra distributed gaz run --config configs/connect_four_cpu_debug.yaml --execution local_multiprocess
```

Quando si tocca Ray, verificare almeno un head locale e fermarlo alla fine:

```bash
uv run --extra cpu --extra distributed gaz cluster head --config configs/connect_four_lan.yaml --host 127.0.0.1 --port 6383 --wait-workers --min-workers 0
uv run --extra cpu --extra distributed gaz run --config configs/connect_four_lan.yaml --execution lan_ray --set cluster.head_address=127.0.0.1:6383
uv run --extra cpu --extra distributed ray stop --force
```

Alla fine di test manuali o benchmark, ripulire artifact generati:

```text
artifacts/runs/*
artifacts/cache/*
artifacts/benchmarks/*
.pytest_cache
.ruff_cache
__pycache__
```

---

## 11. Cosa non introdurre nella base

Non introdurre:

- JAX, Flax, Optax, Orbax, MCTX, Chex, PGX;
- Docker obbligatorio;
- Kubernetes;
- MinIO obbligatorio;
- database server obbligatorio;
- Redis, Kafka, RabbitMQ, Celery;
- microservizi inutili;
- dashboard web obbligatoria;
- API FastAPI obbligatoria;
- path assoluti;
- IP hardcoded;
- configurazioni macchina-specifiche.

Eccezioni future sono ammesse solo se il ciclo locale resta semplice, i test
passano e la nuova tecnologia sta dietro un adapter.
