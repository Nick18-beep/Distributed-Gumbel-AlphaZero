# AGENTS.md â€” Istruzioni permanenti per AI Coding Agent

Questo file contiene le istruzioni operative permanenti per qualunque AI coding agent che lavori su questo repository.

L'obiettivo Ã¨ costruire un framework moderno, estendibile e facile da installare per addestrare agenti self-play su giochi a informazione perfetta, partendo da **Connect Four** e da **Gumbel AlphaZero**, ma lasciando il progetto pronto per:

- altri giochi;
- altri algoritmi oltre Gumbel AlphaZero;
- altri modelli neurali;
- esecuzione locale, multiprocesso e multi-PC in LAN;
- analisi offline, benchmark e confronti riproducibili.

La prioritÃ  assoluta Ã¨ questa esperienza:

```bash
python scripts/bootstrap.py --profile cpu
uv run gaz doctor
uv run gaz run --config configs/connect_four.yaml
```

La modalitÃ  locale deve essere semplice. La modalitÃ  multi-PC deve essere giÃ  prevista dall'architettura, ma non deve rendere difficile il quickstart.

---

## 0. Sintesi architetturale

La migliore architettura per questo progetto Ã¨:

```text
local-first modular monolith + backend di esecuzione pluggabile
```

Questo significa:

- un solo package Python principale;
- una CLI principale `gaz`;
- componenti interni ben separati;
- filesystem locale come storage base;
- nessun Docker obbligatorio;
- nessun database server obbligatorio;
- supporto multi-PC tramite backend opzionale, non tramite riscrittura;
- astrazioni esplicite per gioco, algoritmo, modello, search, replay, scheduler e storage.

Non costruire subito un sistema a microservizi. Prima deve funzionare perfettamente il ciclo locale:

```text
self-play -> replay -> training -> checkpoint -> evaluation -> promotion
```

Poi si estende a multiprocesso e LAN.

---

## 1. Decisioni non negoziabili

1. **Connect Four Ã¨ il primo gioco ufficiale.**
   Il codice non deve perÃ² conoscere Connect Four fuori da `GameAdapter`, config, test specifici e rendering.

2. **Gumbel AlphaZero Ã¨ il primo algoritmo ufficiale.**
   Il codice non deve perÃ² essere hardcoded su Gumbel AlphaZero. Gli algoritmi devono stare dietro `TrainingAlgorithm` e `SearchBackend`.

3. **`uv` Ã¨ il package manager principale.**
   Usare `pyproject.toml` e `uv.lock`. Non usare `requirements.txt` come fonte primaria.

4. **Installazione automatica sÃ¬, modifiche di sistema no.**
   Il progetto puÃ² installare `uv` se manca e puÃ² eseguire `uv sync`. Non deve installare driver GPU, CUDA di sistema, firewall rules, Docker o pacchetti admin senza conferma esplicita.

5. **Nessun Docker obbligatorio.**
   Docker puÃ² essere aggiunto in futuro come comoditÃ , ma il progetto deve essere usabile senza Docker.

6. **Nessun database server obbligatorio.**
   Default: filesystem locale con file atomici, JSON/JSONL, msgpack e zstd. DuckDB/SQLite sono ammessi solo come strumenti locali opzionali di indicizzazione o analisi.

7. **Multi-PC deve essere previsto dall'inizio.**
   Anche se il primo ciclo Ã¨ single-machine, usare interfacce `ExecutionBackend`, `WorkerClient`, `ReplayTransfer`, `CheckpointSync` e `Scheduler`.

8. **Non vendere prototipi come implementazione completa.**
   Una demo che gioca partite ma non ha replay robusto, checkpoint, evaluation, promotion, metriche, resume e test non soddisfa questo file.

9. **Gumbel AlphaZero deve essere reale, non una POC.**
   La prima implementazione deve usare una search Gumbel AlphaZero corretta con legal masking, Gumbel root noise, candidate set, sequential halving, Q transform, improved policy target e value target coerenti. Una policy con rumore, un MCTS generico rinominato o una versione Python lenta nel path caldo non soddisfano il progetto.

10. **Le performance sono un requisito di correttezza ingegneristica.**
    Il path caldo deve essere batchato, jittabile dove sensato, con shape statiche, trasferimenti host/device minimizzati e benchmark riproducibili. Le configurazioni debug sono ammesse, ma devono essere separate dai preset di training reale.

11. **Preferire librerie mature a implementazioni da zero.**
    Se esiste una libreria solida, mantenuta, compatibile con JAX/Python moderno e adatta al contratto del progetto, usarla o incapsularla dietro un adapter invece di reimplementare da zero. Implementare internamente solo quando la libreria non copre il requisito, introduce dipendenze troppo pesanti, rompe performance/JIT/batching, rende il quickstart fragile o impedisce estendibilitÃ .

12. **Ogni dipendenza deve essere dichiarata.**
    Se manca una libreria, aggiungerla a `pyproject.toml` nell'extra corretto e poi usare `uv sync`. Non fare `pip install` manuale nel codice, nei notebook o nei README.

13. **Windows e Linux/WSL sono piattaforme di prima classe.**
    Ogni funzionalitÃ  deve funzionare sia su Windows sia su Linux/WSL, salvo limitazioni esplicite di una dipendenza esterna. Script, path, subprocess, file atomici, multiprocessing, bootstrap, CLI e test devono essere progettati e verificati per entrambe le famiglie di sistemi.

14. **Ogni nuova estensione deve passare da registry e contratti.**
    Nuovi giochi, algoritmi, modelli e search backend devono essere registrati e testati senza rompere il ciclo Connect Four.

---

## 2. Obiettivo del progetto

Costruire un framework self-play stile AlphaZero/Gumbel AlphaZero per giochi a informazione perfetta.

La prima milestone funzionante deve essere:

```text
Connect Four + Gumbel AlphaZero + JAX/Flax/Optax + replay locale + checkpoint + evaluation + CLI
```

Il sistema deve supportare:

- training locale CPU-only;
- training locale con GPU NVIDIA se disponibile;
- self-play batchato e, dove possibile, vettorizzato;
- Connect Four come ambiente iniziale;
- altri giochi tramite adapter;
- Gumbel AlphaZero come algoritmo iniziale;
- altri algoritmi futuri tramite interfacce stabili;
- replay buffer append-only;
- checkpoint robusti e resume;
- evaluation automatica tra checkpoint;
- promotion del modello migliore;
- metriche JSONL;
- benchmark riproducibili;
- CLI semplice;
- modalitÃ  LAN multi-PC opzionale;
- bootstrap automatico con `uv`.

---

## 3. ModalitÃ  di esecuzione

Implementare in questo ordine.

Tutte le modalitÃ , inclusa `single_process`, devono passare da `ExecutionBackend`.
L'interfaccia base e `SingleProcessExecutionBackend` vanno introdotti presto, prima dell'orchestrator, cosÃ¬ l'estensione a multiprocesso e LAN non richiede riscritture del ciclo principale.

### 3.1 `single_process`

ModalitÃ  iniziale e sempre supportata.

```bash
uv run gaz run --config configs/connect_four.yaml
```

Caratteristiche:

- self-play, replay, training, checkpoint ed evaluation nello stesso processo;
- ideale per debug, laptop e primo sviluppo;
- deve rimanere funzionante anche quando vengono aggiunti multiprocesso e LAN;
- Ã¨ la modalitÃ  usata dalla maggior parte dei test smoke.

### 3.2 `local_multiprocess`

```bash
uv run gaz run --config configs/connect_four.yaml --execution local_multiprocess
```

Caratteristiche:

- piÃ¹ processi self-play locali;
- trainer centrale nello stesso PC;
- code bounded;
- backpressure;
- replay writer centralizzato oppure shard temporanei importati atomicamente;
- shutdown pulito con SIGINT/SIGTERM;
- evitare molte compilazioni JAX concorrenti.

### 3.3 `lan_ray`

ModalitÃ  multi-PC opzionale.

```bash
# PC principale
uv run gaz cluster head --config configs/connect_four_lan.yaml --host 0.0.0.0 --port 6379

# Ogni PC worker
uv run gaz cluster worker --head HEAD_IP:6379 --config configs/connect_four_lan.yaml --auto

# Avvio run distribuita dal PC head
uv run gaz run --config configs/connect_four_lan.yaml --execution lan_ray
```

Regole:

- Ray deve stare dietro `ExecutionBackend`;
- `ray` deve essere dipendenza opzionale extra `distributed`;
- il progetto deve funzionare senza Ray installato;
- il protocollo logico deve restare indipendente da Ray: registrazione worker, heartbeat, task lease, replay upload, checkpoint sync;
- il trainer principale vive sul nodo head;
- i worker generano replay e lo caricano verso il nodo head;
- nessun worker deve scrivere direttamente nello storage del trainer se non c'Ã¨ filesystem condiviso esplicito.

### 3.4 Futuro `jax_distributed`

JAX multi-host puÃ² essere considerato solo dopo:

1. ciclo locale stabile;
2. `local_multiprocess` stabile;
3. LAN Ray stabile;
4. benchmark che dimostrano che il collo di bottiglia Ã¨ davvero il training multi-host.

Non introdurre JAX multi-host nella prima implementazione.

---

## 4. Installazione automatica e developer experience

### 4.1 File obbligatori

```text
pyproject.toml
uv.lock
scripts/bootstrap.py
scripts/bootstrap.sh
scripts/bootstrap.ps1
README.md
configs/connect_four.yaml
configs/connect_four_lan.yaml
AGENTS.md
TASKS.md
```

### 4.2 Bootstrap locale

Implementare `scripts/bootstrap.py`.

Comandi desiderati:

```bash
# CPU
python scripts/bootstrap.py --profile cpu

# GPU NVIDIA, secondo extra configurato nel pyproject
python scripts/bootstrap.py --profile cuda

# sviluppo
python scripts/bootstrap.py --profile cpu --profile dev

# distribuito LAN
python scripts/bootstrap.py --profile cpu --profile distributed
```

Il bootstrap deve:

1. rilevare sistema operativo e Python;
2. verificare se `uv` Ã¨ installato;
3. installare `uv` se manca usando metodo ufficiale;
4. eseguire `uv sync` con gli extra richiesti;
5. creare cartelle locali `artifacts/`, `artifacts/runs/`, `artifacts/cache/` se mancano;
6. eseguire `uv run gaz doctor`;
7. stampare il comando successivo consigliato.

### 4.3 Profili e extras

Usare profili utente semplici, mappati a extra `uv`.

Esempio:

```text
profile cpu          -> uv sync --extra cpu
profile cuda         -> uv sync --extra cuda
profile cuda12       -> uv sync --extra cuda12
profile dev          -> uv sync --extra dev
profile distributed  -> uv sync --extra distributed
profile analysis     -> uv sync --extra analysis
```

Nel `pyproject.toml` tenere le dipendenze base in `[project.dependencies]` e mantenere separati gli extra:

```toml
dependencies = [
  "pydantic",
  "typer",
  "pyyaml",
  "numpy",
  "msgpack",
  "zstandard",
]
```

```toml
[project.optional-dependencies]
cpu = [
  "jax",
  "flax",
  "optax",
  "orbax-checkpoint",
  "chex",
  "mctx",
  "pgx",
]
cuda = [
  "jax[cuda13]",
  "flax",
  "optax",
  "orbax-checkpoint",
  "chex",
  "mctx",
  "pgx",
]
cuda12 = [
  "jax[cuda12]",
  "flax",
  "optax",
  "orbax-checkpoint",
  "chex",
  "mctx",
  "pgx",
]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
]
distributed = [
  "ray[default]",
]
analysis = [
  "duckdb",
  "pandas",
  "matplotlib",
]
```

Regole per JAX GPU:

- non usare URL manuali verso vecchie wheel JAX;
- non installare CUDA di sistema automaticamente;
- `gaz doctor` deve verificare `jax.devices()` e dire chiaramente se sta usando CPU o GPU;
- su Windows nativo JAX GPU NVIDIA non deve essere promesso come supportato; indicare WSL2/Linux quando serve CUDA;
- se l'extra `cuda` non Ã¨ corretto per la macchina, aggiungere un extra esplicito come `cuda12` o aggiornare il README secondo la documentazione ufficiale JAX;
- non mascherare fallback CPU come se fosse GPU.

### 4.4 CompatibilitÃ  Windows e Linux/WSL

Regole obbligatorie:

- usare `pathlib` e API standard multipiattaforma per path e filesystem;
- non hardcodare separatori `\` o `/` fuori da stringhe documentative;
- fornire wrapper Windows PowerShell e POSIX shell quando si aggiunge uno script utente;
- testare comandi principali su Windows e Linux/WSL prima di dichiarare completata una fase che tocca CLI, bootstrap, storage, multiprocessing o process management;
- se il runtime Linux/WSL non Ã¨ disponibile sulla macchina corrente, dichiararlo esplicitamente nel report finale e lasciare test/script portabili pronti per CI Linux;
- non usare feature Windows-only o Linux-only nel codice core senza adapter o fallback;
- gestire file atomici, lock, signal handling e multiprocessing con differenze di piattaforma esplicite e testate;
- documentare limiti specifici, per esempio JAX GPU NVIDIA su Windows nativo.

### 4.5 Installazione su piÃ¹ PC

L'installazione automatica remota multi-PC non fa parte della roadmap corrente. Per ora ogni worker deve poter essere installato manualmente con:

```bash
git clone <repo>
cd gumbel-alphazero
python scripts/bootstrap.py --profile cpu --profile distributed
uv run gaz cluster worker --head HEAD_IP:6379 --config configs/connect_four_lan.yaml --auto
```

Se in futuro verra reintrodotto un bootstrap remoto, dovra usare solo host esplicitamente elencati, chiedere conferma prima di eseguire comandi remoti, installare solo ambiente Python/dipendenze progetto e non modificare driver, CUDA, Docker, firewall o pacchetti admin.

### 4.6 `gaz doctor`

Implementare:

```bash
uv run gaz doctor
uv run gaz doctor --fix
uv run gaz doctor --distributed
uv run gaz doctor --cuda
```

Deve controllare:

- versione Python;
- presenza `uv`;
- ambiente virtuale;
- import pacchetti base;
- import JAX/Flax/Optax/Orbax quando il profilo ML Ã¨ installato;
- `jax.devices()`;
- backend JAX effettivo;
- scrittura in `artifacts/`;
- config Connect Four valida;
- presenza checkpoint/replay se si fa resume;
- Ray se richiesto;
- porte LAN usate da Ray/head;
- raggiungibilitÃ  nodo head da worker.

`--fix` puÃ² fare solo fix sicuri:

- `uv sync`;
- creazione cartelle locali;
- rigenerazione config di esempio;
- pulizia cache temporanea del progetto;
- ricostruzione replay index da shard validi.

---

## 5. Stack tecnologico

### 5.1 Base

Usare:

- Python 3.11+;
- `uv`;
- `pyproject.toml`;
- `uv.lock`;
- layout `src/`;
- CLI `gaz`.

Regole:

- committare `uv.lock`;
- non usare `requirements.txt` come fonte primaria;
- nuove dipendenze solo se motivate, ma non evitare una libreria matura solo per ridurre il numero di dipendenze;
- dipendenze pesanti in extra opzionali;
- ogni libreria esterna deve stare dietro un adapter o wrapper quando tocca dominio, execution, storage, search o ambiente;
- import pesanti lazy nei moduli CLI, cosÃ¬ `gaz doctor` puÃ² spiegare cosa manca invece di fallire in import globale.

### 5.2 Machine learning

Usare:

- JAX;
- Flax Linen per la prima implementazione;
- Optax;
- Orbax Checkpoint;
- Chex;
- NumPy;
- jaxtyping opzionale.

Regole:

- `train_step` jittabile;
- inference batchata;
- shape statiche nel path caldo;
- host/device transfer minimizzati;
- seed PRNG esplicito;
- no stato globale random non controllato;
- ogni loss deve essere testata contro NaN/Inf.

### 5.3 Search / planning

Usare:

- `mctx` come backend reference per Gumbel search/MCTS in JAX;
- preferire `mctx.gumbel_muzero_policy` per il backend Gumbel quando l'API installata lo espone;
- wrapper interno `SearchBackend`;
- implementazione algoritmica propria sopra il backend: self-play, target generation, replay, training, evaluation, promotion.

Non basta chiamare una funzione di libreria e dire che il progetto Ã¨ Gumbel AlphaZero. Il backend `mctx` deve essere integrato in un algoritmo completo: preparazione delle root, masking, recurrent function, action selection, target generation, replay schema, training loss, evaluation e metriche search.

Nel contesto di questo progetto, Gumbel AlphaZero significa usare la policy improvement Gumbel con modello perfetto del gioco. Se si usa `mctx.gumbel_muzero_policy`, la `recurrent_fn` deve rappresentare la dinamica reale del `GameAdapter`, non una dinamica appresa MuZero-style.

Regole performance:

- la search usata in self-play deve supportare batch e JIT dove possibile;
- non usare loop Python per simulazioni interne se `mctx` puÃ² eseguire il lavoro in JAX;
- separare chiaramente codice debug leggibile e path caldo ottimizzato;
- misurare compile time, steady-state throughput e uso device prima di dichiarare una modifica completata;
- mantenere fallback CPU corretto, ma non degradare il design GPU/batch per semplificare debug.

Prima di implementare componenti search custom, verificare se `mctx` copre giÃ  il requisito. Codice custom Ã¨ ammesso per glue code, target generation, metriche, validation, compatibility layer e parti non esposte dalla libreria.

### 5.4 Environment

PrioritÃ :

1. `pgx` per giochi board-game JAX-native disponibili e adatti al contratto;
2. Connect Four via adapter `pgx` se copre legal mask, observation, terminal/reward semantics, batch/vmap e performance richieste;
3. Connect Four custom JAX-native solo se offre piÃ¹ controllo, testabilitÃ  o performance rispetto a un adapter libreria;
4. `gymnasium` tramite adapter per compatibilitÃ ;
5. altri ambienti solo se rispettano il contratto `GameAdapter`.

Per Connect Four valutare prima una libreria esistente. Implementare direttamente solo se questo dÃ  piÃ¹ controllo su:

- legal mask;
- canonical observation;
- terminal value;
- batch/vmap;
- test esaustivi.

### 5.5 Distribuzione

Usare Ray come prima opzione opzionale per LAN mode.

Regole:

- Ray dietro `ExecutionBackend`;
- nessuna dipendenza Ray nel codice dominio;
- nessuna dipendenza Ray nel codice gioco;
- nessuna dipendenza Ray nel codice algoritmo;
- Ray solo in `execution/lan_ray.py` e CLI cluster.

---

## 6. Cosa NON usare nella base

Non introdurre nella base:

- Docker obbligatorio;
- Kubernetes;
- MinIO obbligatorio;
- PostgreSQL;
- MongoDB;
- Redis;
- Kafka;
- RabbitMQ;
- Celery;
- microservizi inutili;
- dashboard web obbligatoria;
- API FastAPI obbligatoria;
- path assoluti;
- IP hardcoded;
- configurazioni macchina-specifiche.

Eccezioni future ammesse solo se:

1. il ciclo locale funziona;
2. i test passano;
3. il README resta semplice;
4. la nuova tecnologia sta dietro un'interfaccia;
5. la modalitÃ  base non diventa piÃ¹ difficile.

---

## 7. Repository layout

Usare questa struttura:

```text
gumbel-alphazero/
  AGENTS.md
  TASKS.md
  README.md
  pyproject.toml
  uv.lock

  configs/
    connect_four.yaml
    connect_four_lan.yaml
    connect_four_cpu_debug.yaml
    connect_four_gpu.yaml
    tic_tac_toe.yaml

  scripts/
    bootstrap.py
    bootstrap.sh
    bootstrap.ps1
    benchmark.py
    profile_selfplay.py

  src/
    gumbel_az/
      __init__.py
      py.typed

      cli/
        main.py
        doctor.py
        cluster.py
        inspect.py
        play.py

      config/
        schema.py
        loader.py
        overrides.py
        defaults.py

      domain/
        types.py
        game.py
        algorithm.py
        replay.py
        checkpoint.py
        metrics.py

      envs/
        base.py
        registry.py
        pgx_adapter.py
        gymnasium_adapter.py
        custom/
          connect_four.py
          tic_tac_toe.py

      algorithms/
        registry.py
        base.py
        gumbel_alphazero/
          algorithm.py
          search.py
          policy_improvement.py
          sequential_halving.py
          q_transform.py
      search/
        backend.py
        mctx_backend.py
        custom_backend.py
        outputs.py

      model/
        registry.py
        factory.py
        common.py
        mlp.py
        resnet.py
        loss.py
        optimizer.py
        checkpoint.py
        inference.py

      replay/
        schema.py
        codec.py
        writer.py
        reader.py
        sampler.py
        index.py
        validation.py

      selfplay/
        manager.py
        worker.py
        trajectory.py
        batching.py

      training/
        trainer.py
        train_state.py
        train_loop.py
        batcher.py
        augmentations.py
        schedules.py

      eval/
        arena.py
        rating.py
        opponents.py
        promotion.py

      execution/
        base.py
        single_process.py
        local_multiprocess.py
        lan_ray.py
        messages.py
        heartbeat.py
        task_lease.py

      orchestration/
        run.py
        scheduler.py
        device_planner.py
        resume.py
        stop_conditions.py

      storage/
        filesystem.py
        registry.py
        atomic.py
        transfer.py

      logging/
        setup.py
        metrics.py
        events.py

      analysis/
        replay_report.py
        training_report.py
        search_report.py

  tests/
    test_config.py
    test_doctor.py
    test_connect_four_env.py
    test_game_adapter_contract.py
    test_algorithm_registry.py
    test_gumbel_search.py
    test_sequential_halving.py
    test_q_transform.py
    test_mctx_backend_smoke.py
    test_replay_roundtrip.py
    test_training_step.py
    test_checkpoint_roundtrip.py
    test_scheduler.py
    test_selfplay_smoke.py
    test_eval_arena.py
    test_execution_single_process.py
    test_cli_smoke.py
```

---

## 8. EstendibilitÃ : giochi

### 8.1 `GameAdapter`

Ogni gioco deve implementare un contratto comune.

```python
class GameAdapter(Protocol):
    name: str
    num_players: int
    num_actions: int
    observation_shape: tuple[int, ...]
    max_moves: int
    supports_jit: bool
    supports_vmap: bool

    def init(self, rng_key): ...
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

- nessuna logica del training deve conoscere Connect Four direttamente;
- `legal_action_mask` Ã¨ obbligatoria;
- `rewards(state)` Ã¨ la fonte autorevole per i risultati terminali per-player;
- `value_target` replay Ã¨ sempre dalla prospettiva di `to_play` nel sample;
- `terminal_value(state)` Ã¨ solo una convenience API e deve dichiarare esplicitamente la propria prospettiva;
- seed deterministico;
- batch/vmap quando possibile;
- test di contratto per ogni gioco.

### 8.2 Primo gioco: Connect Four

Requisiti:

- board 6x7;
- 7 azioni, una per colonna;
- legal action se la colonna non Ã¨ piena;
- gravitÃ  pedine;
- cambio player corretto;
- vittoria orizzontale, verticale, diagonale crescente, diagonale decrescente;
- draw se board piena;
- osservazione canonica dalla prospettiva del current player;
- simmetria orizzontale opzionale;
- render testuale per `gaz play`;
- test esaustivi su legal moves e win detection.

### 8.3 Aggiungere un nuovo gioco

Per aggiungere un gioco futuro:

1. creare `src/gumbel_az/envs/custom/<game>.py` oppure adapter a libreria;
2. registrarlo in `envs/registry.py`;
3. aggiungere config `configs/<game>.yaml`;
4. aggiungere test di contratto;
5. aggiungere test self-play smoke;
6. verificare legal mask e terminal value;
7. aggiornare README.

Il training loop non deve cambiare quando si aggiunge un gioco.

---

## 9. EstendibilitÃ : algoritmi

### 9.1 `TrainingAlgorithm`

Ogni algoritmo deve implementare:

```python
class TrainingAlgorithm(Protocol):
    name: str

    def make_initial_state(self, config, game, network): ...
    def select_action(self, state, game_state, network_apply, rng_key): ...
    def generate_targets(self, trajectory, final_outcome): ...
    def loss(self, params, batch, network_apply, config): ...
    def metrics(self, outputs, batch): ...
```

### 9.2 `SearchBackend`

Ogni search backend deve implementare:

```python
class SearchBackend(Protocol):
    name: str

    def search(
        self,
        root_observation,
        root_legal_mask,
        network_apply,
        recurrent_fn,
        rng_key,
        config,
    ) -> SearchOutput: ...
```

`SearchOutput` deve contenere almeno:

```text
policy_target
selected_action
root_value
visit_counts
q_values
prior_logits
search_metadata
```

Convenzione obbligatoria per i target:

- `policy_target` Ã¨ sempre definito nello spazio azioni legale della root del sample;
- `value_target` Ã¨ sempre dalla prospettiva di `to_play` nel sample;
- `root_value` in `SearchOutput` segue la stessa prospettiva di `value_target`;
- quando una trajectory viene convertita in replay, il risultato finale va ribaltato in base al player che doveva muovere in quella posizione.

### 9.3 Primo algoritmo: Gumbel AlphaZero

Implementare Gumbel AlphaZero come prima scelta, in forma completa e prestazionale.

Componenti obbligatori:

- policy logits dalla rete;
- value dalla rete;
- legal action mask;
- Gumbel noise alla root;
- candidate action set;
- Sequential Halving;
- Q transform documentata;
- improved policy target;
- selected action;
- value target dal risultato finale;
- temperature schedule;
- gestione terminal/draw;
- metriche search.

Regole:

- usare `mctx` come reference backend dove adatto;
- usare una integrazione reale con recurrent function, root embedding e policy/value network output coerenti con l'API search;
- la selected action deve provenire dalla policy migliorata/search, non dalla policy raw salvo config esplicita di evaluation;
- separare chiaramente prior policy, improved policy e azione giocata;
- non mischiare logica Gumbel dentro `GameAdapter`;
- non assumere sempre 7 azioni fuori da Connect Four;
- i target devono essere validi anche per nuovi giochi a numero azioni diverso.
- fallire i test se una probabilitÃ  positiva viene assegnata ad azioni illegali;
- includere benchmark minimi per search batchata su Connect Four.

### 9.4 Futuri algoritmi

Il registry deve contenere solo algoritmi realmente implementati e testati. La roadmap corrente richiede:

```text
gumbel_alphazero
```

Futuri algoritmi possibili:

- random_baseline;
- alphazero_puct;
- minimax baseline per giochi piccoli;
- MuZero-like con modello dinamico appreso;
- policy-gradient baseline;
- pure supervised imitation su dataset esterno.

Regola fondamentale: aggiungere un algoritmo non deve richiedere modifiche al gioco, al replay codec o alla CLI principale, salvo nuova config.

---

## 10. EstendibilitÃ : modelli neurali

### 10.1 `NetworkFactory`

Ogni modello deve essere registrato.

```python
class NetworkFactory(Protocol):
    name: str

    def init(self, rng_key, observation_shape, num_actions, config): ...
    def apply(self, params, observations, train: bool = False): ...
```

Output standard:

```text
policy_logits: [batch, num_actions]
value: [batch]
```

### 10.2 Modelli iniziali

Implementare:

```text
mlp_small
resnet_board
```

Connect Four puÃ² partire con `resnet_board`, ma `mlp_small` serve per test rapidi CPU.

Regole:

- nessun modello deve assumere board 6x7 se non dichiarato nella config;
- il numero azioni viene dal gioco;
- policy logits non devono applicare softmax prima della loss;
- value in range coerente, tipicamente `[-1, 1]`;
- test shape obbligatori.

---

## 11. Replay buffer

### 11.1 Default locale

Replay append-only su filesystem.

Usare librerie esistenti per serializzazione, compressione e checkpointing quando possibile (`msgpack`, `zstandard`, Orbax). Non scrivere formati binari, compressori, database o checkpoint manager custom se una libreria consolidata copre il requisito.

Struttura:

```text
artifacts/runs/<run_id>/
  replay/
    shards/
      shard_000000001.msgpack.zst
      shard_000000002.msgpack.zst
    index.json
    quarantine/
  checkpoints/
    ckpt_000001/
    ckpt_000002/
    index.json
  eval/
    matches.jsonl
  logs/
    events.jsonl
    metrics.jsonl
  config.resolved.yaml
  run_state.json
```

### 11.2 Schema replay

Ogni sample deve contenere:

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
- scrittura atomica: temp file -> fsync se opportuno -> rename;
- shard incompleti mai indicizzati;
- shard corrotti in `quarantine/`;
- replay schema versionato;
- `value_target` sempre dalla prospettiva di `to_play`;
- sampler uniforme iniziale;
- replay window configurabile.

### 11.3 Database opzionale

DuckDB/SQLite sono ammessi solo per:

- indicizzare replay metadata;
- fare report offline;
- interrogare training/evaluation metrics;
- accelerare analisi.

Non usarli come requisito per avviare training.

---

## 12. Checkpoint e model registry

Usare Orbax per checkpoint JAX.

Ogni checkpoint deve avere metadata:

```json
{
  "version": 12,
  "created_at": "2026-01-01T12:00:00Z",
  "training_step": 10000,
  "games_seen": 5000,
  "samples_seen": 210000,
  "game": "connect_four",
  "algorithm": "gumbel_alphazero",
  "model": "resnet_board",
  "config_hash": "...",
  "network_hash": "...",
  "git_commit": "...",
  "eval": {
    "win_rate_vs_previous_best": 0.57,
    "games": 200,
    "promoted": true
  }
}
```

Registry locale:

```text
checkpoints/index.json
checkpoints/latest.json
checkpoints/best.json
```

Regole:

- salvataggio atomico;
- load di `latest` e `best`;
- resume da checkpoint;
- checkpoint incompleti ignorati;
- worker LAN scaricano checkpoint in modo atomico;
- non sovrascrivere checkpoint validi.

---

## 13. Training loop

Il training loop deve essere una implementazione reale e performante, non un loop dimostrativo. Deve supportare:

- AdamW;
- learning rate schedule cosine;
- warmup opzionale;
- gradient clipping;
- policy loss cross entropy su improved policy target;
- value loss;
- weight decay;
- mixed precision opzionale;
- checkpoint periodici;
- evaluation periodica;
- promotion policy;
- resume.

Metriche minime:

```text
train_step
policy_loss
value_loss
total_loss
learning_rate
grad_norm
samples_per_sec
replay_samples_available
replay_sample_age_mean
checkpoint_version
```

Regole:

- `train_step` JIT;
- batch shape statiche nel path jittato;
- nessuna conversione host/device dentro `train_step`;
- evitare shape dinamiche non necessarie;
- batch generati da `ReplaySampler`;
- augmentations provenienti da `GameAdapter.symmetries`;
- test per loss finita e gradienti non NaN.
- benchmark steady-state dopo warmup obbligatorio.

---

## 14. Self-play

Self-play deve produrre partite complete e replay samples validi usando il path search reale. Una modalitÃ  random/debug puÃ² esistere, ma non soddisfa `gaz run` con Gumbel AlphaZero.

Pipeline:

```text
init game
while not terminal:
  canonical observation
  legal action mask
  search backend
  action selection
  step environment
  store search stats
final outcome
convert trajectory to replay samples
write shard
```

Metriche minime:

```text
games_per_sec
positions_per_sec
searches_per_sec
mean_game_length
illegal_action_rate
policy_entropy_mean
root_value_mean
model_version_used
replay_write_latency_ms
```

Regole:

- illegal action rate deve essere zero;
- se viene scelta un'azione illegale, fallire forte in debug;
- in release, loggare errore e scartare partita;
- seed deterministico;
- supporto batch dove possibile;
- nessun loop Python inutile nel path caldo se JAX puÃ² vettorizzare.
- i worker devono riusare funzioni compilate e non ricompilare per ogni partita;
- la batch size di self-play deve essere configurabile e misurata.

---

## 15. Evaluation e promotion

Ogni nuovo checkpoint importante deve essere valutato.

Arena minima:

```text
candidate checkpoint vs current best
randomized starting player
N games configurable
win/loss/draw
confidence interval opzionale
promotion threshold
```

Regole:

- alternare player iniziale;
- usare seed controllato;
- non promuovere senza numero minimo di partite;
- registrare risultati in `eval/matches.jsonl`;
- aggiornare `best.json` solo atomicamente;
- supportare avversari: random, previous checkpoint, best checkpoint.

---

## 16. Scheduler e balancing locale

Lo scheduler Ã¨ il control plane del progetto. Non deve essere un dettaglio secondario.

### 16.1 Obiettivo

Bilanciare:

- self-play generation;
- replay writing;
- training;
- evaluation;
- checkpointing;
- report/benchmark.

### 16.2 Segnali osservati

Lo scheduler deve osservare almeno:

```text
replay_samples_available
replay_low_watermark
replay_high_watermark
samples_generated_per_sec
samples_consumed_per_sec
train_steps_per_sec
selfplay_queue_depth
replay_write_queue_depth
checkpoint_pending
evaluation_pending
model_staleness
cpu_utilization
gpu_utilization_if_available
memory_available
```

### 16.3 Policy iniziale

Regole iniziali semplici:

```text
if replay_samples < low_watermark:
    prioritize self-play
    pause or slow training warmup

if replay_samples > high_watermark:
    prioritize training
    reduce self-play in-flight tasks

if evaluation_pending:
    reserve bounded budget for evaluation

if checkpoint_pending:
    avoid starting too many long self-play batches with stale model

if disk queue high:
    increase shard size or pause self-play briefly
```

### 16.4 Backpressure

Implementare backpressure tramite:

- bounded queues;
- max self-play batches in flight;
- replay write queue limit;
- evaluation budget;
- checkpoint version staleness limit;
- graceful pause/resume.

Nessun componente deve produrre dati illimitatamente.

---

## 17. Multi-PC, worker balancing e task leases

### 17.1 Architettura LAN

```text
HEAD PC
  - orchestrator
  - scheduler
  - trainer
  - replay store centrale
  - checkpoint registry
  - metrics collector
  - Ray head opzionale

WORKER PC
  - worker agent
  - hardware detector
  - self-play worker
  - local temporary replay writer
  - replay uploader
  - checkpoint puller
  - heartbeat client
```

### 17.2 Worker capabilities

Ogni worker registra:

```text
worker_id
hostname
ip
os
python_version
cpu_count
ram_total
ram_available
gpu_present
gpu_name
gpu_memory
jax_devices
ray_resources
network_latency_to_head
local_artifacts_path
supported_profiles
current_checkpoint_version
```

### 17.3 Pull-based scheduling

Preferire scheduling pull-based:

1. worker manda heartbeat e capabilities;
2. worker chiede lavoro;
3. head assegna task con lease;
4. worker completa task e invia risultato;
5. head valida risultato e aggiorna stato.

Questo Ã¨ piÃ¹ robusto di spingere task a worker non pronti.

### 17.4 Task lease

Ogni task distribuito deve avere:

```text
task_id
lease_id
task_type
assigned_worker
created_at
lease_expires_at
required_checkpoint_version
config_hash
status
retry_count
```

Regole:

- se lease scade, task riassegnabile;
- replay shard incompleti non importati;
- task idempotenti quando possibile;
- nessun task deve richiedere stato locale non dichiarato.

### 17.5 Ruoli worker

Ruoli supportati:

```text
selfplay_cpu_worker
selfplay_gpu_worker
evaluator_worker
replay_uploader
```

Futuro:

```text
inference_worker
analysis_worker
```

### 17.6 Policy di bilanciamento LAN

CPU-only worker:

- self-play leggero;
- batch moderati;
- checkpoint polling non troppo frequente;
- upload shard compressi;
- evitare inference CPU costosa se GPU remota Ã¨ prevista in futuro.

GPU worker:

- self-play batchato JAX-native;
- evaluation;
- inference locale futura;
- non competere con trainer se Ã¨ sul nodo head.

Head con GPU:

1. prioritÃ  training;
2. evaluation periodica;
3. self-play solo se resta capacitÃ .

### 17.7 Comandi dal nodo head

Il nodo head deve poter inviare comandi:

```text
pause_selfplay
resume_selfplay
reduce_batch_size
increase_shard_size
sync_checkpoint
run_eval_games
shutdown_gracefully
```

### 17.8 Fault tolerance

Regole:

- heartbeat periodico;
- worker `lost` se non manda heartbeat entro timeout;
- task non confermati riassegnabili;
- checkpoint download atomico;
- worker con checkpoint troppo vecchio deve aggiornarsi prima di generare nuovi dati, salvo config contraria;
- errori worker loggati in head metrics.

---

## 18. Device planner

All'avvio rilevare:

- CPU core;
- RAM;
- OS;
- Python;
- JAX devices;
- GPU presenti;
- memoria GPU se disponibile;
- capacitÃ  di scrittura disco;
- spazio libero;
- latenza verso head in LAN mode.

Policy iniziale:

```text
single_process CPU debug/auto:
  small model
  small batch
  low simulations

single_process real preset:
  use the config values as requested unless the user selects a debug profile

single_process GPU:
  bigger batch
  more simulations
  JAX warmup before timing

local_multiprocess CPU:
  reserve 1 core for orchestrator
  N self-play workers bounded by RAM and replay queue

LAN worker:
  advertise capabilities
  let head scheduler assign role and task size
```

Non hardcodare valori macchina-specifici. I default devono essere ragionevoli e sovrascrivibili da config.

---

## 19. Configurazione YAML

Esempio `configs/connect_four.yaml`:

```yaml
run:
  name: connect-four-gumbel-az
  seed: 42
  output_dir: artifacts/runs

execution:
  backend: single_process

cluster:
  enabled: false
  head_address: null

install:
  profile: cpu

storage:
  backend: filesystem
  root: artifacts/runs

logging:
  level: INFO
  format: jsonl

game:
  name: connect_four
  implementation: custom_jax

algorithm:
  name: gumbel_alphazero

search:
  backend: mctx
  simulations_per_move: 128
  max_num_considered_actions: 7
  gumbel_scale: 1.0
  q_transform: completed_by_mix_value

model:
  name: resnet_board
  channels: 64
  blocks: 4

selfplay:
  games_per_iteration: 128
  batch_size: 128
  temperature_moves: 12
  shard_max_samples: 4096

replay:
  window_samples: 200000
  min_samples_to_train: 4096
  low_watermark: 8192
  high_watermark: 100000

training:
  batch_size: 256
  steps_per_iteration: 100
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  checkpoint_every_steps: 1000

eval:
  enabled: true
  games: 100
  promotion_win_rate: 0.55

stop:
  max_iterations: null
  max_train_steps: null
  max_games: null
  max_wall_time_sec: null
```

I test config devono intercettare errori YAML o schema prima di avviare una run.

`configs/connect_four.yaml` deve essere il preset locale reale per Gumbel AlphaZero, non una demo. `configs/connect_four_cpu_debug.yaml` serve per smoke test, CI e sviluppo rapido. Preset piÃ¹ aggressivi per GPU devono stare in `configs/connect_four_gpu.yaml`.

Regole:

- schema Pydantic obbligatorio;
- config risolta salvata nella run directory;
- stop condition esplicite obbligatorie per test, benchmark e smoke run;
- override CLI solo per parametri comuni;
- no env vars come configurazione primaria;
- env vars ammesse solo per override runtime e CI.

---

## 20. Logging e metriche

Usare JSONL.

Esempio evento:

```json
{
  "timestamp": "2026-01-01T12:00:00Z",
  "level": "INFO",
  "component": "selfplay",
  "event": "game_completed",
  "game_id": "abc",
  "moves": 31,
  "model_version": 12,
  "duration_sec": 0.42
}
```

Metriche minime:

```text
selfplay/games_per_sec
selfplay/positions_per_sec
selfplay/illegal_action_rate
search/policy_entropy
search/root_value_mean
replay/samples_available
replay/write_throughput
train/policy_loss
train/value_loss
train/total_loss
train/samples_per_sec
eval/win_rate
eval/draw_rate
scheduler/decision
system/cpu_percent
system/gpu_percent_if_available
```

Regole:

- eventi e metriche separati se utile;
- log leggibili anche senza dashboard;
- nessuna dashboard obbligatoria;
- report offline da JSONL.

---

## 21. Analisi obbligatorie

Aggiungere comandi:

```bash
uv run gaz inspect run artifacts/runs/<run_id>
uv run gaz inspect replay artifacts/runs/<run_id>/replay
uv run gaz inspect checkpoint artifacts/runs/<run_id>/checkpoints/best.json
```

Analisi da produrre:

### 21.1 Correctness analysis

- illegal action rate;
- terminal value distribution;
- draw rate;
- value perspective check;
- replay schema validation;
- duplicate/corrupt shard count.

### 21.2 Learning analysis

- policy loss trend;
- value loss trend;
- total loss trend;
- entropy trend;
- win rate vs previous best;
- checkpoint promotion history.

### 21.3 Search analysis

- visit count distribution;
- selected action entropy;
- root value distribution;
- Q-value distribution;
- average search simulations/sec;
- fraction of masked actions.

### 21.4 Systems analysis

- self-play throughput;
- training throughput;
- replay read/write throughput;
- checkpoint save/load time;
- queue depth;
- worker utilization in LAN mode.

---

## 22. Testing

Test minimi:

- config valida/non valida;
- CLI smoke;
- `gaz doctor`;
- GameAdapter contract;
- Connect Four legal moves;
- Connect Four win detection;
- Connect Four draw;
- canonical observation;
- symmetries;
- NetworkFactory shape;
- train step;
- Gumbel target generation;
- Sequential Halving;
- Q transform;
- action masking;
- replay roundtrip;
- corrupted replay shard;
- checkpoint save/load;
- evaluation arena;
- scheduler watermarks;
- single_process smoke;
- local_multiprocess smoke quando implementato;
- LAN mocks per heartbeat e task lease.

Comandi:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

CompatibilitÃ  piattaforme:

- le suite smoke devono passare su Windows e Linux/WSL;
- bootstrap CPU deve essere verificato con `scripts/bootstrap.ps1` su Windows e `scripts/bootstrap.sh` su Linux/WSL;
- ogni test che usa path, subprocess o filesystem deve evitare assunzioni su separatori, permessi, symlink e shell;
- se una verifica Linux/WSL non puÃ² essere eseguita localmente, deve essere coperta da CI Linux o segnalata come non verificata nel report.

---

## 23. Benchmark

Implementare:

```bash
uv run gaz benchmark --config configs/connect_four.yaml
```

Misurare:

- JAX compile time;
- warmup time;
- self-play games/sec;
- positions/sec;
- search simulations/sec;
- replay write throughput;
- replay read throughput;
- train samples/sec;
- checkpoint save/load time;
- evaluation games/sec.

Scrivere risultati in:

```text
artifacts/benchmarks/*.jsonl
```

Regole:

- distinguere compile time da steady-state throughput;
- indicare CPU/GPU backend;
- salvare config e git commit;
- benchmark brevi per CI, benchmark lunghi opzionali.

---

## 24. README richiesto

Il README deve contenere:

1. cosa fa il progetto;
2. quickstart CPU;
3. quickstart GPU;
4. quickstart LAN;
5. comandi `gaz doctor`, `gaz run`, `gaz play`, `gaz benchmark`;
6. struttura artifacts;
7. come aggiungere un gioco;
8. come aggiungere un algoritmo;
9. come aggiungere un modello;
10. troubleshooting JAX/CUDA;
11. limiti noti.

Il quickstart CPU deve stare entro pochi comandi:

```bash
git clone <repo>
cd gumbel-alphazero
python scripts/bootstrap.py --profile cpu
uv run gaz run --config configs/connect_four.yaml
```

---

## 25. Definition of Done

Il progetto Ã¨ considerato funzionante quando:

1. `python scripts/bootstrap.py --profile cpu` completa;
2. `uv run gaz doctor` passa;
3. `uv run pytest` passa;
4. `uv run gaz run --config configs/connect_four.yaml` parte;
5. self-play genera partite Connect Four;
6. illegal action rate Ã¨ zero;
7. replay shard vengono scritti;
8. trainer consuma replay;
9. checkpoint viene salvato;
10. evaluation viene eseguita;
11. best checkpoint viene aggiornato;
12. `uv run gaz play --config configs/connect_four.yaml` permette umano vs agente;
13. `uv run gaz resume <run_dir>` riprende una run;
14. `uv run gaz benchmark --config configs/connect_four.yaml` produce JSONL;
15. i contratti e i registry permettono estensioni future senza modificare il ciclo Connect Four;
16. i quickstart e gli smoke test passano sia su Windows sia su Linux/WSL o CI Linux.

---

## 26. PrioritÃ  implementativa

Ordine obbligatorio:

1. repository skeleton;
2. packaging `uv`;
3. bootstrap e doctor;
4. config schema;
5. logging e run directory;
6. `ExecutionBackend` base e `SingleProcessExecutionBackend`;
7. GameAdapter;
8. Connect Four;
9. NetworkFactory;
10. train step;
11. checkpoint;
12. SearchBackend;
13. Gumbel AlphaZero;
14. replay;
15. self-play;
16. trainer;
17. evaluation/promotion;
18. orchestrator single-process;
19. scheduler locale;
20. resume;
21. `gaz play`;
22. inspect/report;
23. benchmark;
24. local_multiprocess;
25. LAN Ray;
26. LAN replay e checkpoint;
27. documentazione;
28. criteri di successo;
29. pulizia finale.

---

## 27. Regola finale per AI coding agent

Quando lavori sul repository:

- leggi sempre `AGENTS.md` prima di iniziare;
- aggiorna `TASKS.md` quando completi task reali;
- non introdurre dipendenze non dichiarate;
- non introdurre infrastruttura non necessaria;
- mantieni il quickstart semplice;
- mantieni Connect Four sempre funzionante;
- mantieni interfacce stabili per giochi/algoritmi/modelli;
- aggiungi test quando aggiungi logica;
- misura throughput prima di ottimizzare;
- mantieni compatibilitÃ  Windows e Linux/WSL per ogni modifica;
- quando completi una fase, verifica la piattaforma corrente e, se Linux/WSL non Ã¨ disponibile, segnala chiaramente cosa non Ã¨ stato eseguito;
- se devi scegliere tra architettura elegante ma fragile e sistema semplice che funziona, scegli il sistema semplice che funziona;
- se devi scegliere tra prototipo veloce e implementazione robusta, consegna una slice stretta ma reale: corretta, testata, riproducibile e misurata.
