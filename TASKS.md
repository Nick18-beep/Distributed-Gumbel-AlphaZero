# TASKS.md â€” Roadmap spuntabile

Questa checklist serve per implementare il progetto uno step alla volta. Spunta manualmente con `[X]` quando completi un task.

---

## Fase 0 â€” Setup repository

- [X] Creare repository `gumbel-alphazero`.
- [X] Aggiungere `AGENTS.md` alla root.
- [X] Aggiungere `TASKS.md` alla root.
- [X] Creare struttura cartelle `src/gumbel_az/`, `configs/`, `scripts/`, `tests/`.
- [X] Creare `README.md` iniziale.
- [X] Creare `.gitignore` per Python, uv, artifacts, cache, checkpoint, replay e log.
- [X] Creare cartella `artifacts/` ignorata da git salvo `.gitkeep` se necessario.
- [X] Verificare che il repository sia importabile con layout `src/`.

---

## Fase 1 â€” Packaging con uv

- [X] Creare `pyproject.toml`.
- [X] Definire package `gumbel_az` con layout `src/`.
- [X] Definire comando CLI `gaz`.
- [X] Aggiungere dipendenze base leggere: Pydantic, Typer, PyYAML, NumPy, msgpack, zstandard.
- [X] Aggiungere extra `cpu` con JAX, Flax, Optax, Orbax, Chex, mctx, pgx.
- [X] Aggiungere extra `cuda` con JAX GPU CUDA 13, mctx e pgx secondo documentazione ufficiale aggiornata.
- [X] Aggiungere extra `cuda12` con mctx e pgx se serve supporto esplicito CUDA 12.
- [X] Aggiungere extra `dev` con pytest, pytest-cov, ruff, mypy.
- [X] Aggiungere extra `distributed` con Ray.
- [X] Aggiungere extra `analysis` con DuckDB, Pandas, Matplotlib.
- [X] Documentare che JAX GPU NVIDIA non Ã¨ supportato su Windows nativo e richiede Linux/WSL2 dove applicabile.
- [X] Valutare librerie mature prima di introdurre implementazioni custom.
- [X] Tenere librerie esterne dietro adapter/wrapper dove toccano dominio, search, storage o execution.
- [X] Generare `uv.lock`.
- [X] Verificare `uv sync --extra cpu`.
- [X] Verificare `uv run python -c "import gumbel_az"`.
- [X] Verificare che import base non richieda JAX se extra ML non installati.

---

## Fase 2 â€” Bootstrap automatico

- [X] Creare `scripts/bootstrap.py`.
- [X] Far rilevare se `uv` Ã¨ installato.
- [X] Installare `uv` automaticamente se manca.
- [X] Supportare `--profile cpu`.
- [X] Supportare `--profile cuda`.
- [X] Supportare `--profile cuda12` se configurato.
- [X] Supportare `--profile dev`.
- [X] Supportare `--profile distributed`.
- [X] Supportare `--profile analysis`.
- [X] Mappare profili a `uv sync --extra ...`.
- [X] Creare cartelle locali `artifacts/`, `artifacts/runs/`, `artifacts/cache/`.
- [X] Eseguire `uv run gaz doctor` a fine bootstrap.
- [X] Stampare comando successivo consigliato.
- [X] Creare wrapper `scripts/bootstrap.sh`.
- [X] Creare wrapper `scripts/bootstrap.ps1`.
- [X] Documentare bootstrap nel README.
- [X] Verificare che bootstrap non installi driver, Docker o pacchetti admin.

---

## Fase 3 â€” CLI minima

- [X] Creare `src/gumbel_az/cli/main.py`.
- [X] Implementare comando `gaz --help`.
- [X] Implementare comando `gaz init`.
- [X] Implementare comando `gaz doctor`.
- [X] Registrare comando `gaz run --config ...` con validazione config e errore chiaro finchÃ© il backend non Ã¨ collegato.
- [X] Registrare comando `gaz resume RUN_DIR` con validazione path e errore chiaro finchÃ© resume non Ã¨ implementato.
- [X] Registrare comando `gaz selfplay --config ...` come comando dev, separato dal path principale.
- [X] Registrare comando `gaz train --config ...` come comando dev, separato dal path principale.
- [X] Registrare comando `gaz eval --config ...` come comando dev, separato dal path principale.
- [X] Registrare comando `gaz play --config ...` con errore chiaro finchÃ© il player non Ã¨ implementato.
- [X] Registrare comando `gaz benchmark --config ...` con errore chiaro finchÃ© benchmark non Ã¨ implementato.
- [X] Registrare comando `gaz inspect ...` con errore chiaro finchÃ© inspect non Ã¨ implementato.
- [X] Aggiungere test smoke CLI.

---

## Fase 4 â€” Doctor

- [X] Controllare versione Python.
- [X] Controllare presenza uv.
- [X] Controllare ambiente virtuale.
- [X] Controllare sistema operativo e distinguere Windows, Linux e WSL.
- [X] Controllare import pacchetti base.
- [X] Controllare import JAX se profilo ML installato.
- [X] Stampare `jax.devices()` se JAX Ã¨ installato.
- [X] Distinguere chiaramente backend CPU/GPU.
- [X] Controllare import Flax.
- [X] Controllare import Optax.
- [X] Controllare import Orbax.
- [X] Controllare scrittura in `artifacts/`.
- [X] Controllare presenza config Connect Four.
- [X] Implementare `gaz doctor --fix` per fix sicuri.
- [X] Implementare `gaz doctor --distributed` per controllare Ray.
- [X] Implementare `gaz doctor --cuda` per diagnosi GPU.
- [X] Aggiungere output chiaro sui limiti CUDA/JAX per Windows nativo.
- [X] Aggiungere test per doctor.
- [X] Eseguire test doctor su Windows e Linux/WSL o CI Linux.

---

## Fase 5 â€” Config

- [X] Creare `src/gumbel_az/config/schema.py`.
- [X] Definire schema Pydantic per `run`.
- [X] Definire schema Pydantic per `execution`.
- [X] Definire schema Pydantic per `cluster`.
- [X] Definire schema Pydantic per `install`.
- [X] Definire schema Pydantic per `storage`.
- [X] Definire schema Pydantic per `game`.
- [X] Definire schema Pydantic per `algorithm`.
- [X] Definire schema Pydantic per `search`.
- [X] Definire schema Pydantic per `model`.
- [X] Definire schema Pydantic per `selfplay`.
- [X] Definire schema Pydantic per `replay`.
- [X] Definire schema Pydantic per `training`.
- [X] Definire schema Pydantic per `eval`.
- [X] Definire schema Pydantic per `logging`.
- [X] Definire schema Pydantic per `stop`.
- [X] Creare loader YAML.
- [X] Implementare override CLI semplici.
- [X] Salvare config risolta in run directory.
- [X] Creare `configs/connect_four.yaml` come preset reale locale per Gumbel AlphaZero.
- [X] Creare `configs/connect_four_lan.yaml`.
- [X] Creare `configs/connect_four_cpu_debug.yaml`.
- [X] Creare `configs/connect_four_gpu.yaml`.
- [X] Verificare che i preset abbiano stop condition adatte a test e smoke run.
- [X] Verificare che i preset debug/CI siano separati dal preset reale.
- [X] Aggiungere test config valida.
- [X] Aggiungere test config non valida.
- [X] Verificare path relativi config su Windows e Linux/WSL.

---

## Fase 6 â€” Run directory e logging

- [X] Creare modulo `storage/atomic.py`.
- [X] Implementare scrittura atomica file JSON.
- [X] Implementare scrittura atomica file YAML se necessaria.
- [X] Creare run directory timestampata.
- [X] Creare pointer cross-platform `artifacts/runs/latest.json`.
- [X] Scrivere `config.resolved.yaml`.
- [X] Scrivere `logs/events.jsonl`.
- [X] Scrivere `logs/metrics.jsonl`.
- [X] Scrivere `run_state.json`.
- [X] Implementare logger JSONL.
- [X] Implementare metric writer JSONL.
- [X] Aggiungere test scrittura log.
- [X] Testare path, rename atomico e scrittura log su Windows e Linux/WSL.

---

## Fase 6.5 â€” ExecutionBackend base

- [X] Creare `src/gumbel_az/execution/base.py`.
- [X] Definire protocollo `ExecutionBackend`.
- [X] Definire messaggi/task minimi per run locale.
- [X] Implementare `SingleProcessExecutionBackend`.
- [X] Collegare `gaz run` al backend single-process.
- [X] Aggiungere test unitari execution backend base.
- [X] Verificare che il dominio non importi backend Ray o multiprocessing.
- [X] Verificare comportamento backend base su Windows e Linux/WSL.

---

## Fase 7 â€” GameAdapter

- [X] Valutare librerie ambiente esistenti (`pgx`, `gymnasium`) prima di implementare adapter custom.
- [X] Definire protocollo `GameAdapter`.
- [X] Definire tipi base per stato, observation, action mask, rewards.
- [X] Creare registry giochi.
- [X] Implementare test di contratto generico per giochi.
- [X] Creare adapter PGX placeholder.
- [X] Creare adapter Gymnasium placeholder.
- [X] Documentare come aggiungere un gioco.
- [X] Verificare che training loop non importi Connect Four direttamente.

---

## Fase 8 â€” Connect Four

- [X] Valutare prima Connect Four via `pgx` adapter.
- [X] Motivare implementazione custom solo se migliore per semantica, testabilita, JAX batching o performance.
- [X] Implementare stato Connect Four.
- [X] Implementare `init`.
- [X] Implementare `legal_action_mask`.
- [X] Implementare `step`.
- [X] Implementare gravita pedine.
- [X] Implementare cambio player.
- [X] Implementare win orizzontale.
- [X] Implementare win verticale.
- [X] Implementare win diagonale crescente.
- [X] Implementare win diagonale decrescente.
- [X] Implementare draw.
- [X] Implementare terminal value.
- [X] Implementare canonical observation.
- [X] Implementare simmetria orizzontale.
- [X] Implementare `render_text`.
- [X] Rendere funzioni compatibili con JAX dove possibile.
- [X] Aggiungere test legal moves board vuota.
- [X] Aggiungere test legal moves colonne piene.
- [X] Aggiungere test vittoria orizzontale.
- [X] Aggiungere test vittoria verticale.
- [X] Aggiungere test vittoria diagonale crescente.
- [X] Aggiungere test vittoria diagonale decrescente.
- [X] Aggiungere test draw.
- [X] Aggiungere test current player.
- [X] Aggiungere test canonical observation.
- [X] Aggiungere test simmetria orizzontale.
- [X] Aggiungere test batch/vmap se implementato.

---

## Fase 9 â€” NetworkFactory

- [X] Definire protocollo `NetworkFactory`.
- [X] Creare registry modelli.
- [X] Usare Flax Linen per la prima implementazione.
- [X] Implementare `mlp_small`.
- [X] Implementare `resnet_board`.
- [X] Policy head output `[batch, num_actions]`.
- [X] Value head output `[batch]`.
- [X] Applicare tanh o bound coerente per value.
- [X] Test init deterministico con seed.
- [X] Test forward pass MLP.
- [X] Test forward pass ResNet.
- [X] Test shape policy/value.
- [X] Verificare che modello usi `num_actions` dal gioco.

---

## Fase 10 â€” Optimizer e train state

- [X] Definire train state.
- [X] Implementare AdamW.
- [X] Implementare cosine schedule.
- [X] Implementare warmup opzionale.
- [X] Implementare gradient clipping.
- [X] Implementare policy loss.
- [X] Implementare value loss.
- [X] Implementare total loss.
- [X] Implementare `train_step` JAX-jitted.
- [X] Loggare grad norm.
- [X] Aggiungere test singolo train step.
- [X] Aggiungere test loss finita.
- [X] Aggiungere test gradienti non NaN.

---

## Fase 11 â€” Checkpoint

- [X] Usare Orbax come checkpoint manager, senza checkpoint format custom salvo metadata/index locali.
- [X] Integrare Orbax.
- [X] Salvare parametri modello.
- [X] Salvare optimizer state.
- [X] Salvare training step.
- [X] Salvare metadata checkpoint.
- [X] Creare `checkpoints/index.json`.
- [X] Implementare `latest`.
- [X] Implementare `best`.
- [X] Implementare scrittura atomica registry.
- [X] Implementare load checkpoint.
- [X] Ignorare checkpoint incompleti.
- [X] Aggiungere test save/load.
- [X] Aggiungere test registry atomico.
- [X] Aggiungere test resume da checkpoint.

---

## Fase 12 â€” SearchBackend

- [X] Verificare quali parti sono giÃ  coperte da `mctx` prima di implementare search custom.
- [X] Definire `SearchBackend` protocol.
- [X] Definire `SearchOutput`.
- [X] Implementare action masking utility.
- [X] Implementare Q transform utility.
- [X] Implementare Sequential Halving utility.
- [X] Integrare backend `mctx`.
- [X] Usare `mctx.gumbel_muzero_policy` quando disponibile nell'API installata.
- [X] Creare `MctxSearchBackend`.
- [X] Implementare root preparation batchata per `mctx`.
- [X] Implementare recurrent function compatibile con JIT.
- [X] Evitare loop Python per simulazioni interne nel path caldo.
- [X] Fare smoke test su stato Connect Four.
- [X] Verificare illegal action probability zero.
- [X] Verificare determinismo con seed.
- [X] Aggiungere test Q transform.
- [X] Aggiungere test Sequential Halving.
- [X] Aggiungere test search output shape.
- [X] Aggiungere benchmark search batchata post-warmup.

---

## Fase 13 â€” Gumbel AlphaZero algorithm

- [X] Definire `TrainingAlgorithm` protocol.
- [X] Creare registry algoritmi.
- [X] Implementare `gumbel_alphazero`.
- [X] Verificare che `gumbel_alphazero` usi davvero search Gumbel, non policy raw o random fallback.
- [X] Usare modello perfetto del gioco nella recurrent function, non dinamica appresa MuZero-style.
- [X] Separare prior logits, Gumbel noise, candidate set, improved policy.
- [X] Generare selected action.
- [X] Generare policy target.
- [X] Generare value target dalla prospettiva di `to_play`.
- [X] Gestire terminal state.
- [X] Gestire draw.
- [X] Gestire temperature schedule.
- [X] Gestire resign opzionale solo se testato.
- [X] Aggiungere test target policy valida.
- [X] Aggiungere test target value prospettiva `to_play`.
- [X] Aggiungere test algoritmo smoke su Connect Four.
- [X] Verificare che algoritmo non assuma 7 azioni fuori dalla config gioco.
- [X] Aggiungere test che azioni illegali abbiano probabilitÃ  zero nel target.
- [X] Aggiungere confronto deterministico su seed fisso per search/action selection.

---

## Fase 14 â€” Replay

- [X] Usare `msgpack` e `zstandard` invece di codec/compressione custom.
- [X] Definire replay schema versionato.
- [X] Implementare codec msgpack.
- [X] Implementare compressione zstd.
- [X] Implementare replay writer.
- [X] Implementare replay reader.
- [X] Implementare replay index JSON.
- [X] Implementare scrittura shard atomica.
- [X] Implementare replay sampler uniforme.
- [X] Implementare replay window.
- [X] Implementare validazione sample.
- [X] Validare che `value_target` sia dalla prospettiva di `to_play`.
- [X] Implementare quarantena shard corrotti.
- [X] Aggiungere test roundtrip.
- [X] Aggiungere test corrupted shard.
- [X] Aggiungere test sampler.
- [X] Aggiungere test schema version mismatch.

---

## Fase 15 â€” Self-play

- [X] Definire struttura trajectory.
- [X] Implementare self-play singola partita.
- [X] Implementare self-play batchato.
- [X] Integrare SearchBackend.
- [X] Usare il path Gumbel AlphaZero reale nel comando `gaz run`.
- [X] Separare eventuale self-play random/debug dal path principale.
- [X] Riusare funzioni JAX compilate tra partite e batch.
- [X] Salvare search stats.
- [X] Convertire trajectory in replay samples.
- [X] Scrivere replay shard.
- [X] Loggare games/sec.
- [X] Loggare positions/sec.
- [X] Loggare illegal action rate.
- [X] Loggare policy entropy.
- [X] Loggare root value mean.
- [X] Aggiungere self-play smoke test.
- [X] Aggiungere test seed deterministico.
- [X] Fallire in debug se viene scelta un'azione illegale.
- [X] Aggiungere benchmark self-play batchato post-warmup.

---

## Fase 16 â€” Trainer

- [X] Implementare ReplaySampler -> batch training.
- [X] Implementare augmentations da simmetrie Connect Four.
- [X] Implementare train loop per N steps.
- [X] Garantire batch shape statiche in `train_step`.
- [X] Evitare trasferimenti host/device dentro `train_step`.
- [X] Loggare policy loss.
- [X] Loggare value loss.
- [X] Loggare total loss.
- [X] Loggare learning rate.
- [X] Loggare train samples/sec.
- [X] Loggare replay sample age.
- [X] Salvare checkpoint periodici.
- [X] Aggiungere test trainer smoke.
- [X] Aggiungere benchmark train step post-warmup.

---

## Fase 17 â€” Evaluation

- [X] Implementare opponent random.
- [X] Implementare arena checkpoint vs checkpoint.
- [X] Alternare player iniziale.
- [X] Salvare risultati in `eval/matches.jsonl`.
- [X] Calcolare win/loss/draw.
- [X] Implementare promotion threshold.
- [X] Aggiornare `best` nel registry.
- [X] Aggiungere test arena.
- [X] Aggiungere test promotion.
- [X] Evitare promotion con numero partite troppo basso.

---

## Fase 18 â€” RunOrchestrator

- [X] Implementare orchestrator single-process.
- [X] Collegare self-play -> replay -> train -> checkpoint -> eval.
- [X] Implementare stop conditions.
- [X] Implementare salvataggio run_state.
- [X] Implementare resume base.
- [X] Gestire Ctrl+C con shutdown pulito.
- [X] Aggiungere test run smoke breve.
- [ ] Verificare `uv run gaz run --config configs/connect_four.yaml`.
- [ ] Verificare run smoke su Windows e Linux/WSL.

---

## Fase 19 â€” Scheduler e balancing locale

- [X] Implementare replay low watermark.
- [X] Implementare replay high watermark.
- [X] Dare prioritÃ  self-play quando replay basso.
- [X] Dare prioritÃ  training quando replay alto.
- [X] Limitare self-play batches in flight.
- [X] Limitare replay write queue.
- [X] Limitare tempo evaluation.
- [X] Regolare shard flush frequency.
- [X] Gestire checkpoint pending.
- [X] Gestire model staleness.
- [X] Loggare decisioni scheduler.
- [X] Aggiungere test backpressure.

---

## Fase 20 â€” Resume

- [X] Caricare run_state.
- [X] Caricare config resolved.
- [X] Caricare checkpoint latest.
- [X] Caricare replay index.
- [ ] Riprendere training step.
- [X] Evitare doppia importazione shard.
- [X] Gestire checkpoint incompleti.
- [X] Gestire replay shard incompleti.
- [X] Ricostruire replay index da shard validi.
- [X] Aggiungere test resume run interrotta.
- [ ] Verificare resume con path Windows e path POSIX.

---

## Fase 21 â€” Play umano vs agente

- [X] Implementare `gaz play`.
- [X] Render testuale board.
- [X] Input colonna utente.
- [X] Validazione mossa legale.
- [X] Mossa agente via policy/search.
- [X] Messaggio vittoria/sconfitta/draw.
- [X] Supportare scelta checkpoint latest/best.
- [X] Aggiungere test funzioni non interattive.

---

## Fase 22 â€” Inspect e report

- [X] Implementare `gaz inspect run PATH`.
- [X] Implementare `gaz inspect replay PATH`.
- [X] Implementare `gaz inspect checkpoint PATH`.
- [X] Implementare report metriche principali.
- [X] Implementare analisi illegal action rate.
- [X] Implementare analisi policy entropy.
- [X] Implementare analisi value target distribution.
- [X] Implementare analisi replay age.
- [X] Implementare analisi checkpoint promotion history.
- [X] Aggiungere test inspect smoke.

---

## Fase 23 â€” Benchmark

- [X] Implementare `gaz benchmark`.
- [X] Misurare JAX compile time.
- [X] Misurare warmup.
- [X] Misurare self-play games/sec.
- [X] Misurare positions/sec.
- [X] Misurare search simulations/sec.
- [X] Misurare train samples/sec.
- [X] Misurare replay write throughput.
- [X] Misurare replay read throughput.
- [X] Misurare checkpoint save/load time.
- [X] Misurare evaluation games/sec.
- [X] Salvare JSONL in `artifacts/benchmarks/`.
- [X] Aggiungere benchmark CPU smoke.
- [X] Distinguere metriche debug da metriche preset reale.
- [X] Salvare backend JAX, device, model config, search config e git commit.

---

## Fase 23.5 â€” Performance acceptance

- [X] Verificare che `configs/connect_four.yaml` non usi fallback random o policy-only.
- [X] Verificare che search, self-play e train step abbiano warmup separato dalla misura.
- [X] Verificare che non ci siano loop Python per simulazioni MCTS interne nel path principale.
- [X] Verificare che la search batchata produca target validi e masked correttamente.
- [X] Verificare che il throughput venga scritto in JSONL per confronto futuro.
- [ ] Documentare baseline CPU e, se disponibile, GPU.

---

## Fase 24 â€” local_multiprocess

- [X] Implementare `LocalMultiprocessExecutionBackend`.
- [X] Creare worker process self-play.
- [X] Creare queue bounded.
- [X] Evitare compilazioni JAX concorrenti inutili.
- [X] Gestire shutdown processi.
- [X] Importare replay shard locali.
- [X] Aggiungere test unitari execution backend.
- [X] Aggiungere smoke test local_multiprocess breve.
- [X] Gestire differenze multiprocessing Windows spawn e Linux fork/spawn.
- [X] Eseguire smoke test local_multiprocess su Windows.
- [ ] Eseguire smoke test local_multiprocess su Linux/WSL o CI Linux.

---

## Fase 25 â€” LAN Ray base

- [X] Aggiungere extra `distributed` con Ray.
- [X] Implementare `LanRayExecutionBackend`.
- [X] Implementare `gaz cluster head`.
- [X] Implementare `gaz cluster worker`.
- [X] Implementare `gaz cluster status`.
- [X] Implementare actor head controller.
- [X] Implementare actor worker.
- [X] Implementare registrazione worker.
- [X] Implementare `WorkerCapabilities`.
- [X] Implementare heartbeat.
- [X] Implementare timeout worker.
- [X] Implementare task lease.
- [X] Implementare retry task scaduti.
- [X] Aggiungere test unitari heartbeat.
- [X] Aggiungere test registrazione worker mock.
- [X] Aggiungere test task lease mock.
- [X] Verificare Ray opzionale su Windows e Linux/WSL o documentare limiti.

---

## Fase 26 â€” LAN replay e checkpoint

- [X] Worker scarica checkpoint latest/best.
- [X] Download checkpoint atomico.
- [X] Worker genera replay shard temporanei.
- [X] Worker comprime shard.
- [X] Worker carica shard verso head.
- [X] Head valida shard ricevuto.
- [X] Head importa shard nel replay index.
- [X] Head mette in quarantena shard corrotti.
- [X] Head invia comandi pause/resume ai worker.
- [X] Head puÃ² chiedere sync checkpoint.
- [X] Loggare throughput upload.
- [X] Aggiungere test upload mock.
- [ ] Aggiungere smoke test LAN locale se possibile.

---

## Fase 27 â€” Documentazione

- [X] Scrivere quickstart CPU.
- [X] Scrivere quickstart GPU.
- [X] Scrivere quickstart LAN.
- [X] Scrivere quickstart Windows.
- [X] Scrivere quickstart Linux/WSL.
- [X] Documentare `gaz doctor`.
- [X] Documentare `gaz run`.
- [X] Documentare `gaz resume`.
- [X] Documentare `gaz play`.
- [X] Documentare `gaz benchmark`.
- [X] Documentare struttura artifacts.
- [X] Documentare come aggiungere un gioco.
- [X] Documentare come aggiungere un algoritmo.
- [X] Documentare come aggiungere un modello.
- [X] Documentare troubleshooting JAX/CUDA.
- [X] Documentare differenze note tra Windows nativo, WSL e Linux.
- [X] Documentare limiti noti.

---

## Fase 28 â€” Primo criterio di successo

- [X] Bootstrap CPU completato.
- [X] `gaz doctor` passa.
- [X] Test passano.
- [X] Run Connect Four locale parte.
- [X] Self-play genera partite usando Gumbel AlphaZero reale.
- [X] Illegal action rate zero.
- [X] Replay shard scritti.
- [X] Trainer consuma replay.
- [X] Checkpoint salvato.
- [X] Evaluation eseguita.
- [X] Best checkpoint aggiornato.
- [X] `gaz play` funziona.
- [X] `gaz resume` funziona.
- [X] Benchmark CPU produce JSONL.
- [X] Benchmark include search/self-play/train throughput post-warmup.
- [X] Nessun fallback POC Ã¨ usato dal preset `configs/connect_four.yaml`.
- [X] Smoke test CPU passa su Windows.
- [ ] Smoke test CPU passa su Linux/WSL o CI Linux.

---

## Fase 29 â€” Pulizia finale

- [X] Eseguire `uv run ruff check .`.
- [X] Eseguire `uv run ruff format --check .`.
- [X] Eseguire `uv run pytest`.
- [X] Eseguire test completi su Windows.
- [ ] Eseguire test completi su Linux/WSL o CI Linux.
- [X] Eseguire benchmark smoke.
- [X] Aggiornare README.
- [X] Aggiornare TASKS.md.
- [X] Verificare che non ci siano dipendenze non dichiarate.
- [X] Verificare che non siano state reimplementate funzionalitÃ  giÃ  coperte da librerie mature senza motivo tecnico documentato.
- [X] Verificare che le librerie esterne critiche siano isolate dietro adapter/wrapper.
- [X] Verificare che non ci siano path assoluti.
- [X] Verificare che non ci siano assunzioni Windows-only o Linux-only nel codice core.
- [X] Verificare che Docker non sia necessario.
- [X] Verificare che database server non sia necessario.
- [X] Verificare che Connect Four funzioni ancora.
- [X] Verificare che aggiunta gioco/algoritmo sia documentata.
