## Context

The current experiment stack was deliberately built around a typed `WorkflowPlanner`, generated stage execution YAML, one subprocess per stage, run-local artifacts and summaries, completed-prefix resume, multi-method jobs, ablation aliases, and MLflow parent/baseline-child runs. That architecture is functional, but it makes the repository responsible for DAG construction, process execution, cache validation, retry state, dependency projection, and tracking projection.

The resulting `graph_memory/experiment/` package is now larger and harder to follow than the scientific flow it represents. Expensive outputs are owned by `runs/<run-name>/`, so changing only a downstream method, evaluation option, display name, or study grouping cannot naturally reuse compatible assets from an independent experiment.

Prefect 3.7.8 is already present in `pyproject.toml` and `uv.lock`, and the user has completed a successful basic Prefect spike in the repository environment. This change therefore treats Prefect availability as an accepted prerequisite rather than an implementation task.

The repository has two storage requirements that are not interchangeable:

- reusable inputs to future computation, including prepared splits, evidence graphs, training pairs, predictions, checkpoints, and model directories, belong under `data/processed/`;
- experiment presentation and delivery files belong under `runs/<run-name>/`, and nothing below `runs/` may be read back by the workflow as computation input.

Historical runs and MLflow rows remain useful evidence. The new architecture applies only to newly created runs and does not migrate or reinterpret old workflow state.

## Goals / Non-Goals

**Goals:**

- Make the complete stage chain for one final method visible in one ordinary Python Prefect Flow.
- Make one Hydra job select exactly one final method and at most one method variant.
- Use Prefect Tasks for expensive, reusable, and observable stages.
- Reuse compatible Task outputs across names, studies, Hydra jobs, methods, variants, and repeated invocations.
- Keep large reusable artifacts under `data/processed/` and pass stable typed references between Tasks.
- Keep `runs/<run-name>/` as a complete output-only small-file surface that remains compatible with run delivery collection.
- Record one comparable MLflow run per final method/variant whether stages execute or hit cache.
- Preserve current dataset adapters, retrieval algorithms, training algorithms, evaluation definitions, output schemas, and leakage constraints.
- Remove superseded planner, stage protocol, subprocess, run-state, aggregate, parent/child tracking, and compatibility code after parity is demonstrated.

**Non-Goals:**

- Building a general workflow platform, plugin system, DAG registry, custom cache database, cache lock layer, retry framework, or custom distributed scheduler.
- Adding Prefect deployments, work pools, remote workers, Dask, Ray, Kubernetes, or scheduled execution in the first implementation.
- Running Tasks concurrently by default; direct synchronous Task calls are the initial execution contract.
- Accepting an in-Flow ablation variant list or using `task.submit()` to fan complete variant workflows out inside one Python process.
- Using `MlflowClient`, child runs, or run-ID lookup/reuse in active experiment tracking.
- Automatically garbage-collecting reusable assets below `data/processed/`.
- Treating `runs/`, MLflow, or run names as cache authorities.
- Automatically repairing a Prefect cache record whose referenced processed asset was manually deleted.
- Migrating existing run directories, stage summaries, MLflow rows, or completed OpenSpec changes.
- Preserving direct execution of generated stage YAML or maintaining dual old/new runner paths.

## Decisions

### 1. One final method and variant per Hydra job

The root experiment configuration replaces `methods: list[...]`, `method_configs`, global ablation enablement, and an ablation variant list with one discriminated `method` configuration. R-GCN methods expose one singular effective `method.variant`, defaulting to `full_rgcn`; methods without variants reject that field. A job MUST reject `ablation.variants=[...]` or any other list-valued variant contract.

The selected method configuration contains the complete configuration needed by its explicit branch. A composite method such as Dense-FT-seeded R-GCN contains both its Dense-FT seed-training configuration and its R-GCN configuration. Hydra may compose a small shared stage-config fragment where the same real stage configuration is reused, but no generic dependency metadata or method-plan registry is introduced.

Multi-baseline studies and ablation suites use independent Hydra jobs, launched through Hydra multirun, separate commands, or an external launcher. Each job is its own process, Prefect Flow run, and MLflow run. Users may assign jobs to `cuda:0`, `cuda:1`, and other devices for multi-GPU parallelism, while compatible upstream Tasks share cache entries across processes.

The first implementation intentionally does not loop over a variant list or use `task.submit()` for variant fan-out. Under the proposed local execution model that would make several complete train/retrieve/evaluate chains share one Python and CUDA runtime, introduce GPU resource scheduling and failure-cancellation semantics, require per-variant output coordination, and force one active MLflow run either to prefix every variant metric or to recreate child runs. Independent jobs keep the final scientific result as the unit of configuration, failure, tracking, and delivery.

Concurrent jobs do not write to run-local shared computation paths. Prefect owns cache records, while opaque processed-asset destinations and atomic publication prevent jobs from overwriting one another. Two simultaneous first requests may duplicate uncached work; avoiding that stampede is not a first-version repository responsibility. A variant that changes pair construction, training, retrieval, or evaluation receives a different identity at the first affected Task and therefore publishes distinct downstream assets.

Alternative considered: retain a list of methods or variants and let the Flow loop or submit over it. Rejected because it recreates multiple final results, aggregation, dependency ownership, GPU scheduling, output coordination, and multi-run tracking inside one Flow.

### 2. The Flow is the only workflow definition

`graph_memory/experiment/workflow.py` owns one `@flow` function and calls the thin stage-level Tasks directly and synchronously. It does not add ordinary forwarding wrappers around Task calls, compile a plan, create an invocation registry, write stage YAML, call `.submit()` for stages or ablation variants, or launch subprocesses.

The Flow uses explicit `if`/`elif` branches for the supported final methods. Hidden train dependencies are written directly in the owning branch; for example, Dense-FT training appears before seeded R-GCN training. Shared ranking, evaluation, benchmark, output, and tracking work forms one visible common tail after the method-specific branch.

Alternative considered: keep `WorkflowPlanner` and execute its nodes through Prefect. Rejected because Prefect would then observe an already compiled custom DAG and the repository would keep two orchestration authorities.

### 3. Stage implementations are pure importable services

Existing stage CLI bodies are moved into `graph_memory/stages/` services that accept typed domain inputs or processed artifact references and return typed results. Prefect Task wrappers own caching; stage services own scientific computation and atomic artifact publication. The first implementation does not add automatic Task retry policy or mirror Prefect Task-run state into domain results.

The old scripts may temporarily call the extracted service during migration, but generated stage YAML and direct stage entrypoints are deleted when the Prefect acceptance workflows pass. Dataset acquisition utilities that populate raw source data remain separate from the experiment Flow.

Alternative considered: import and call each script's `main()` in-process. Rejected because those functions still reconstruct CLI state, write run-local paths, and couple scientific work to stage lifecycle summaries.

### 4. Task signatures are the cache boundary

Each cached Task receives only inputs that affect its output:

- a prepared-split Task receives a source reference with content identity plus resolved count, offset, sampling seed, and dataset conversion configuration;
- a graph Task receives a prepared split reference and effective graph configuration;
- a pair Task receives prepared inputs/labels, optional graph reference, pair-stage configuration, and encoder/model identity when used, but not a final method name used only for presentation or storage namespacing;
- a train Task receives the exact prepared, graph, pair, model dependency, trainer, selection, and device inputs that can affect its checkpoint;
- retrieval and evaluation receive their exact upstream references and effective method/evaluation configurations.

Display name, Hydra job number, run directory, Prefect flow/task IDs, MLflow run ID, tracking configuration, output presentation configuration, and unrelated downstream options never enter a reusable Task signature.

Cached Tasks use persistent Prefect result storage and an explicit cross-flow policy based on effective inputs plus Task source. Each important Task also carries a short stage-specific implementation version because helper changes may not be represented by the decorated function source alone. The whole Git commit is not included in every key.

Alternative considered: pass the resolved root config into every Task. Rejected because unrelated overrides would invalidate reusable upstream computation.

### 5. Processed assets are immutable data-plane objects

Prefect result storage persists small typed result objects. Large files and directories are published below `data/processed/` using kind-specific roots such as:

```text
data/processed/
  prefect/
    results/
  datasets/<dataset>/<artifact-id>/
  evidence_graphs/<artifact-id>/
  training_pairs/<artifact-id>/
  models/<method>/<artifact-id>/
  predictions/<method>/<artifact-id>/
  .staging/<task-run-id>/
```

The exact leaf is an opaque generated artifact ID, not a run name and not a second cache key. A Task writes within `.staging`, validates every declared output, writes a manifest, atomically publishes on the same filesystem, and returns an `ArtifactRef` containing URI, kind, content digest, manifest URI, size/shape metadata, and origin identifiers.

Pydantic validates artifact-reference and manifest structure. The returned reference also carries its declared payload map, so consumers resolve a role without reparsing and rehashing the complete artifact on every payload access. Publication computes content identity once; consumption checks that the requested declared payload still exists and otherwise fails with the missing path and refresh instruction. Full digest re-auditing is an explicit diagnostic operation, not an implicit read-path cost.

Alternative considered: serialize graphs, models, or complete directories directly through Prefect results. Rejected because current assets reach hundreds of MiB and need ordinary file/directory access.

### 6. Prefect owns stage state and cross-run reuse

`run_state.yaml`, stage summaries as cache truth, status-derived completed prefixes, and `stages.from/to` are removed. Re-running a full Flow is the only recovery command: completed compatible Tasks are cached, failed Tasks run again, and changed effective inputs invalidate only the affected stage and its downstream consumers.

`cache.refresh=true` is an operational Flow-scoped Prefect setting and is excluded from scientific Task inputs. The first version refreshes all reusable Tasks in the Flow; it does not repeat `.with_options(refresh_cache=...)` at every call or introduce a task-selection mini-language.

Tasks do not retry automatically in the first implementation. A repeated full Flow command is the recovery interface: compatible completed Tasks remain cached and the failed Task runs again. If a future acquisition boundary has a demonstrated transient failure mode, its retry policy is specified locally rather than applied to every scientific stage.

Prefect remains the sole Task-state authority. Domain results, run outputs, and MLflow records do not duplicate Task-run states, retry counts, or cache-hit flags. This keeps observability in Prefect and scientific results independent of orchestration metadata.

### 7. `runs/<run-name>` is output-only and remains collectable

The Flow creates a single-run directory or Hydra multirun job directory only for presentation files. A new run output contains, where applicable:

```text
runs/<run-name>/[<job-selector>/]
  config/resolved.yaml
  config/overrides.yaml
  workflow/summary.yaml
  assets/manifest.yaml
  metrics/final.metrics.csv
  tables/main_results.csv
  tables/path_results.csv
  tables/efficiency_results.csv
  training/train_metrics.jsonl
  debug/failure_cases.jsonl
  report/...
```

These files are projections from final Task results. They are written after computation by uncached Flow-level output code and are never passed into Tasks. Large processed assets are represented in `assets/manifest.yaml` by URI, digest, kind, size, and origin; they are not copied into the run directory.

`scripts/deliver/collect_run_artifacts.py --name <name>` continues to mirror the named output tree into `results/<name>`, including every Hydra job. It records reusable asset references in the delivery manifest without traversing or copying `data/processed/`.

Alternative considered: eliminate local run outputs and export exclusively from MLflow. Rejected because the repository intentionally supports inspectable, offline, small-file run delivery.

### 8. Aggregate comparison leaves the single experiment Flow

The ordinary Flow evaluates one final method and produces one metric row. It does not run an aggregate Task or create an ablation index. Cross-method and cross-variant comparison uses MLflow Compare Runs and, when a paper table is required, a separate read-only reporting/export utility that consumes experiment outputs or MLflow records, not the experiment computation Flow.

Alternative considered: retain aggregate as the last Task. Rejected because a single-method Flow has no in-flow collection to aggregate and making it query other runs would couple computation to external study state.

### 9. One active MLflow run mirrors one Flow

The Hydra entrypoint opens exactly one top-level MLflow run for its final method/variant, invokes the Flow within that active-run context, and ends that same run. Parent runs, baseline children, dependency children, and stage children are removed. All final runs use the same method-independent `final.*` metric keys and carry method, variant, dataset, profile, seed, study/run name, Prefect flow-run ID, processed asset references, and other effective parameters or tags.

Active tracking uses MLflow's fluent functions such as `set_tracking_uri`, `set_experiment`, `start_run`, `log_params`, `log_metrics`, `set_tags`, artifact logging, and `end_run(status=...)`. `graph_memory/experiment/tracking.py` does not instantiate `MlflowClient`, search for an existing run ID, reuse a prior run, mutate a non-active run, or encode parent/child relationships. A separate read-only reporting utility may use the fluent `mlflow.search_runs` interface to compare completed peer runs, but it is outside the experiment Flow and cannot write the current experiment.

MLflow logging that must appear on every experiment occurs after Task results are available in the Flow, so cached Tasks still produce a complete current MLflow run. Task bodies do not own required experiment-level logging. Small training histories and reports may be projected from model assets and are labelled with their asset origin; they are not presented as current execution work.

Tracking failure fails the Flow after local output and reusable asset state are kept internally consistent. MLflow never participates in a cache key or artifact-validity decision.

### 10. Runtime benchmark measurements are not cache artifacts

Cached Task metadata may retain original production duration for provenance, but a current experiment MUST NOT log that value as current runtime. Quality metrics may reuse cached rankings and evaluation results.

When comparable efficiency measurements are requested, the Flow invokes an explicit non-cached benchmark Task after obtaining cached scientific inputs/models. That Task performs the required warmup/repetition protocol and returns benchmark measurements without replacing the reusable ranking/model asset. Current-run `benchmark.*` or eligible final efficiency metrics come only from this fresh benchmark result. Otherwise current runtime fields are omitted or marked unavailable, while historical asset-production timing remains metadata.

Alternative considered: force every retrieval and graph stage to rerun so timing is always fresh. Rejected because it defeats cross-experiment caching for ordinary scientific iteration.

### 11. Direct cutover with no compatibility branch

Once the acceptance matrix passes, old planner, execution, invocation, resume, state, stage CLI, stage config generation, run-local computation layout, parent/child tracking, plan/status/reset behavior, and obsolete tests are removed in the same change. There is no feature flag or dual-write mode.

Historical directories remain readable by existing delivery and manual inspection tools. New code does not read historical run-local computation artifacts.

## Risks / Trade-offs

- [A cached result references a processed asset that was manually deleted] → Resolve the declared payload from the typed reference, fail with the missing URI and recovery command, and require `cache.refresh=true`; do not repeatedly rehash every payload during ordinary reads.
- [Helper code changes without changing the decorated Task source] → Keep narrow implementation-version inputs per semantic stage and add invalidation tests around helper changes represented by version bumps.
- [Model or raw-data content changes at the same path] → Convert external sources into typed references with content digest or immutable revision before they reach a cached Task.
- [Parallel Hydra jobs request the same uncached stage] → Accept possible duplicate first computation; opaque destinations and atomic publication prevent overwrite, and later calls reuse the Prefect cache record.
- [Users want parallel ablations across several GPUs] → Launch one singular-variant job/process per GPU. Variant-specific stages receive distinct identities and no process writes another job's run directory.
- [Training fails late within a Task] → Re-run the complete Flow; completed dependencies remain cached and the failed training Task starts cleanly because current trainers expose no safe resume boundary.
- [Run directories no longer contain computation artifacts] → Preserve complete small summaries, metrics, debug outputs, and an asset manifest so delivery remains self-contained and explicit about omitted data.
- [Cached rankings make old timing look current] → Separate provenance timing from an explicit non-cached benchmark Task and restrict current-run efficiency metric logging to fresh measurements.
- [Single-method jobs create more Hydra/MLflow runs] → Use a stable study/name tag and identical final metric keys; this is intentional because the final method is the unit of comparison.
- [The change removes recently implemented workflow code] → Verify scientific payload parity before deletion and use a direct cutover rather than carrying the complexity indefinitely.

## Migration Plan

1. Freeze semantic acceptance fixtures from the current runner for representative stateless, Dense-FT, R-GCN, Dense-FT-seeded R-GCN, provenance, and ablation paths. Compare behavior-bearing payloads rather than timestamps, paths, or latency tie-break bytes.
2. Add processed artifact reference/publication types and configure Prefect result/key storage below `data/processed/prefect/` without changing current stage execution.
3. Extract current script bodies into importable `graph_memory/stages/` services and prove direct service parity while the old runner still invokes them.
4. Replace multi-method and variant-list configuration with the singular discriminated method/variant contract and add Hydra composition plus independent-job/multi-GPU launch tests.
5. Implement the explicit Prefect Flow and cached Task wrappers, including Dense-FT dependency reuse, invalidation behavior, Flow-scoped refresh, and missing-payload failure without retry or Task-state mirroring.
6. Implement output-only run projection, the asset manifest, fluent-API one-run MLflow tracking with no `MlflowClient`, optional fresh benchmarking, and updated delivery collection.
7. Run the cache/invalidation matrix and actual smoke workflows across the representative method and dataset families. Run a repeated experiment under a different name and prove upstream cache reuse.
8. Delete the old runner and compatibility surfaces, update active documentation, and run the full test/static/OpenSpec validation gates.

Rollback before adoption is a normal source rollback. Processed assets and new MLflow/Prefect records created during validation may remain historical and are not rewritten. After adoption there is no runtime switch back to the old runner.

## Open Questions

None. Prefect availability and the `data/processed/` versus output-only `runs/` storage boundary are confirmed inputs to this design. Exact stage implementation-version strings and benchmark repetition counts are implementation values governed by existing scientific configuration and focused acceptance tests.
