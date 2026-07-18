## 1. Freeze the replacement contract

- [x] 1.1 Add semantic parity fixtures for one stateless evidence-retrieval method and record behavior-bearing prepared, ranking, and evaluation payloads from the current runner.
- [x] 1.2 Add semantic parity fixtures for Dense-FT, ordinary evidence R-GCN, and Dense-FT-seeded R-GCN, excluding timestamps, paths, transient timing, and documented latency tie-break bytes.
- [x] 1.3 Add semantic parity fixtures for execution-provenance retrieval, execution-provenance R-GCN, and one executable R-GCN ablation.
- [x] 1.4 Add failing architecture tests that forbid new scientific inputs below `runs/` and identify the planner, subprocess, generated-stage, multi-method, parent/child tracking, and run-state surfaces that must be absent at cutover.
- [x] 1.5 Document the accepted direct-cutover matrix for single runs, Hydra multiruns, cache hits, failures, delivery, and historical-run non-migration.

## 2. Build the processed artifact foundation

- [x] 2.1 Define closed typed external source references, processed `ArtifactRef` variants, artifact manifests, and stage result objects with URI, kind, digest/revision, declared payloads, origin, size/shape, metrics, and metadata contracts.
- [x] 2.2 Implement streaming file and deterministic directory identity for raw data, local encoders, checkpoints, and other mutable-path inputs.
- [x] 2.3 Implement kind-specific processed roots below `data/processed/` for datasets, evidence graphs, training pairs, models, predictions, and Prefect results.
- [x] 2.4 Implement task-specific staging workspaces, declared-output validation, manifest writing, atomic same-filesystem publication, and failed-attempt cleanup.
- [x] 2.5 Use Pydantic for processed-reference structure and resolve declared payloads without repeated whole-artifact hashing; missing payloads report the refresh recovery command without reading a run-local substitute.
- [x] 2.6 Configure shared Prefect result storage and serialization below `data/processed/prefect/` without a repository-owned cache lock layer.
- [x] 2.7 Add focused tests for file/directory identity, atomic publication, incomplete staging rejection, typed payload resolution, and missing assets.

## 3. Extract importable stage services

- [x] 3.1 Extract the HotpotQA, MuSiQue, 2Wiki, and execution-provenance split preparation bodies into typed `graph_memory/stages/` services while preserving dataset validation and sampling semantics.
- [x] 3.2 Extract evidence-graph construction into an importable service that reads processed split references and publishes validated graph artifacts.
- [x] 3.3 Extract training-pair construction into an importable service that owns graphless/graph-backed requests, negative sampling, and optional dense encoder inputs.
- [x] 3.4 Refactor model training services to consume processed references and publish final checkpoint/model artifacts plus small training histories without run-local output ownership.
- [x] 3.5 Refactor retrieval services to consume processed split/graph/model references and publish validated ranking artifacts without run-local output ownership.
- [x] 3.6 Refactor evaluation services to consume processed prediction/label/graph references and return validated metrics plus failure-case artifacts.
- [x] 3.7 Freeze legacy stage payloads, prove focused semantic parity through the extracted services, and remove the generated-stage CLI contracts in the direct cutover.

## 4. Replace multi-method configuration

- [x] 4.1 Replace root `methods`, `method_configs`, `stages`, cache enablement, and global ablation selection with one discriminated final `method`, one optional singular effective variant, cache refresh, and benchmark configuration; reject list-valued variants.
- [x] 4.2 Create singular Hydra method configs for stateless methods, Dense-FT, evidence R-GCN, Dense-FT-seeded R-GCN, execution-provenance retrieval, and execution-provenance R-GCN.
- [x] 4.3 Give Dense-FT-seeded R-GCN an explicit Dense-FT seed-training section and R-GCN section without generic dependency metadata or runtime dependency traversal.
- [x] 4.4 Compose only real shared encoder or training-stage configuration fragments and add tests that the public Dense-FT config and composite seed config resolve to one canonical effective stage contract.
- [x] 4.5 Replace ablation enable/list semantics with one validated singular `method.variant`, default `full_rgcn`; reject `ablation.variants`, variant lists, and variants on unsupported methods.
- [x] 4.6 Preserve deterministic single-run and multirun output directories below `runs/<run-name>/` while excluding run and job identity from resolved scientific Task inputs.
- [x] 4.7 Add Hydra/Pydantic tests for every legal method/dataset family, representative invalid combinations, baseline sweeps, and ablation sweeps.
- [x] 4.8 Document and test independently launched variant jobs, including one-command-per-GPU examples, shared study identity, and cross-process upstream cache reuse without in-Flow fan-out.

## 5. Add cached Prefect Tasks

- [x] 5.1 Add thin cached prepare-split Tasks with exact source, conversion, split, count, offset, and seed inputs and stage-specific implementation identity.
- [x] 5.2 Add thin cached graph and pair Tasks whose signatures exclude unrelated final-method labels, evaluation, tracking, and run-output configuration.
- [x] 5.3 Add cached Dense-FT, evidence R-GCN, Dense-FT-seeded R-GCN, and provenance R-GCN training Tasks with explicit model/device identities and atomically published processed outputs.
- [x] 5.4 Add cached ranking and evaluation Tasks that return complete results needed by downstream computation and current-run tracking on cache hits.
- [x] 5.5 Configure no automatic retries for scientific Tasks; rerunning the full Flow is the recovery interface.
- [x] 5.6 Keep current trainers free of attempt-state publication because they expose no safe mid-run resume boundary.
- [x] 5.7 Apply `cache.refresh` once through a Flow-scoped Prefect setting without placing it in scientific cache inputs or adding per-stage refresh syntax.
- [x] 5.8 Keep Task state, retry history, and cache-hit observation in Prefect instead of copying them into every domain result, run output, and MLflow run.

## 6. Implement the explicit single-method Flow

- [x] 6.1 Create `graph_memory/experiment/workflow.py` with one synchronous Prefect Flow and direct Task calls, with no `.submit()` for stages or ablation variants, planner, invocation registry, generated stage YAML, or subprocess execution.
- [x] 6.2 Implement the explicit BM25, Dense, GraphRAG, and execution-provenance stateless branches.
- [x] 6.3 Implement the explicit Dense-FT branch from prepared splits through pairs, training, ranking, and evaluation.
- [x] 6.4 Implement the explicit ordinary evidence R-GCN and execution-provenance R-GCN branches.
- [x] 6.5 Implement the explicit Dense-FT-seeded R-GCN branch and prove it reuses an equal prior Dense-FT Task result without creating a separate final result.
- [x] 6.6 Apply one effective R-GCN variant inside the selected branch and remove in-Flow variant expansion, aliasing, and ablation indexing.
- [x] 6.7 Return one typed final experiment result and remove aggregate computation from the Flow.
- [x] 6.8 Update `experiment/run.py` to compose Hydra configuration, initialize one final MLflow run, invoke the Flow, and report one final method/variant result.

## 7. Preserve output-only runs and delivery

- [x] 7.1 Implement uncached Flow-level run projection for resolved config, overrides, workflow summary, final metrics, applicable small training/debug/report files, and asset manifests.
- [x] 7.2 Ensure single runs write below `runs/<run-name>/` and Hydra jobs write below deterministic multirun child directories without passing those paths to scientific Tasks.
- [x] 7.3 Add runtime guards and architecture scans proving no new stage service or Prefect Task reads a path below `runs/`.
- [x] 7.4 Update `scripts/deliver/collect_run_artifacts.py` to collect the new output-only single-run surface and include reusable asset references in its delivery manifest.
- [x] 7.5 Update multirun collection to preserve every independent single-method job and its deterministic selector without traversing or copying `data/processed/`.
- [x] 7.6 Add offline-delivery tests proving `results/<run-name>` identifies config, method, variant, seed, final metrics, and referenced asset URI/digest without Prefect or MLflow access.

## 8. Replace experiment tracking and timing

- [x] 8.1 Replace parent/baseline-child tracking with one top-level MLflow run per Hydra job and final method/variant using only fluent active-run APIs; delete `MlflowClient`, run-ID lookup/reuse, and all writes to non-active runs.
- [x] 8.2 Log resolved config, identity tags, common `final.*` metrics, small reports, and processed asset references from the Flow on both fresh execution and cache hits.
- [x] 8.3 Preserve shared method-independent final metric keys and reject fabricated numeric values for inapplicable metrics.
- [x] 8.4 Project training history with explicit asset-origin metadata and keep historical duration/resource use out of current execution metrics.
- [x] 8.5 Add an explicit non-cached benchmark Task with configured warmup/repetition for requested fresh efficiency measurements.
- [x] 8.6 Ensure cached production timing remains provenance metadata and is omitted from current runtime/benchmark metrics when no fresh benchmark executes.
- [x] 8.7 Preserve strict tracking failure propagation while proving MLflow state never participates in cache keys or processed asset resolution.
- [x] 8.8 Add peer-run study tags for Hydra baseline and ablation multiruns without creating an MLflow parent.

## 9. Remove superseded runner surfaces

- [x] 9.1 Delete the public plan entrypoint and remove active documentation/examples for planning, stage ranges, generated stage commands, and completed-prefix resume.
- [x] 9.2 Delete planner, invocation, subprocess execution, resume, run-state, stage-status, and run-local computation-layout code that has no new caller.
- [x] 9.3 Delete generated stage models/config persistence and obsolete direct stage CLI entrypoints after service and Flow parity passes.
- [x] 9.4 Delete multi-method, ablation list expansion/alias, in-Flow variant fan-out, aggregate-stage, `MlflowClient`, run-ID lookup/reuse, and parent/child MLflow code plus their accidental-shape tests.
- [x] 9.5 Remove obsolete status/reset semantics tied to run-local cache truth while retaining only narrow discovery or output-management commands with real callers.
- [x] 9.6 Sweep README, active operations/config/design documentation, validation scripts, delivery docs, and OpenSpec references for the retired contracts and new commands.
- [x] 9.7 Add residual scans proving there is no workflow subprocess execution, `WorkflowPlanner`, `StageInvocation`, generated stage YAML, `methods` or ablation-variant list, variant `.submit()` fan-out, `stages.from/to`, run-state cache truth, aggregate Task, `MlflowClient`, run-ID reuse, or MLflow child-run lifecycle in active production code.

## 10. Verify cache semantics and real workflows

- [x] 10.1 Prove an identical experiment under a different run name hits every compatible scientific Task cache and writes a new complete run output and MLflow run.
- [x] 10.2 Prove changing only evaluation configuration recomputes evaluation without rebuilding data, graphs, pairs, training, or rankings.
- [x] 10.3 Prove changing pair, trainer, model, variant, source digest, encoder digest, and implementation version invalidates exactly the intended stage and downstream consumers.
- [x] 10.4 Prove `cache.refresh=true`, missing processed payloads, interrupted Tasks without automatic retry, and non-overwriting concurrent publication follow the specified behavior.
- [x] 10.5 Run fresh-name smoke workflows for a stateless evidence method, Dense-FT, ordinary evidence R-GCN, Dense-FT-seeded R-GCN, execution-provenance retrieval, execution-provenance R-GCN, and one ablation variant.
- [x] 10.6 Run a Hydra baseline multirun plus independently launched ablation jobs on separate GPU assignments, verify peer MLflow runs, cross-process cache reuse after publication, and distinct variant-specific assets, then collect the complete named output trees into `results/`.
- [x] 10.7 Compare new and frozen old-run behavior-bearing scientific payloads and document any accepted non-semantic differences.
- [x] 10.8 Run the focused workflow tests, full `uv run pytest -q`, isolated Python 3.10 workflow tests, basedpyright, Ruff, compileall, and `git diff --check`.
- [x] 10.9 Run repository-wide retired-surface scans and `openspec validate replace-experiment-runner-with-prefect-workflow --strict` with no remaining violations.

10.6 completed across local and deployment validation: the local baseline multirun, peer MLflow runs, shared cache, distinct variant assets, and collector passed, and the user confirmed independent GPU assignment acceptance on deployment hardware.

## 11. Correct the first implementation toward the minimal workflow contract

- [x] 11.1 Remove automatic retries, retry hooks/counters, custom filesystem cache locks, unused attempt/cache-key directories, and repeated per-call cache-refresh options.
- [x] 11.2 Remove Flow forwarding wrappers and Task-state capture; call stage Tasks directly from visible method branches and use one common ranking/evaluation/benchmark tail.
- [x] 11.3 Narrow pair, train, ranking, and evaluation cache inputs so final-method labels, unrelated trainer/pair configuration, duplicated runtime identity, and presentation metadata cannot invalidate scientific work.
- [x] 11.4 Carry declared payload metadata in Pydantic artifact references, remove repeated whole-artifact validation and digest scans, and retain only publication-time identity plus missing-payload checks at consumption.
- [x] 11.5 Remove duplicated Task-run/cache metadata from final results, run outputs, MLflow logging, delivery tests, and active documentation.
- [x] 11.6 Remove unused artifact/config abstractions and redundant root-propagation validation while retaining cross-record scientific/leakage checks that Pydantic cannot express locally.
- [x] 11.7 Add focused regression tests for direct Flow structure, no retry/lock/state mirror surfaces, precise cross-method pair/ranking cache inputs, typed payload resolution, and Flow-scoped refresh.
- [x] 11.8 Run focused and full tests, Ruff, basedpyright, compileall, `git diff --check`, retired-surface scans, and strict OpenSpec validation.
