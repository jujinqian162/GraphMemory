## 1. Freeze Legacy Behavior

- [x] 1.1 Add golden plan and resolved-config fixtures for the default HotpotQA quick workflow, the full eight-method registry, and representative 2Wiki and MuSiQue workflows.
- [x] 1.2 Add focused parity fixtures for Memory Stream external importance/split constraints and the hidden Dense-FT seed dependency of `dense_ft_rgcn_graph_retriever`.
- [x] 1.3 Add cache/status fixtures covering missing, complete, stale, alias, failed residual outputs, and completed-prefix resume.
- [x] 1.4 Add ablation fixtures covering changed-dimension invalidation, baseline aliasing, variant filtering, and ablations-only prerequisites.
- [x] 1.5 Record current prediction, failure-case, selected-config, checkpoint metadata, stage metric, and aggregate CSV schemas used by migration tests.
- [x] 1.6 Declare exact deterministic parity rules and tolerance-based trainable-method comparison rules in test helpers and migration documentation.

## 2. Dependency and Runtime Spikes

- [x] 2.1 Add stable Python 3.10-compatible `hydra-core`, Pydantic V2, and MLflow dependency constraints and update `uv.lock`.
- [x] 2.2 Add and run a real Python 3.10 smoke that imports Hydra, Pydantic, MLflow, and the project from the locked environment.
- [x] 2.3 Add Hydra spike tests for root composition, unknown overrides, `hydra.job.chdir=false`, BasicLauncher, and sequential multirun output paths.
- [x] 2.4 Add MLflow SQLite spike tests for shared backend initialization, parent/child runs, resume by persisted parent id, and curated artifact logging.

## 3. Typed Configuration and Hydra Groups

- [x] 3.1 Create closed Pydantic primitives for scientific scalars, split windows/capacities, count policies, artifact references, stage bounds, cache, and ablation selection.
- [x] 3.2 Create discriminated Pydantic method configuration contracts for BM25, Dense, Memory Stream, graph rerank, R-GCN, Dense-FT, and Dense-FT-seeded R-GCN without wide optional fields.
- [x] 3.3 Create the root `ExperimentConfig` and typed resolver with unknown-key rejection, unresolved-value rejection, scientific scalar validation, capacity checks, and normalized comparison output.
- [x] 3.4 Create Hydra dataset groups for HotpotQA, 2WikiMultiHopQA, and MuSiQue with split sources, adapters, and logical-window capacities.
- [x] 3.5 Create Hydra profile groups for quick, smoke, cloud-full, and every retained profile using `fixed`/`all_available` policies plus trainable-method settings.
- [x] 3.6 Create all eight stable `method_configs.<method>` groups, graph-rerank and Memory Stream search-space groups, tracking defaults, and ablation defaults.
- [x] 3.7 Create the default root config and retained special root presets, including `2wiki_tiny`, with no recipe abstraction and no code-owned business defaults.
- [x] 3.8 Add composition/validation tests for dataset replacement, profile capacity resolution, all-method multi-select layout, method-specific override isolation, seed/device propagation, method subsets, and legal ablation values.
- [x] 3.9 Add migration tests proving every retained legacy JSON experiment/method/profile/search-space value has a YAML/Pydantic representation.

## 4. Typed Experiment Core and Plan Parity

- [x] 4.1 Implement `RunLayout` as the exclusive owner of absolute single-run, multirun, stage-config, artifact, variant, alias, state, override, and resolved-config paths.
- [x] 4.2 Implement typed method registry projections for lifecycle, tuning, train dependencies, checkpoint kinds, seed methods, and artifact roles across all eight public methods.
- [x] 4.3 Implement discriminated stage config models and typed `StageInvocation` contracts with declared inputs, outputs, dependencies, and resolved YAML paths.
- [x] 4.4 Implement `WorkflowPlanner` for shared stages, public method subsets, hidden dependencies, tune bindings, stage bounds, prerequisite validation, and stable ordering.
- [x] 4.5 Implement domain ablation planning with changed-dimension invalidation, baseline aliases, variant namespaces, filtering, and ablations-only prerequisites.
- [x] 4.6 Implement atomic resolved-config, override-list, and stage-YAML persistence without executing stage processes.
- [x] 4.7 Implement shared plan formatting data and prove plan/run command parity.
- [x] 4.8 Add `experiment/plan.py` as a thin Hydra entrypoint with implicit initialization and no MLflow run or stage execution.
- [x] 4.9 Add plan snapshot/parity tests for three datasets, all method families, hidden Dense-FT dependencies, Memory Stream, intermediate stage ranges, and ablations.

## 5. Typed State, Status, and Resume

- [x] 5.1 Implement versioned Pydantic `RunState` with atomic YAML persistence, normalized config/mode reuse checks, selected methods/stages, artifact references, and optional MLflow parent id.
- [x] 5.2 Implement versioned Pydantic `StageRunSummary` with atomic persistence for running, successful, and failed attempts.
- [x] 5.3 Implement the shared stage lifecycle helper for timestamps, exception preservation, counts/timings, summaries, and optional active tracking hooks.
- [x] 5.4 Port stage-specific expected-value and artifact-kind validation to typed invocations while preserving file and directory checkpoint behavior.
- [x] 5.5 Implement live `missing`/`complete`/`stale`/`alias` status derivation without copied state or MLflow cache decisions.
- [x] 5.6 Implement completed-prefix resume and `cache.enabled=false` behavior without node hashes or cross-run reuse.
- [x] 5.7 Add `experiment/status.py` and status/resume regression tests for every frozen cache fixture.
- [x] 5.8 Add run identity tests for exact resume, config mismatch, and single-run/multirun mode mismatch.

## 6. Direct Stage YAML Migration

- [x] 6.1 Add the common `--config <resolved-yaml>` parser/loader and tests that reject extra flags, unknown fields, and legacy JSON inputs.
- [x] 6.2 Migrate dataset prepare dispatch and dataset-specific prepare scripts to typed stage YAML while retaining split/projector/leakage validation.
- [x] 6.3 Migrate graph building to typed stage YAML and shared lifecycle summaries.
- [x] 6.4 Migrate train-pair building to typed stage YAML and shared lifecycle summaries.
- [x] 6.5 Migrate graph-rerank and Memory Stream tuning to typed stage YAML while retaining candidate tables and selected-config validation.
- [x] 6.6 Migrate R-GCN and Dense-FT training to typed stage YAML while retaining local metric JSONL, checkpoint metadata, seed/device propagation, and file/directory artifact validation.
- [x] 6.7 Migrate retrieval to method-discriminated typed stage YAML while retaining request boundaries, selected-config/checkpoint validation, and prediction schemas.
- [x] 6.8 Migrate evaluation to typed stage YAML while retaining split/leakage validation, metric files, and failure-case schemas.
- [x] 6.9 Migrate aggregate to typed stage YAML while retaining main/path/efficiency/ablation CSV schemas.
- [x] 6.10 Add direct-script integration tests for each migrated stage and sequential subprocess fail-fast tests.

## 7. Run Execution and MLflow Tracking

- [x] 7.1 Implement the strict MLflow adapter with one shared SQLite store/artifact root, flattened params, provenance/environment tags, and no silent fallback.
- [x] 7.2 Implement deny-by-default curated artifact logging and path/size/kind/role metadata for prohibited large artifacts.
- [x] 7.3 Implement parent run creation/reuse for Hydra jobs and nested child runs only for executed stage attempts.
- [x] 7.4 Mirror epoch-step training metrics, tuning best metrics/candidate tables, retrieval/evaluation metrics, aggregate tables, timings, status, and exceptions while retaining local facts.
- [x] 7.5 Implement sequential subprocess execution, plan-equivalent command printing, first-failure stop, state refresh, and tracking failure propagation.
- [x] 7.6 Add `experiment/run.py` and integration tests for new runs, exact resume, cache hits without child runs, failed tracking after output creation, and sequential multirun.
- [x] 7.7 Prove direct stage scripts without parent context create no orphan MLflow runs.

## 8. Inspection, Reset, Delivery, and Documentation

- [x] 8.1 Add `experiment/inspect.py` for stages, methods, datasets, profiles, configs, and ablations, with no recipe kind.
- [x] 8.2 Add `experiment/reset.py` as the only destructive named-run operation with strict path containment and mode-independent cleanup.
- [x] 8.3 Update delivery collection and report/export tools to consume typed run state and preserved aggregate/local artifact contracts.
- [x] 8.4 Update README, operations commands, active plans, and examples to the Hydra entrypoints and document the old-to-new command mapping and MLflow UI startup.
- [x] 8.5 Add inspection, reset-containment, delivery, and active-documentation contract tests.

## 9. Legacy Removal

- [x] 9.1 Delete all legacy JSON experiment/method/profile/search-space configs and JSON-only config codec/converter/patch/loader code.
- [x] 9.2 Delete the positional/subcommand `scripts/experiment.py`, recipe discovery/listing, custom run-root, force, no-cache, and legacy CLI tests/adapters.
- [x] 9.3 Delete migrated scientific argparse defaults, wide/generic stage config wrappers, and legacy generated JSON stage config paths.
- [x] 9.4 Delete `scripts/workflow/` after all typed planning/state/status/execution ownership has moved into `graph_memory/experiment/`.
- [x] 9.5 Delete duplicate per-stage run-summary lifecycle boilerplate and observability forwarding APIs not used by typed summaries or MLflow.
- [x] 9.6 Run residual scans proving forbidden JSON loaders/configs, argparse defaults, recipes, `_target_` runtime construction, duplicate workflow registries, compatibility adapters, and old command examples are gone.

## 10. Full Verification

- [x] 10.1 Run full pytest, Ruff, basedpyright error-level, compileall, `git diff --check`, and strict OpenSpec validation.
- [x] 10.2 Run a locked real-Python-3.10 Hydra composition and MLflow SQLite verification suite.
- [x] 10.3 Run HotpotQA smoke across all eight registry methods, using the dedicated Memory Stream fixture/artifact where required.
- [x] 10.4 Run default HotpotQA quick across the approved seven methods and verify predictions plus main/path/efficiency tables.
- [x] 10.5 Run complete supported-method workflow smoke for 2WikiMultiHopQA and MuSiQue.
- [x] 10.6 Run R-GCN ablation, interrupted-resume, cache-disabled rerun, and sequential Hydra multirun workflows.
- [x] 10.7 Compare deterministic artifacts exactly and trainable metrics against declared tolerances; record any intentional differences.
- [x] 10.8 Re-run delivery collection and verify local outputs remain sufficient without MLflow artifact access.
