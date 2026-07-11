## 1. Freeze the Correction Boundary

- [x] 1.1 Record the current accepted plan identifiers, stage configs, inputs, outputs, aliases, summaries, aggregate schemas, and deterministic outputs for the default HotpotQA quick workflow before deleting any structure.
- [x] 1.2 Record focused parity fixtures for all-eight-method smoke, 2Wiki, MuSiQue, Memory Stream importance splits, hidden Dense-FT training dependency, R-GCN ablations, interrupted resume, cache-disabled rerun, and multirun identity.
- [x] 1.3 Add a production call-site inventory test/helper that distinguishes real callers from test-only callers for every optional/default parameter named by this change.
- [x] 1.4 Add failing structural tests for the five `experiment/*.py` files, flattened `configs/` root, canonical `config.yaml`, fixed `runs/.mlflow/` paths, and absence of `_base.yaml`, root presets, command configs, `mlruns/`, and package-override defaults.
- [x] 1.5 Mark `refactor-experiment-config-workflow` as superseded by this corrective change in its handoff/status documentation without changing its historical artifacts.

## 2. Flatten and Minimize Configuration

- [x] 2.1 Move the Hydra authoring root from `configs/experiment/` to `configs/` and update composition/tests to load `configs/config.yaml` directly.
- [x] 2.2 Merge `_base.yaml` and the forwarding `config.yaml` into one canonical root and delete `_base.yaml`.
- [x] 2.3 Replace per-method package override defaults with one Hydra method-config multi-select that resolves all eight methods under `method_configs.<method>`.
- [x] 2.4 Delete `2wiki_tiny.yaml`, `2wiki_rgcn_ablation_7methods.yaml`, `hotpotqa_dev_full.yaml`, and `hotpotqa_rgcn_ablation_selected.yaml`; add copyable CLI equivalents to active operations documentation.
- [x] 2.5 Replace `hotpotqa_memory_stream.yaml` with a complete importance-backed dataset option that does not select methods or profile values.
- [x] 2.6 Audit profile files against production workflows and delete aliases or profiles without a distinct retained scale policy.
- [x] 2.7 Collapse the one-option tracking group into the root config and consolidate search spaces into the smallest group structure that still represents real alternatives.
- [x] 2.8 Keep a graph config group only if at least two production-selectable graph configurations remain; otherwise place the single graph configuration in the canonical root.
- [x] 2.9 Remove duplicate `seed_method` values from graph-rerank method YAML and remove all other runtime lifecycle/dependency metadata from authored config.
- [x] 2.10 Remove dead config fields `dataset.adapter`, `render_training_curves`, and aggregate `input_dir`, updating resolved-config parity fixtures for these intentional structural deletions.
- [x] 2.11 Set fixed tracking values to `runs/.mlflow/tracking.db` and `runs/.mlflow/artifacts/` and verify absolute resolution from the repository root.
- [x] 2.12 Add residual tests proving no `configs/experiment`, `_base`, deleted root preset, command YAML, one-option tracking group, package override, duplicate seed method, or dead field remains.

## 3. Restore the Public Interface

- [x] 3.1 Create thin `experiment/plan.py` and `experiment/run.py` Hydra entrypoints using `configs/config.yaml` and one reusable initialization service call.
- [x] 3.2 Create thin `experiment/status.py`, `experiment/inspect.py`, and `experiment/reset.py` entrypoints with shared narrow `key=value` parsing into closed Pydantic command models.
- [x] 3.3 Delete `status_command.yaml`, `inspect_command.yaml`, and `reset_command.yaml` and prove the three commands create no Hydra output directories.
- [x] 3.4 Remove or make internal the module-entrypoint implementations under `graph_memory/experiment/{plan,run,status,inspect,reset}.py` so there is one documented public command path and no compatibility facade.
- [x] 3.5 Add subprocess tests for all five top-level file commands, including missing required values, method overrides, multirun job selection, reset containment, and inspect catalog output.

## 4. Simplify Root Config Validation and Persistence

- [x] 4.1 Narrow composed-config validation to the actual Hydra `DictConfig` input and delete the unused Mapping conversion branch and redundant root-object check.
- [x] 4.2 Delete the recursive unresolved-value scan and prove `OmegaConf.to_container(..., throw_on_missing=True)` plus Pydantic reject every missing mandatory value.
- [x] 4.3 Replace repeated device validators with one reusable typed device field contract.
- [x] 4.4 Replace `_resolve_method_paths()` full model dump/revalidation with direct typed path resolution or config interpolation that changes only path-owning fields.
- [x] 4.5 Remove `validate_composed_config` from the public package export surface if it has no external production caller and update tests to compose through the public service boundary.
- [x] 4.6 Narrow `read_yaml_model` to model classes, remove the unused TypeAdapter alternative, and inline/remove one-line persistence wrappers that own no invariant.
- [x] 4.7 Replace broad persistence `Any` return/value annotations with the repository's closed JSON/YAML value contract.

## 5. Replace Wide Optional Models

- [x] 5.1 Split raw and importance dataset split configs so labels and importance bindings are required only by the importance variant and absent from raw variants.
- [ ] 5.2 Split BM25 graph-rerank and Dense graph-rerank tune configs so a Dense encoder is required only by the Dense variant.
- [ ] 5.3 Split ordinary R-GCN and Dense-FT-seeded R-GCN train configs so the seed model directory is required only by the seeded method.
- [x] 5.4 Split ordinary and ablation aggregate configs so ablation index, output, and typed selections are either all required or all absent.
- [ ] 5.5 Split normal and alias artifact references so alias source is required by the alias variant and absent from ordinary artifacts.
- [ ] 5.6 Model running, successful, and failed stage-summary states so terminal timestamp and error fields are required or absent according to status.
- [ ] 5.7 Replace persisted `dict[str, Any]` effective-config/observation fields with closed Pydantic or recursive JSON-value contracts.
- [ ] 5.8 Add negative tests proving invalid half-configurations and string/bool coercions fail at the typed boundary without downstream `None` checks.

## 6. Establish One Method and Ablation Authority

- [ ] 6.1 Delete `ExperimentMethodSpec`, `ExperimentMethodRegistry`, `build_experiment_method_registry`, its single-use constant, unused string-input handling, and the impossible derived-order check.
- [ ] 6.2 Move/retain train-dependency traversal on the existing runtime method registry without copying lifecycle, graph, tuning, artifact-kind, seed, or dependency fields.
- [ ] 6.3 Extend the existing ablation registry as needed so each executable variant owns its typed config patch and earliest invalidated stage.
- [ ] 6.4 Replace planner variant-name conditionals with application of the registered typed ablation patch.
- [ ] 6.5 Add registry-boundary tests for all eight methods, hidden Dense-FT dependency, file/directory checkpoint kinds, graph-rerank seed methods, full-R-GCN alias, and every executable `wo_*` variant.

## 7. Persist One Stage Execution Contract

- [ ] 7.1 Define one closed persisted stage execution contract containing identity, typed payload, absolute script/config/summary paths, declared inputs, outputs, dependency identifiers, method/split/variant identity, and artifact kinds.
- [ ] 7.2 Make planning, formatting, persistence, subprocess execution, lifecycle summary creation, status, and resume consume that contract directly.
- [ ] 7.3 Standardize every stage on one top-level summary-path field and delete dynamic `summary` versus `outputs.summary` discovery.
- [ ] 7.4 Delete `stage_cli._artifact_bindings()` and the second `StageInvocation` reconstruction path; direct scripts shall load the persisted contract without fabricating empty dependencies.
- [ ] 7.5 Delete the generic recursive absolute-path walker and enforce path invariants in the typed execution contract.
- [ ] 7.6 Make script path required at the contract construction boundary and delete the `sys.argv[0]` fallback.
- [ ] 7.7 Remove `_external`'s unused kind parameter and narrow layout/method helpers to repository enums used by production callers.
- [ ] 7.8 Replace `RunLayout`'s optional single/multirun field fallbacks with explicit validated construction that never uses `override_dirname or ""` after validation.
- [ ] 7.9 Add round-trip tests proving every planned stage contract reads back identically in its direct script for all stage and method variants.

## 8. Simplify Planner, Status, Resume, and State Ownership

- [ ] 8.1 Reduce `WorkflowPlanner` to workflow selection, stable ordering, range selection, and dependency coordination; move or delete method-specific projection code so it does not remain a monolithic second registry.
- [ ] 8.2 Delete planner `_invocation_state()` and `_stage_summary_path()` copies and use the typed status authority for prerequisite validation.
- [x] 8.3 Narrow status summary comparison to `StageRunSummary` and remove the impossible post-validation `isinstance` branch.
- [x] 8.4 Remove the default from `prune_completed_prefix(cache_enabled)` and update every production and test call to pass the policy explicitly.
- [x] 8.5 Remove the optional index/default output mode from invocation formatting and require an explicit execution-order index.
- [x] 8.6 Remove unused lifecycle summary-path and MLflow-child-id parameters, the test-only summary hook Protocol, and its optional publication branch.
- [x] 8.7 Make corrupt existing summaries explicit stale/corrupt inputs and delete the silent attempt-1 fallback.
- [x] 8.8 Stop rewriting `RunState` after each stage when only `updated_at` changes; persist only creation, parent-id changes, and meaningful terminal metadata changes.
- [ ] 8.9 Re-run frozen missing/complete/stale/alias, directory checkpoint, failed residual, external prerequisite, and completed-prefix resume fixtures against the single status path.

## 9. Remove Legacy Script Carriers and Dynamic Field Discovery

- [ ] 9.1 Delete `PrepareHotpotQAArgs`, `PrepareTwoWikiArgs`, and `PrepareMuSiQueArgs`; make preparation functions consume typed stage variants or explicit required domain arguments.
- [ ] 9.2 Delete unreachable prepare branches for absent `max_examples`, absent combined output, and string-based source discrimination while preserving raw invalid-example policy and Memory Stream importance behavior.
- [ ] 9.3 Replace retrieval `getattr` discovery of graphs, selected config, importance, checkpoint, and model directory with direct access on discriminated retrieval variants.
- [ ] 9.4 Remove test-only provider parameters from `scripts/train_method.main` and patch/test lower trainer/provider boundaries instead.
- [x] 9.5 Delete aggregate dead `input_dir`, `_looks_like_metric_file`, and the unused default selection argument; consume the ordinary or ablation aggregate contract directly.
- [ ] 9.6 Remove every remaining dead field, dead helper, no-op branch, unused union alternative, and forwarding wrapper identified by the production call-site audit.
- [ ] 9.7 Add direct-script integration tests for prepare, graph, pairs, tune, R-GCN train, Dense-FT train, every retrieval family, evaluate, ordinary aggregate, and ablation aggregate after the cleanup.

## 10. Simplify Execution and Typed Ablation Flow

- [x] 10.1 Remove the test-only optional tracking adapter from the public execution signature and assemble tracking at the application composition boundary; rewrite failure tests by patching that boundary.
- [x] 10.2 Delete the no-op outer `try/except BaseException: raise` and preserve only exception handling that changes summary or MLflow state.
- [x] 10.3 Replace string `method=variant` ablation selections with a typed record across plan, execution, index persistence, stage YAML, and aggregate.
- [x] 10.4 Delete repeated partition/separator validation for repository-generated ablation selections and retain string parsing only at CLI/display input.
- [ ] 10.5 Verify first-failure stop, child/parent termination, failed-summary handling, cache-hit child suppression, and exact resume after execution simplification.

## 11. Repair MLflow Observability

- [x] 11.1 Migrate tracking initialization and documentation from `mlruns/` to `runs/.mlflow/tracking.db` and `runs/.mlflow/artifacts/` with no fallback path.
- [ ] 11.2 Preserve one parent per Hydra job, one child per executed stage attempt, parent reuse on exact resume, and no child creation for cache hits.
- [ ] 11.3 Define stable MLflow metric keys that include method and variant identity for evaluation and multi-row aggregate tables.
- [ ] 11.4 Log every numeric row from main, path, efficiency, and ablation result tables instead of skipping CSV files with more than one row.
- [ ] 11.5 Preserve curated upload of resolved config, overrides, summaries, selected configs, candidate tables, final tables, report images, and bounded failure summaries.
- [ ] 11.6 Prove datasets, graphs, pairs, predictions, checkpoints, and model directories remain metadata-only and local artifacts remain sufficient for offline delivery.
- [ ] 11.7 Add cross-name and sequential-multirun MLflow integration tests against the shared SQLite store, including tracking-write failure propagation.

## 12. Rewrite Tests and Active Documentation

- [ ] 12.1 Rewrite tests that directly require `configs/experiment`, `_base`, `2wiki_tiny`, command YAML, public `validate_composed_config`, lifecycle hooks, optional script fallback, or injected tracking/provider parameters.
- [ ] 12.2 Add forbidden-surface scans for deleted public module commands, root presets, package overrides, duplicate experiment registry, duplicate binding/status helpers, dynamic retrieval `getattr`, legacy Prepare Args, broad persisted Any, no-op exception blocks, and audited unused defaults.
- [ ] 12.3 Update README and active overview/design/operations/config documentation to the five top-level files, flattened config tree, copyable override examples, fixed MLflow paths, and single ownership model.
- [ ] 12.4 Remove stale active command examples for deleted JSON configs, `scripts/experiment.py`, module-only commands, root presets, and `mlruns/`; retain historical references only where clearly marked as superseded history.
- [x] 12.5 Document why `???` is allowed only for the required root experiment name and why narrow commands no longer need Hydra command configs.

## 13. Full Verification

- [ ] 13.1 Run config composition and resolved-config parity for every retained dataset, profile, method, search space, root override, and Memory Stream input variant.
- [ ] 13.2 Run plan parity for default seven methods, all eight methods, 2Wiki, MuSiQue, hidden Dense-FT dependencies, stage ranges, ablations, cache-disabled execution, and multirun.
- [ ] 13.3 Run the complete direct-stage subprocess integration matrix and verify local artifact, checkpoint metadata, prediction, failure-case, metric, and aggregate schemas.
- [ ] 13.4 Run status/resume/alias/corrupt-summary regression tests and prove no cache feature or semantics were added.
- [ ] 13.5 Run full pytest, Ruff, basedpyright error-level, compileall, `git diff --check`, and strict OpenSpec validation.
- [ ] 13.6 Run the locked Python 3.10 Hydra/Pydantic/MLflow verification suite from a clean environment.
- [ ] 13.7 Run real HotpotQA all-eight smoke and default seven-method quick workflows, plus complete supported-method smoke for 2Wiki and MuSiQue.
- [ ] 13.8 Run real Memory Stream, R-GCN ablation, interrupted resume, cache-disabled rerun, sequential multirun, MLflow cross-run comparison, and offline delivery workflows.
- [ ] 13.9 Compare deterministic outputs exactly and trainable results against the existing declared tolerances; record and approve any intentional structural difference before marking this change complete.
- [ ] 13.10 Run final residual call-site and filesystem scans proving every audit-listed dead parameter, default, branch, field, helper, preset, duplicate authority, and wrong public/tracking path is gone.
