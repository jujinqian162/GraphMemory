## Context

`refactor-experiment-config-workflow` replaced the legacy JSON/argparse workflow with Hydra, Pydantic, typed stage YAML, local run summaries, and MLflow. Its scientific workflows are substantially functional: the current checkout has passing full tests and successful artifacts for the eight-method HotpotQA smoke, default seven-method quick run, cross-dataset workflows, ablation, and resume exercises.

The implementation nevertheless violates the governing `REFACTOR-7-9` goals in two ways. First, it changes locked public/config/tracking contracts: it exposes module entrypoints rather than `experiment/*.py`, nests all author configs under `configs/experiment/`, uses `_base.yaml` and several root presets, restores the rejected Hydra package-override syntax, and stores MLflow under `mlruns/` rather than `runs/.mlflow/`. Second, it optimizes for theoretical reuse and test convenience instead of minimum readable production code.

The read-only audit confirmed the following implementation debt that this change must remove rather than document as acceptable:

- `config.yaml` is an empty forwarder to `_base.yaml`; `_base.yaml` exists only to support root presets that ordinary CLI overrides can express.
- `2wiki_tiny`, both R-GCN ablation roots, and `hotpotqa_dev_full` add no capability. `hotpotqa_memory_stream` mixes dataset shape, profile, method selection, and artifact binding in one root preset.
- `tracking/default.yaml` is a one-option config group, graph parameters are embedded in the root despite the earlier group design, and method configs duplicate `seed_method` already owned by the runtime registry.
- `DatasetConfig.adapter`, `TrainingReportingConfig.render_training_curves`, `AggregateStageConfig.input_dir`, and `_looks_like_metric_file()` have no production consumer.
- `initialize_from_hydra.repository_root`, `stage_lifecycle.summary_path`, and `stage_lifecycle.mlflow_child_run_id` have no caller that supplies a value. `stage_lifecycle.hook`, `execute_experiment.tracking`, and several script provider parameters exist only for tests.
- `prune_completed_prefix.cache_enabled=True`, `format_invocation.index=None`, `_external.kind="file"`, `_indexed_ablation_rows.selection_values=()`, and `load_stage_execution.script=None` retain defaults or alternatives that production callers do not use.
- `validate_composed_config` accepts a `Mapping` path no caller uses, checks again that a known root `DictConfig` became a dict, and recursively searches for `???` after `OmegaConf.to_container(..., throw_on_missing=True)` already rejects missing values.
- `read_yaml_model` accepts a `TypeAdapter` path no caller uses; `write_overrides_atomic` and tracking's `_summary_path` are one-line forwarding wrappers without a distinct contract.
- `execute_experiment` contains an outer `try/except BaseException: raise`, rewrites `RunState` after every stage even though only `updated_at` changes, and then rewrites it again after the loop.
- `RunLayout` validates multirun identity and then still falls back through `override_dirname or ""`; its helpers accept `str | RetrievalMethodId` although production callers use the enum.
- `ExperimentMethodRegistry` copies the existing runtime registry, accepts unused string inputs, exposes a factory used once, and validates an order that is guaranteed because both sides are produced from the same source iteration.
- method and ablation facts are duplicated: config YAML contains seed-method metadata, `ExperimentMethodSpec` copies runtime metadata, and `_apply_rgcn_ablation` hard-codes patches instead of consuming the registered override.
- `planning.py` duplicates status/summary checks and owns stage projection, method dispatch, dependencies, ablation, aliasing, range selection, artifact binding, path binding, and formatting in one large module.
- `stage_cli.py` reconstructs a second incomplete `StageInvocation` through a large `isinstance` tree, repeats every stage's artifact bindings, assigns empty dependencies, and performs a hand-written recursive absolute-path validation.
- summary paths have two shapes (`summary` and `outputs.summary`), causing duplicate dynamic `getattr` lookup in planning and state.
- status accepts a summary as `object` and performs an impossible runtime type check after `read_stage_summary()` has already validated it.
- `_next_attempt()` silently converts a corrupt existing summary into attempt 1 rather than surfacing corruption.
- prepare scripts convert Pydantic stage configs back into legacy `Prepare*Args` dataclasses. Required `count` and `combined` values become optional again, leaving unreachable `None` branches.
- retrieval still uses `getattr(config, "graphs", None)` and `getattr(config, "selected_config", None)` despite having a method-discriminated config union.
- raw versus importance splits, BM25 versus Dense tuning, ordinary versus seeded R-GCN, ordinary versus ablation aggregation, and ordinary versus alias artifacts use wide optional fields instead of contracts that make each variant complete.
- `StageRunSummary` and observation records retain broad `dict[str, Any]` fields.
- ablation selections are serialized to `method=variant` strings, parsed again during execution, then parsed again during aggregation, creating validation branches for strings that repository code itself just generated.
- MLflow aggregate logging skips every multi-row results CSV, so the normal seven-method tables are artifacts but not queryable metrics.
- tests currently freeze several accidental shapes (`configs/experiment`, `2wiki_tiny`, public `validate_composed_config`, lifecycle hooks, injected tracking adapters) instead of only testing public behavior and ownership boundaries.

Constraints for the correction are strict:

- preserve verified scientific results and artifact schemas;
- preserve artifact-plus-summary cache truth and completed-prefix resume without adding cache features;
- preserve Hydra only where composition is useful, Pydantic at external boundaries, subprocess isolation, and MLflow as a strict mirror;
- introduce no compatibility adapters, alternate config paths, recipe layer, fallback backend, content hash, or parallel launcher;
- support Python 3.10.

## Goals / Non-Goals

**Goals:**

- Make the authored config tree obvious from `configs/` with one canonical root and only meaningful choice groups.
- Restore the exact five public file entrypoints and keep their implementations thin.
- Reduce production APIs to parameters and modes exercised by real production callers.
- Replace wide optional models with complete discriminated variants where absence changes the legal shape.
- Give method metadata, ablation patches, stage invocation bindings, summary paths, and status comparison one owner each.
- Remove dead fields, dead functions, unreachable branches, silent fallbacks, duplicate conversions, and legacy config carriers.
- Keep direct stage execution and orchestration on one persisted typed execution contract.
- Move MLflow to the locked repository-wide location and make normal aggregate results queryable.
- Keep the current verified workflow and local-artifact behavior unchanged.

**Non-Goals:**

- Adding methods, datasets, metrics, cache features, recipes, launchers, remote tracking, or new experiment modes.
- Replacing Hydra, Pydantic, MLflow, subprocess execution, or local artifacts.
- Redesigning retrieval, graph construction, evaluation, training algorithms, or scientific parameter values.
- Keeping old root presets, module-only public commands, JSON config compatibility, or test-only production injection points.
- Removing every `None` mechanically. Optionality that represents a real runtime state, such as multirun job selection, new-versus-resumed MLflow parent, failure error data, or ordinary-versus-ablation identity, remains where a discriminated model would not improve clarity.

## Decisions

### 1. One config root with no `_base` or root preset system

The authored tree is:

```text
configs/
  config.yaml
  dataset/
    hotpotqa.yaml
    2wiki.yaml
    musique.yaml
    hotpotqa-memory-stream.yaml   # only if the importance-backed split remains necessary
  profile/
    <retained profiles only>
  method_configs/
    <one complete config per public method>
  graph/                         # only if more than one real graph choice remains
    default.yaml
  search_spaces/
    default.yaml
```

`config.yaml` owns root defaults and Hydra run/sweep behavior directly. It uses Hydra multi-select for all method configs and does not use per-method package override entries. `tracking` is inline because the change supports exactly one backend/store policy. A graph group is retained only if two real choices are exercised; otherwise graph values are inline.

The following root presets are deleted: `2wiki_tiny`, `2wiki_rgcn_ablation_7methods`, `hotpotqa_dev_full`, and `hotpotqa_rgcn_ablation_selected`. Their documentation becomes explicit copyable overrides. The necessary Memory Stream split binding, if retained, is a dataset option rather than a root preset because it changes the input contract.

Alternative considered: keep `_base.yaml` as a conventional Hydra composition layer. Rejected because it exists only to share content between non-required presets and makes the canonical root invisible.

### 2. Hydra composes experiments, not narrow read-only commands

`experiment/plan.py` and `experiment/run.py` are the only Hydra composition entrypoints. `experiment/status.py`, `experiment/inspect.py`, and `experiment/reset.py` accept the same `key=value` spelling but parse a narrow dot-list into closed Pydantic command models without starting a Hydra job or requiring command YAML files. They create no Hydra output directory.

`???` remains valid only in the canonical Hydra experiment root for the required experiment name. Command YAML files and their repeated temporary-output Hydra blocks are removed.

Alternative considered: retain Hydra on all five commands. Rejected because status, inspect, and reset compose no experiment groups, and the extra jobs/config files are pure machinery.

### 3. Public adapters are files; reusable logic stays under `graph_memory`

The public surface is exactly:

```text
experiment/plan.py
experiment/run.py
experiment/status.py
experiment/inspect.py
experiment/reset.py
```

These files contain entrypoint decoration/parsing and one service call. `graph_memory/experiment/` contains reusable types and behavior but its modules are not documented as the primary CLI. No dual public path or compatibility wrapper is kept.

Alternative considered: continue documenting `python -m graph_memory.experiment.*`. Rejected because it contradicts the locked public contract and exposes internal package layout.

### 4. Production signatures follow production callers

Every function changed by this correction receives a call-site audit. A parameter remains optional or defaulted only when at least two production calls exercise the different modes or the domain state genuinely permits absence.

Specific locked removals include:

- `initialize_from_hydra.repository_root`;
- the Mapping branch, redundant root-shape check, and redundant unresolved-value walk in composed config validation;
- `prune_completed_prefix`'s default cache flag;
- `format_invocation`'s optional index;
- `stage_lifecycle.summary_path`, `stage_lifecycle.mlflow_child_run_id`, and its test-only hook protocol;
- `load_stage_execution`'s optional script fallback;
- `read_yaml_model`'s unused adapter alternative;
- `_external`'s unused kind parameter;
- `_indexed_ablation_rows`' unused default;
- forwarding helpers without a distinct invariant;
- the no-op outer execution exception block;
- test-only tracking/provider injection from public production entry functions.

Tests inject at lower domain boundaries or patch constructors/factories rather than expanding production signatures.

Alternative considered: keep dependency-injection seams because they make tests easy. Rejected because they are not application composition points and materially increase the public control flow the user must read.

### 5. One runtime registry and one ablation patch authority

The existing `Registry.methods` remains the method authority. The duplicate `ExperimentMethodRegistry`, projection dataclass, single-use builder, duplicate seed-method YAML fields, unused string alternatives, and impossible order check are removed. Any experiment-specific dependency traversal consumes the existing typed definitions directly.

`graph_memory.registry.ablations` owns executable variant identity, changed dimensions, invalidated stage, and the typed config patch. Planning applies the registered patch; it does not switch on variant names to recreate the patch.

Alternative considered: keep a derived experiment registry for isolation. Rejected because it copies every relevant field and supplies no independent invariant.

### 6. Persist and consume one complete stage execution contract

The planner builds the single typed stage execution contract used for plan display, status, resume, persistence, and subprocess execution. The persisted YAML includes complete identity, script/config payload, declared inputs, outputs, dependency identifiers, and the single summary path. Direct scripts validate and consume this contract rather than reconstructing an invocation from method-specific `getattr` and `isinstance` branches.

Every stage uses the same top-level summary-path field. Artifact paths are validated by the typed contract; `stage_cli.py` does not recursively traverse arbitrary Pydantic objects. A direct script receives a complete contract, not a `StageInvocation` with fabricated empty dependencies.

Alternative considered: keep config-only YAML and a shared `_artifact_bindings()` reconstruction helper. Rejected because the planner and direct script then own separate binding maps and can drift.

### 7. Discriminated models replace wide optional models

The following variants are explicit:

- raw split versus importance-backed split;
- BM25 graph-rerank tune versus Dense graph-rerank tune;
- ordinary R-GCN train versus Dense-FT-seeded R-GCN train;
- ordinary aggregate versus ablation aggregate;
- normal artifact versus alias artifact;
- running, successful, and failed stage summaries where status controls required timestamps/error data.

Method, dataset, split, artifact kind, and stage fields use repository enums/literals rather than broad strings. Stage observations use a closed JSON-value contract or explicit typed observation fields; broad `Any` does not cross the persisted boundary.

Alternative considered: retain `Path | None` because Pydantic can validate it. Rejected because the absence changes which other fields are required and produces repeated `if` checks.

### 8. Migrated scripts consume Pydantic contracts directly

`PrepareHotpotQAArgs`, `PrepareTwoWikiArgs`, and `PrepareMuSiQueArgs` are removed. Preparation functions accept the discriminated stage payload or explicit required domain arguments. Required `count` and combined-output paths do not become optional again, and source-kind branching happens once.

Retrieval switches on the validated method variant once and accesses required fields directly. It does not use `getattr` to rediscover optional graphs, selected config, or importance fields.

Aggregate receives either an ordinary or ablation contract. Dead `input_dir`, `_looks_like_metric_file`, unused selection defaults, and half-valid combinations of `ablation_index`/`ablation` are removed. The `render_training_curves` and dataset `adapter` fields are removed unless a production consumer is identified before implementation.

Alternative considered: retain legacy Args as internal domain dataclasses. Rejected because they carry CLI/config concerns and recreate optionality already eliminated by the stage contract.

### 9. State, status, and resume have one comparison path

Status owns invocation-versus-summary comparison. Planning reuses it for external prerequisite checks rather than maintaining `_invocation_state()` and `_stage_summary_path()` copies. A corrupt existing summary is explicitly stale or raises a corruption error; attempt numbering never silently resets to 1.

Run state is written when created, when the MLflow parent id changes, and once after execution if meaningful state metadata changes. It is not rewritten after every stage merely to update a timestamp. Stage status remains live and is not copied into run state.

Alternative considered: preserve per-stage `updated_at` writes as progress heartbeats. Rejected because no reader treats that timestamp as a heartbeat and stage summaries already provide progress.

### 10. Ablation selections are typed, not encoded strings

Planner, execution, aggregate configuration, and index generation share a typed `AblationSelection(method, variant)` contract. Repository code does not serialize to `method=variant`, validate its separator, parse it, serialize it again, and parse it again. String formatting exists only at the CLI/display boundary.

Alternative considered: keep strings because they are easy to put in YAML. Rejected because Pydantic serializes typed records cleanly and the current form creates impossible parser-failure branches.

### 11. Tracking uses the locked store and records comparable results

Tracking is fixed at:

```text
runs/.mlflow/tracking.db
runs/.mlflow/artifacts/
```

Parent/child semantics, strict failure propagation, local artifact authority, and the curated artifact allowlist are preserved. Evaluation and aggregate results are logged under stable metric keys that retain method/row identity; multi-row result tables are not skipped. Tables are still uploaded as curated artifacts.

Alternative considered: retain `mlruns/` because it is conventional. Rejected because the approved repository contract explicitly fixed `runs/.mlflow/` and all run-related state belongs below `runs/`.

### 12. Tests protect behavior, not accidental extension points

Tests compose from `configs/`, execute the five top-level file entrypoints, verify the absence of `_base`, root presets, command configs, package-override syntax, dead fields/functions, duplicate registry/binding helpers, and forbidden optional/default parameters. Existing scientific parity fixtures and real workflow smoke remain.

Tests that currently require public `validate_composed_config`, lifecycle hooks, injected tracking adapters, injected script providers, or `2wiki_tiny` are rewritten around the actual public boundary or lower domain services.

Alternative considered: retain current tests and only simplify untested code. Rejected because several current tests are the only callers keeping the unwanted APIs alive.

## Risks / Trade-offs

- [Removing root presets makes some commands longer] → Document copyable overrides for the few retained workflows; do not reintroduce a saved-command abstraction.
- [Changing the config search root can break relative Hydra defaults] → Migrate composition tests first and require exact resolved-config parity before deleting the old tree.
- [Removing test injection can make heavyweight tests harder] → Patch lower-level model/provider constructors or test domain services directly; keep subprocess smoke small.
- [A single persisted stage contract can be verbose] → Prefer explicit repeated path data in one authoritative cross-process contract over two independent binding registries.
- [Splitting wide models creates more Pydantic classes] → Keep variants only where required fields truly differ; do not create wrappers for naming alone.
- [Removing defensive fallbacks can surface existing corrupt artifacts] → Fail with explicit paths and reset/new-name guidance; never silently reinterpret corruption.
- [Planner simplification can alter stage order or dependency completion] → Compare frozen plan identifiers, inputs, outputs, configs, aliases, and completed-prefix decisions before and after each ownership move.
- [MLflow metric-key changes affect existing UI queries] → Treat the current uncommitted `mlruns/` data as disposable migration output; document the new stable keys and rerun the acceptance workflows.

## Migration Plan

1. Add failing structural tests for the top-level entrypoints, flattened config tree, canonical root, fixed MLflow paths, forbidden root presets/package overrides, and dead config fields.
2. Move configuration to `configs/`, compose exact parity snapshots, replace root presets with documented overrides, and remove command YAML/Hydra jobs for status, inspect, and reset.
3. Add the top-level public entrypoint files and move reusable command parsing/service calls behind them.
4. Introduce the single persisted typed stage execution contract and migrate plan, direct scripts, summaries, status, resume, and tracking to consume it.
5. Remove the duplicate experiment registry and make planning consume runtime method and ablation authorities directly.
6. Replace wide optional models with the locked discriminated variants; remove broad `Any` from persisted config/state boundaries.
7. Remove legacy prepare Args, retrieval `getattr`, dead aggregate/training/dataset fields, dead functions, unused wrappers, optional parameters, defaults, no-op branches, and silent fallbacks identified in the audit.
8. Simplify execution state writes, typed ablation flow, and MLflow aggregate metric logging.
9. Update active documentation and rewrite tests that preserve accidental implementation seams.
10. Run config/plan parity, direct-stage integration, status/resume/ablation tests, full pytest, Ruff, basedpyright, compileall, strict OpenSpec validation, Python 3.10 verification, and the real dataset/method workflow matrix.

Rollback before deletion is a repository revert to the current uncommitted refactor tree. No runtime dual path or compatibility layer is created.

## Open Questions

None. The correction deliberately chooses the smallest single path. If implementation discovers a genuinely required second mode, it must first identify a production caller and update this design rather than add an optional parameter or fallback inline.
