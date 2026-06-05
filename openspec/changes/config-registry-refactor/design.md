## Context

The accepted core package refactor has already split most domain ownership, but the configuration and composition boundary still trails that package layout. Public script parsers, config-file/profile layering, retrieval method metadata, method-family builder dispatch, workflow planning, ablation variants, and training config helper slicing are still distributed across scripts, `graph_memory.retrieval`, `graph_memory.retrieval_registry`, `scripts.workflow`, and model-specific helpers.

The main compatibility constraint is strict behavior preservation at the repository boundary: script commands, CLI flags, workflow manifests, artifact schemas, public method names, ranking semantics, tuning behavior, and training outputs must continue to work while internal config and registry ownership changes.

## Goals / Non-Goals

**Goals:**

- Establish `graph_memory.config` as the mechanism-only layer for loading, merging, structuring, and serializing stage-root configs.
- Establish `graph_memory.registry` as the application composition boundary for stage specs, method ids, settings unions, builder maps, projections, and ablation patches.
- Keep scripts thin: select a stage config, load artifacts, validate artifacts, call a fixed stage runner or domain service, write outputs, and write summaries.
- Use method-specific settings unions instead of optional method-family bags.
- Move retrieval metadata source-of-truth from retrieval catalog/root registry ports into `graph_memory.registry` while keeping compatibility projections.
- Add typed pair, train, retrieve, and evaluate stage roots without changing external CLI contracts.
- Make workflow manifest generation prefer typed stage config projections while preserving old manifest JSON readability.
- Introduce schema v2 method config files without breaking existing training config paths.

**Non-Goals:**

- Do not change public CLI flag names, choices, defaults, or output artifact shapes.
- Do not change retrieval ranking, graph-rerank scoring, tuning objective/tie-break behavior, or trainable checkpoint semantics.
- Do not introduce dynamic plugin discovery, a dependency injection container, or a generic workflow engine.
- Do not move model internals unless required to expose a narrow registry builder boundary.
- Do not require YAML in this change; the codec boundary should merely keep YAML possible later.

## Decisions

### Decision: Loader takes a stage spec and argv

`ConfigLoader.load(spec, argv)` is the only public loader entrypoint. The loader gets all stage-specific knowledge from `StageConfigSpec`: parser factory, config path resolution, profile selection, CLI patching, and registry patching. The loader itself owns the fixed layer order: base config without `profiles`, selected profile patch, registry patch, then CLI patch last.

Alternative considered: expose smaller helpers such as `load_profiled_file()` or `load_cli_config()`. That would make callers learn intermediate states and would preserve the current spread of config layering decisions.

### Decision: Registry owns dispatch, stages own orchestration

`graph_memory.registry` owns method ids, settings unions, capability metadata, and settings-type-to-builder maps. Stage runners are normal functions under `graph_memory/stages/`; registry does not expose `Registry.stages.<stage>.run()`.

Alternative considered: make registry execute stages directly. That would mix declaration/lookup with artifact orchestration and recreate a central workflow engine.

### Decision: Keep compatibility projections narrow

`graph_memory.retrieval.catalog` and `graph_memory.retrieval_registry` remain import-compatible during migration, but they become projections over registry-owned metadata. Legacy fields such as `builder_id` may remain in the projection while not acting as the new runtime dispatch source.

Alternative considered: delete compatibility ports immediately. That would force broad workflow/doc/test edits before the typed registry boundary is fully in place.

### Decision: Use stage-root configs, not method-family option bags

Each script/workflow stage gets a root config with `io` and `job` sections. Retrieval and training jobs are discriminated unions of method-specific settings. BM25 settings do not carry dense fields; checkpoint-backed settings do not carry graph-rerank fields.

Alternative considered: keep one wide config with optional `dense`, `graph_rerank`, and `trainable` branches. That would keep the old method-family conditional shape under a new name.

### Decision: Migrate in behavior-preserving slices

The migration is split into loader/specs, retrieve dispatch, pair/train/evaluate configs, typed workflow projection, config schema cleanup, and old helper removal. Each slice must add focused tests before production edits and preserve the full validation gate.

Alternative considered: rewrite all scripts and config files in one pass. That would make CLI/workflow regressions harder to isolate.

## Risks / Trade-offs

- [Risk] Compatibility projections can accidentally become a second source of truth. -> Mitigation: add tests that projection values are generated from registry metadata and architecture tests that forbid independent production registries.
- [Risk] Typed `Path` serialization can change run-summary path strings on Windows. -> Mitigation: keep artifact schemas tested and compare paths semantically where workflow status needs path equality.
- [Risk] Direct script config precedence can diverge from workflow-resolved config precedence. -> Mitigation: add direct CLI tests for pairs/train/evaluate and keep CLI patch last in `ConfigLoader`.
- [Risk] Training migration can expose large model-specific helper assumptions. -> Mitigation: introduce `Registry.training.build(config.job, deps)` behind focused stage tests before deleting old helpers.
- [Risk] Schema v2 can strand existing `configs/training/.../base.json` users. -> Mitigation: keep old path support through a migration/alias adapter and ensure scripts/stages do not know which path was used.

## Migration Plan

1. Freeze public parser contracts and record a validation baseline.
2. Add `graph_memory.config` and `StageConfigSpec`/stage config registry roots.
3. Add retrieval method settings unions, settings-type builder registry, and retrieve stage runner.
4. Migrate retrieval metadata source-of-truth to `graph_memory.registry` and downgrade old catalog/root registry modules to projections.
5. Migrate `scripts/run_retrieval.py` to typed retrieve stage config while preserving artifact IO, validation, and summaries.
6. Migrate pair-build, train, and evaluate scripts to typed stage configs and introduce the training builder registry.
7. Move workflow manifest planning to typed stage config projections and registry-owned ablation patches.
8. Add schema v2 method configs and compatibility for existing config paths.
9. Remove old method-string dispatch and old dict-slicing helper calls after focused tests and residual searches pass.

Rollback is by slice: keep compatibility projections and old config path adapters until the next slice has passing focused tests and the full gate is green.

## Open Questions

None for the current plan. Dense fine-tuning remains a future method family used only to validate that `stages/train.py` does not need method-specific branches.
