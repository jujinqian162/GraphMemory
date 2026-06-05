## Why

The current configuration and retrieval-construction path still spreads stage parsing, config layering, method metadata, and runtime dispatch across scripts, retrieval catalog/factory/resolver modules, and workflow helpers. This makes CLI precedence, profile resolution, and new method registration hard to reason about as the repository moves toward typed package boundaries.

This change introduces a stage-root, typed, registry-driven configuration and builder boundary while preserving public script commands, workflow behavior, artifact schemas, and retrieval/training semantics.

## What Changes

- Add a single `ConfigLoader.load(StageConfigSpec, argv)` entrypoint that owns config-file reading, profile patching, registry patching, CLI overrides, typed structure, and resolved-config serialization.
- Add `graph_memory.registry` as the source of truth for stage config specs, public method ids, method settings unions, builder maps, workflow-facing projections, and ablation patches.
- Migrate retrieval execution so scripts choose only their stage config, retain artifact IO/validation/run summary responsibilities, and call fixed stage runner functions.
- Migrate pair-build, train, and evaluate scripts to typed stage root configs without changing their external parser contracts.
- Move workflow manifest planning toward typed stage config projections while keeping existing manifest JSON readable.
- Add config schema v2 support with shallow method config files and compatibility for the existing training config path.
- Remove old method-string dispatch and legacy dict-slicing helpers once typed configs and registry builders cover the affected paths.
- No breaking public CLI, workflow command, artifact schema, method-name, ranking, tuning, or training behavior changes are intended.

## Capabilities

### New Capabilities

- `stage-config-registry`: Stage-root typed config loading, registry-owned method/stage metadata, and registry-driven builder/projection boundaries.

### Modified Capabilities

None.

## Impact

- Affected production areas: `graph_memory/config/`, `graph_memory/registry/`, `graph_memory/stages/`, `scripts/run_retrieval.py`, `scripts/build_train_pairs.py`, `scripts/train_graph_retriever.py`, `scripts/evaluate_retrieval.py`, `scripts/workflow/manifest.py`, `scripts/workflow/workflows.py`, `scripts/workflow/registry.py`, retrieval catalog/factory/resolver compatibility paths, training config helpers, and config files under `configs/`.
- Affected tests: CLI contract tests, config loader/stage registry tests, retrieval registry builder/projection tests, run-retrieval smoke tests, pair/training/evaluation tests, experiment runner/workflow orchestration tests, config schema migration tests, and architecture boundary tests.
- Validation gates: focused pytest slices, full pytest, `uv run basedpyright --outputjson --level error`, `uv run ruff check .`, and `openspec validate --all --strict`.
