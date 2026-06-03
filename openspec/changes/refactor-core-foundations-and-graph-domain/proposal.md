## Why

The current core package has grown into several large multi-domain modules, making it difficult to audit contracts, validation, graph construction, retrieval inputs, and evaluation behavior independently. This change starts the behavior-preserving refactor described in `docs/10-plans/graph-memory-core-package-refactor-design.md` by freezing regression baselines before moving production code and then migrating only the low-level foundation, dataset/text, graph, and evaluation domains.

## What Changes

- Add regression coverage for CLI parser contracts, workflow planning contracts, and small deterministic domain fixtures before production code is moved.
- Split foundational artifact contracts, validators, IO/run-summary helpers, dataset/text helpers, graph construction helpers, and evaluation helpers into explicit domain packages.
- Preserve all existing CLI argument names, defaults, choices, artifact schemas, scoring semantics, graph construction semantics, metric semantics, and workflow behavior.
- Keep `scripts/workflow/` unchanged and retain only the documented narrow root-level workflow integration ports.
- Defer retrieval factory/context removal, graph rerank/tuning migration, training-pair migration, trainable model reorganization, final old-module deletion, and durable docs promotion to later changes.

## Capabilities

### New Capabilities
- `core-refactor-baseline-freeze`: Regression snapshots and golden fixtures that prove the refactor starts from the current CLI, workflow, and deterministic domain behavior.
- `core-foundation-domain-boundaries`: Behavior-preserving package boundaries for contracts, validation, infrastructure, datasets/text, graph construction, and evaluation.

### Modified Capabilities

## Impact

- Affected production areas: `graph_memory/types.py`, `graph_memory/validation.py`, `graph_memory/io.py`, `graph_memory/observability.py`, `graph_memory/hotpotqa.py`, `graph_memory/splits.py`, `graph_memory/text.py`, `graph_memory/entities.py`, `graph_memory/graphs.py`, `graph_memory/evaluation.py`, and their direct importers.
- Affected tests: parser contract tests, workflow planning tests, deterministic fixture tests, validation tests, IO/run-summary tests, graph tests, and evaluation tests.
- Public CLI, workflow commands, artifact schemas, model checkpoints, and retrieval method names are not intended to change.
