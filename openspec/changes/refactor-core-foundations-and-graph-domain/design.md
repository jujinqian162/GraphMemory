## Context

The accepted architecture plan is documented in `docs/10-plans/graph-memory-core-package-refactor-design.md`. This OpenSpec change implements only the first proposed change group: Batch 0 through Batch 4.

Current core modules mix several domains:

- `graph_memory/types.py` combines artifact contracts, evaluation records, retrieval config, graph config, model config, and tensor batch types.
- `graph_memory/validation.py` validates artifacts, configs, metrics, and trainable model tensors in one large module.
- `graph_memory/io.py` and `graph_memory/observability.py` are used by workflow code and must remain as narrow integration ports during this change.
- Dataset parsing, text helpers, graph construction, and evaluation currently live in root modules that are difficult to reason about independently.

The key constraint is behavior preservation. This change must not modify public CLI behavior, workflow behavior, artifact schemas, ranking semantics, graph construction semantics, metrics, checkpoint schemas, or training behavior.

## Goals / Non-Goals

**Goals:**

- Freeze current behavior with focused parser, workflow, and deterministic fixture tests before moving production code.
- Establish low-level domain packages for `contracts`, `validation`, `infrastructure`, `datasets`, `text`, `graphs`, and `evaluation`.
- Keep root `io.py` and `observability.py` as narrow workflow integration ports.
- Keep `scripts/workflow/` unchanged.
- Avoid new production dependencies.

**Non-Goals:**

- Do not remove `graph_memory/types.py` in this change.
- Do not remove `RetrievalBuildContext`.
- Do not reorganize retrieval factories, graph rerank/tuning, training pairs, or trainable graph model code.
- Do not change workflow orchestration internals.
- Do not add new baselines, datasets, graph rules, scoring formulas, or performance optimizations.
- Do not promote durable docs until the final cleanup change.

## Decisions

### Decision: Start with regression-freeze tests

Batch 0 creates tests that capture parser actions, workflow plan output, and deterministic domain behavior before production code moves.

Alternative considered: move files first and rely on existing tests. That would make later regressions harder to distinguish from test blind spots, especially around parser defaults, workflow stage planning, and deterministic graph/evaluation outputs.

### Decision: Split by domain, not by legacy file

Moved definitions and functions will land where their domain owner is clear:

- Artifact-shaped `TypedDict` contracts go to `graph_memory/contracts/`.
- Validators go to `graph_memory/validation/`.
- IO and runtime summary helpers go to `graph_memory/infrastructure/`.
- HotpotQA parsing/conversion and splits go to `graph_memory/datasets/`.
- Token, lexical, and entity helpers go to `graph_memory/text/`.
- Graph config, construction, indexes, statistics, and views go to `graph_memory/graphs/`.
- Metric primitives, connectivity, tables, and failure cases go to `graph_memory/evaluation/`.

Alternative considered: split each large root module into same-named submodules. That would reduce file size but preserve unclear ownership and cross-domain imports.

### Decision: Preserve workflow integration ports

The workflow package currently imports root `graph_memory.io` and `graph_memory.observability`. During this change those modules remain, but they must only re-export or thinly adapt approved infrastructure functions consumed by workflow code.

Alternative considered: update workflow imports now. That would violate the design boundary that freezes `scripts/workflow/` until a later workflow-specific change.

### Decision: Keep migration incremental

`graph_memory/types.py` remains as a shrinking temporary migration file for domains not yet moved. This avoids creating future retrieval/model/training-pair packages before their dedicated changes.

Alternative considered: delete `types.py` during this change. That would force broad facade modules or premature later-domain packages and make the batch boundary less auditable.

## Risks / Trade-offs

- Parser or workflow contract tests may expose pre-existing ambiguity in script APIs -> encode the current behavior directly from parser/action state instead of comparing formatted help text.
- Golden fixtures may become too broad -> keep fixtures tiny, deterministic, and in-memory where possible.
- Narrow integration ports could slowly grow -> add architecture/import tests or direct assertions that root ports expose only approved names.
- Pure file moves can accidentally alter imports or defaults -> run focused tests after each domain batch and full validation before marking tasks complete.
- Retaining `types.py` temporarily leaves some centralization in place -> prohibit new `from graph_memory.types` imports for migrated domains and shrink it only within the current batch boundary.

## Migration Plan

1. Create Batch 0 tests and record baseline commands without moving production code.
2. Split contracts, validation, and infrastructure while preserving root `io.py` and `observability.py` integration ports.
3. Split dataset and text helpers.
4. Split graph construction, graph indexes/statistics/views, and graph validation connections.
5. Split evaluation metric/connectivity/table/failure-case logic.
6. Run focused tests, full pytest, type checking at error level, and strict OpenSpec validation.

Rollback is file-level: each batch is behavior-preserving and can be reverted independently if its focused tests identify a regression.

## Open Questions

- None for Change A. Retrieval factory/context deletion, graph rerank/tuning migration, training-pair migration, trainable model reorganization, final old-module deletion, and durable docs promotion are intentionally deferred.
