## Context

Change A moved foundation, dataset, text, graph, and evaluation ownership into domain packages. Change B moved retrieval, graph-rerank, and tuning ownership. Change C moved train-pair generation and trainable graph retriever ownership. The remaining work is to remove temporary compatibility surfaces that were intentionally left until all domains had moved, then lock the dependency direction with tests and durable docs.

The package-refactor design lists several root names for deletion, but some of those names now exist as real domain packages rather than old root modules. `graph_memory.validation`, `graph_memory.graphs`, `graph_memory.evaluation`, `graph_memory.text`, and `graph_memory.retrieval` must remain as domain packages. The deletion target is old root modules and compatibility packages, not the new owned packages.

## Goals / Non-Goals

**Goals:**

- Delete obsolete root compatibility modules and the temporary `graph_memory.types` aggregation layer.
- Move any remaining records owned by `types.py` to their domain package paths.
- Add architecture dependency tests for approved package direction and root integration ports.
- Update durable docs so navigation points at the post-refactor module tree.
- Verify no behavior drift through full tests, type checking, OpenSpec validation, and a same-config quick R-GCN workflow comparison.

**Non-Goals:**

- Do not alter `scripts/workflow/` internals.
- Do not change CLI arguments, artifact schemas, checkpoint schema, ranking formulas, training objectives, random seed behavior, or method names.
- Do not keep broad old import facades for tests or historical internal callers.
- Do not archive A/B/C/D in this change implementation step.

## Decisions

### Decision: Delete old module files, retain domain packages

Remove `graph_memory/types.py`, `graph_memory/hotpotqa.py`, `graph_memory/splits.py`, `graph_memory/entities.py`, old `graph_memory/indexes/`, and old `graph_memory/learned/` compatibility surfaces once imports are clean. Keep the domain package directories that replaced former root modules.

Alternative considered: preserve root compatibility re-exports for old internal import paths. That contradicts the approved non-goal of retaining broad facades and would make future dependency tests ineffective.

### Decision: Move remaining `types.py` records to owning domains before deletion

`NegativeSamplingConfig` belongs in `training_pairs.config`. `NodeFeatureConfig`, `TrainableModelConfig`, `TrainableTrainingConfig`, `GraphBatch`, and `TrainingBatch` belong in `models.graph_retriever.config.records` or `models.graph_retriever.internals` as already planned. Tests and scripts must import from those owned paths.

Alternative considered: keep a small `types.py` with only remaining records. The plan explicitly treats `types.py` as a temporary migration exception that must end in Batch 9, so this is not acceptable.

### Decision: Architecture tests use AST import scanning

Use standard-library AST scanning so dependency rules are cheap, deterministic, and independent of runtime import side effects. The test must distinguish package directories from removed root files and must whitelist only the five workflow integration ports: `io.py`, `observability.py`, `retrieval_registry.py`, `training_config.py`, and `experiment.py`.

Alternative considered: rely on `rg` commands in the final checklist only. Manual searches are useful verification but do not prevent future regressions.

### Decision: Behavior equivalence is verified outside production code

The implementation must not add production behavior just for comparison. The same-config quick R-GCN run should create a new workflow run name, then compare stable intermediate artifacts against the existing baseline run after normalizing allowed run-name/path/timestamp differences if needed.

Alternative considered: compare only unit tests and type checking. That is weaker than the user-requested workflow-level check and could miss manifest/config/runtime wiring drift.

## Risks / Trade-offs

- Removed internal import paths may break tests that still assert temporary compatibility -> Update tests to target domain-owned paths and add architecture checks for the new boundary.
- Architecture tests may flag generated caches or permission-restricted temp folders -> Scan only repository source roots and Python files, excluding known temp/cache directories.
- Workflow output comparison can include legitimate run-name or path fields -> Compare behavior-bearing artifacts directly and normalize run-local paths/timestamps in run-summary style files.
- `__init__.py` imports can accidentally load heavy dependencies -> Keep package `__init__` files lightweight unless a domain package already has an explicit public re-export contract.

## Migration Plan

1. Add focused failing tests for removed old paths, owned import paths, root port allowlists, and architecture dependency direction.
2. Move remaining `types.py` definitions to domain-owned modules and update scripts/tests imports.
3. Delete obsolete root compatibility files and empty old packages.
4. Update durable docs with the final module navigation and retained workflow ports.
5. Run focused tests, full tests, basedpyright error-level type checking, ruff, OpenSpec strict validation, and the quick R-GCN workflow equivalence comparison.

## Open Questions

None. The only implementation caveat is that old root module names in the plan must be interpreted as old files/facades, not as the new domain package directories with the same import root.

## Validation Note

The final workflow comparison used a fresh `rgcn_quick_train_after_refactor` run initialized with the same quick profile method and stage selection as `rgcn_quick_train`. Shared stable input, graph, pair, and R-GCN ranking artifacts matched the baseline behavior. The comparison also exposed an existing reproducibility limitation in graph-rerank tuning: `select_best_config` uses `Retrieval Latency / Query` as a tie-breaker after equal objective metrics, so tied BM25 graph-rerank candidates can select `max_hops=1` or `max_hops=2` depending on runtime latency. This is not changed by Change D, but it means strict byte-for-byte equality across all intermediate files is not a valid acceptance criterion for the full quick workflow unless tuning tie-breaking is made deterministic in a separate behavior change.
