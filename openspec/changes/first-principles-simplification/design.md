## Context

The repository currently supports eight retrieval methods across two genuinely different domains:

- evidence retrieval over HotpotQA, 2Wiki, and MuSiQue with flat candidates or native `EvidenceGraph` inputs;
- ISETrace retrieval over canonical trajectories, source spans, and native `ProvenanceGraph` inputs.

The algorithms are not the main source of size. The same lifecycle is represented repeatedly in `experiment/workflow.py`, Prefect tasks, stage materializers, training payloads/trainers, registry builders, execution tasks, and result envelopes. Configuration is similarly projected into method, stage, train-stage, ranking, and resolved forms.

The starting branch passes all 194 tests. The refactor must preserve all eight methods and both graph domains. Existing artifact directories are historical data, not a compatibility target for removed internal Python classes. Every implementation tranche must remain runnable and produce a net code reduction.

## Goals / Non-Goals

**Goals:**

- Make a scientific run traceable through no more than one orchestration function, one stage/materialization boundary, and one concrete algorithm entrypoint.
- Give method selection, training parameters, request context, result validation, and persisted output one owner each.
- Preserve current preparation, training, ranking, evaluation, reproducibility metadata, and persisted result behavior.
- Keep only abstractions with at least two real implementations or consumers.
- Deliver the refactor as independently testable, net-deleting tranches rather than a big-bang rewrite.

**Non-Goals:**

- Removing or redesigning any of the eight methods in this change.
- Combining `EvidenceGraph` and `ProvenanceGraph` or their construction/tensorization semantics.
- Changing dataset-specific parsing, labels, negative-sampling algorithms, graph edges/motifs, model architecture, or metric definitions.
- Adding a replacement intermediate representation, experiment plan, method context, generic graph request, registry, adapter, validator, policy, or compatibility facade.
- Deleting input digests, source revisions, manifests, rankings, metrics, per-task results, or model provenance required for reproducibility.

## Decisions

### 1. Preserve behavior before reducing the method set

All current methods remain executable through the same experiment entrypoint during this refactor. Simplification targets framework structure, not the paper's method matrix.

Alternative considered: immediately keep only BM25, Dense, and the two R-GCN methods. Rejected because it mixes architectural simplification with a scientific-scope decision and makes review parity harder.

### 2. Keep the two graph domains separate

`EvidenceGraph` and `ProvenanceGraph` retain separate contracts, builders, candidate semantics, relation vocabularies, and tensorization. Existing `TaskGraphTensor`, `GraphBatch`, neural scoring core, and optimizer loop remain shared because both domains consume them.

Alternative considered: a unified graph representation or generic graph request. Rejected because it would move semantic differences into optional fields and validators instead of removing complexity.

### 3. Delete wrappers from the outside inward

Implementation proceeds in four tranches:

1. result/DTO/output wrappers with no algorithmic behavior;
2. retrieval registry and execution transport;
3. training payload/trainer/factory and repeated config projections;
4. workflow/task duplication and peripheral run integrations.

Each tranche updates callers and tests before deleting the old layer. No dual runtime path or compatibility adapter remains at tranche completion.

Alternative considered: rewrite the workflow first. Rejected because the workflow currently depends on all lower transport layers and would require a big-bang cutover.

### 4. Use one explicit retrieval dispatch

The retrieval stage performs one closed dispatch over the existing validated method settings and directly constructs the concrete method plus its concrete requests. `RetrievalRegistry`, `RetrievalBuilderSpec`, settings/payload type maps, and `RetrievalExecutionTask` are deleted.

The dispatch is intentionally static. Eight repository-owned methods do not justify runtime plugin registration.

Alternative considered: replace the registry with a new capability table or plan object. Rejected because that preserves the same indirection under a new name.

### 5. Concrete requests own execution context

A concrete method request is the only request entering the retrieval loop. It carries the fields its algorithm needs; a duplicate base text request is not paired beside it. Task/candidate consistency is checked once where results are assembled or persisted.

`RankedResult` remains the result contract. Envelope and batch models used only for repeated validation are removed.

Alternative considered: strengthen cross-object validators. Rejected because eliminating duplicated context removes the invalid state entirely.

### 6. Stage materializers call concrete training functions

`stages/models.py` remains the IO/materialization boundary and directly calls Dense-FT, evidence R-GCN, or provenance R-GCN training. `stages/train_payloads.py`, `stages/trainers.py`, train dependency carriers, and single-implementation model/trainer factories are deleted.

Algorithm-level reusable helpers remain when they encode real operations such as example construction, batching, task-local evaluation, checkpoint loading, and optimization.

Alternative considered: merge all training into one generic trainer. Rejected because Dense-FT and R-GCN have different data, model, and evaluation semantics.

### 7. Fixed scientific protocol values have one owner

Method-specific training and ranking values are read once from the parsed method configuration. Projection helpers that only copy those fields into stage/train/ranking/resolved models are removed. Values that have one supported production value are fixed at the concrete algorithm boundary, not repeated through profiles and payloads.

This tranche does not change numerical values. A parameter is fixed only after call-site and config searches prove one effective production value or after all callers can use the existing canonical value.

Alternative considered: create one canonical resolved configuration object. Rejected because it would add another representation rather than removing projections.

### 8. Express orchestration once

After lower layers are direct, `experiment/workflow.py` owns the readable sequential lifecycle. Prefect task wrappers are either deleted or reduced to a decorator plus one direct call when caching/resume still requires a physical task boundary. Stage functions continue to own filesystem materialization and may be invoked directly in tests.

Tracking, input provisioning, benchmark collection, and report publication remain peripheral calls around the scientific lifecycle; they do not select method behavior or introduce alternate results.

Alternative considered: remove Prefect and MLflow immediately. Rejected because their current cache, resume, and provenance behavior must first be separated from scientific control flow and parity-tested.

### 9. Persist three categories, not three new wrapper types

The durable categories are prepared data, optional trained model, and result outputs. Existing files remain where required for current behavior, but intermediate pairs/embeddings are treated as implementation caches and no longer create public type hierarchies.

Digest, origin, source revision, and manifest metadata remain because they define reproducibility boundaries. Duplicate combined DTOs and redundant CSV projections are removed when no current consumer requires them.

Alternative considered: introduce a single artifact object. Rejected because it would create another universal transport type.

### 10. Tests protect behavior and deletion boundaries

Retained tests assert deterministic preparation, labels, graph semantics, training inputs, ranking order/scores within existing tolerances, metrics, persisted outputs, cache identity, and resume behavior. Tests that only instantiate deleted wrappers, freeze deprecated exports, or assert duplicate files are removed or rewritten.

Each tranche records source/test line counts and must reduce total lines after replacement code and tests are included.

## Risks / Trade-offs

- [Internal imports may be used by undocumented external scripts] -> Search the full repository before each deletion; treat removed internal Python APIs as breaking and do not add compatibility aliases.
- [Static dispatch is longer than a registry lookup] -> Keep all method construction in one function and factor only algorithmic helpers with multiple callers.
- [Removing repeated validation may hide malformed data] -> Retain validation at raw/config/persisted boundaries and result assembly; remove only checks made necessary by duplicate transport objects.
- [Artifact cleanup can break resume or downstream analysis] -> Separate wrapper deletion from persisted-file deletion; prove cache/resume and search all scripts/docs before removing a file.
- [Workflow flattening can alter Prefect cache keys or MLflow lineage] -> Defer it until direct retrieval/training behavior is stable, then compare manifests, rankings, metrics, and tracking tests.
- [Existing tests may be coupled to implementation] -> Replace them with boundary tests in the same tranche, never delete coverage without identifying the retained behavior.
- [Large diff becomes hard to review] -> Keep four coherent tranche commits available to the user; do not mix scientific changes into structural commits.

## Migration Plan

1. Freeze baseline tests, method/config inventory, representative persisted shapes, and production/test line counts.
2. Remove dead DTOs, result wrappers, single-use factories/helpers, and redundant output projections while retaining observable output compatibility unless explicitly proven unused.
3. Replace retrieval registry/builders/execution tasks with one explicit stage dispatch and direct requests; run all method-focused tests.
4. Replace training payloads/trainers/factories and repeated config projections with direct concrete calls; run Dense-FT and both graph-family training tests.
5. Flatten workflow/task orchestration and move peripheral concerns outside the scientific method branches; run cache/resume/tracking and full tests.
6. Update active docs/configs, archive or mark superseded OpenSpec changes whose requirements describe deleted internals, and validate this change strictly.
7. Run full pytest, Ruff, basedpyright, compileall, `git diff --check`, config composition, and representative smoke workflows available within local resources.

Rollback is by tranche commit. No runtime migration switch or compatibility layer is introduced.
