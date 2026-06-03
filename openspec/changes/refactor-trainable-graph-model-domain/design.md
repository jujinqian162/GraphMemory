## Context

Change A established lower-level core packages and Change B moved retrieval, flat methods, graph-rerank, and tuning into owned retrieval boundaries. Change C now covers Batch 7 and Batch 8 from `docs/10-plans/graph-memory-core-package-refactor-design.md`: move training pair generation out of `learned`, and reorganize trainable graph retriever internals under `models/graph_retriever`.

The current `graph_memory.learned` package has two distinct domains. `data.py` creates pair artifacts and is useful beyond the trainable graph model. The rest of the package owns trainable model runtime concerns, but its internal dependencies are not layered: `inference.py` imports `build_model_from_config()` from `training.py`, and scripts import model-facing protocols from `learned.features`. The refactor must preserve behavior while reducing those ownership leaks.

## Goals / Non-Goals

**Goals:**

- Establish `graph_memory/training_pairs/` for pair generation, config ownership, and negative sampler composition.
- Establish `graph_memory/models/graph_retriever/` for trainable graph retriever config, checkpoint, batching, tensorization, neural model, factory, training, dev evaluation, text embeddings, and inference.
- Add a retrieval-owned `TrainableGraphRetrievalMethod` adapter under `graph_memory/retrieval/methods/`.
- Keep `dense_rgcn_graph_retriever` available through the existing retrieval factory and scripts.
- Ensure inference depends on model factory, not training.
- Keep `graph_memory/training_config.py` as a narrow integration port for workflow-facing config loading.
- Update imports in scripts and tests to use owned modules rather than adding broad learned facades.

**Non-Goals:**

- Do not modify `scripts/workflow/`.
- Do not change CLI flags, parser defaults, required/optional status, choices, output paths, artifact schemas, checkpoint schema, relation vocab order, ablation mapping, or training/retrieval math.
- Do not introduce Dense-FT, dense batching, shared dense encoding service, new negative sampling ratios, new neural encoders, plugin discovery, or dependency-injection containers.
- Do not delete final old root modules assigned to Change D, such as `types.py`, `validation.py`, or final compatibility ports.
- Do not promote durable docs; final docs synchronization belongs to Change D.

## Decisions

### Decision: Move training pairs before model internals

Training-pair generation is a lower-level artifact-building domain that does not depend on the trainable graph model. Moving it first lets scripts and training consume a stable `training_pairs` API before model internals move.

Alternative considered: move the entire `learned` package at once. That would blur behavior-equivalence failures across pair sampling, tensorization, model construction, and training.

### Decision: Use parallel sampler objects without changing sampler algorithms

`TrainPairBuilder` will orchestrate `EasyRandomNegativeSampler`, `BM25HardNegativeSampler`, `DenseHardNegativeSampler`, and `GraphNeighborNegativeSampler`. Each sampler keeps its existing algorithm, order, random state behavior, de-duplication key behavior, and truncation semantics.

Alternative considered: unify all samplers behind one generic rank-and-filter helper. That would reduce repetition but risks changing ordering and makes the current domain rules less explicit.

### Decision: Put model construction in a factory shared by training and inference

`GraphScoringModelFactory` becomes the shared model construction boundary. Training and inference both depend on the factory, eliminating the current inference-to-training dependency.

Alternative considered: keep `build_model_from_config()` in training and add a compatibility re-export. That would keep the wrong dependency direction and obscure the architecture issue Change C is meant to fix.

### Decision: Split retrieval adapter from model inference

Model inference will return model-level rankings and edges from checkpoint-backed scoring. `TrainableGraphRetrievalMethod` adapts that model inference into the retrieval `RetrievalMethod` contract. The retrieval factory should import the adapter, not model internals directly.

Alternative considered: keep the old `TrainableGraphRetriever` class in the model package implementing retrieval contracts. That forces the model package to know retrieval execution semantics.

### Decision: Preserve narrow compatibility ports only where external boundaries require them

`graph_memory/training_config.py` remains as the workflow-facing config-loading integration port. Existing `learned` modules should not become a wide facade for internal imports; callers should move to the owned domains during this change.

Alternative considered: keep `graph_memory.learned.*` as broad re-export modules. That would make the refactor look complete while preserving the old ownership surface.

## Risks / Trade-offs

- Pair artifact ordering can change if samplers are refactored too aggressively -> add focused tests that compare exact pair records and summaries.
- Relation vocab or tensor concatenation ordering can change during module moves -> keep tensorization and model forward golden tests in the verification set.
- Checkpoint loading can drift if config imports move carelessly -> keep checkpoint round-trip tests and schema validation intact.
- Inference can accidentally keep importing training through a re-export -> add an architecture test that inspects model inference imports.
- The old `learned` package may still exist during the transition -> reject production/script/test imports of `graph_memory.learned.data`, and only leave temporary modules when needed for Change D boundaries.

## Migration Plan

1. Add focused architecture/import tests that fail while train pair callers still depend on `graph_memory.learned.data` and while inference imports training.
2. Create `training_pairs/` modules and move pair generation into builder/samplers/config while preserving public result/config dataclasses.
3. Update `scripts/build_train_pairs.py`, training code, and tests to import from `training_pairs`.
4. Create `models/graph_retriever/` modules for contracts, text embeddings, internals, config, factory, checkpoint, dev evaluation, training, and inference.
5. Move the trainable retrieval adapter into `retrieval/methods/trainable_graph.py` and update retrieval factory/request wiring.
6. Update scripts and tests to import owned model modules without changing parser contracts or artifact schemas.
7. Run pair, tensorization, model, training, retrieval, parser contract, full pytest, type checking at error level, ruff, and strict OpenSpec validation.

Rollback is batch-local: pair-domain changes are verified before model-domain moves, so failures can be isolated to the latest boundary.

## Open Questions

- None for Change C. Final old-module deletion, architecture dependency hardening beyond these boundaries, and durable docs promotion remain assigned to Change D.
