## Context

The accepted architecture plan in `docs/10-plans/graph-memory-core-package-refactor-design.md` splits this refactor into multiple OpenSpec changes. Change A has already established lower-level domain packages for contracts, validation, infrastructure, datasets/text, graphs, and evaluation. Change B now handles Batch 5 and Batch 6: retrieval core boundaries, flat retrieval methods, graph rerank, and graph-rerank tuning.

The current retrieval path still centers on `graph_memory/retrieval.py`. That file constructs every method family through `RetrievalBuildContext`, carries dense-specific `query_prefix` and `passage_prefix` through high-level orchestration, performs graph-rerank score-pipeline construction, assembles ranked artifacts, and contains temporary trainable checkpoint wiring. `graph_memory/rerank.py`, `graph_memory/rerank_config.py`, and `graph_memory/tuning.py` are also still root modules despite belonging to retrieval and graph-rerank subdomains.

The key constraint is behavior preservation. This change must not modify public CLI behavior, workflow behavior, artifact schemas, method names, ranking semantics, graph-rerank scoring semantics, tuning objective, candidate ordering, or trainable graph retrieval results.

## Goals / Non-Goals

**Goals:**

- Establish `graph_memory/retrieval/` as the retrieval domain package.
- Move flat BM25/dense method construction into `retrieval/methods/flat/`.
- Replace `RetrievalBuildContext` with method-family build request dataclasses and runtime composition objects.
- Keep dense prefix configuration inside dense config/runtime objects once the CLI boundary has been resolved.
- Move result assembly and token approximation into retrieval execution modules.
- Move graph-rerank config, engine, components, candidate expansion, normalization, debug helpers, method adapter, and tuning into retrieval subpackages.
- Keep `graph_memory/retrieval_registry.py` as the narrow workflow integration port.
- Keep scripts and tests aligned to the new domain imports while preserving parser contracts.

**Non-Goals:**

- Do not modify `scripts/workflow/`.
- Do not change public retrieval method names or CLI flags.
- Do not change graph-rerank formulas, weights, candidate expansion order, normalization, tie-breaks, or tuning sort keys.
- Do not reorganize trainable graph model internals; Change C owns that work.
- Do not implement dense batching, dense cache artifacts, new retrievers, plugin discovery, dependency injection containers, or a pipeline framework.
- Do not delete `graph_memory/types.py`; only move retrieval-owned types that belong to this change.
- Do not promote durable docs; final docs promotion belongs to Change D.

## Decisions

### Decision: Split retrieval by method family and operation

Retrieval construction will be split into contracts, requests, resolver, factory, execution, flat methods, graph-rerank methods, tuning, and signals. This follows the accepted design principle that method families can share the top-level output contract without sharing a universal construction context.

Alternative considered: move the current `retrieval.py` content into a same-named package module with minimal structural change. That would reduce file size but preserve the wide context and dense parameter leakage that Change B is explicitly meant to remove.

### Decision: Use composition objects instead of context inheritance

The factory will receive precise request objects such as flat, graph-rerank, and trainable graph build requests. Dense-specific settings are grouped into a dense runtime/config object. Graph inputs are grouped through `GraphIndex`. Trainable graph runtime remains a temporary adapter around the existing learned inference path.

Alternative considered: create `BaseContext`, `DenseContext`, `GraphContext`, and `TrainableContext`. That would keep an inheritance axis where the real relationship is capability composition, and it would keep encouraging cross-family optional fields.

### Decision: Keep CLI optionality at the application boundary

The public `run_retrieval()` signature and scripts may continue accepting broad CLI-shaped arguments during this change, because public CLI compatibility is frozen. The resolver is responsible for turning that broad request into a precise method-family build request before factory construction.

Alternative considered: change script signatures to accept only exact family-specific inputs. That would be cleaner internally but would violate the external compatibility boundary for this refactor.

### Decision: Move graph-rerank and tuning together

Graph-rerank scoring and tuning both depend on the same config parsing, initial-score cache, and rerank execution semantics. Moving them in the same change avoids leaving tuning attached to old root module paths while the graph-rerank method has already moved.

Alternative considered: move graph-rerank first and leave tuning as a root module. That creates a temporary mixed ownership path and increases the chance that new imports keep relying on old root modules.

### Decision: Preserve trainable retrieval as a temporary factory branch

The `dense_rgcn_graph_retriever` method remains available through the new factory, but the underlying implementation still delegates to the existing learned inference module until Change C. This respects the OpenSpec split: Change B removes retrieval construction context and graph-rerank boundaries; Change C reorganizes trainable model internals.

Alternative considered: move learned inference now. That would mix Change B with training-pair and model-domain work and make behavior-equivalence failures harder to isolate.

## Risks / Trade-offs

- A package/file name conflict can occur when replacing root `graph_memory/retrieval.py` with `graph_memory/retrieval/` -> migrate imports and remove the old module only after focused tests pass.
- Moving tuning while preserving precomputed initial-score behavior can accidentally change latency accounting -> keep the initial-score cache and rerank latency addition behavior covered by existing tests.
- Dense prefixes can still appear in script parsers and trainable model configs by design -> assert only that high-level retrieval construction no longer passes them as loose fields after request resolution.
- Tests currently import some internal root modules -> update tests to import the new owned modules instead of adding wide compatibility facades.
- Some old type definitions may remain in `graph_memory/types.py` until Change C/D -> move only the retrieval/graph-rerank-owned types for this change and avoid creating a new aggregate re-export.

## Migration Plan

1. Add focused architecture/import tests that fail while `RetrievalBuildContext` and old root retrieval/rerank/tuning modules are still present.
2. Create retrieval package modules for contracts, requests, resolver, factory, execution, flat methods, signals, graph-rerank, and tuning.
3. Move flat BM25/dense construction and seed retriever logic into `retrieval/methods/flat/`.
4. Move graph-rerank config, scoring, debug helpers, method adapter, and tuning into retrieval subpackages.
5. Update scripts and tests to import the new domain paths while preserving parser contracts and public method names.
6. Remove old root modules only after residual import searches show no production/script/test callers.
7. Run focused retrieval tests, parser contract tests, smoke tests, type checking at error level, and strict OpenSpec validation.

Rollback is file-level within each batch: each moved module keeps behavior-preserving tests, so a failure can be isolated to the latest moved boundary.

## Open Questions

- None for Change B. Trainable model internals, final old-module deletion beyond this change, architecture dependency tests for final cleanup, and durable docs promotion remain assigned to later changes.
