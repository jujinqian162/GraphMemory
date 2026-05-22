## Context

Phase 1 retrieval now supports four public methods: `bm25`, `dense`, `bm25_graph_rerank`, and `dense_graph_rerank`. The public contract is stable: every method writes a complete ranked node list plus a top-k retrieved subgraph in the shared `RankedResult` schema.

The implementation is less stable than the contract. `run_retrieval` currently branches on method strings to select a retriever, decide whether graph inputs are required, run graph rerank, and assemble graph edges. This is acceptable for Phase 1, but the original experiment plan names future baselines such as Dense-FT, Memory Stream, GraphRAG, MemGPT-style memory, and a trainable graph retriever. Some of these are score-combination methods; some are not.

## Goals / Non-Goals

**Goals:**

- Keep public method names, CLI arguments, JSON outputs, validators, tuning behavior, and evaluation behavior unchanged.
- Introduce a stable `RetrievalMethod` boundary for any baseline that can produce ranked results.
- Introduce a score-pipeline implementation for baselines that are naturally composed from node-score components.
- Move current flat and graph-rerank methods onto score-pipeline recipes.
- Make graph requirements explicit in the constructed method instead of scattered through the retrieval loop.
- Update docs so future contributors understand when to use score components and when to implement a separate retrieval method.

**Non-Goals:**

- Do not implement Dense-FT, Memory Stream, GraphRAG, MemGPT-style memory, or trainable graph retrieval in this change.
- Do not introduce dynamic plugin discovery, YAML-defined method loading, or a broad package hierarchy.
- Do not change graph construction, evaluation metrics, tuning objective, or ranked-result schema.
- Do not persist reusable score caches yet.

## Decisions

### Decision 1: Top-level abstraction is `RetrievalMethod`, not weighted scoring

The top-level contract should be:

```text
task input + optional graph -> ranked result
```

This protects future methods whose core behavior is graph traversal, hierarchy selection, or learned message passing. Those methods still produce ranked nodes, but their internals may not be a simple weighted sum.

Alternative considered: make `run_retrieval` directly combine `GraphRerankScore + BaselineScore + ...`. That works for Phase 1 graph rerank but would force GraphRAG, MemGPT-style memory, and trainable graph methods into an unnatural shape.

### Decision 2: Score pipeline is the first concrete `RetrievalMethod`

The current four methods are naturally expressed as:

```text
bm25 = BM25 score
dense = Dense score
bm25_graph_rerank = BM25 score + graph query/neighbor/bridge components
dense_graph_rerank = Dense score + graph query/neighbor/bridge components
```

The score pipeline owns:

- running reusable `NodeScoreComponent`s
- normalizing component scores
- combining weighted scores
- applying graph candidate gating for graph components
- building the retrieved subgraph

This keeps the current behavior understandable without prematurely creating separate classes for every method name.

### Decision 3: Keep graph-rerank scoring behavior equivalent

The score-pipeline implementation must preserve existing graph-rerank semantics:

- normalize the initial BM25/dense score with min-max normalization
- select graph candidates from top `seed_top_s` seeds expanded up to `max_hops`
- apply `lambda_init`, `lambda_query`, `lambda_neighbor`, and `lambda_bridge`
- keep `lambda_path` inert until a path component is implemented
- return every original memory node exactly once

This avoids changing experiment results while improving the shape of the code.

### Decision 4: Keep builders explicit and local

Use a small in-code registry from method name to recipe. Do not add plugin discovery or external method config yet. The current project needs reproducible, reviewable experiment baselines more than runtime extensibility.

## Risks / Trade-offs

- [Risk] A score-pipeline abstraction could hide the graph-rerank formula behind too many classes. -> Mitigation: keep component classes small, deterministic, and covered by focused tests; leave `rerank.py` helpers reusable.
- [Risk] Changing the retrieval service could alter scores or ordering. -> Mitigation: add tests that compare old graph-rerank helper output with the pipeline output and run the full retrieval test suite.
- [Risk] Future methods may not fit weighted scoring. -> Mitigation: keep `RetrievalMethod` as the top-level boundary and document score pipeline as one implementation style, not the universal model.
- [Risk] Registry-based recipes can become another hidden dispatch table. -> Mitigation: keep recipe names identical to CLI method names and keep unsupported methods fail-fast.

## Migration Plan

1. Add score-pipeline tests before production changes.
2. Add small scoring and method dataclasses/protocols to `types.py` or `retrieval.py`, choosing the smallest surface that keeps signatures readable.
3. Refactor `run_retrieval` to construct a `RetrievalMethod` once and delegate per-task execution.
4. Preserve `graph_rerank` and `graph_rerank_with_breakdown` as compatibility functions backed by the same component logic or equivalent helper behavior.
5. Update architecture, abstraction, handoff, and testing docs.
6. Run OpenSpec validation and retrieval/full test verification.
