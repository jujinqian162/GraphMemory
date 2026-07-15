## Context

Phase 1 graph rerank tuning evaluates a grid of graph-rerank configurations. The current implementation calls `run_retrieval(...)` for each candidate, so dense-seeded tuning reloads and reuses the dense retriever through the full retrieval path for every candidate. On CPU, the repeated dense encoding dominates runtime and caused quick100 `dense_graph_rerank` tuning to time out.

The retrieval layer already has `ScorePipelineMethod.rank_task_from_scores(...)`, which can apply graph rerank scoring to precomputed initial scores. The change should use that boundary rather than introducing a new artifact schema.

## Goals / Non-Goals

**Goals:**
- Compute seed retriever scores once per task for a `tune_graph_rerank(...)` invocation.
- Reuse those scores for every graph-rerank grid candidate.
- Keep public Python and CLI interfaces compatible.
- Preserve existing candidate-row, selected-config, ranked-result, and metric schemas.
- Keep BM25 and dense tuning behavior semantically equivalent to the old per-candidate retrieval path.

**Non-Goals:**
- Add persistent on-disk score caches.
- Change graph-rerank scoring quality, graph construction, or tuning objective.
- Implement GraphRAG, Memory Stream, or Dense-FT baselines.
- Change `run_retrieval(...)` behavior for normal retrieval commands.

## Decisions

1. **Use an in-memory initial-score cache inside `tune_graph_rerank(...)`.**
   - Rationale: the immediate problem is repeated work within one tuning process. In-memory caching avoids new cache invalidation rules and artifact contracts.
   - Alternative considered: add `--initial_scores_cache` file paths. Rejected for this change because it would create a new artifact lifecycle before the internal contract is proven.

2. **Expose a small retrieval helper for initial score computation and config-specific graph rerank assembly.**
   - Rationale: `tuning.py` should not duplicate how retrieval methods are constructed or how ranked results are assembled.
   - Alternative considered: keep calling `run_retrieval(...)` and rely on global memoization inside dense retriever. Rejected because the repeated retrieval loop would still obscure the intended tuning flow and make tests harder.

3. **Keep `run_retrieval(...)` untouched as the public execution path.**
   - Rationale: normal CLI retrieval commands should remain stable and easy to validate against existing tests.
   - Alternative considered: add optional initial-score parameters to `run_retrieval(...)`. Rejected because it would broaden the public API for an internal tuning optimization.

## Risks / Trade-offs

- Cached initial scores may increase peak memory for larger tuning splits. Mitigation: cache only `dict[node_id, score]` per task, not embeddings or model tensors.
- Refactoring retrieval helpers could change rankings accidentally. Mitigation: add an equivalence test comparing cached tuning candidate metrics with the existing `run_retrieval(...)` path on a small fixture.
- Dense model runtime may still be high for the one initial retrieval pass. Mitigation: this change removes the grid-size multiplier; persistent embedding caches can be proposed separately if needed.
