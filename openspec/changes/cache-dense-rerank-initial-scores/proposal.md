## Why

Dense-seeded graph rerank tuning currently recomputes dense retrieval for every graph-rerank grid candidate. On quick100 this caused `dense_graph_rerank` tuning to exceed a 20-minute CPU run window, blocking comparison against the strongest flat baseline.

## What Changes

- Cache per-task initial retrieval scores once per tuning invocation.
- Reuse cached initial scores across graph-rerank grid candidates.
- Keep the existing `tune_graph_rerank(...)` and `scripts/tune_graph_rerank.py` interfaces compatible.
- Preserve existing ranked-result schemas and selected-config/candidate-row outputs.
- No breaking changes.

## Capabilities

### New Capabilities
- `efficient-graph-rerank-tuning`: Defines that graph-rerank tuning reuses initial retrieval scores instead of recomputing the seed retriever for each candidate.

### Modified Capabilities

## Impact

- Affected modules: `graph_memory/tuning.py`, `graph_memory/retrieval.py`.
- Affected tests: Phase 1 retrieval/tuning tests.
- No new runtime dependencies.
- Existing CLI commands remain valid.
