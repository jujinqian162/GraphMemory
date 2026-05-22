## Why

`run_retrieval` currently mixes baseline selection, graph-method validation, rerank control flow, and result assembly in one function. This is readable for four Phase 1 methods, but it will become brittle as Phase 2 adds Dense-FT, Memory Stream, GraphRAG-style, and ablation baselines.

This change introduces a small internal retrieval-method abstraction that keeps the public method names and ranked-result schema stable while allowing score-based baselines to be composed from reusable scoring components.

## What Changes

- Introduce a top-level `RetrievalMethod` execution boundary for any baseline that can return a `RankedResult`.
- Introduce a `ScorePipelineMethod` implementation for baselines that are naturally expressed as weighted node-score components.
- Move BM25, dense, and current graph-rerank methods onto score-pipeline recipes.
- Keep `bm25`, `dense`, `bm25_graph_rerank`, and `dense_graph_rerank` method names unchanged.
- Keep existing CLI arguments, ranked output schema, graph validation behavior, tuning calls, and evaluation compatibility unchanged.
- Update design docs to distinguish the stable retrieval-method contract from the optional score-pipeline implementation style.

## Capabilities

### New Capabilities
- `score-pipeline-retrieval`: Defines composable node-score pipeline behavior for score-based retrieval baselines.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `graph_memory/types.py`
  - `graph_memory/retrieval.py`
  - `tests/test_phase1_real_retrieval.py`
  - `tests/test_type_contracts.py`
- Affected docs:
  - `docs/30-design/architecture.md`
  - `docs/30-design/abstractions.md`
  - `docs/40-operations/implementation-handoff.md`
- No breaking changes to CLI commands, JSON artifacts, method names, or evaluation tables.
