## Why

Graph rerank execution now has two overlapping shapes: public retrieval methods route through the score-pipeline abstraction, while `graph_memory/rerank.py` still contains graph-rerank helpers and compatibility entry points. At the same time, the `type_weights` config field looks parallel to `lambda_*` even though it only calibrates graph edge types inside neighbor-style scoring, which makes tuned configs easy to misread.

This change clarifies the graph-rerank boundary before more experiments depend on it: graph rerank scoring abstractions should live in `rerank.py`, retrieval orchestration should stay in `retrieval.py`, and `type_weights` should be renamed to `neighbor_type_weights` so it no longer appears to duplicate component lambdas.

## What Changes

- Move graph-rerank score component abstractions and combination helpers out of `graph_memory/retrieval.py` and into `graph_memory/rerank.py`.
- Keep `graph_memory/retrieval.py` focused on seed retriever selection, latency measurement, result assembly, and public method orchestration.
- Rename graph-rerank config field `type_weights` to `neighbor_type_weights`.
- Treat `neighbor_type_weights` as internal edge-type calibration for neighbor propagation and bridge-neighbor scoring, not as final component weights.
- Keep query-overlap scoring controlled only by `lambda_query`; it SHALL NOT use `neighbor_type_weights["query_overlap"]`.
- Preserve public method names, ranked-result schema, tuning objective, and graph-rerank ranking behavior except for the config field rename.
- Provide a compatibility migration path for existing selected config JSON files and search-space files during the rename.
- **BREAKING**: newly written graph-rerank config artifacts should use `neighbor_type_weights`; `type_weights` becomes deprecated compatibility input only.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `score-pipeline-retrieval`: Clarify graph-rerank ownership, config naming, and component-vs-edge weight semantics.

## Impact

- Affected code:
  - `graph_memory/types.py`
  - `graph_memory/rerank.py`
  - `graph_memory/retrieval.py`
  - `graph_memory/tuning.py`
  - `graph_memory/validation.py`
  - `scripts/tune_graph_rerank.py`
- Affected configs and artifacts:
  - `configs/search_spaces/graph_rerank.json`
  - existing `runs/**/tuned/*graph_rerank*.json` files as compatibility inputs
- Affected tests:
  - retrieval/rerank unit tests for component parity and config migration
  - validation tests for renamed config fields
  - CLI or smoke tests that read graph-rerank grid config
- Affected docs:
  - `docs/20-contracts/phase1-data-contracts.md`
  - `docs/30-design/abstractions.md`
  - `docs/30-design/architecture.md`
  - `docs/40-operations/commands.md`
  - `docs/40-operations/implementation-handoff.md`
