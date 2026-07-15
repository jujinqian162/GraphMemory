## 1. Tests First

- [x] 1.1 Add config tests proving `neighbor_type_weights` is the canonical graph-rerank field and deprecated `type_weights` is rejected.
- [x] 1.2 Add tests proving `query_overlap` is not required in `neighbor_type_weights` and query-overlap scoring is controlled only by `lambda_query`.
- [x] 1.3 Add rerank/retrieval parity tests proving graph-rerank rankings and retrieved subgraph edges remain unchanged after moving graph scoring ownership to `rerank.py`.

## 2. Config Rename

- [x] 2.1 Rename `GraphRerankConfig.type_weights` to `neighbor_type_weights` and update default weights to include memory-to-memory graph edge types only.
- [x] 2.2 Add config loading that rejects deprecated `type_weights`, including records that also contain `neighbor_type_weights`.
- [x] 2.3 Update graph-rerank validation so `neighbor_type_weights` is required for canonical configs and `query_overlap` is not required as a neighbor type weight.
- [x] 2.4 Update grid parsing and selected-config serialization so newly written tuning artifacts emit `neighbor_type_weights` only.
- [x] 2.5 Update `configs/search_spaces/graph_rerank.json` to use `neighbor_type_weights`.

## 3. Rerank Boundary Refactor

- [x] 3.1 Move graph-rerank score context, component protocols/classes, and component-combination helpers from `graph_memory/retrieval.py` to `graph_memory/rerank.py`.
- [x] 3.2 Expose a small rerank module entry point that ranks from initial scores and returns ranked nodes plus induced top-k edges for retrieval orchestration.
- [x] 3.3 Refactor `graph_memory/retrieval.py` so graph methods select the seed retriever, compute initial scores, delegate graph scoring to `rerank.py`, and assemble the existing `RankedResult` schema.
- [x] 3.4 Keep `graph_rerank(...)` and `graph_rerank_with_breakdown(...)` compatible by routing through the canonical rerank implementation or sharing the same combination helper.

## 4. Documentation Updates

- [x] 4.1 Update data-contract docs to document `neighbor_type_weights`, deprecated `type_weights` rejection, and the distinction from `lambda_*`.
- [x] 4.2 Update architecture and abstraction docs so `rerank.py` owns graph-rerank scoring while `retrieval.py` owns orchestration.
- [x] 4.3 Update operations docs and handoff notes to use `neighbor_type_weights` in examples and explain that old `type_weights` configs must be converted before reuse.

## 5. Verification

- [x] 5.1 Run focused retrieval/rerank/config tests.
- [x] 5.2 Run CLI smoke tests that read graph-rerank search-space config and write selected config output.
- [x] 5.3 Run OpenSpec validation for `clarify-graph-rerank-weights`.
- [x] 5.4 Review the diff and confirm no source behavior beyond the planned config rename and rerank-boundary refactor changed.
