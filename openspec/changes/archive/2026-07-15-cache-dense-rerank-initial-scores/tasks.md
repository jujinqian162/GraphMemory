## 1. Tests

- [x] 1.1 Add a regression test proving graph-rerank tuning invokes the seed retriever once per task across multiple configs.
- [x] 1.2 Add an equivalence test proving cached candidate evaluation matches the normal retrieval path for a fixed config.

## 2. Retrieval Helpers

- [x] 2.1 Add an internal retrieval helper that computes initial score maps once per task.
- [x] 2.2 Add an internal retrieval helper that assembles ranked results from precomputed initial score maps for a graph-rerank config.

## 3. Tuning Integration

- [x] 3.1 Refactor `tune_graph_rerank(...)` to precompute seed scores once and reuse them for every candidate.
- [x] 3.2 Preserve the existing `tune_graph_rerank(...)` function signature and CLI arguments.

## 4. Verification

- [x] 4.1 Run focused retrieval/tuning tests.
- [x] 4.2 Run `openspec status` and validate the change is apply-ready.
