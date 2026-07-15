## 1. Tests First

- [x] 1.1 Add retrieval tests that exercise score-pipeline method construction, flat graph-free execution, graph-method requirement failures, and pipeline-vs-graph-rerank equivalence.
- [x] 1.2 Run the focused retrieval tests and confirm the new score-pipeline tests fail for the expected missing abstraction.

## 2. Score Pipeline Implementation

- [x] 2.1 Add retrieval-method, score-context, and node-score component abstractions without changing public artifact types.
- [x] 2.2 Implement BM25 and dense baseline components and graph query/neighbor/bridge components with normalization and candidate gating.
- [x] 2.3 Refactor `run_retrieval` to build a retrieval method once and delegate per-task execution while preserving CLI-visible method behavior.

## 3. Documentation

- [x] 3.1 Update architecture and abstraction docs to describe `RetrievalMethod` as the top-level contract and score pipeline as one implementation style.
- [x] 3.2 Update implementation handoff and testing docs with the new review path and extension guidance.

## 4. Verification

- [x] 4.1 Run focused retrieval/type tests.
- [x] 4.2 Run OpenSpec validation for `refactor-retrieval-score-pipeline`.
- [x] 4.3 Run the full test suite or the repository-local verified fallback command, and review the diff for scope.
