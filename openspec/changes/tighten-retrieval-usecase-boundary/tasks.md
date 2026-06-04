## 1. Boundary Tests

- [x] 1.1 Add failing tests for the application retrieval use-case request and narrowed execution signature.
- [x] 1.2 Strengthen retrieval boundary scans so loose dense prefix fields cannot reappear in execution or tuning internals.
- [x] 1.3 Strengthen architecture import scans so domain packages cannot import retained root workflow integration ports.

## 2. Application Retrieval Use Case

- [x] 2.1 Add `graph_memory.application.run_retrieval` with `RunRetrievalRequest` and typed runtime/config request construction.
- [x] 2.2 Narrow `graph_memory.retrieval.execution.service.run_retrieval` to execute an already-built `RetrievalMethod`.
- [x] 2.3 Update retrieval scripts and tests to import and call the application use case.

## 3. Tuning Boundary Cleanup

- [x] 3.1 Move `InitialScoreCache` and initial-score precomputation into `graph_memory.retrieval.tuning.initial_scores`.
- [x] 3.2 Update graph-rerank tuning internals and script wiring to pass `DenseRuntime` instead of loose dense prefix fields.

## 4. Guardrail Fixes and Docs

- [x] 4.1 Replace domain imports from root workflow integration ports with owned implementation-module imports.
- [x] 4.2 Update durable architecture, abstraction, retrieval contract, testing strategy, handoff, and docs index references.

## 5. Validation

- [x] 5.1 Run focused boundary and retrieval behavior tests.
- [x] 5.2 Run lint/type/OpenSpec validation or document any environment blocker with exact command output.
