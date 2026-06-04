## Why

The core package refactor moved retrieval into domain packages, but the top-level retrieval run still mixes use-case orchestration with method execution. `graph_memory.retrieval.execution.service` still accepts loose dense/checkpoint parameters, which weakens the Single Level of Abstraction rule and leaves the old universal-context pressure in a new location.

## What Changes

- Add an application-owned retrieval run use case that accepts a single `RunRetrievalRequest`, resolves method-family build requests, builds the retrieval method, and delegates execution.
- Narrow `retrieval.execution.service` so it executes an already-built `RetrievalMethod` instead of resolving methods or constructing dense runtime state.
- Move graph-rerank initial-score precomputation out of execution and make tuning pass typed dense runtime state instead of loose dense prefix parameters.
- Update scripts and internal tests to call the application use case while preserving public CLI arguments, defaults, method names, and artifact schemas.
- Strengthen architecture tests so boundary checks cover retrieval execution and domain imports from root workflow integration ports.
- Update durable docs to describe the application boundary and the stricter dependency constraints.

## Capabilities

### New Capabilities
- `retrieval-usecase-boundary`: Application-level retrieval run orchestration and narrowed retrieval execution boundaries.
- `core-boundary-guardrails`: Automated and documented guardrails for application/retrieval layering and root workflow integration ports.

### Modified Capabilities

None.

## Impact

- Affected code: `graph_memory/application/`, `graph_memory/retrieval/execution/`, `graph_memory/retrieval/tuning/`, `scripts/run_retrieval.py`, `scripts/run_trainable_retrieval.py`, `scripts/tune_graph_rerank.py`, and focused tests.
- Affected docs: durable architecture, abstraction, retrieval contract, testing strategy, implementation handoff, and docs index.
- External compatibility: public script CLI contracts and artifact schemas remain unchanged.
