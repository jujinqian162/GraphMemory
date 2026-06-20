## Why

The current request-first refactor removes most dataset record leakage, but retrieval execution still assembles method-family requests by inspecting concrete method classes, and evaluation still assumes one fixed HotpotQA-style evidence metric table. This blocks the intended LongMemEval path where a dataset adapter owns projection while retriever methods and metric computation stay reusable.

## What Changes

- Make retrieval execution request-authoritative: callers pass execution tasks that already contain the exact method-family request to run.
- Remove concrete method-class branching from retrieval execution; stage or registry adapters assemble `TextRankingRequest`, `GraphRankingRequest`, or `TemporalMemoryRankingRequest` before execution.
- Restore a typed `RetrievalMethod.rank_task` boundary instead of accepting unconstrained request objects.
- Make checkpoint-backed R-GCN inference use the graph carried by `GraphRankingRequest` as the authoritative graph for that invocation.
- Introduce a metric-suite evaluation boundary so evidence metrics and future LongMemEval metrics can coexist without putting dataset-specific metric columns in retrievers.
- Keep current HotpotQA public CLI behavior, method names, ranking formulas, graph-rerank scoring math, Dense-FT behavior, R-GCN model math, and existing evidence metric formulas unchanged.
- Do not add LongMemEval data ingestion in this change; this change prepares the production boundary that LongMemEval will use.

## Capabilities

### New Capabilities
- `request-authoritative-retrieval-execution`: Retrieval execution runs preassembled method-family requests and does not infer request shape from concrete method implementations.
- `metric-suite-evaluation-boundary`: Evaluation selects a metric suite that owns its eval units, aggregate rows, validation, and failure-case behavior.

### Modified Capabilities

## Impact

- Affected production areas: `graph_memory/retrieval/contracts.py`, `graph_memory/retrieval/execution/`, `graph_memory/stages/retrieve.py`, `graph_memory/registry/retrieval*.py`, `graph_memory/models/graph_retriever/inference.py`, `graph_memory/evaluation/`, `graph_memory/contracts/metrics.py`, `graph_memory/validation/metrics.py`, scripts/stages that call evaluation, tests, and durable docs.
- Dataset projectors remain in dataset-aware modules. HotpotQA stages may still load HotpotQA artifacts, but they must project before calling reusable retrieval or evaluation services.
- No new runtime dependency is introduced.
