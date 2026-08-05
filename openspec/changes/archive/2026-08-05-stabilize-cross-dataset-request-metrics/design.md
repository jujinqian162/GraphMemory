## Context

The current cross-dataset refactor has moved HotpotQA records behind dataset projectors and introduced consumer-owned requests such as `TextRankingRequest`, `GraphRankingRequest`, `TemporalMemoryRankingRequest`, and `EvidenceEvaluationRequest`. That is enough for BM25 and dense retrieval, but two production boundaries still prevent future datasets such as LongMemEval from being mostly adapter/projection work.

First, `graph_memory.retrieval.execution.service` still decides which request to build by checking concrete method classes. That makes retrieval execution a hidden projection layer and forces new method families or dataset-specific request assembly to edit retrieval execution. Second, evaluation still has one global evidence metric table and one global metric validator. LongMemEval-style turn support, session support, and answer quality metrics cannot be added cleanly without either expanding the HotpotQA evidence evaluator or bypassing it.

This change stabilizes those two boundaries while preserving current HotpotQA behavior. It does not add LongMemEval parsing or experiment recipes.

## Goals / Non-Goals

**Goals:**

- Make retrieval execution consume a preassembled method-family request rather than deriving request shape from the concrete method instance.
- Restore a typed `RetrievalMethod.rank_task` protocol over the supported request union.
- Move GraphRerank, Memory Stream, and R-GCN request assembly into stage/registry adapter code where method requirements are already known.
- Make checkpoint-backed R-GCN inference use the `GraphRankingRequest.graph` carried by the request as the authoritative graph for that invocation.
- Introduce metric-suite evaluation so the existing evidence metric suite and future LongMemEval metric suite can have different eval units, rows, validators, and failure-case logic.
- Preserve current public method names, HotpotQA CLI behavior, artifact file names, ranking formulas, R-GCN model math, Dense-FT behavior, and existing evidence metric formulas.

**Non-Goals:**

- Do not add LongMemEval ingestion, raw parsing, prepared artifacts, graph rules, or experiment configs in this change.
- Do not implement a full dataset/projection/workflow capability planner.
- Do not introduce new graph edge types or change R-GCN relation vocab semantics in this change.
- Do not redesign GraphRerank scoring, Memory Stream scoring, Dense-FT training, or R-GCN training.
- Do not remove HotpotQA-aware scripts or stages; they remain valid dataset-aware boundaries.

## Decisions

### Decision: Execution tasks carry the exact method request

`RetrievalExecutionTask` will carry the request object that should be passed to `RetrievalMethod.rank_task`, plus the `TextRankingRequest` needed for ranked-result validation and token accounting. Retrieval execution will loop over tasks, call `rank_task(task.method_request, top_k=...)`, assemble ranked results from the text request and trace, and validate results.

Alternative considered: keep `RetrievalExecutionTask(text_request, graph, temporal_metadata)` and improve the existing `_request_for_method()` helper. That still leaves execution responsible for method-family projection and continues the concrete-class branching problem.

### Decision: Request assembly belongs to stage/registry adapters

The retrieve stage or registry adapter will assemble method requests after it has built the retrieval method and loaded required artifacts. Flat methods receive `TextRankingRequest`. GraphRerank receives `GraphRankingRequest` with graph and explicit initial scores. Memory Stream receives `TemporalMemoryRankingRequest` with selected importance and temporal metadata. R-GCN receives `GraphRankingRequest` with graph and seed scores from its seed signal provider.

Alternative considered: make every method expose a `prepare_request()` hook. That would push orchestration into method implementations and make retriever methods responsible for artifact joins that belong at the application/stage boundary.

### Decision: R-GCN inference consumes request graph authority

`GraphRetrieverInference.rank_task()` will batch from `request.graph` rather than looking up a graph from internal `graph_by_task_id`. Loader-level graph indexes may remain as compatibility or construction helpers, but they must not override the graph attached to the request being ranked.

Alternative considered: keep internal graph lookup for performance. This is unnecessary for single-task inference because the request already carries the graph, and the lookup makes projection results non-authoritative.

### Decision: Evaluation is suite-driven, not one fixed table

The existing evidence metrics become an evidence metric suite. A metric suite owns its accepted evaluation request or eval units, aggregate row shape, validation, and optional failure-case generation. Evaluation stages select the suite explicitly. This allows LongMemEval to add turn/session/answer metrics later without expanding retriever code or forcing every metric row to include evidence-only columns.

Alternative considered: add optional LongMemEval columns to the existing `MetricRow`. That would make a single table schema represent unrelated metric families and would keep future datasets coupled to HotpotQA evidence reporting.

### Decision: Keep the current workflow shape for now

This change updates retrieve/evaluate stage boundaries and low-level scripts as needed, but it does not replace `scripts/workflow/` with the planning skeleton under `abstraction/`. Dataset-specific workflow selection remains a later change.

Alternative considered: land the full capability planner now. That would mix this boundary stabilization with dataset registry, projection registry, and workflow DAG changes before LongMemEval has real production artifacts to validate the design.

## Risks / Trade-offs

- Request assembly can be duplicated between scripts, stages, and tests -> centralize helper functions in stage/registry adapter modules and cover GraphRerank, Memory Stream, and R-GCN with focused tests.
- Changing `RetrievalExecutionTask` can touch many tests -> migrate tests by constructing the explicit method request they expect the method to consume instead of adding compatibility branches.
- R-GCN graph authority change may expose tests that relied on loader graph lookup -> add a focused test where the request graph is different from the loader graph and assert the request graph wins.
- Metric-suite abstractions can grow too broad -> implement only the evidence suite and generic suite protocol needed by current evaluation, leaving LongMemEval suite implementation to the dataset change.
- Existing aggregate/report code expects evidence metric columns -> keep evidence suite output unchanged and route current report/table code through the evidence suite path.

## Migration Plan

1. Add failing architecture and behavior tests for request-authoritative retrieval execution and suite-owned metric validation.
2. Update retrieval request contracts and `RetrievalMethod` typing.
3. Replace execution-time method branching with preassembled method requests.
4. Move GraphRerank, Memory Stream, and R-GCN request assembly into retrieve stage or registry adapter helpers.
5. Update R-GCN inference to batch from `GraphRankingRequest.graph`.
6. Introduce metric suite protocols and migrate current evidence metrics into the first suite.
7. Update metric validation, failure-case generation, evaluate stage, scripts, tests, and docs.
8. Run focused retrieval/evaluation tests, architecture boundary tests, full pytest, lint, type checking, and strict OpenSpec validation.

Rollback is batch-local: retrieval execution and evaluation suite migration are separable. If metric-suite migration is delayed, the request-authoritative retrieval execution change can still ship independently.

## Open Questions

None for this change. New LongMemEval edge semantics and metric definitions should be decided in the later LongMemEval dataset change.
