## 1. Boundary Tests And Inventory

- [x] 1.1 Add architecture tests proving `graph_memory.retrieval.execution.service` does not import or branch on concrete retrieval method classes.
- [x] 1.2 Add a focused retrieval execution test where a fake graph method receives an already-built `GraphRankingRequest`.
- [x] 1.3 Add a focused retrieval execution test where a fake temporal method receives an already-built `TemporalMemoryRankingRequest`.
- [x] 1.4 Add a focused R-GCN inference test where `request.graph` differs from a cached loader graph and the request graph is used.
- [x] 1.5 Add metric validation tests showing evidence rows keep strict evidence columns while a non-evidence suite can define different columns.

## 2. Request-Authoritative Retrieval Execution

- [x] 2.1 Replace the unconstrained `RetrievalMethod.rank_task` request parameter with a named request union covering text, graph, and temporal ranking requests.
- [x] 2.2 Update `RetrievalExecutionTask` to carry both the method request and the text request used for result assembly and validation.
- [x] 2.3 Remove `_request_for_method` and concrete method-class imports from retrieval execution.
- [x] 2.4 Update ranked-result assembly calls so execution derives task id, candidates, token counts, latency, and trace from request-level inputs.
- [x] 2.5 Update retrieval execution tests and dense batching tests to construct explicit execution tasks.

## 3. Stage And Registry Request Assembly

- [x] 3.1 Add stage or registry adapter helpers that turn flat runs into `TextRankingRequest` execution tasks.
- [x] 3.2 Add graph-rerank request assembly that computes seed initial scores before creating `GraphRankingRequest` execution tasks.
- [x] 3.3 Add checkpoint-backed R-GCN request assembly that computes seed scores from the configured seed signal provider before creating `GraphRankingRequest` execution tasks.
- [x] 3.4 Add Memory Stream request assembly that selects importance records and temporal metadata before creating `TemporalMemoryRankingRequest` execution tasks.
- [x] 3.5 Update `graph_memory.stages.retrieve`, `scripts/run_retrieval.py`, and registry builder tests to use the new execution task shape without changing public CLI flags.

## 4. R-GCN Request Graph Authority

- [x] 4.1 Update `GraphRetrieverInference.rank_task` to build full-ranking batches from `request.graph`.
- [x] 4.2 Remove or demote internal graph lookup from inference so cached graphs cannot override request graphs.
- [x] 4.3 Preserve checkpoint loader behavior and provenance while ensuring request assembly supplies the graph at inference time.
- [x] 4.4 Update R-GCN retrieval, tensorization, and checkpoint tests for the request-authoritative graph path.

## 5. Metric Suite Evaluation Boundary

- [x] 5.1 Introduce a small metric suite protocol or callable interface that owns evaluation input, aggregate rows, validation, and optional failure cases.
- [x] 5.2 Move current `evaluate_results` evidence behavior behind an evidence metric suite while preserving all existing formulas and column names.
- [x] 5.3 Change metric row contracts to allow suite-specific row dictionaries while keeping evidence metric rows strongly validated.
- [x] 5.4 Update metric validation so validation is selected by suite instead of one global fixed evidence column list.
- [x] 5.5 Move current evidence failure-case generation behind the evidence suite or an explicitly paired evidence failure-case builder.

## 6. Evaluate Stage, Scripts, And Reports

- [x] 6.1 Update `graph_memory.stages.evaluate` to select the evidence suite for current HotpotQA runs after projecting labels.
- [x] 6.2 Update evaluation scripts and tests so reusable evaluation service code does not import HotpotQA projectors.
- [x] 6.3 Keep aggregate table/report behavior on the evidence-suite path unchanged.
- [x] 6.4 Add docs that explain metric suites as the extension point for LongMemEval turn/session/answer metrics.

## 7. Verification

- [x] 7.1 Run focused retrieval execution, GraphRerank, Memory Stream, R-GCN retrieval, and dense batching tests.
- [x] 7.2 Run focused evaluation, metric validation, failure-case, aggregate table, and stage tests.
- [x] 7.3 Run architecture/source guard tests for dataset and method boundary leaks.
- [x] 7.4 Run `uv run pytest tests -q` outside the Windows filesystem sandbox.
- [x] 7.5 Run `uv run ruff check`, `uv run basedpyright --level error`, `python -m compileall graph_memory scripts tests`, `git diff --check`, and `openspec validate --all --strict`.
