## 1. Contract Tests

- [x] 1.1 Add metric contract tests proving evidence and LongMemEval rows use distinct row types without casts to one evidence-shaped row.
- [x] 1.2 Add table schema tests proving column groups come from an explicit suite schema and do not depend on `Turn Recall@5` row sniffing.
- [x] 1.3 Add workflow or aggregation tests proving empty metric inputs still write the configured suite columns.
- [x] 1.4 Add Memory Stream tests proving real-time recency is read from typed request data, not metadata keys.
- [x] 1.5 Add formal retrieval and tuning tests proving request-owned importance validation uses the same behavior.

## 2. Suite-Owned Metric Rows

- [x] 2.1 Split `MetricRow` into explicit `EvidenceMetricRow`, `LongMemEvalMetricRow`, and a generic suite/table row alias for shared dictionary handling.
- [x] 2.2 Add a metric table schema contract that exposes main, path, efficiency, and wide columns per suite.
- [x] 2.3 Move evidence and LongMemEval column definitions behind their metric suites or a suite schema selector.
- [x] 2.4 Update metric validation to dispatch through suite-owned validators without relying on row-shape sniffing.
- [x] 2.5 Update failure-case and evaluation return types so suite-specific rows flow without unsafe casts.

## 3. Evaluation And Aggregation Wiring

- [x] 3.1 Update `run_evaluate_stage` and `scripts/evaluate_retrieval.py` to write metrics with the selected suite wide schema.
- [x] 3.2 Update aggregation to accept an explicit suite/schema source from workflow or CLI context.
- [x] 3.3 Update `scripts/workflow/workflows.py` and related stage planning so aggregate commands carry the suite/schema context.
- [x] 3.4 Remove `metric_columns_for_rows()` row-content detection and replace callers with explicit schema selection.
- [x] 3.5 Update existing HotpotQA, 2Wiki, and LongMemEval tests for the new schema path while preserving output column names.

## 4. Typed Temporal Memory Signals

- [x] 4.1 Add typed temporal recency spec classes or typed dictionaries to `graph_memory/retrieval/requests.py`.
- [x] 4.2 Update HotpotQA and 2Wiki temporal projectors to emit position recency specs.
- [x] 4.3 Update LongMemEval temporal projector to emit a real-time recency spec.
- [x] 4.4 Update Memory Stream scoring to dispatch on the typed recency spec instead of metadata keys.
- [x] 4.5 Keep non-authoritative metadata available for provenance/debug output without scorer-required keys.

## 5. Shared Importance Validation

- [x] 5.1 Extract request-owned importance validation into a Memory Stream helper shared by builder and tuning paths.
- [x] 5.2 Update `graph_memory/registry/retrieval_builders.py` to call the shared helper when no external importance artifact is supplied.
- [x] 5.3 Update `graph_memory/retrieval/tuning/memory_stream.py` to call the shared helper with the tuning completeness requirement.
- [x] 5.4 Remove duplicated validation branches and keep error messages specific enough for failed task/item diagnosis.

## 6. Verification

- [x] 6.1 Run targeted metric/evaluation/table tests.
- [x] 6.2 Run targeted Memory Stream, LongMemEval projector, and retrieval builder tests.
- [x] 6.3 Run `uv run pytest -q` outside the Windows filesystem sandbox.
- [x] 6.4 Run `uv run ruff check .` and `uv run basedpyright --level error`.
- [x] 6.5 Run `openspec validate stabilize-metric-row-contracts --strict` and `git diff --check`.

## 7. Review Follow-Ups

- [x] 7.1 Route graph-rerank tuning through the selected metric suite and selection key instead of the evidence-only evaluator.
- [x] 7.2 Include dataset and metric-suite context in workflow status checks so stale artifacts are not marked complete after contract changes.
