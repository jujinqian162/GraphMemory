## Context

The current branch has the right high-level dataset boundary: LongMemEval parsing and projection live under `graph_memory/datasets/longmemeval`, and retrievers consume request contracts instead of raw dataset records. The remaining issues are narrower contract leaks introduced while making LongMemEval run end to end.

First, `MetricRow` still describes an evidence-shaped row with required evidence columns, but `LongMemEvalMetricSuite` returns rows with turn and session columns instead. The implementation then uses casts and first-row column sniffing to keep CSV output working. That is a table contract mismatch, not a LongMemEval feature.

Second, Memory Stream scoring now supports position recency and real-time recency, but the selected mode and required timestamps are encoded as free-form metadata keys. That keeps LongMemEval out of retriever imports, but it leaves scorer-required data outside the typed request contract.

## Goals / Non-Goals

**Goals:**

- Make metric row shape suite-owned by introducing explicit row types such as `EvidenceMetricRow` and `LongMemEvalMetricRow`.
- Make metric table columns suite-owned through a table schema object or equivalent protocol.
- Remove row-shape sniffing from generic table selection.
- Keep generic workflow/table code driven by suite or schema selection rather than hard-coded LongMemEval conditionals.
- Make Memory Stream recency a typed field on `TemporalMemoryRankingRequest`.
- Keep dataset projectors as the only place that translates dataset-specific temporal information into request contracts.
- Share request-owned importance validation between formal retrieval and tuning.

**Non-Goals:**

- Do not redesign evaluation formulas or report metrics.
- Do not add new LongMemEval metrics beyond the current turn and session retrieval metrics.
- Do not add a general dataset capability planner.
- Do not remove legacy position recency support.
- Do not change public experiment recipe names, method ids, or output artifact locations.

## Decisions

### Decision: Split metric rows by suite

`MetricRow` will stop being a single evidence-shaped typed dict used for all suites. Evidence metrics will use `EvidenceMetricRow`; LongMemEval metrics will use `LongMemEvalMetricRow`; shared table rows can use a generic `MetricTableRow` or `SuiteMetricRow` alias when only dictionary operations are needed.

Alternative considered: keep one `MetricRow` with all suite-specific columns marked `NotRequired`. That would make unrelated metric families share one expanding shape and keep the original contract leak.

### Decision: Metric suites expose table schema

Each metric suite will expose the column groups needed by evaluation and aggregation: main, path, efficiency, and wide columns. `evaluate_retrieval.py` will write the suite wide schema directly after selecting the suite. Aggregation will receive or derive the suite schema from workflow/stage context, not from row contents.

Alternative considered: keep `metric_columns_for_rows()` and make row sniffing more robust. That still makes a table utility infer semantics from incidental columns and will fail when a mixed or empty input appears.

### Decision: Generic workflow code must not special-case LongMemEval tables

Generic table and workflow code can select a metric suite through existing dataset/stage config or a small registry, but it must not branch directly on `dataset == "longmemeval"` to repair table output. Dataset-to-suite selection belongs in a narrow evaluation/schema selector, analogous to dataset-to-request projection.

Alternative considered: add a direct LongMemEval branch in `aggregate_tables.py`. That would solve the immediate CSV issue while preserving the patch pattern this change is meant to remove.

### Decision: Temporal recency is part of the request

`TemporalMemoryRankingRequest` will carry a typed recency spec. The spec should distinguish legacy position recency from real-time recency with explicit fields, for example `PositionRecencySpec(position_by_item_id=...)` and `RealTimeRecencySpec(question_datetime=..., datetime_by_item_id=...)`.

Alternative considered: keep using `metadata["recency_mode"]` but validate it more strictly. That improves errors but still leaves scorer-required fields in free-form metadata.

### Decision: Share request-owned importance validation

Formal retrieval and tuning will call the same helper for request-owned importance scores. The builder may still wrap scores into `TaskImportanceRecord`, and tuning may still build `task_id -> scores`, but numeric validation and missing/extra candidate handling should live in one place.

Alternative considered: leave duplication because tests currently pass. That keeps an avoidable drift point between tuning and retrieval paths.

## Risks / Trade-offs

- Metric suite schema selection can become a hidden dataset registry -> keep the selector narrow and limited to evaluation/table schema, not a general workflow planner.
- Splitting row types can touch many annotations -> prioritize production contracts and tests over exhaustive typing churn in reporting helpers.
- Adding typed recency may require updating several test fixtures -> migrate tests to construct the same request shape as real projectors.
- Aggregation inputs might be empty -> explicit suite/schema selection should make empty aggregation deterministic.

## Migration Plan

1. Add tests that fail on the current contract leaks: LongMemEval metric rows must type-check without evidence columns, table columns must come from suite schema, and Memory Stream must reject missing typed recency rather than missing metadata keys.
2. Split metric row types and add suite table schema ownership.
3. Update evaluation script and aggregation workflow to use explicit schema selection instead of row sniffing.
4. Add typed temporal recency specs and update dataset projectors plus Memory Stream scoring.
5. Extract request-owned importance validation and update builder/tuning callers.
6. Run focused metric, workflow, Memory Stream, and LongMemEval tests, then full validation.

Rollback is straightforward: metric row/schema work and temporal request work are separable. If one half uncovers broader churn, the other half can still land independently.

## Open Questions

None. The chosen direction is to formalize existing contracts, not add new dataset-specific behavior.
