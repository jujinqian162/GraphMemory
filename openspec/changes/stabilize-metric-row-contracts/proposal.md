## Why

The LongMemEval adaptation exposed two contract leaks: metric rows are now suite-specific but still typed as one evidence-shaped `MetricRow`, and Memory Stream real-time recency is required by scoring but is carried through free-form metadata keys. These leaks make the current branch work, but they invite dataset-specific patches when the next dataset or metric suite is added.

## What Changes

- Split metric row contracts into explicit suite-owned row types, including `EvidenceMetricRow` and `LongMemEvalMetricRow`, instead of making one `MetricRow` pretend every suite has the same columns.
- Add a suite-owned metric table schema so evaluation and aggregation select columns from the selected metric suite, not by sniffing the first row for LongMemEval-only columns.
- Route workflow aggregation through an explicit metric suite or dataset-provided schema while avoiding ad hoc `if dataset == "longmemeval"` checks in generic table code.
- Replace Memory Stream recency metadata magic keys with a typed temporal recency spec on `TemporalMemoryRankingRequest`.
- Keep dataset projectors responsible for translating dataset records into consumer requests: HotpotQA and 2Wiki use position recency, LongMemEval uses real-time recency.
- Extract request-owned Memory Stream importance validation into one shared helper used by formal retrieval and tuning.
- Preserve current metric formulas, method names, artifact paths, and LongMemEval workflow behavior.

## Capabilities

### New Capabilities

- `suite-owned-metric-rows`: Metric suites own their row type, table schema, validation, and failure-case behavior.
- `typed-temporal-memory-signals`: Temporal memory requests carry typed recency and validated request-owned importance signals instead of scorer-required metadata keys.

### Modified Capabilities

## Impact

- Affected code areas: `graph_memory/contracts/metrics.py`, `graph_memory/evaluation/suites.py`, `graph_memory/evaluation/tables.py`, `graph_memory/validation/metrics.py`, `graph_memory/stages/evaluate.py`, `scripts/evaluate_retrieval.py`, `scripts/aggregate_tables.py`, `scripts/workflow/workflows.py`, `graph_memory/retrieval/requests.py`, `graph_memory/retrieval/methods/memory_stream/`, `graph_memory/retrieval/tuning/memory_stream.py`, dataset projectors, tests, and docs.
- No new external dependency.
- No public method id, dataset id, or existing experiment recipe is removed.
