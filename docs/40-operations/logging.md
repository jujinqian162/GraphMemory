# Logging And Run Records

Date: 2026-05-20

Status: Working reference.

## Goal

Logging should support debugging, auditability, and experiment reproduction without becoming a monitoring framework. A failed or surprising result should be traceable from final metrics back to config, input artifacts, graph statistics, retrieval settings, and per-task diagnostics.

## Logging Principles

- Human console logs should explain progress and major decisions.
- Structured run records should preserve reproducibility details.
- Logs must not silently hide failed assumptions.
- Logs should not contain label-only fields during retrieval or graph construction.
- Debug details should be opt-in when they can become large.
- Every script should make the effective configuration inspectable.

## Output Types

| Output | Format | Purpose |
|---|---|---|
| Console log | human-readable text | Show progress, counts, paths, and failures. |
| Run summary | JSON | Reproduce and audit a run. |
| Debug records | JSON/JSONL | Inspect selected task-level details. |
| Result artifacts | JSON/CSV | Scientific outputs consumed by later steps. |

## Console Logging

Use concise stage-based messages:

```text
[prepare_hotpotqa] read raw examples: 90447
[prepare_hotpotqa] sampled split: train count=5000 seed=13 offset=0
[prepare_hotpotqa] wrote inputs: data/hotpotqa/processed/train_memory_tasks.input.json
[prepare_hotpotqa] wrote labels: data/hotpotqa/processed/train_memory_tasks.labels.json
```

Recommended levels:

| Level | Use |
|---|---|
| `INFO` | Stage starts/ends, counts, artifact paths, effective method/config summary. |
| `WARNING` | Non-fatal but suspicious conditions that do not change outputs. |
| `ERROR` | About to fail because a contract or runtime assumption is violated. |
| `DEBUG` | Per-task details, score components, sampled examples. Off by default. |

Fail-fast rule:

- If an issue invalidates the experiment, raise an exception instead of logging a warning and continuing.

## Logging Boundaries

Logging should live at orchestration boundaries, not inside low-level pure algorithms.

| Layer | Should log? | Reason |
|---|---|---|
| CLI scripts | Yes | They know command context, paths, and user-visible stages. |
| Batch service functions | Limited | They know counts, timings, method choice, and per-run progress. |
| Validators | Limited | They may log validation stage start/end, then raise on violation. |
| Core metric functions | No | Pure functions should be deterministic and quiet. |
| Text/entity utility functions | No | Too low-level; logs would be noisy. |
| Graph edge scoring helpers | No | Debug should be returned as data if needed. |
| Retriever `rank()` | Normally no | The caller should log method-level progress and timing. |
| Rerank formula | No | Score components should be returned, not logged internally. |
| I/O helpers | No | Scripts should log paths before/after calling I/O. |

Rule of thumb:

```text
If a function is easy to unit test with a tiny fixture, it probably should not log.
If a function represents a user-visible experiment stage, it probably should log.
```

## Level Policy By Situation

### Use `INFO` For

- Script start and successful finish.
- Effective config summary.
- Input and output artifact paths.
- Number of records read or written.
- Dataset split count, seed, and offset.
- Method name and top-k cutoff.
- Dense encoder model and prefixes.
- Graph rerank config path and selected parameters.
- Graph aggregate statistics.
- Average retrieval latency and total runtime.
- Metric output paths.

Examples:

```text
INFO [run_retrieval] method=dense tasks=1000 top_k=10
INFO [run_retrieval] dense_model=intfloat/e5-base-v2 query_prefix="query: " passage_prefix="passage: "
INFO [build_graphs] graphs=1000 avg_nodes=52.4 avg_edges=311.8
```

### Use `WARNING` For

Only use warnings for conditions that are suspicious but explicitly do not change scientific outputs.

Acceptable warnings:

- Optional compatibility output was not requested.
- Optional debug output is disabled.
- Optional environment metadata could not be collected.
- spaCy was requested but an optional model name was not provided, if the command explicitly allows heuristic fallback.
- A debug limit truncated debug records.
- A non-result file in `results/` was skipped by aggregation with a clear reason.

Do not warn-and-continue for:

- Missing required input files.
- Invalid schema.
- Unsupported method.
- Split overlap.
- Missing graph for graph rerank.
- Gold labels missing during evaluation.
- Prediction task IDs that do not match labels.
- NaN or infinite scores.

Those should raise.

### Use `ERROR` For

Use `ERROR` at script boundaries immediately before exiting due to an exception, if the logging setup catches top-level failures.

Examples:

```text
ERROR [evaluate_retrieval] prediction task IDs do not match label task IDs
ERROR [build_graphs] graph edge references missing node: task_id=hotpot_000123 node_id=m52
```

Rules:

- Do not log an error deep inside a helper and then also raise if the top-level script will log it again.
- Prefer one clear top-level error with the exception message and failing artifact path.

### Use `DEBUG` For

Debug logs are off by default and should be bounded.

Use `DEBUG` for:

- First N sampled task IDs.
- Per-task graph node/edge counts for debug-limited tasks.
- Top-k ranked nodes without gold labels during retrieval.
- Score component summaries for graph rerank.
- Metric intermediate values during evaluation, where labels are allowed.

Rules:

- If debug output is large or structured, write a debug artifact instead of console logs.
- Retrieval/debug logs must not print `gold_answer`, `gold_evidence_sentence_ids`, `supporting_facts`, or `is_gold` fields.
- Evaluation debug may include gold labels because it is label-aware, but it must be clearly produced by evaluation.

## Places That Should Not Log

These should return values or raise exceptions instead:

- `content_tokens`
- `compute_idf`
- `lexical_score`
- `heuristic_entities`
- `extract_entities`
- `recall_at`
- `full_support_at`
- `mrr`
- `connected_evidence_at`
- small graph edge scoring helpers
- score normalization helpers
- `induced_retrieved_subgraph`
- dataclass constructors and type conversion helpers

Reasons:

- They are called many times.
- Logging would make tests noisy.
- Their behavior is better inspected through return values and focused tests.
- Reproducibility is better captured at script/service boundaries.

## Places That Should Log

These should produce `INFO` logs and run summary fields:

- `scripts/prepare_hotpotqa.py`
- `scripts/build_graphs.py`
- `scripts/run_retrieval.py`
- `scripts/tune_graph_rerank.py`
- `scripts/evaluate_retrieval.py`
- `scripts/aggregate_tables.py`

These may produce limited logs if implemented as service functions:

- `run_retrieval(...)`
- `tune_graph_rerank(...)`
- high-level graph building batch functions
- high-level evaluation aggregation functions

Recommendation:

- Scripts own user-facing logs.
- Service functions may return counts/timings/debug records to scripts instead of logging directly.
- If service functions log, pass a logger explicitly or use a module logger; do not create ad hoc print statements.

## Run Summary

Each runnable script must write a compact run summary near its main output.

Suggested file names:

```text
results/run_summary_{script}_{method}.json
data/hotpotqa/processed/run_summary_{script}_{split}.json
```

Shape:

```json
{
  "script": "run_retrieval.py",
  "started_at": "2026-05-20T12:00:00+08:00",
  "finished_at": "2026-05-20T12:05:00+08:00",
  "status": "success",
  "effective_config": {},
  "inputs": {},
  "outputs": {},
  "counts": {},
  "timings": {},
  "environment": {},
  "notes": []
}
```

Required fields:

| Field | Meaning |
|---|---|
| `script` | Script entry point. |
| `started_at`, `finished_at` | Timestamped run boundaries. |
| `status` | `success` or `failed`. |
| `effective_config` | Defaults + config file + CLI overrides. |
| `inputs` | Input artifact paths. |
| `outputs` | Output artifact paths. |
| `counts` | Number of examples, graphs, predictions, rows, etc. |
| `timings` | Wall-clock durations for major stages. |
| `environment` | Python version, platform, important dependency versions when available. |
| `notes` | Explicit caveats such as missing optional metadata. |

## Per-Script Logging Requirements

### `prepare_hotpotqa.py`

Log and summarize:

- input raw path
- split count, seed, offset
- number of raw examples read
- number of input tasks written
- number of label records written
- output paths
- compatibility output path, if requested

### `build_graphs.py`

Log and summarize:

- input task path
- graph config
- number of graphs
- average nodes per graph
- average edges per graph
- edge count by type
- number of isolated memory nodes
- output path

### `run_retrieval.py`

Log and summarize:

- method
- task path
- graph path, if used
- encoder model and prefixes, if dense is used
- graph rerank config, if used
- top-k cutoff for retrieved subgraph
- number of tasks
- average latency
- output path

### `tune_graph_rerank.py`

Log and summarize:

- method
- dev task/label/graph paths
- grid size
- objective definition
- selected config
- selected dev metrics
- tie-break decisions if any
- output config path

### `evaluate_retrieval.py`

Log and summarize:

- prediction path
- label path
- graph path
- number of matched task IDs
- method name
- output metric path
- failure-case debug path, if requested

### `aggregate_tables.py`

Log and summarize:

- input result directory
- included metric files
- output table paths
- skipped files, if any, with explicit reason

## Debug Artifacts

Debug artifacts should be optional and bounded.

Detailed debug artifact formats are defined in `docs/40-operations/debug-artifacts.md`.

Recommended debug outputs:

| Debug artifact | Purpose |
|---|---|
| graph stats JSON | Inspect graph density and edge distribution. |
| score breakdown JSONL | Inspect graph rerank score components per selected task. |
| failure cases JSONL | Inspect examples where full support or connectivity failed. |
| leakage check report | Confirm input/graph artifacts contain no label-only fields. |

Rules:

- Debug artifacts must include `task_id`.
- Debug artifacts should include enough config context to interpret values.
- Large debug outputs should support `--debug_limit`.
- Retrieval debug outputs should not include gold labels unless produced by evaluation.

## Reproducibility Requirements

A completed experiment should answer:

- Which raw data files were used?
- Which split seed and offset were used?
- Which config file was used?
- Which CLI overrides were applied?
- Which encoder model and prefixes were used?
- Which graph config was used?
- Which graph rerank config was selected?
- Which output artifacts were produced?
- How many tasks were processed?
- What command/script produced each result?

These answers should be visible from run summaries and command documentation.

## Recommended Implementation Direction

- Use Python `logging` for console logs.
- Use explicit JSON writing for run summaries.
- Do not make a global logging singleton.
- Pass script name, effective config, paths, counts, and timings into small observability helper functions.
- Keep debug output optional and controlled by CLI flags.
- Prefer returning debug data from core functions over logging inside them.
- Avoid `print()` except possibly in a script's final user-facing CLI message; normal progress should use `logging`.
- When a script fails after it knows where to write its run summary, it should attempt to write a partial summary with `status = "failed"` and the error message.

## Extension Decisions

- Failed-run summaries are required when the script has enough context to safely write them.
- If a failure happens before output paths are known, the console error is sufficient.
