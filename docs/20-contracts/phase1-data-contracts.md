# Phase 1 Data Contracts

Date: 2026-05-20

Status: Historical Phase 1 reference. The maintained project-level artifact contracts now live in `data-contracts.md`.

This document is preserved for Phase 1 provenance and implementation history. Do not treat it as the current source of truth for new work; promote stable schema changes into `data-contracts.md`, `retrieval-contracts.md`, or `model-contracts.md` as appropriate.

## Contract Principles

- Every artifact must have a clear producer and consumer.
- Core fields are strict. Unknown core fields should raise an error unless they are placed under an explicit `metadata` or `debug` object.
- Retrieval and graph construction must consume input-visible artifacts only.
- Evaluation and tuning may consume label artifacts.
- Test-time input artifacts must not contain label-only fields.
- Violations should fail fast with clear exceptions instead of fallback behavior.
- JSON artifacts should be UTF-8 encoded and deterministic when practical.

## Artifact Overview

| Artifact | Producer | Consumer | Purpose |
|---|---|---|---|
| `*_memory_tasks.input.json` | `prepare_hotpotqa.py` | graph construction, retrieval | Input-visible query and memory sentence records. |
| `*_memory_tasks.labels.json` | `prepare_hotpotqa.py` | evaluation, dev tuning | Gold answers and evidence labels. |
| `*_memory_tasks.json` | `prepare_hotpotqa.py` | humans, compatibility only | Combined compatibility artifact. Must not be used by retrieval or graph construction. |
| `*_graphs.json` | `build_graphs.py` | graph rerank, evaluation connectivity | Typed graph over question and memory sentence nodes. |
| `ranked_results_{method}.json` | `run_retrieval.py` | evaluation, analysis | Complete per-task ranking and optional retrieved subgraph. |
| graph rerank config JSON | `tune_graph_rerank.py` or human | graph rerank retrieval | Fixed graph-rerank parameters selected on dev. |
| per-method metric CSV | `evaluate_retrieval.py` | aggregation, reporting | Wide method-level metric rows used as aggregation input. |
| aggregate result CSVs | `aggregate_tables.py` | paper tables, README | Final Phase 1 `main_results.csv`, `path_results.csv`, and `efficiency_results.csv` tables. |
| `run_summary.json` | each runnable script | humans, debugging | Effective config, input/output paths, counts, timings, environment notes. |

## Shared ID Rules

### Task IDs

- `task_id` is the primary join key across inputs, labels, graphs, predictions, and metrics.
- `task_id` must be unique within each artifact.
- The same split must use the same `task_id` values across all derived artifacts.
- Missing or duplicate `task_id` values should raise an error.
- For HotpotQA Phase 1, `task_id` must be derived from the raw HotpotQA `_id`, not from sampled position.
- Required HotpotQA form:

```text
task_id = "hotpot_" + raw_example["_id"]
```

- A raw HotpotQA example without `_id` should fail conversion instead of receiving a generated position-based ID.

### Node IDs

- `q` is reserved for the question node inside graph artifacts.
- Memory sentence node IDs use `m{position}`, for example `m0`, `m1`, `m2`.
- `position` is the flattened sentence index within a task.
- A node referenced by a graph edge, gold label, or ranked result must exist in that task.

## Memory Task Input Contract

File pattern:

```text
data/hotpotqa/processed/{split}_memory_tasks.input.json
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "query": "question text",
    "memory_items": [
      {
        "id": "m0",
        "node_type": "document_sentence",
        "text": "sentence text",
        "source": "Document_Title",
        "sentence_id": 0,
        "position": 0
      }
    ]
  }
]
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Stable task identifier. |
| `query` | string | Retrieval query. |
| `memory_items` | array | Candidate memory sentences. |

Required `memory_items` fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Memory node id, usually `m{position}`. |
| `node_type` | string | Must be `document_sentence` in Phase 1. |
| `text` | string | Sentence text. |
| `source` | string | HotpotQA document title. |
| `sentence_id` | integer | Sentence index within the source document. |
| `position` | integer | Flattened sentence index within the task. |

Forbidden fields:

- `gold_answer`
- `gold_evidence_nodes`
- `gold_dependency_edges`
- `supporting_facts`
- `is_gold`
- `is_gold_evidence`
- `is_gold_edge`

Invariants:

- `memory_items` must not be empty.
- `memory_items[*].id` must be unique within the task.
- `position` must match flattened order.
- `id` should match `m{position}` unless a later dataset explicitly documents a different rule.

## Memory Task Label Contract

File pattern:

```text
data/hotpotqa/processed/{split}_memory_tasks.labels.json
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "gold_answer": "answer text",
    "gold_evidence_nodes": ["m1", "m7"],
    "gold_dependency_edges": []
  }
]
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Stable task identifier matching input artifact. |
| `gold_answer` | string | Gold answer, used only for optional analysis. |
| `gold_evidence_nodes` | array of strings | Gold supporting sentence node IDs. |
| `gold_dependency_edges` | array | Empty for HotpotQA Phase 1 unless a dataset provides dependency labels. |

Invariants:

- Every label record must match exactly one input task.
- Every `gold_evidence_nodes` entry must refer to a memory item in the matching input task.
- `gold_dependency_edges` is empty for HotpotQA Phase 1.

## Graph Contract

File pattern:

```text
data/hotpotqa/processed/{split}_graphs.json
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "nodes": [
      {
        "id": "q",
        "node_type": "question",
        "text": "question text"
      },
      {
        "id": "m0",
        "node_type": "document_sentence",
        "text": "sentence text",
        "source": "Document_Title",
        "sentence_id": 0,
        "position": 0
      }
    ],
    "edges": [
      {
        "source": "q",
        "target": "m0",
        "edge_type": "query_overlap",
        "weight": 2.5,
        "directed": true
      }
    ]
  }
]
```

Allowed node types:

- `question`
- `document_sentence`

Allowed Phase 1 edge types:

- `sequential`
- `query_overlap`
- `entity_overlap`
- `bridge`

Required edge fields:

| Field | Type | Meaning |
|---|---|---|
| `source` | string | Source node id. |
| `target` | string | Target node id. |
| `edge_type` | string | One of the allowed edge types. |
| `weight` | number | Non-negative edge weight. |
| `directed` | boolean | Whether traversal direction matters. |

Forbidden fields:

- Any gold label field.
- Any field whose value directly reveals supporting facts or answer labels.

Invariants:

- Each graph must contain exactly one question node `q`.
- All memory nodes from the input task must appear in the graph.
- Every edge endpoint must exist in `nodes`.
- Edge weights must be finite numbers.
- Graph construction must be reproducible from input-visible task fields and graph config.

## Ranked Result Contract

File pattern:

```text
results/ranked_results_{method}.json
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "method": "bm25_graph_rerank",
    "ranked_nodes": [
      {
        "node_id": "m3",
        "score": 12.4
      }
    ],
    "retrieved_subgraph": {
      "nodes": ["m3"],
      "edges": []
    },
    "latency_ms": 23.5,
    "input_tokens": 640
  }
]
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Matching task id. |
| `method` | string | Retrieval method name. |
| `ranked_nodes` | array | Complete ranked list of memory nodes. |
| `retrieved_subgraph` | object | Induced subgraph for top-k analysis. |
| `latency_ms` | number | Retrieval latency for the task. |
| `input_tokens` | integer | Approximate input token count or `0` if unavailable. |

Invariants:

- `ranked_nodes` must include every memory node exactly once.
- Scores must be finite numbers.
- Ranking order must be descending by score.
- Flat methods may output an empty edge list in `retrieved_subgraph`.
- Graph methods should output the induced top-k subgraph used for analysis.

## Graph Rerank Config Contract

Shape:

```json
{
  "lambda_init": 1.0,
  "lambda_query": 0.1,
  "lambda_neighbor": 0.2,
  "lambda_bridge": 0.1,
  "lambda_path": 0.0,
  "seed_top_s": 30,
  "max_hops": 2,
  "neighbor_type_weights": {
    "sequential": 0.3,
    "entity_overlap": 0.7,
    "bridge": 1.0
  }
}
```

Invariants:

- All lambda values must be finite non-negative numbers.
- `lambda_path` must remain `0.0` for HotpotQA-only Phase 1 unless a fully unsupervised path bonus is explicitly documented and tested.
- `seed_top_s` and `max_hops` must be positive integers.
- `lambda_*` fields weight final score components after normalization.
- `neighbor_type_weights` calibrates memory-to-memory graph edge types used by neighbor propagation and bridge-neighbor scoring.
- `query_overlap` is not a neighbor type weight; query-overlap contribution is controlled by `lambda_query`.
- Deprecated `type_weights` input remains readable for historical run artifacts. Compatibility loading ignores historical `type_weights.query_overlap` and newly written configs must use `neighbor_type_weights`.

## Metric CSV Contract

`evaluate_retrieval.py` should produce a wide per-method metric CSV so aggregation has one complete input row per method.

Wide per-method metric columns:

```text
Method
Recall@2
Recall@5
Recall@10
Evidence F1@5
Evidence F1@10
Full Support@5
Full Support@10
MRR
Connected Evidence Recall@5
Connected Evidence Recall@10
Query-Evidence Connectivity@10
Path Recall@10
Edge Recall@10
Retrieval Latency / Query
```

HotpotQA-only values:

- `Path Recall@10` should be `N/A`.
- `Edge Recall@10` should be `N/A`.

Invariants:

- Metrics must be computed from prediction artifacts, label artifacts, and graph artifacts.
- Metrics must not read gold fields from input task artifacts.
- All numeric metrics except latency should be in `[0.0, 1.0]`.

### Query-Evidence Connectivity

For `Query-Evidence Connectivity@10`:

```text
selected_nodes = top-10 ranked memory node IDs
if any gold evidence node is missing from selected_nodes: score = 0
otherwise build the induced graph over {"q"} union selected_nodes
directed edges are traversed source -> target
undirected edges are traversed both ways
score = 1 only if every gold evidence node is reachable from q
otherwise score = 0
```

HotpotQA Phase 1 label records must contain at least one gold evidence node; an empty gold set is invalid rather than vacuously connected.

### Aggregate Table Outputs

`aggregate_tables.py` must split the wide per-method metric rows into canonical final outputs:

`results/main_results.csv`:

```text
Method
Recall@2
Recall@5
Recall@10
Evidence F1@5
Evidence F1@10
Full Support@5
Full Support@10
MRR
```

`results/path_results.csv`:

```text
Method
Connected Evidence Recall@5
Connected Evidence Recall@10
Query-Evidence Connectivity@10
Path Recall@10
Edge Recall@10
```

`results/efficiency_results.csv`:

```text
Method
Index Build Time
Graph Construction Time
Retrieval Latency / Query
Memory Size
Avg Retrieved Nodes
Avg Retrieved Edges
```

## Run Summary Contract

Each runnable script must write a summary record next to its output.

Shape:

```json
{
  "script": "run_retrieval.py",
  "started_at": "2026-05-20T12:00:00+08:00",
  "finished_at": "2026-05-20T12:05:00+08:00",
  "effective_config": {},
  "inputs": {},
  "outputs": {},
  "counts": {},
  "timings": {},
  "notes": []
}
```

Purpose:

- Make temporary CLI overrides visible.
- Record artifact paths and counts.
- Support debugging without adding a heavy logging system.

## Validation Strategy

Recommended validators:

- `validate_memory_task_inputs(records)`
- `validate_memory_task_labels(records, inputs_by_task_id)`
- `validate_graphs(graphs, inputs_by_task_id)`
- `validate_ranked_results(predictions, inputs_by_task_id)`
- `validate_metric_rows(rows)`

Validation should run at script boundaries:

- After reading input artifacts.
- Before writing output artifacts.
- Before computing metrics.

Validation should raise exceptions for contract violations.

## Extension Decisions

- Unknown top-level fields should fail unless they are placed under explicit `metadata` or `debug` objects.
- Label-only fields remain forbidden in input-visible artifacts even if nested under extension containers.
- Formal JSON Schema files are deferred until contract validation becomes too large for readable Python validators.
