# Data Contracts

Status: Maintained project-level reference.

This document defines disk artifact contracts for the Graph Memory project across supported phases. It supersedes phase-specific schema definitions in `phase1-data-contracts.md` and the artifact sections of `phase2-trainable-retriever-contracts.md`.

## Contract Principles

- Every artifact has a clear producer and consumer.
- Core fields are strict. Unknown core fields should raise an error unless they are placed under an explicitly documented `metadata` or `debug` object.
- Input-visible artifacts must not contain label-only fields.
- Retrieval and graph construction consume input-visible artifacts only.
- Evaluation, tuning, and training may consume label artifacts when explicitly documented.
- JSON artifacts are UTF-8 encoded and deterministic when practical.
- Validators fail fast with clear exceptions and do not repair records.

## Artifact Overview

| Artifact | Producer | Consumer | Purpose |
|---|---|---|---|
| `*_memory_tasks.input.json` | `scripts/prepare_hotpotqa.py` | HotpotQA projectors for graph construction, retrieval, pair building, and model input | HotpotQA ranking records with `question` and `candidate_sentences`. File name stays stable for workflow compatibility. |
| `*_memory_tasks.labels.json` | `scripts/prepare_hotpotqa.py` | evaluation, tuning, pair building, trainable dev evaluation | HotpotQA label records with gold answer and supporting sentence IDs. |
| `*_memory_tasks.json` | `scripts/prepare_hotpotqa.py` | humans, compatibility only | Combined compatibility artifact. Must not be used by retrieval or graph construction. |
| `*_graphs.json` | `scripts/build_graphs.py` | graph rerank, trainable graph retriever, evaluation connectivity | Typed graph over question and evidence candidate nodes. |
| `{split}_pairs.json` | `scripts/build_train_pairs.py` | trainable graph retriever training | Supervised query-node examples for training. |
| `{split}_pairs.summary.json` | `scripts/build_train_pairs.py` | humans, reproducibility checks | Negative sampling summary and effective sampling config. |
| `ranked_results_{method}.json` | `scripts/run_retrieval.py`, trainable inference | evaluation, analysis | Complete per-task ranking and optional retrieved subgraph. |
| graph rerank config JSON | `scripts/tune_graph_rerank.py` or human | graph rerank retrieval | Fixed graph-rerank parameters selected on dev. |
| per-method metric CSV | `scripts/evaluate_retrieval.py` | aggregation, reporting | Wide method-level metric rows used as aggregation input. |
| aggregate result CSVs | `scripts/aggregate_tables.py` | paper tables, README | Final `main_results.csv`, `path_results.csv`, and `efficiency_results.csv` tables. |
| `run_summary.json` | each runnable script | humans, debugging | Effective config, paths, counts, timings, and environment notes. |

## Shared ID Rules

### Task IDs

- `task_id` is the primary join key across ranking records, labels, graphs, predictions, metrics, and train pairs.
- `task_id` must be unique within each artifact.
- The same split must use the same `task_id` values across all derived artifacts.
- Missing or duplicate `task_id` values are invalid.
- For HotpotQA, `task_id` is derived from raw HotpotQA `_id`:

```text
task_id = "hotpot_" + raw_example["_id"]
```

- A raw example without `_id` must fail conversion instead of receiving a generated position-based ID.

### Candidate And Node IDs

- `q` is reserved for the question node inside graph artifacts and model batches.
- HotpotQA candidate sentence IDs use `m{position}`, for example `m0`, `m1`, `m2`.
- `position` is the flattened sentence index within a task.
- Any candidate ID referenced by graph edges, gold labels, train pairs, or ranked results must exist in the matching HotpotQA ranking record.

## HotpotQA Ranking Records

File pattern:

```text
data/hotpotqa/processed/{split}_memory_tasks.input.json
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "question": "question text",
    "candidate_sentences": [
      {
        "sentence_id": "m0",
        "title": "Document_Title",
        "sentence_index": 0,
        "position": 0,
        "text": "sentence text"
      }
    ]
  }
]
```

Required task fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Stable HotpotQA task identifier. |
| `question` | string | Input-visible question text. |
| `candidate_sentences` | array | Candidate evidence sentences. |

Required `candidate_sentences` fields:

| Field | Type | Meaning |
|---|---|---|
| `sentence_id` | string | Candidate sentence ID, usually `m{position}`. |
| `title` | string | Source document title. |
| `sentence_index` | integer | Sentence index within the source document. |
| `position` | integer | Flattened sentence index within the task. |
| `text` | string | Sentence text. |

Forbidden fields:

- `gold_answer`
- `gold_evidence_sentence_ids`
- `gold_dependency_edges`
- `supporting_facts`
- `is_gold`
- `is_gold_evidence`
- `is_gold_edge`

Invariants:

- `candidate_sentences` must not be empty.
- `candidate_sentences[*].sentence_id` must be unique within the task.
- `position` must match flattened order.
- `sentence_id` should match `m{position}` unless a later dataset documents a different rule in its own record contract.

## HotpotQA Label Records

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
    "gold_evidence_sentence_ids": ["m1", "m7"],
    "gold_dependency_edges": []
  }
]
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Stable task identifier matching the ranking record. |
| `gold_answer` | string | Gold answer, used only for optional analysis. |
| `gold_evidence_sentence_ids` | array of strings | Gold supporting sentence IDs. |
| `gold_dependency_edges` | array | Dependency edge labels if a dataset provides them; empty for HotpotQA sentence evidence. |

Invariants:

- Every label record must match exactly one ranking record.
- Every `gold_evidence_sentence_ids` entry must refer to a candidate sentence in the matching ranking record.
- HotpotQA label records must contain at least one gold evidence sentence ID.

## Consumer Request Contracts

Dataset artifacts are not method inputs. Dataset-specific projectors translate records into stable consumer-side requests:

```text
HotpotQARankingRecord -> HotpotQA-owned prepared dataset artifact
HotpotQALabelRecord -> HotpotQA-owned label artifact
Dataset-specific record -> dataset-specific projector -> consumer request
```

Current stable request contracts:

| Request | Owner | Main consumers |
|---|---|---|
| `TextRankingRequest` | `graph_memory.retrieval.requests` | BM25, Dense, Dense-FT, text seed providers. |
| `GraphBuildRequest` | `graph_memory.graphs.requests` | graph construction. |
| `GraphRankingRequest` | `graph_memory.retrieval.requests` | Graph Rerank and trainable graph retrieval. |
| `TemporalMemoryRankingRequest` | `graph_memory.retrieval.requests` | Memory Stream. |
| `EvidenceEvaluationRequest` | `graph_memory.evaluation.requests` | evidence metric suite aggregation and failure-case analysis. |

A shared task view can be introduced later only if multiple dataset projectors genuinely share the same stable intermediate semantics.
## Memory Graphs

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
        "node_type": "graph_item",
        "node_kind": "document_sentence",
        "text": "sentence text",
        "source_ref": "Document_Title",
        "group_key": "document:Document_Title",
        "sequence_index": 0,
        "metadata": {
          "title": "Document_Title",
          "position": 0
        }
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
- `graph_item`

Graph item node fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Domain graph item id matching the projected request item id. |
| `node_type` | string | Must be `graph_item` for non-query nodes. |
| `node_kind` | string | Dataset/projector-provided item kind, such as `document_sentence`. |
| `text` | string | Text used by graph rules and model features. |
| `source_ref` | string, optional | Human-readable source reference, such as a document title or session id. |
| `group_key` | string, optional | Grouping key for sequential or same-source rules. |
| `sequence_index` | integer, optional | Order within `group_key` or source. |
| `metadata` | object, optional | Dataset-owned extra fields that are not graph-domain required fields. |

Allowed graph edge types:

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

Invariants:

- Each graph must contain exactly one question node `q`.
- All projected graph item ids from the graph build request must appear in the graph.
- Every edge endpoint must exist in `nodes`.
- Edge weights must be finite non-negative numbers.
- Graph construction must be reproducible from input-visible task fields and graph config.
- Graph artifacts must not contain any gold label field or answer-derived marker.

## Train Pair Artifact

File pattern:

```text
data/hotpotqa/processed/{split}_pairs.json
```

Producer:

```text
scripts/build_train_pairs.py
```

Consumers:

```text
scripts/train_method.py --method dense_rgcn_graph_retriever
graph_memory.training_pairs
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "node_id": "m7",
    "label": 1,
    "sample_type": "positive"
  }
]
```

Allowed `sample_type` values:

- `positive`
- `easy_random`
- `hard_bm25`
- `hard_dense`
- `hard_graph_neighbor`

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | string | Task join key matching ranking, label, and graph artifacts. |
| `node_id` | string | Supervised memory node id; must not be `q`. |
| `label` | integer | Binary evidence label, `1` for gold evidence and `0` for sampled negative. |
| `sample_type` | string | Sampling source used to create this row. |

Invariants:

- `task_id` must exist in ranking, label, and graph artifacts.
- `node_id` must be a memory node in the task, never `q`.
- `label=1` rows must exactly come from `gold_evidence_sentence_ids`.
- `label=0` rows must not include any gold evidence sentence ID.
- `sample_type="positive"` requires `label=1`.
- All other sample types require `label=0`.
- Duplicate `(task_id, node_id, sample_type)` rows are invalid.
- Unknown top-level fields are invalid unless this contract is explicitly extended.

## Train Pair Build Summary

File pattern:

```text
data/hotpotqa/processed/{split}_pairs.summary.json
```

Shape:

```json
{
  "positive_count": 4000,
  "negative_count_by_type": {
    "easy_random": 8000,
    "hard_bm25": 8000,
    "hard_dense": 8000,
    "hard_graph_neighbor": 4000
  },
  "avg_positive_per_task": 2.0,
  "avg_negative_per_task": 14.0,
  "tasks_with_no_positive": [],
  "sampling_config": {
    "random_seed": 13,
    "easy_random_per_positive": 2,
    "hard_bm25_per_positive": 2,
    "hard_dense_per_positive": 2,
    "hard_graph_neighbor_per_positive": 1,
    "hard_pool_size": 30
  }
}
```

Invariants:

- `tasks_with_no_positive` must be empty for normal HotpotQA training data.
- All negative counts must use documented negative sample types.
- `sampling_config` records the effective config, not only user-provided overrides.

## Ranked Results

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
| `method` | string | Public retrieval method name. |
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

## Graph Rerank Config

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
- `lambda_path` remains `0.0` for HotpotQA-only runs unless an unsupervised path bonus is explicitly documented and tested.
- `seed_top_s` and `max_hops` must be positive integers.
- `lambda_*` fields weight final score components after normalization.
- `neighbor_type_weights` calibrates memory-to-memory graph edge types used by neighbor propagation and bridge-neighbor scoring.
- `query_overlap` is not a neighbor type weight; query-overlap contribution is controlled by `lambda_query`.
- Deprecated `type_weights` input is rejected; use `neighbor_type_weights`.

## Metric CSVs

`scripts/evaluate_retrieval.py` currently selects the evidence metric suite and produces a wide per-method metric CSV so aggregation has one complete input row per method. Metric rows are suite-owned: the evidence suite keeps the columns below, while a future LongMemEval suite can define turn/session/answer metric columns without adding evidence-only columns to retrievers.

Wide metric columns:

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

- Metrics are computed by the selected metric suite from prediction artifacts, label artifacts, and graph artifacts.
- Metrics must not read gold fields from ranking records.
- All numeric metrics except latency must be in `[0.0, 1.0]`.

Aggregate outputs:

```text
results/main_results.csv
results/path_results.csv
results/efficiency_results.csv
```

## Query-Evidence Connectivity

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

An empty gold evidence set is invalid for HotpotQA rather than vacuously connected.

## Run Summary

Each runnable script writes a summary record next to its main output.

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

## Validators

Recommended validators:

```text
validate_hotpotqa_ranking_records(records)
validate_hotpotqa_label_records(records, ranking_records_by_task_id)
validate_graphs(graphs, ranking_records_by_task_id)
validate_train_pairs(records, ranking_records_by_task_id, labels_by_task_id, graphs_by_task_id)
validate_train_pair_build_summary(summary)
validate_ranked_results(predictions, ranking_records_by_task_id)
validate_metric_rows(rows)
validate_metric_rows(rows, metric_suite=custom_suite)
```

Validation should run:

- after reading input artifacts.
- before writing output artifacts.
- before computing metrics.

Validation raises exceptions for contract violations.
