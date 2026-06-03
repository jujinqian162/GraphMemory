# Naming Conventions

Date: 2026-05-20

Status: Working reference.

## Goal

Names should make the experiment readable as a system. Prefer explicit domain language over short clever names. A reader should understand what a module, function, or variable owns without tracing its internals.

## General Rules

- Use Python standard naming style unless a data contract requires a specific external field name.
- Prefer full domain words over abbreviations.
- Use consistent terms from the project: `task`, `memory_item`, `graph`, `edge`, `retriever`, `reranker`, `prediction`, `label`, `metric`.
- Do not use generic names such as `data`, `obj`, `item`, `result`, `info` when a domain name is available.
- Avoid ambiguous terms like `record` unless the code is truly artifact-generic.
- Keep public names stable and boring.

## Package And Module Names

Use short, noun-based module names that match responsibilities:

| Package or module | Responsibility |
|---|---|
| `datasets/hotpotqa/` | Raw HotpotQA parsing and conversion. |
| `datasets/splits.py` | Deterministic split sampling. |
| `text/` | Tokenization, lexical scoring, and entity extraction. |
| `graphs/` | Graph construction, graph views, and graph statistics. |
| `retrieval/` | Method request resolution, runtime construction, execution, rerank, tuning, and adapters. |
| `training_pairs/` | Train-pair construction and negative sampling. |
| `models/graph_retriever/` | Trainable graph retriever config, tensors, model, checkpoint, training, and inference. |
| `evaluation/` | Metrics, connectivity, aggregate rows, and failure cases. |
| `validation/` | Artifact and model contract checks. |
| `infrastructure/` | JSON, CSV, config read/write helpers, run summaries, and runtime environment capture. |
| `contracts/` | Type aliases and artifact-shaped `TypedDict`s. |
| root integration ports | `io.py`, `observability.py`, `retrieval_registry.py`, `training_config.py`, and `experiment.py` stay thin for workflow compatibility. |

Rule:

- A module name should describe its domain, not its implementation trick.

## Function Names

Use verb-first names for actions:

| Pattern | Example |
|---|---|
| Convert raw data | `convert_hotpotqa_examples` |
| Build artifacts | `build_graph`, `build_graphs` |
| Validate artifacts | `validate_memory_task_inputs` |
| Compute metrics | `recall_at`, `full_support_at` |
| Run service | `run_retrieval`, `tune_graph_rerank` |
| Load/write files | `read_json`, `write_json`, `write_csv` |

Rules:

- Use singular names for single-task functions: `build_graph`.
- Use plural names for batch functions: `build_graphs`.
- Metric functions should include the cutoff style when relevant: `recall_at`, `connected_evidence_at`.
- Validation functions should be named `validate_*` and should raise on failure.
- Avoid names that hide side effects. If a function writes output, its name should say `write_*`.

## Class And Dataclass Names

Use noun names for stable concepts:

| Concept | Name |
|---|---|
| Single ranked node | `RankedNode` |
| Graph rerank output | `RerankResult` |
| Graph score components | `ScoreComponents` |
| Graph build parameters | `GraphBuildConfig` |
| Graph rerank parameters | `GraphRerankConfig` |
| Dense retriever parameters | `DenseConfig` |

Rules:

- Config classes end with `Config`.
- Result classes end with `Result`.
- Protocols describe behavior: `Retriever`, `Reranker`.
- Avoid suffixes like `Manager`, `Handler`, `Processor`, or `Helper` unless no clearer domain word exists.

## Variable Names

Prefer names that state the artifact role:

| Prefer | Avoid |
|---|---|
| `task_input` | `task`, when labels may also exist |
| `task_labels` | `gold`, if the object contains more than gold nodes |
| `memory_items` | `items` |
| `gold_nodes` | `labels`, when only node IDs are meant |
| `graph_by_task_id` | `graphs` when lookup shape matters |
| `ranked_nodes` | `results` |
| `initial_ranking` | `ranking1` |
| `reranked_nodes` | `new_results` |
| `effective_config` | `config`, after merge |

Rules:

- Include `_by_task_id` for dictionaries keyed by task id.
- Include `_ids` for lists or sets of IDs.
- Use `*_path` for filesystem paths and `*_dir` for directories.
- Use `*_records` for raw JSON rows only when the type is artifact-generic.

## Artifact Field Names

External JSON/CSV fields follow the data contract, not Python naming preferences.

Examples:

- `task_id`
- `memory_items`
- `gold_evidence_nodes`
- `ranked_nodes`
- `retrieved_subgraph`
- `latency_ms`
- `input_tokens`

Rule:

- Do not rename external fields in code for convenience unless conversion is explicit and tested.

## Method Names

Use stable lowercase method identifiers:

```text
bm25
dense
bm25_graph_rerank
dense_graph_rerank
```

Rules:

- Method names are artifact values and should remain stable.
- Use method names in output files: `ranked_results_bm25.json`.
- Use human-readable display names only in final tables if needed.

## Graph Names

Use these consistently:

| Concept | Name |
|---|---|
| Current query node | `q` |
| Memory sentence node | `m{position}` |
| Adjacent sentence edge | `sequential` |
| Query-to-sentence lexical edge | `query_overlap` |
| Entity-sharing edge | `entity_overlap` |
| Cross-document connector | `bridge` |

Rule:

- Do not invent alternate names such as `doc_edge`, `semantic_edge`, or `cross_edge` unless the contract changes.

## Naming Anti-Patterns

Avoid:

- `process_data`
- `handle_result`
- `do_eval`
- `graph_func`
- `calc`
- `tmp`, except in tiny local scopes
- `x`, `y`, `z`, except mathematical formulas with nearby explanation
- one-letter aliases for domain objects

## When A Name Gets Long

Long names are acceptable when they prevent ambiguity, but repeated long names may signal a missing abstraction.

Prefer extracting a named concept over compressing important meaning into abbreviations.
