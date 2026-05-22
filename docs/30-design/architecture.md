# Architecture

Date: 2026-05-20

Status: Working reference.

## Core Decision

Use a library-core architecture with thin CLI scripts.

```text
Artifacts are the external contract.
Domain objects are the internal language.
CLI scripts are adapters, not the system.
```

## External Structure

Respect the original experiment structure:

```text
data/
  hotpotqa/
    raw/
    processed/
results/
scripts/
```

The script names and artifact names from the project plan remain stable:

- `scripts/prepare_hotpotqa.py`
- `scripts/build_graphs.py`
- `scripts/run_retrieval.py`
- `scripts/tune_graph_rerank.py`
- `scripts/evaluate_retrieval.py`
- `scripts/aggregate_tables.py`
- `*_memory_tasks.input.json`
- `*_memory_tasks.labels.json`
- `*_graphs.json`
- `ranked_results_{method}.json`
- final metric CSVs

## Package Shape

Start with a mostly flat package. Do not introduce a broad subpackage hierarchy before Phase 1 is runnable.

```text
graph_memory/
  __init__.py
  types.py
  validation.py
  io.py
  hotpotqa.py
  splits.py
  text.py
  entities.py
  graphs.py
  indexes/
    __init__.py
    bm25.py
    dense.py
  retrieval.py
  rerank.py
  tuning.py
  evaluation.py
  observability.py
```

`indexes/` is the small exception because BM25 and dense retrieval are naturally parallel implementations and the Phase 1 plan already names this module.

## Layer Responsibilities

| Layer | Responsibility |
|---|---|
| `scripts/` | Parse CLI/config, call library functions, log progress, write run summaries. |
| `types.py` | Shared aliases, `TypedDict`s, dataclasses, and protocols. |
| `validation.py` | Fail-fast artifact contract validation. |
| `io.py` | JSON, CSV, and config file helpers. |
| `hotpotqa.py` | Raw HotpotQA conversion into input and label artifacts. |
| `splits.py` | Deterministic split sampling. |
| `text.py`, `entities.py` | Text normalization, lexical scoring, and entity extraction. |
| `graphs.py` | Typed graph construction and graph statistics. |
| `indexes/` | Flat retriever implementations. |
| `retrieval.py` | Retrieval method construction, score-pipeline execution for score-based baselines, and ranked-result assembly. |
| `rerank.py` | Reusable graph reranking helpers and compatibility functions over explicit initial scores. |
| `tuning.py` | Dev-set graph rerank parameter selection. |
| `evaluation.py` | Metrics, aggregation, and failure-case selection. |
| `observability.py` | Run summaries, graph stats, and debug record builders. |

## Dependency Direction

Allowed flow:

```text
scripts
  -> io / validation / observability
  -> domain modules

graphs
  -> text / entities

retrieval
  -> indexes
  -> rerank helpers for graph score components

evaluation
  -> validation
  -> graph connectivity helpers

hotpotqa
  -> no graph, retrieval, tuning, or evaluation imports
```

Avoid reverse dependencies:

- Dataset conversion must not import graph construction.
- Graph construction must not import labels, retrieval, tuning, or evaluation.
- Retrievers must not import evaluation metrics.
- Evaluation must not import raw dataset conversion.
- Core algorithms must not write files.

## Tuning And Cache Boundary

Phase 1 does not use a persistent score cache. Keep score reuse explicit and bounded:

- Flat retrievers produce complete initial rankings.
- Graph rerank consumes an explicit `initial_scores` mapping plus a graph and config.
- Score-pipeline methods may combine baseline scores and graph scores in memory for one task at a time.
- Dev grid search precomputes seed-retriever scores once per tuning invocation and reuses them across graph-rerank candidates.
- Graph score components are calibrated per task before weighted combination, and neighbor propagation is degree-normalized before component normalization.
- The graph-rerank tuning grid includes a pure initial-score fallback so dev tuning can select "no graph bonus" without changing public method names or artifacts.
- Persistent score reuse should be added as a named artifact and validation boundary only if full-dev tuning still becomes a practical blocker.

## Script Boundary

Scripts own:

- CLI/config parsing.
- File paths.
- top-level logging.
- run summary writing.
- invoking validators at boundaries.

Scripts do not own:

- core algorithms.
- metric definitions.
- graph scoring formulas.
- retrieval implementation details.

## Observability Boundary

Observability attaches to script and service boundaries. Core pure functions should return values or debug data, not log internally.

Mandatory:

- run summary for every script.
- graph statistics for graph construction runs.

Optional and bounded:

- score breakdown JSONL.
- failure-case JSONL.
- leakage check reports.

## Future Extraction Rule

Do not create plugin registries or deep package hierarchies in Phase 1. Extract new subpackages only when a module grows multiple independent implementations or becomes hard to navigate.

The retrieval service now has two abstraction levels:

```text
RetrievalMethod
  -> produces a ranked result for any baseline

ScorePipelineMethod
  -> one RetrievalMethod implementation for weighted node-score baselines
```

Use `ScorePipelineMethod` for BM25, dense, Memory Stream-style scores, and current graph rerank variants. Use a separate `RetrievalMethod` implementation when a future baseline is primarily graph traversal, hierarchical memory selection, or learned message passing rather than a transparent weighted sum.
