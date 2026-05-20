# Implementation Handoff

Date: 2026-05-20

Status: Phase 1 implementation handoff.

## Purpose

This document is for code review. `commands.md` explains how to run the project; this document explains how to read and review the implementation.

## Review Entry Points

Recommended reading order:

1. `docs/00-overview/project-overview.md`
   Confirms the Phase 1 scientific boundary: HotpotQA evidence tracing, not answer generation.
2. `docs/20-contracts/phase1-data-contracts.md`
   Defines the artifact schemas that validators enforce.
3. `docs/30-design/architecture.md` and `docs/30-design/abstractions.md`
   Explains the library-core, thin-CLI design and abstraction boundaries.
4. `graph_memory/types.py`
   Contains shared aliases, TypedDict records, frozen config dataclasses, `RankedNode`, and `Retriever`.
5. `graph_memory/validation.py`
   Enforces fail-fast artifact contracts and leakage checks.
6. `graph_memory/hotpotqa.py`, `graph_memory/graphs.py`, `graph_memory/retrieval.py`, `graph_memory/rerank.py`, `graph_memory/evaluation.py`
   Core Phase 1 control flow from conversion to metrics.
7. `scripts/*.py`
   CLI adapters that parse arguments, call core functions, validate artifacts, and write run summaries.

## Main Control Flow

```text
scripts/prepare_hotpotqa.py
  -> graph_memory.splits.sample_split
  -> graph_memory.hotpotqa.parse_hotpotqa_examples
  -> graph_memory.hotpotqa.convert_hotpotqa_examples
  -> validate_memory_task_inputs
  -> validate_memory_task_labels

scripts/build_graphs.py
  -> graph_memory.graphs.build_graphs
  -> validate_graphs
  -> graph_memory.observability.graph_statistics

scripts/run_retrieval.py
  -> graph_memory.retrieval.run_retrieval
  -> BM25TaskRetriever.rank or DenseTaskRetriever.rank
  -> graph_memory.rerank.graph_rerank for graph methods
  -> validate_ranked_results

scripts/tune_graph_rerank.py
  -> graph_memory.tuning.tune_graph_rerank
  -> run_retrieval on dev configs
  -> graph_memory.evaluation.evaluate_results
  -> graph_memory.tuning.select_best_config

scripts/evaluate_retrieval.py
  -> graph_memory.evaluation.evaluate_results
  -> metric primitives and shared-graph connectivity helpers

scripts/aggregate_tables.py
  -> graph_memory.evaluation.split_metric_tables
```

## Key Abstractions

| Abstraction | Location | What it does | Must not do | Tests |
|---|---|---|---|---|
| `MemoryTaskInput` | `graph_memory/types.py` | Input-visible query and memory sentence artifact shape. | Contain labels or answer text. | `tests/test_phase1_real_validation.py`, `tests/test_phase1_real_data_structures.py` |
| `MemoryTaskLabels` | `graph_memory/types.py` | Gold answer and evidence labels for evaluation/tuning. | Feed graph construction or retrieval. | `tests/test_phase1_real_validation.py` |
| `HotpotQAExample` / `HotpotQAConversionResult` | `graph_memory/hotpotqa.py` | Typed raw HotpotQA parse result and named conversion output. | Expose raw `dict` records or tuple-packed outputs. | `tests/test_phase1_real_data_structures.py` |
| `MemoryGraph` | `graph_memory/types.py`, `graph_memory/graphs.py` | Typed graph over `q` and memory sentence nodes. | Read label-only fields. | `tests/test_phase1_real_graphs.py` |
| `RankedNode` / `RankedResult` | `graph_memory/types.py`, `graph_memory/retrieval.py` | Complete per-task ranking and persisted result schema. | Drop unselected memory nodes. | `tests/test_phase1_real_retrieval.py` |
| `Retriever` | `graph_memory/types.py` | Single-task complete ranking protocol. | Compute metrics or read labels. | `tests/test_phase1_real_retrieval.py` |
| `GraphRerankConfig` / `graph_rerank` | `graph_memory/types.py`, `graph_memory/rerank.py` | Graph score propagation over explicit initial scores. | Run BM25/Dense itself or use labels. | `tests/test_phase1_real_retrieval.py` |
| Validators | `graph_memory/validation.py` | Enforce contracts and fail fast. | Repair, sort, drop, or infer data. | `tests/test_phase1_real_validation.py` |
| Metric primitives | `graph_memory/evaluation.py` | Compute node and connectivity metrics. | Re-run retrieval or read task inputs for gold fields. | `tests/test_phase1_real_evaluation.py` |
| Run summaries | `graph_memory/observability.py` | Preserve config, paths, counts, timings, environment, and notes. | Change algorithm behavior. | `tests/test_phase1_real_io_observability.py` |

## File Map

| Area | Files | What to review |
|---|---|---|
| CLI adapters | `scripts/prepare_hotpotqa.py`, `scripts/build_graphs.py`, `scripts/run_retrieval.py`, `scripts/tune_graph_rerank.py`, `scripts/evaluate_retrieval.py`, `scripts/aggregate_tables.py` | Argument names, config visibility, validation calls, run summaries, output paths. |
| Contracts/types | `graph_memory/types.py`, `graph_memory/validation.py` | Field names, forbidden fields, strict invariants, readable type annotations. |
| Data conversion | `graph_memory/hotpotqa.py`, `graph_memory/splits.py` | Stable task IDs, supporting-fact mapping, split determinism, label separation. |
| Text/entity | `graph_memory/text.py`, `graph_memory/entities.py` | Stopword filtering, lexical scoring, deterministic heuristic entities, optional spaCy behavior. |
| Graph construction | `graph_memory/graphs.py` | Edge semantics, edge limits, deterministic sorting, no label access. |
| Retrieval | `graph_memory/indexes/bm25.py`, `graph_memory/indexes/dense.py`, `graph_memory/retrieval.py` | Complete rankings, dense encoder prefixes, method dispatch, graph-method requirements. |
| Reranking | `graph_memory/rerank.py` | Score normalization, candidate expansion, bridge/query/neighbor components, all-node preservation. |
| Evaluation | `graph_memory/evaluation.py` | Metric definitions, exact joins, shared-graph connectivity, N/A path metrics. |
| Operations | `graph_memory/io.py`, `graph_memory/observability.py`, `configs/*.json` | Deterministic writes, config defaults, run summary fields. |

## Review Checklist

- Input artifacts contain no label-only fields.
- Graph construction reads only `*_memory_tasks.input.json` fields.
- Retrieval methods return complete rankings over every memory node.
- Graph rerank is independent from BM25/Dense retrievers.
- Evaluation reads labels from label artifacts only.
- Dev tuning and test evaluation are separate.
- Every script writes a run summary when output paths are known.
- `docs/40-operations/commands.md` matches actual script arguments.
- Tests pass or skip only for documented local-model reasons.
- Phase 1 scope exclusions are preserved.

## Test And Verification Summary

Focused verification commands run during implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_validation.py tests/test_phase1_real_io_observability.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_data_structures.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_graphs.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_retrieval.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_evaluation.py -q -p no:cacheprovider
```

Full-suite verification run during final implementation check:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
37 passed in 0.21s
```

`uv run pytest` was attempted earlier, but this sandbox could not access the local uv cache and global pytest temp directories. The verified fallback uses the repository-local `.venv` and a repository-local pytest base temp.

CLI smoke path run on `tests/fixtures/hotpotqa_smoke.json`:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_hotpotqa.py --input tests\fixtures\hotpotqa_smoke.json --output_input .pytest-tmp\smoke\test_memory_tasks.input.json --output_labels .pytest-tmp\smoke\test_memory_tasks.labels.json --output_combined .pytest-tmp\smoke\test_memory_tasks.json --max_examples 1 --seed 13 --offset 0
.\.venv\Scripts\python.exe scripts\build_graphs.py --input .pytest-tmp\smoke\test_memory_tasks.input.json --output .pytest-tmp\smoke\test_graphs.json --max_query_overlap 20 --max_entity_neighbors 10 --max_bridge_edges 50
.\.venv\Scripts\python.exe scripts\run_retrieval.py --method bm25 --tasks .pytest-tmp\smoke\test_memory_tasks.input.json --output .pytest-tmp\smoke\ranked_results_bm25.json --top_k 10
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --pred .pytest-tmp\smoke\ranked_results_bm25.json --labels .pytest-tmp\smoke\test_memory_tasks.labels.json --graphs .pytest-tmp\smoke\test_graphs.json --output .pytest-tmp\smoke\main_results_bm25.csv --failure_cases_output .pytest-tmp\smoke\failure_cases_bm25.jsonl --failure_case_limit 5
.\.venv\Scripts\python.exe scripts\aggregate_tables.py --input_dir .pytest-tmp\smoke --output_main .pytest-tmp\smoke\main_results.csv --output_path .pytest-tmp\smoke\path_results.csv --output_efficiency .pytest-tmp\smoke\efficiency_results.csv
```

Result: all smoke commands exited with code `0`.

Leakage check run on smoke input/graph artifacts:

```powershell
rg "gold_answer|gold_evidence_nodes|supporting_facts|is_gold" .pytest-tmp\smoke -g "*input*.json" -g "*graphs*.json"
```

Result: no matches.

## Known Limitations

Phase 1 intentionally does not implement:

- answer generation
- Dense-FT
- trainable GNN retriever
- GraphRAG
- Memory Stream
- MemGPT-style memory
- 2WikiMultiHopQA
- MuSiQue
- tool trajectories
- persistent score caching

Current implementation limitations:

- Dense retrieval uses Sentence-Transformers at runtime and needs the configured model available locally or downloadable.
- Entity extraction is heuristic unless the caller explicitly enables spaCy and provides an environment with spaCy installed.
- Graph-rerank dev tuning recomputes initial rankings for clarity; no score cache is introduced in Phase 1.

## Extension Notes

- Add a new retriever by implementing `Retriever.rank(task_input)` and extending `graph_memory/retrieval.py` dispatch.
- Add a new graph reranker by keeping the boundary `initial_scores + graph + config -> complete ranking`.
- Add graph ablations by extending `GraphBuildConfig` or adding named graph-transform functions before retrieval.
- Add a new dataset converter by producing the same `MemoryTaskInput` and `MemoryTaskLabels` artifacts.
- Add new metrics by introducing pure metric primitives first, then adding aggregate columns in `evaluate_results` and table split helpers.
