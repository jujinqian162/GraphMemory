# Implementation Handoff

Date: 2026-05-20

Status: Required post-implementation deliverable. Fill this document after Phase 1 implementation is complete.

## Purpose

This document is for code review and project handoff. It should tell a reviewer where to start, how the control flow works, where the main abstractions live, and which files deserve the most attention.

`commands.md` explains how to run the project. This document explains how to read and review the implementation.

## Required Sections

### Review Entry Points

List the recommended reading order for a reviewer.

Example shape:

```text
1. docs/00-overview/project-overview.md
2. docs/30-design/architecture.md
3. graph_memory/types.py
4. graph_memory/retrieval.py
5. graph_memory/rerank.py
6. scripts/run_retrieval.py
```

Explain why each file is on the path.

### Main Control Flow

Describe the end-to-end Phase 1 flow and map each step to scripts and core functions:

```text
prepare_hotpotqa.py
  -> convert_hotpotqa_examples
  -> validate_memory_task_inputs
  -> validate_memory_task_labels

build_graphs.py
  -> build_graphs
  -> validate_graphs

run_retrieval.py
  -> run_retrieval
  -> Retriever.rank
  -> graph_rerank, when graph method is selected

tune_graph_rerank.py
  -> tune_graph_rerank
  -> evaluate_results on dev

evaluate_retrieval.py
  -> evaluate_results
  -> metric primitives

aggregate_tables.py
  -> aggregate final CSVs
```

Update names to match the implemented code.

### Key Abstractions

Identify where these live in code:

- `MemoryTaskInput`
- `MemoryTaskLabels`
- `MemoryGraph`
- `RankedNode`
- `RankedResult`
- `Retriever`
- graph rerank function or `Reranker`
- validation functions
- evaluation metric functions
- run summary/debug artifact helpers

For each abstraction, explain:

- what it does
- what it must not do
- which tests cover it

### File Map

Group implementation files by purpose:

| Area | Files | What to review |
|---|---|---|
| CLI adapters | `scripts/*.py` | Argument parsing, config merge, validation calls, run summaries. |
| Contracts/types | `graph_memory/types.py`, `graph_memory/validation.py` | Data shape, invariants, fail-fast behavior. |
| Data conversion | `graph_memory/hotpotqa.py`, `graph_memory/splits.py` | Label mapping, split determinism, leakage separation. |
| Graph construction | `graph_memory/graphs.py` | Edge semantics, no label access, graph stats. |
| Retrieval | `graph_memory/indexes/*`, `graph_memory/retrieval.py` | Complete rankings, dense/BM25 behavior, schema assembly. |
| Reranking | `graph_memory/rerank.py` | Score normalization, graph propagation, score components. |
| Evaluation | `graph_memory/evaluation.py` | Metric definitions, task joins, connectivity logic. |
| Operations | `graph_memory/io.py`, `graph_memory/observability.py` | Reproducibility, run summaries, debug artifacts. |

Update this table to match the implemented files.

### Review Checklist

The implementation handoff must include a checklist covering:

- input artifacts contain no label-only fields
- graph construction reads only input-visible fields
- retrieval methods return complete rankings
- graph rerank is independent from BM25/Dense retrievers
- evaluation reads labels from label artifacts only
- dev tuning and test evaluation are separated
- run summaries are written by every script
- command documentation is complete
- tests pass or skip only for documented local-model reasons
- Phase 1 scope exclusions are preserved

### Test And Verification Summary

List the verification commands run after implementation and summarize results.

Include:

- unit test command
- any CLI smoke test command
- leakage check command
- dense model skip/pass notes

Do not claim passing tests without fresh command output.

### Known Limitations

State what Phase 1 intentionally does not implement:

- answer generation
- Dense-FT
- trainable GNN retriever
- GraphRAG
- Memory Stream
- MemGPT-style memory
- 2WikiMultiHopQA
- MuSiQue
- tool trajectories

Also list any implementation limitations discovered during build.

### Extension Notes

Point to likely future extension points:

- adding a new retriever
- adding a new reranker
- adding graph ablations
- adding a new dataset converter
- adding new metrics

Keep this section practical and file-oriented.

## Completion Rule

Phase 1 implementation should not be considered review-ready until this handoff document is filled with real file paths, implemented function names, verification commands, and known limitations.
