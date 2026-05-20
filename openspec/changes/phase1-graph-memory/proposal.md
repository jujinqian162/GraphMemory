## Why

Phase 1 needs to turn the current documentation and experiment plan into a runnable, reviewable HotpotQA evidence-tracing system. The current repository is only a scaffold, so there is no leakage-safe data pipeline, graph construction, retrieval output, tuning path, or evaluation table implementation yet.

## What Changes

- Add leakage-safe HotpotQA conversion that writes input-visible task artifacts separately from label artifacts.
- Add strict artifact validators, deterministic I/O helpers, config defaults, logging/run summaries, and debug artifact foundations.
- Add deterministic Phase 1 typed graph construction over question and sentence nodes.
- Add BM25, frozen dense retrieval, BM25-seeded graph rerank, and dense-seeded graph rerank under one ranked-result schema.
- Add dev-set graph-rerank tuning and test-set evaluation without using test labels for tuning.
- Add metric computation, aggregation, command documentation, and implementation handoff documentation.
- Keep Phase 1 retrieval-only; do not add answer generation, Dense-FT, GraphRAG, Memory Stream, MemGPT-style memory, trainable GNN retrievers, or additional datasets.

## Capabilities

### New Capabilities

- `leakage-safe-hotpotqa-artifacts`: Convert labeled HotpotQA data into deterministic input and label artifacts with fail-fast validation and no label leakage into retrieval-visible files.
- `typed-graph-memory-retrieval`: Build typed memory graphs and run flat plus graph-aware retrieval methods that emit complete ranked results in a shared schema.
- `phase1-evidence-evaluation`: Tune graph rerank parameters on dev, evaluate predictions against label artifacts and graphs, aggregate Phase 1 tables, and preserve reproducibility records.

### Modified Capabilities

- None.

## Impact

- Adds the `graph_memory/` package, `scripts/` entry points, `configs/` defaults, and Phase 1 tests.
- Updates `pyproject.toml` dependencies for Python 3.12 experiment execution.
- Updates root `README.md`, `docs/40-operations/commands.md`, and `docs/40-operations/implementation-handoff.md`.
- Produces artifacts under `data/hotpotqa/processed/`, `results/`, and `configs/` when commands are run.
