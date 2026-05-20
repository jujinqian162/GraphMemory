## 1. Foundation Contracts, I/O, And Observability

- [x] 1.1 Update `pyproject.toml` dependencies and create package/script/config/test directories.
- [x] 1.2 Add tests for critical validator negative cases: label leakage, missing graph endpoint, duplicate ranked node, task ID mismatch, and non-finite scores.
- [x] 1.3 Implement `graph_memory/types.py` with aliases, TypedDict contracts, config dataclasses, ranked-node dataclass, and retriever protocol.
- [x] 1.4 Implement `graph_memory/validation.py` fail-fast validators and `ContractValidationError`.
- [x] 1.5 Implement deterministic JSON/CSV/config helpers in `graph_memory/io.py`.
- [x] 1.6 Implement run summary and compact debug/stat helpers in `graph_memory/observability.py`.
- [x] 1.7 Run focused foundation/validation tests and mark this group complete only after expected tests pass.

## 2. HotpotQA Conversion And Split Sampling

- [x] 2.1 Write conversion and split tests for stable task IDs, title/sentence label mapping, leakage separation, deterministic sampling, and oversized split failure.
- [x] 2.2 Implement `graph_memory/hotpotqa.py` conversion from raw labeled HotpotQA examples to input and label records.
- [x] 2.3 Implement `graph_memory/splits.py` deterministic `sample_split`.
- [x] 2.4 Implement `scripts/prepare_hotpotqa.py` with leakage-safe outputs, optional compatibility output, validation, logging, and run summary.
- [x] 2.5 Run `uv run pytest tests/test_phase1_real_data_structures.py -q`.

## 3. Text, Entity, And Typed Graph Construction

- [x] 3.1 Write graph/text tests for stopword-safe content tokens, lexical scoring, heuristic entities, edge semantics, graph limits, and no-label graph output.
- [x] 3.2 Implement `graph_memory/text.py` content tokens, IDF, and lexical score utilities.
- [x] 3.3 Implement `graph_memory/entities.py` title aliases, heuristic entities, and optional spaCy-backed extraction.
- [x] 3.4 Implement `graph_memory/graphs.py` graph build config, typed edge construction, batch graph building, and graph statistics.
- [x] 3.5 Implement `scripts/build_graphs.py` with validation, logging, graph stats, and run summary.
- [x] 3.6 Run `uv run pytest tests/test_phase1_real_graphs.py -q`.

## 4. Flat Retrieval And Shared Ranked Schema

- [x] 4.1 Write retrieval tests for BM25 and dense ranked-result schema, complete rankings, finite scores, top-k subgraph shape, and dense local-model skip behavior.
- [x] 4.2 Implement `graph_memory/indexes/bm25.py` per-task BM25 retriever.
- [x] 4.3 Implement `graph_memory/indexes/dense.py` frozen dense retriever with configurable encoder, prefixes, and batch size.
- [x] 4.4 Implement initial `graph_memory/retrieval.py` service for `bm25` and `dense` methods.
- [x] 4.5 Implement `scripts/run_retrieval.py` flat-method CLI path with validation, logging, and run summary.
- [x] 4.6 Run `uv run pytest tests/test_phase1_real_retrieval.py -q`.

## 5. Graph Rerank And Graph-Aware Retrieval

- [x] 5.1 Write rerank tests for score normalization, bridge/entity promotion, candidate expansion, full-node preservation, graph config validation, and retrieved subgraph extraction.
- [x] 5.2 Implement `graph_memory/rerank.py` graph rerank config, score normalization, candidate expansion, final scoring, score components, and induced subgraph extraction.
- [x] 5.3 Extend `graph_memory/retrieval.py` for `bm25_graph_rerank` and `dense_graph_rerank`.
- [x] 5.4 Extend `scripts/run_retrieval.py` graph-method CLI path requiring graphs and graph config.
- [x] 5.5 Run `uv run pytest tests/test_phase1_real_retrieval.py -q`.

## 6. Dev Tuning

- [x] 6.1 Write tuning tests for objective calculation, grid generation, deterministic config selection, and latency tie-break behavior.
- [x] 6.2 Implement `graph_memory/tuning.py` objective, graph-rerank grid, config selection, and dev tuning service.
- [x] 6.3 Implement `scripts/tune_graph_rerank.py` with dev labels, graph-rerank methods, selected config output, logging, and run summary.
- [x] 6.4 Run `uv run pytest tests/test_phase1_real_retrieval.py -q`.

## 7. Evaluation And Aggregation

- [x] 7.1 Write evaluation tests for Recall@k, Evidence F1@k, Full Support@k, MRR, Connected Evidence Recall@k, Query-Evidence Connectivity@10, task ID mismatch, and HotpotQA N/A path metrics.
- [x] 7.2 Implement `graph_memory/evaluation.py` metric primitives, label-aware aggregate evaluation, connectivity helpers, failure-case selection, and table aggregation helpers.
- [x] 7.3 Implement `scripts/evaluate_retrieval.py` with `--labels` and `--gold` alias, validation, metric CSV output, optional failure cases, logging, and run summary.
- [x] 7.4 Implement `scripts/aggregate_tables.py` final main/path/efficiency table output.
- [x] 7.5 Run `uv run pytest tests/test_phase1_real_evaluation.py -q`.

## 8. Reproducibility, Documentation, And Verification

- [x] 8.1 Add `configs/phase1_default.json` and `configs/phase1_graph_rerank_grid.json`.
- [x] 8.2 Update root `README.md` with short Phase 1 quick-start and links to the command runbook.
- [x] 8.3 Fill `docs/40-operations/commands.md` with actual leakage-safe commands, compatibility notes, leakage check, and test commands.
- [x] 8.4 Fill `docs/40-operations/implementation-handoff.md` with real review entry points, control flow, key abstractions, file map, verification summary, limitations, and extension notes.
- [x] 8.5 Run `openspec validate phase1-graph-memory`.
- [x] 8.6 Run `uv run pytest tests -q`.
- [x] 8.7 Run at least one tiny CLI smoke path or clearly report why no local raw data/model smoke path was possible.
- [x] 8.8 Review git diff for scope, stage relevant files, and commit the Phase 1 implementation.
