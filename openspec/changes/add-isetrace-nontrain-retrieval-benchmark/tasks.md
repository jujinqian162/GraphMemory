## 1. OpenSpec and contracts

- [x] 1.1 Add proposal, design, capability deltas, and implementation checklist for the ISETrace non-training benchmark.
- [x] 1.2 Add dataset-owned benchmark task/label contracts, explicit review/label policies, and strict graph/query/node alignment validation.
- [x] 1.3 Add graph-owned query-independent logical output dependency contracts and projection.

## 2. ISETrace preparation and evaluation projection

- [x] 2.1 Extend experiment config/source identity for test-only ISETrace query + trajectory inputs pinned to the fixed revision.
- [x] 2.2 Materialize ToolOutput candidate tasks, labels, combined records, counts, and unique provenance graphs without label-conditioned topology.
- [x] 2.3 Project ISETrace records to `TextRankingRequest`, output-only evaluation graphs, and evidence labels under the selected policy.
- [x] 2.4 Add adapter tests for reuse of one graph by multiple queries, unknown graph/output failure, revision identity, review filtering, and intent-aware labels.

## 3. Flat and entity-graph baselines

- [x] 3.1 Route BM25 and frozen Dense over execution-provenance requests with identical candidates and no graph input.
- [x] 3.2 Add title-free shared-entity groups when GraphRAG receives candidates without title groups; keep document behavior and exact Dense fallback unchanged.
- [x] 3.3 Add tests proving GraphRAG can intervene from text-only shared entities and never requires a provenance graph.

## 4. Training-free provenance retrieval

- [x] 4.1 Add a strict `ProvenancePathRequest`, settings/build payload, method ID, registry entry, config, and Hydra YAML.
- [x] 4.2 Implement bounded bidirectional logical dependency search, protected-prefix stable insertion, deterministic conflicts, full-list score-slot preservation, and exact Dense fallback.
- [x] 4.3 Add the closed provenance-path native trace and validation for dependencies, proposals, emitted edges, candidate context, and fallback/intervention consistency.
- [x] 4.4 Add behavior tests for feeds, artifact flow, reverse lookup, multi-hop completion, temporal-only abstention, deterministic ties, top-k edge emission, and Dense identity fallback.

## 5. Workflow and metrics

- [x] 5.1 Integrate test-only execution-provenance preparation, ranking, evaluation graphs, benchmark timing, Prefect caching, tracking, and run output.
- [x] 5.2 Mark query-evidence connectivity unsupported for query-independent ISETrace graphs while preserving support, connected evidence, path, and edge metrics.
- [x] 5.3 Add end-to-end tests showing BM25, Dense, GraphRAG, and provenance-path each produce aligned full rankings without invoking training tasks.

## 6. Documentation and verification

- [x] 6.1 Add operations/config/contract documentation for pilot and accepted-only commands, candidate rendering, policies, method boundaries, and diagnostics.
- [x] 6.2 Run focused tests, full pytest, Ruff, basedpyright, compileall, and `git diff --check`.
- [x] 6.3 Record verification evidence, mark completed tasks, and leave manual query acceptance and paper results explicitly pending.
