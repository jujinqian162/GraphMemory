## Context

The parent refactor correctly separated evidence graphs, execution-provenance graphs, and GraphRAG's private entity graph, but the implementation stopped at thin prototypes. GraphRAG currently constructs a token co-occurrence graph inside `rank_task`; both new methods call the encoder directly and therefore bypass configured query/passage prefixes; provenance expansion keeps only the shortest route to each node; several declared typed edges have no transition validation; native traces are generic dictionaries; and experiment planning repeats method-ID sets already represented by Registry metadata.

This change is a current-only correction before the unfinished parent refactor is committed. It does not preserve the prototype request or trace shapes as compatibility APIs. The locked root plan remains authoritative.

## Goals / Non-Goals

**Goals:**

- Restore the intended deterministic FastGraphRAG-derived entity-search pipeline with an explicit assembler boundary.
- Reuse one dense encoding service for query/passages/entities and preserve encoder prefixes, batch size, and normalization.
- Make execution-provenance paths semantically typed, retain alternative bounded paths, and apply score components exactly once.
- Introduce distinct typed native traces and validate their serialized representation.
- Make Registry input/artifact metadata drive workflow scheduling.
- Add behavioral tests that reproduce every review finding rather than only testing happy-path fixtures.

**Non-Goals:**

- No TRAJECT-Bench adapter, trainable provenance model, LLM entity extraction, community summaries, or new public method ID.
- No compatibility loader for the prototype GraphRAG request/native trace or deleted Memory Stream/beam contracts.
- No attempt to calibrate final provenance weights on a benchmark that is not yet adapted.

## Decisions

### 1. Assemble a concrete entity graph before GraphRAG execution

`GraphRAGRequest` will contain `EntityKnowledgeGraph`. The request value objects live with retrieval consumer contracts; deterministic extraction, alias resolution, index construction, and request assembly live under `retrieval/methods/graphrag`. The Registry builder converts each `TextRankingRequest` into a `GraphRAGRequest`; `GraphRAGMethod.rank_task` only searches the supplied graph.

This keeps dataset adapters free of entity logic while making the method input inspectable and independently testable. Building the graph inside `rank_task` was rejected because it collapses assembly, indexing, search, and scoring into one untestable lifecycle method.

### 2. Restore a bounded deterministic FastGraphRAG-derived pipeline

The assembler builds an alias catalog from candidate metadata and deterministic mentions, normalizes aliases, merges unambiguous aliases, records candidate membership, and creates weighted co-occurrence relations. Search combines query-link, lexical, and dense entity seeds, runs weighted PPR, then projects both entity and relation contributions to candidates with a dense fallback.

The implementation remains non-LLM and dependency-light. It reuses repository text utilities and ports only the behavior-bearing parts of the earlier FastGraphRAG implementation; optional spaCy extraction and old evidence-graph wiring are not restored.

### 3. Reuse `DenseEncodingService` for all semantic scoring

GraphRAG and execution provenance receive a configured `DenseEncodingService` rather than a bare encoder. GraphRAG uses `DenseTaskRetriever` for candidate/entity scores. Provenance uses the same service with a `TextRankingRequest` over retrievable execution nodes. This makes query/passage prefixes and batch size authoritative and removes duplicate matrix validation and cosine code.

### 4. Validate every provenance edge type with an explicit transition table

The graph contract defines allowed source and target node sets for every `ProvenanceEdgeType`. Strong dependency traversal is limited to `returns`, `feeds`, `grounds`, `supports`, and `depends_on`. `contains` and `precedes` are not promoted to dependency edges. Revision edges mark invalid support but are not themselves traversed as evidence paths.

The table is intentionally broad enough for optional Claim/Verification/Decision records but never falls back to accepting arbitrary combinations. Dataset-specific additions require an explicit contract change.

### 5. Enumerate bounded alternative simple paths before scoring

For each semantic seed, deterministic bounded search enumerates simple paths through legal dependency edges up to `max_hops` and `max_path_expansions`. It does not use one `visited_distance` per node, so distinct routes to the same candidate remain scoreable. Exact duplicate edge/node sequences are removed; all path candidates are scored, sorted deterministically, and only then truncated to `top_paths`.

Path scoring returns a typed breakdown. Candidate projection uses the best selected path score once; candidates outside selected paths retain semantic fallback scores. Lifecycle metadata and revision targets contribute invalidation penalties once.

### 6. Use a closed union for native traces

`RetrievalTrace.native_trace` is either `GraphRAGTrace`, `ExecutionProvenanceTrace`, or `None`. Each trace owns typed edge/path records. Result assembly serializes the union to `metadata.native_trace`, and ranked-result validation rejects unknown trace kinds, fields, invalid endpoints, non-finite values, or paths that reference edges/nodes inconsistently.

### 7. Registry artifact metadata drives graph scheduling

Planner helpers query `MethodDefinition.input_spec.required_artifact` instead of maintaining R-GCN ID sets. Train/pair dependencies still use train artifact/dependency metadata where their lifecycle differs, but retrieval and evaluation graph inputs are derived from the semantic Registry contract.

## Risks / Trade-offs

- [Entity extraction behavior differs from the temporary token graph] → Treat this as intended correction before commit and lock deterministic snapshots in tests.
- [Alternative path enumeration can grow exponentially] → Enforce `max_hops`, simple paths, deterministic ordering, and `max_path_expansions` per seed.
- [Transition rules may be too strict for a future dataset] → Fail early and extend the explicit schema with source-backed evidence instead of adding a generic escape hatch.
- [Trace shape is breaking] → The parent implementation is uncommitted and current-only; update all in-repo consumers in the same change.
- [Old completed OpenSpec changes still describe superseded behavior] → Archive superseded changes only after this repair is verified; remove empty deleted-change directories immediately so CLI status is truthful.

## Migration Plan

1. Add failing tests for explicit GraphRAG assembly, dense prefix preservation, alternative path retention, transition rejection, trace validation, and Registry-driven planning.
2. Implement GraphRAG contracts/index/search modules and change the Registry builder boundary.
3. Replace bare encoder calls with shared dense runtime services.
4. Implement complete provenance transition validation and bounded path search/scoring.
5. Introduce typed native traces and migrate serialization/validation/tests.
6. Replace planner method sets with Registry metadata and clean stale active surfaces.
7. Run targeted tests, full pytest, Ruff, basedpyright, compileall, OpenSpec strict validation, symbol scans, and real workflow smoke.

Rollback is a revert of this repair change together with the unfinished parent worktree; no runtime compatibility switch is added.

## Open Questions

None. Benchmark-specific transition extensions and weight calibration remain part of the future TRAJECT-Bench adaptation change.
