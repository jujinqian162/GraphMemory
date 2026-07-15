## 1. Lock review regressions with failing tests

- [x] 1.1 Add GraphRAG request/index tests for explicit entity graph assembly, alias canonicalization, deterministic relations, and relation-to-candidate projection
- [x] 1.2 Add recording-encoder tests proving configured query/passages prefixes are used by GraphRAG and execution-provenance scoring
- [x] 1.3 Add provenance tests proving invalid typed transitions fail and complete bound paths remain competitive with shorter weak paths
- [x] 1.4 Add native trace serialization/validation tests and Registry-driven planner tests

## 2. Restore GraphRAG domain implementation

- [x] 2.1 Add typed EntityKnowledgeGraph entity/relation contracts and GraphRAG request alignment validation
- [x] 2.2 Implement deterministic mention extraction, normalization, alias catalog, and weighted co-occurrence index assembly
- [x] 2.3 Implement query linking, lexical/dense entity seeding, weighted PPR, and entity/relation candidate projection
- [x] 2.4 Move entity graph assembly into the Registry builder and make GraphRAGMethod consume only the assembled graph

## 3. Unify dense semantic runtime

- [x] 3.1 Construct prefix-aware DenseEncodingService/DenseTaskRetriever runtimes from DenseEncoderSettings in shared Registry helpers
- [x] 3.2 Remove duplicate raw encoder matrix scoring from GraphRAG and execution-provenance methods
- [x] 3.3 Verify injected encoders, configured prefixes, batch size, and complete rankings for both methods

## 4. Harden execution provenance graph and search

- [x] 4.1 Define and enforce an explicit source/target transition matrix for every ProvenanceEdgeType
- [x] 4.2 Replace shortest-distance pruning with bounded deterministic alternative simple-path enumeration
- [x] 4.3 Add typed path score breakdown and project selected path scores without nested reweighting
- [x] 4.4 Apply explicit revision-edge and lifecycle-metadata invalidation while excluding chronology/revision edges from strong dependency traversal

## 5. Introduce typed native trace contracts

- [x] 5.1 Replace generic native path/edge dictionaries with GraphRAGTrace and ExecutionProvenanceTrace value objects
- [x] 5.2 Serialize the closed trace union into ranked-result metadata
- [x] 5.3 Validate trace kind, fields, scores, endpoints, duplicates, and path consistency in ranked-result validation
- [x] 5.4 Migrate retrieval methods, docs, and tests to the new trace contract without compatibility aliases

## 6. Make workflow metadata authoritative

- [x] 6.1 Add Registry helpers for required artifacts and use them in retrieval/evaluation planning
- [x] 6.2 Remove duplicated R-GCN method sets from planner graph scheduling while preserving train/pair dependencies
- [x] 6.3 Remove the empty Memory Stream change surface and archive or otherwise deactivate superseded beam/graph-rerank OpenSpec changes

## 7. Final verification

- [x] 7.1 Run focused GraphRAG, provenance, trace, Registry, planner, and R-GCN regression tests
- [x] 7.2 Run full pytest, Ruff, basedpyright, and compileall
- [x] 7.3 Run OpenSpec strict validation, current-only symbol scans, git diff check, and minimum real evidence workflow smoke
- [x] 7.4 Reconcile both parent and repair task checklists with verified implementation state
