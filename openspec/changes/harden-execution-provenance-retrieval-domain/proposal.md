## Why

The execution-provenance retrieval refactor established the intended public method matrix, but review found that the two new retrieval methods and several integration boundaries only implement runnable approximations. The current behavior drops configured dense-encoder semantics, cannot compare alternative provenance paths correctly, accepts invalid typed transitions, and duplicates Registry knowledge in the workflow planner, so the change is not yet safe to treat as the maintained domain architecture.

## What Changes

- Replace the token-only GraphRAG prototype with the deterministic non-LLM entity extraction, normalization, alias catalog, co-occurrence index, query linking, lexical/dense seeding, PPR, and candidate projection behavior locked by the root plan.
- Make `GraphRAGRequest` carry a method-owned `EntityKnowledgeGraph` assembled from `TextRankingRequest` before retrieval.
- Route GraphRAG and execution-provenance semantic scoring through the shared dense encoding boundary so query/passages use configured prefixes and batch size.
- Enforce a complete execution-provenance node/edge transition matrix, including support, verification, dependency, contradiction, revision, and impact edges.
- Replace shortest-distance pruning with bounded deterministic top-path search that can retain and score alternative paths to the same node.
- Separate path scoring from candidate projection so each configured component is applied exactly once.
- Replace generic native trace dictionaries with distinct typed GraphRAG and execution-provenance trace records and validate serialized traces.
- Make evidence-graph scheduling and evaluation derive from Registry artifact metadata instead of duplicated method-ID sets.
- Remove stale active OpenSpec surfaces that still expose deleted Memory Stream, beam R-GCN, or graph-rerank behavior as current capabilities.

## Capabilities

### New Capabilities

- `method-native-retrieval-traces`: Typed, serializable, and validated GraphRAG and execution-provenance native trace contracts.

### Modified Capabilities

- `entity-search-graphrag`: Require the locked FastGraphRAG-derived entity pipeline and explicit entity-graph request assembly.
- `execution-provenance-retrieval`: Require legal typed transitions, alternative top-path retention, single-pass score composition, and lifecycle invalidation handling.
- `typed-retrieval-requests`: Require GraphRAG requests to carry the assembled method-owned entity graph and preserve graph-family isolation.
- `semantic-method-registry`: Require shared dense runtime semantics and make Registry artifact metadata authoritative for workflow routing.
- `retrieval-workflow-matrix`: Require evidence-graph stages and evaluation inputs to be derived from Registry declarations without duplicated method lists.

## Impact

Affected areas include `graph_memory/retrieval/requests`, `retrieval/methods/graphrag`, `retrieval/methods/execution_provenance`, `graphs/provenance`, dense embedding services, Registry builders and semantics, ranked-result validation, experiment planning, focused behavior tests, and active OpenSpec documentation. Public method IDs and the seven-method matrix do not change; GraphRAG and native trace request/result shapes change before the unfinished refactor is committed.
