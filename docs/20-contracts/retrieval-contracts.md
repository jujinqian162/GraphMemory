# Retrieval contracts

Root constraint: [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Concrete requests

| Request | Meaning | Consumers |
| --- | --- | --- |
| `TextRankingRequest` | query plus flat text candidates | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | query, candidates, and the Registry-assembled `EntityKnowledgeGraph` | GraphRAG |
| `EvidenceGraphRankingRequest` | query, candidates, `EvidenceGraph`, and initial scores | two R-GCN methods |
| `ExecutionProvenanceRankingRequest` | query, retrievable candidates, and native `ExecutionProvenanceGraph` | stateless and R-GCN Execution-Provenance retrievers |

There is no generic graph request. A provenance request cannot be routed to an EvidenceGraph R-GCN, and an evidence request cannot be routed to either provenance retriever.

## Registry matrix

| Method ID | Request | Families | Required artifact | Trainable |
| --- | --- | --- | --- | --- |
| `bm25` | `TextRankingRequest` | evidence, provenance | none | no |
| `dense` | `TextRankingRequest` | evidence, provenance | none | no |
| `dense_ft` | `TextRankingRequest` | evidence | none | encoder |
| `graphrag` | `GraphRAGRequest` | evidence, provenance | none | no |
| `dense_rgcn_graph_retriever` | `EvidenceGraphRankingRequest` | evidence | `EvidenceGraph` | yes |
| `dense_ft_rgcn_graph_retriever` | `EvidenceGraphRankingRequest` | evidence | `EvidenceGraph` | yes |
| `execution_provenance_retriever` | `ExecutionProvenanceRankingRequest` | provenance | request-native | no |
| `execution_provenance_rgcn_retriever` | `ExecutionProvenanceRankingRequest` | provenance | request-native | yes |

Builders accept concrete flat, GraphRAG, Evidence-RGCN, stateless-provenance, or Provenance-RGCN payloads. Registry validation checks payload class, request type, task family, checkpoint family, and required artifact before retrieval begins.

## Result contract

All methods return the full ranked candidate list. `retrieved_subgraph` remains the evidence-evaluation compatibility surface. GraphRAG and provenance methods place method-native actual traces under `metadata.native_trace`; entity edges are never presented as evidence or provenance edges.

`metadata.native_trace` is a closed union selected by `trace_kind`:

| Trace kind | Required content |
| --- | --- |
| `entity_search` | all entity IDs, query-linked IDs, seed IDs, and typed entity relations with candidate projections |
| `execution_provenance` | traced graph-context node IDs, selected paths with score components, and typed traversed edges with optional `feeds` binding |

Serialization rejects unknown kinds or fields, non-finite weights/scores, duplicate IDs, relations, paths, or edges, unknown endpoints, and path steps without a corresponding traced edge. The runtime value objects are `GraphRAGTrace` and `ExecutionProvenanceTrace`; there are no generic dictionary-list compatibility fields.
