# Retrieval contracts

Root constraint: [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Concrete requests

| Request | Meaning | Consumers |
| --- | --- | --- |
| `TextRankingRequest` | query plus flat text candidates | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | query, candidates, and the Registry-assembled typed mentions/title groups | GraphRAG |
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

`execution_provenance_retriever` is a single non-trained implementation; its `variant` field picks a frozen preset (`typed_beam` default, `dependency_path` ablation) rather than a different method.

Builders accept concrete flat, GraphRAG, Evidence-RGCN, stateless-provenance, or Provenance-RGCN payloads. Registry validation checks payload class, request type, task family, checkpoint family, and required artifact before retrieval begins.

## Result contract

All methods return the full ranked candidate list. `retrieved_subgraph` remains the evidence-evaluation compatibility surface. GraphRAG and provenance methods place method-native actual traces under `metadata.native_trace`; entity edges are never presented as evidence or provenance edges.

`metadata.native_trace` is a closed union selected by `trace_kind`:

| Trace kind | Required content |
| --- | --- |
| `typed_local_bridge` | Dense ranks, linked entities, typed mentions/title groups, sentence resolver evidence, local bridge proposals, gates, displacement, fallback identity, and emitted promotion edges |
| `execution_provenance` | existing selected-path trace used by the trainable provenance R-GCN path |
| `execution_provenance_local` | Dense ranks, bounded path proposals over existing edge weights, structural gates, displacement, fallback identity, scorer identity, active EPGM `variant`, and emitted promotion edges |

Serialization rejects unknown kinds or fields, non-finite weights/scores, duplicate IDs, paths, or edges, unknown endpoints, and path steps without a corresponding traced edge. Stateless traces report only accepted/rejected local interventions; when every proposal abstains, the ranked nodes and scores are byte-for-byte Dense identity.

Collapsed candidate-level edges in `retrieved_subgraph` always use the `feeds` edge type, the only provenance dependency type in `ALLOWED_EDGE_TYPES`; the finer traversed relation is reported per edge in `native_trace.emitted_edges`. A bidirectional walk orients each collapsed edge along the stored graph direction, so a mostly-reverse path is flipped rather than emitted as an inverted dependency.
