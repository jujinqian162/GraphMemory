# Retrieval contracts

## Requests

| Request | Consumers |
|---|---|
| `TextRankingRequest` | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | GraphRAG |
| `EvidenceGraphRankingRequest` | Dense R-GCN, Dense-FT R-GCN |
| `ProvenancePathRequest` | training-free provenance path |
| `ProvenanceRgcnRequest` | trainable provenance R-GCN |

No generic graph request. Config resolution rejects unsupported dataset/method combinations, and each concrete method rejects the wrong request type at its direct boundary.

## Method matrix

| Method ID | Request | Families | Trainable |
|---|---|---|---|
| `bm25` | text | evidence, execution provenance | no |
| `dense` | text | evidence, execution provenance | no |
| `dense_ft` | text | evidence, execution provenance | encoder |
| `graphrag` | GraphRAG | evidence, execution provenance | no |
| `provenance_path` | ProvenancePath | execution provenance | no |
| `provenance_rgcn` | ProvenanceRgcn | execution provenance | yes |
| `dense_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |

The deleted label-conditioned provenance stack and legacy EPGM IDs remain retired. On ISETrace, BM25, Dense, Dense-FT, and GraphRAG share the same flat trajectory chunks; Dense-FT maps exact gold spans to overlapping chunks and never receives a provenance graph. `ProvenancePathRequest` and `ProvenanceRgcnRequest` are current, label-free method requests over the query-independent graph. The `provenance_rgcn` identity is not a compatibility alias: config resolution permits it only for execution provenance, and loading requires a strict current checkpoint whose method and model config match.

## Results

All methods return the full ranked candidate list.

- `retrieved_subgraph` is the shared evaluation surface for candidate-level structure.
- Method-native detail lives under `metadata.native_trace` with a closed `trace_kind` union, including `fast_graphrag_ppr` and `provenance_path`.
- Connector-only nodes never enter the ranked candidate list.
