# Retrieval contracts

## Requests

| Request | Consumers |
|---|---|
| `TextRankingRequest` | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | GraphRAG |
| `EvidenceGraphRankingRequest` | Dense R-GCN, Dense-FT R-GCN |
| `ProvenancePathRequest` | training-free provenance path |
| `ProvenanceRgcnRequest` | trainable provenance R-GCN |

No generic graph request. Cross-domain routing is rejected at Registry validation.

## Method matrix

| Method ID | Request | Families | Trainable |
|---|---|---|---|
| `bm25` | text | evidence, execution provenance | no |
| `dense` | text | evidence, execution provenance | no |
| `dense_ft` | text | evidence | encoder |
| `graphrag` | GraphRAG | evidence, execution provenance | no |
| `provenance_path` | ProvenancePath | execution provenance | no |
| `provenance_rgcn` | ProvenanceRgcn | execution provenance | yes |
| `dense_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |

The deleted label-conditioned provenance stack and legacy EPGM IDs remain retired. `ProvenancePathRequest` and `ProvenanceRgcnRequest` are current, label-free method requests over the query-independent graph. The new `provenance_rgcn` identity is not a compatibility alias: Registry accepts it only for execution provenance and loads only a strict current checkpoint whose method and model config match.

## Results

All methods return the full ranked candidate list.

- `retrieved_subgraph` is the shared evaluation surface for candidate-level structure.
- Method-native detail lives under `metadata.native_trace` with a closed `trace_kind` union, including `fast_graphrag_ppr` and `provenance_path`.
- Connector-only nodes never enter the ranked candidate list.
