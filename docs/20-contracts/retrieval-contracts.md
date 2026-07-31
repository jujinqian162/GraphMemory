# Retrieval contracts

## Requests

| Request | Consumers |
|---|---|
| `TextRankingRequest` | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | GraphRAG |
| `EvidenceGraphRankingRequest` | Dense R-GCN, Dense-FT R-GCN |
| `ProvenancePathRequest` | training-free provenance path |

No generic graph request. Cross-domain routing is rejected at Registry validation.

## Method matrix

| Method ID | Request | Families | Trainable |
|---|---|---|---|
| `bm25` | text | evidence, execution provenance | no |
| `dense` | text | evidence, execution provenance | no |
| `dense_ft` | text | evidence | encoder |
| `graphrag` | GraphRAG | evidence, execution provenance | no |
| `provenance_path` | ProvenancePath | execution provenance | no |
| `dense_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |

The deleted label-conditioned `ExecutionProvenanceRankingRequest`, legacy EPGM IDs, and Provenance R-GCN ID remain retired. ISETrace integration uses the new `ProvenancePathRequest` over the query-independent M2 graph and a new `provenance_path` identity; it is not a compatibility alias for the deleted stack.

## Results

All methods return the full ranked candidate list.

- `retrieved_subgraph` is the shared evaluation surface for candidate-level structure.
- Method-native detail lives under `metadata.native_trace` with a closed `trace_kind` union, including `typed_local_bridge` and `provenance_path`.
- Connector-only nodes never enter the ranked candidate list.
