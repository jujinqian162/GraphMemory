# Retrieval contracts

## Requests

| Request | Consumers |
|---|---|
| `TextRankingRequest` | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | GraphRAG |
| `EvidenceGraphRankingRequest` | Dense R-GCN, Dense-FT R-GCN |

No generic graph request. Cross-domain routing is rejected at Registry validation.

## Method matrix

| Method ID | Request | Families | Trainable |
|---|---|---|---|
| `bm25` | text | evidence, future provenance | no |
| `dense` | text | evidence, future provenance | no |
| `dense_ft` | text | evidence, future flat supervision | encoder |
| `graphrag` | GraphRAG | evidence, future provenance | no |
| `dense_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |

The legacy `ExecutionProvenanceRankingRequest`, EPGM method IDs, and Provenance R-GCN method ID have been removed. They must not be revived as compatibility aliases when ISETrace is integrated.

## Results

All methods return the full ranked candidate list.

- `retrieved_subgraph` is the shared evaluation surface for candidate-level structure.
- Method-native detail lives under `metadata.native_trace` with a closed `trace_kind` union for currently implemented methods.
- Connector-only nodes never enter the ranked candidate list.
