# Retrieval contracts

## Requests

| Request | Consumers |
|---|---|
| `TextRankingRequest` | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | GraphRAG |
| `EvidenceGraphRankingRequest` | Dense R-GCN, Dense-FT R-GCN |
| `ExecutionProvenanceRankingRequest` | EPGM, Provenance R-GCN |

No generic graph request. Cross-domain routing is rejected at Registry validation.

## Method matrix

| Method ID | Request | Families | Trainable |
|---|---|---|---|
| `bm25` | text | evidence, provenance | no |
| `dense` | text | evidence, provenance | no |
| `dense_ft` | text | both (flat supervised) | encoder |
| `graphrag` | GraphRAG | evidence, provenance | no |
| `dense_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `execution_provenance_retriever` | provenance | provenance | no |
| `execution_provenance_rgcn_retriever` | provenance | provenance | yes |

EPGM is one implementation. Reported default variant: `ppr_steiner`. `typed_beam` and `dependency_path` are historical diagnostics under the same method id.

## Results

All methods return the full ranked candidate list.

- `retrieved_subgraph` is the shared evaluation surface for collapsed candidate-level structure.
- Method-native detail lives under `metadata.native_trace` with a closed `trace_kind` union (`typed_local_bridge`, `execution_provenance`, `execution_provenance_local`, `execution_provenance_subgraph`).
- Provenance collapsed edges use logical type `feeds`; finer relations stay in the native trace.
- Connector-only nodes never enter the ranked candidate list.
