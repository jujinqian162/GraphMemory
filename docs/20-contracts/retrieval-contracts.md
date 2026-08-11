# Retrieval contracts

## Requests

| Request | Consumers |
|---|---|
| `TextRankingRequest` | BM25, Dense, Dense-FT |
| `GraphRAGRequest` | GraphRAG |
| `EvidenceGraphRankingRequest` | Dense R-GCN, Dense-FT R-GCN |
| `ExecutionProvenanceRankingRequest` | provenance path, provenance R-GCN |

No generic graph request. The two provenance methods share one label-free request because they consume the same query-independent graph; static method dispatch selects their different algorithms. Config resolution rejects unsupported dataset/method combinations, and each concrete method rejects the wrong request family at its direct boundary.

## Method matrix

| Method ID | Request | Families | Trainable |
|---|---|---|---|
| `bm25` | text | evidence, execution provenance | no |
| `dense` | text | evidence, execution provenance | no |
| `dense_ft` | text | evidence, execution provenance | encoder |
| `graphrag` | GraphRAG | evidence, execution provenance | no |
| `provenance_path` | execution provenance | execution provenance | no |
| `provenance_rgcn` | execution provenance | execution provenance | yes |
| `dense_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraph | evidence | yes |

The deleted label-conditioned provenance stack and legacy EPGM IDs remain retired. Dense and Dense-FT add a candidate-view variant without adding method IDs: `flat` is the default on every supported family, while `provenance_unit` is valid only on ISETrace. The provenance-unit variants rank the prepared source-backed content units through `TextRankingRequest`; they do not load or receive a provenance graph. Dense-FT maps exact gold spans to every overlapping candidate in the selected view, records the variant in pair/model/checkpoint identity, and rejects a checkpoint from the other view. `provenance_path` and `provenance_rgcn` alone consume `ExecutionProvenanceRankingRequest` over the query-independent graph. The `provenance_rgcn` identity is not a compatibility alias: config resolution permits it only for execution provenance, and loading requires a strict current checkpoint whose method and model config match.

## Results

All methods return the full ranked candidate list.

- `retrieved_subgraph` is the shared evaluation surface for candidate-level structure.
- Method-native detail lives under `metadata.native_trace` with a closed `trace_kind` union, including `fast_graphrag_ppr` and `provenance_path`.
- Connector-only nodes never enter the ranked candidate list.
