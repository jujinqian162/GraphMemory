# Naming conventions

Names describe semantic ownership, not only shape.

- Use `EvidenceGraph*` for dataset-derived question/evidence artifacts and requests.
- Use `ExecutionProvenance*` for source-native agent/tool execution history.
- Use `GraphRAG*` for method-owned entity search.
- Use `TextRankingRequest` only for flat candidates.
- Use `EvidenceRgcn*` for R-GCN settings and build payloads.
- Use `native_trace` for method-specific typed paths/edges. Only contracted candidate-to-candidate dependency edges may enter `retrieved_subgraph.edges` for shared path metrics.

Public method IDs are exactly `bm25`, `dense`, `dense_ft`, `graphrag`, `dense_rgcn_graph_retriever`, `dense_ft_rgcn_graph_retriever`, `execution_provenance_retriever`, and `execution_provenance_rgcn_retriever`.

Use `Epgm*` for the non-trained execution-provenance retriever's config and method-owned algorithm types. The reported default is the query-conditioned PPR/connected-subgraph architecture; historical path strategies remain diagnostic `variant` values behind the same `execution_provenance_retriever` method ID, never separate public methods.

Workflow artifacts use `evidence_graphs` when referring to the prebuilt evidence graph stage or files. Generic `graphs` remains acceptable only as the Python package namespace containing graph domains.
