# Naming conventions

Names describe semantic ownership, not only shape.

- Use `EvidenceGraph*` for dataset-derived question/evidence artifacts and requests.
- Use `ExecutionProvenance*` for source-native agent/tool execution history.
- Use `GraphRAG*` for method-owned entity search.
- Use `TextRankingRequest` only for flat candidates.
- Use `EvidenceRgcn*` for R-GCN settings and build payloads.
- Use `native_trace` for method-specific actual paths/edges; never place entity or provenance edges in `retrieved_subgraph.edges`.

Public method IDs are exactly `bm25`, `dense`, `dense_ft`, `graphrag`, `dense_rgcn_graph_retriever`, `dense_ft_rgcn_graph_retriever`, and `execution_provenance_retriever`.

Workflow artifacts use `evidence_graphs` when referring to the prebuilt evidence graph stage or files. Generic `graphs` remains acceptable only as the Python package namespace containing graph domains.
