# Debug artifacts

Failure cases and metric tables are written under each run's `debug/` and `metrics/` directories. R-GCN diagnostics use EvidenceGraph node/edge identifiers. GraphRAG diagnostics use the `entity_search` relation records from `metadata.native_trace`; provenance diagnostics use the `execution_provenance` selected paths, score components, and traversed typed edges from the same metadata namespace.

Do not reinterpret one trace type as another. In particular, GraphRAG entity edges are not evidence dependency labels, and an induced top-k evidence subgraph is not an execution path.
